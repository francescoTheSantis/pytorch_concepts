"""MarkovNetwork: an undirected probabilistic graphical model (Markov random field)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import torch

from ..factors.potential import ParametricPotential
from .probabilistic_model import ProbabilisticModel
from ..variable import Variable


class MarkovNetwork(ProbabilisticModel):
    """Undirected graphical model (Markov random field) over energy-based potentials.

    The undirected special case of :class:`ProbabilisticModel`: a list of
    :class:`Variable`s wired to a list of undirected :class:`ParametricPotential`
    factors. The joint is ``p(x) ∝ exp(-Σ_f E_f(scope_f ; conditioning_f))``. There
    is no topological order and no per-variable "one factor" constraint — a variable
    may appear in any number of potentials.

    Use :class:`BayesianNetwork` for directed models and a plain
    :class:`ProbabilisticModel` for mixed (partially-directed / chain) graphs.
    Inference is via :class:`BeliefPropagation`.

    Parameters
    ----------
    variables : list of Variable
        All random variables in the model. Names must be unique.
    factors : list of ParametricPotential
        The undirected potentials. Every variable in a potential's ``scope`` must
        be one of ``variables`` (the same object). Conditioning inputs (e.g. an
        embedding) are observed and are *not* part of any scope.
    """

    def __init__(
        self,
        variables: List[Variable],
        factors: List[ParametricPotential],
    ):
        factors = list(factors)
        for f in factors:
            if not isinstance(f, ParametricPotential):
                raise TypeError(
                    "MarkovNetwork factors must be ParametricPotential instances "
                    f"(got {type(f).__name__}). Use a BayesianNetwork for directed "
                    "models, or a plain ProbabilisticModel for mixed (chain) graphs."
                )
        # ProbabilisticModel registers factors ({potential name: potential}),
        # validates scopes, and builds the bipartite adjacency. The undirected
        # scope validation is exactly the base one, so nothing is overridden.
        super().__init__(variables, factors)

    def energy(
        self,
        values: Any,
        factors: Optional[List[str]] = None,
        **layer_kwargs,
    ) -> torch.Tensor:
        """Total energy ``Σ_f E_f`` of a joint assignment, shape ``(*leading,)``.

        The unnormalized negative log-density: ``p(x) ∝ exp(-energy(x))``. The
        partition function is never touched, which is the whole point — a
        gradient-based sampler needs only ``∇ energy``, and contrastive divergence
        needs only energy *differences*.

        Parameters
        ----------
        values : dict, InferenceOutput or AnnotatedTensor
            One value per scope variable, keyed by name, shaped
            ``(*leading, *event)``. A superset is fine — each potential reads only
            its own scope. An engine's output is split by its annotation labels, so
            CD reads as ``mrf.energy(sampler.query(query=batch))``.
        factors : list of str, optional
            Restrict the sum to these potential names. Defaults to every potential.
        **layer_kwargs
            Forwarded to every potential's energy module, which
            :meth:`ParametricPotential.energy` already accepts. This is the channel
            a *conditional* energy needs — an NCSN-style ``E(x, sigma)`` reads its
            noise level through it, so one set of clique nets covers the whole
            noise ladder.

        Returns
        -------
        torch.Tensor
            Shape ``(*leading,)``.

        Notes
        -----
        A Markov network's variables are never plates, so that is the
        only address a value can have — no member/plate resolution to do.
        """
        values = self.as_mapping(values)
        names = list(self._factors) if factors is None else factors
        return sum(
            self._factors[name].energy(
                {v: values[v.name] for v in self._factors[name].scope}, **layer_kwargs
            )
            for name in names
        )

    @staticmethod
    def as_mapping(values: Any) -> Dict[str, torch.Tensor]:
        """Coerce an engine's output into the name-keyed mapping the model wants.

        A mapping passes through. Anything else is taken to be an
        :class:`~torch_concepts.nn.modules.outputs.InferenceOutput` — unwrapped to
        its ``samples`` — or an :class:`~torch_concepts.AnnotatedTensor` already, and
        is split into one block per annotation label. That is what lets a sampler's
        output be scored directly, without the caller re-keying it by hand.
        """
        if isinstance(values, Mapping):
            return values
        samples = getattr(values, "samples", values)
        return {n: samples[[n]].tensor for n in samples.annotation.labels}

    def compute_score(
        self,
        values: Any,
        factors: Optional[List[str]] = None,
        create_graph: bool = True,
        **layer_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """The score ``s(x) = -∇_x E(x)``, name-keyed like :meth:`energy`'s input.

        The quantity every gradient-based method over this model actually consumes:
        a Langevin step follows it, and denoising score matching regresses onto it.
        Because ``E = Σ_c E_c`` is factorized, ``∂E/∂x_i`` collects only the cliques
        containing ``i`` — so the score of a variable depends on nothing outside its
        Markov blanket, and the graph structure survives into the gradient for free.

        Parameters
        ----------
        values : dict, InferenceOutput or AnnotatedTensor
            As :meth:`energy`. Must cover every variable whose score is wanted.
        factors : list of str, optional
            Restrict the energy to these potentials before differentiating.
        create_graph : bool, default True
            Keep the graph so the result stays differentiable w.r.t. the parameters.
            Score matching needs this — its loss differentiates *through* the score,
            which is the one double-backward a training step costs. A sampler only
            reads the value and should pass ``False``.
        **layer_kwargs
            Forwarded to :meth:`energy`; an NCSN-style conditional energy takes its
            noise level this way, as ``compute_score(values, sigma=s)``.

        Returns
        -------
        dict
            ``{name: -dE/dx_name}``, each entry shaped like its input.

        Notes
        -----
        The inputs are **detached** first. The score is a property of the *point*,
        so gradients w.r.t. whatever computed that point are not part of its
        definition — and leaving them attached would silently extend a training
        graph through the sampler that produced the point.
        """
        values = self.as_mapping(values)
        # ``enable_grad`` for the same reason the Langevin chain needs it: the score
        # is required even when the caller wants nothing differentiated.
        with torch.enable_grad():
            points = {
                name: value.detach().requires_grad_(True)
                for name, value in values.items()
            }
            energy = self.energy(points, factors, **layer_kwargs)
            grads = torch.autograd.grad(
                energy.sum(), list(points.values()), create_graph=create_graph
            )
        return {name: -grad for name, grad in zip(points, grads)}
