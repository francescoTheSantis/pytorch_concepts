"""ParametricPotential — undirected, energy-based factor over a clique."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn

from .factor import ParametricFactor
from ..variable import Variable


class GaussianFourierEmbedding(nn.Module):
    """``sigma -> [sin(2*pi*W*log sigma), cos(2*pi*W*log sigma)]``.

    Turns the noise level of a conditional energy into something a network can
    actually use. Fed in raw, ``sigma`` is one scalar among the scope's features and
    is easy to ignore; worse, two adjacent rungs of a geometric ladder differ by a
    few percent, while the scores they demand differ completely. Projecting onto
    random high frequencies makes nearby levels far apart in the embedding, which is
    what lets one set of weights cover the whole ladder.

    ``W`` is a fixed buffer, not a parameter — learning it is the standard failure
    mode, since gradient descent shrinks the frequencies back down and undoes the
    separation. This is the score-SDE / NCSN++ default.

    Args:
        size (int): Output width. Rounded down to even (half sine, half cosine).
        scale (float): Standard deviation of ``W``; larger means higher frequencies.
    """

    def __init__(self, size: int = 32, scale: float = 16.0):
        super().__init__()
        self.size = int(size) // 2 * 2
        self.register_buffer("frequencies", torch.randn(self.size // 2) * scale)

    def extra_repr(self) -> str:
        return f"size={self.size}"

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """``(...,)`` or a scalar -> ``(..., size)``."""
        projected = 2 * math.pi * sigma.reshape(-1, 1).log() * self.frequencies
        return torch.cat([projected.sin(), projected.cos()], dim=-1)


class ParametricPotential(ParametricFactor):
    """Undirected, energy-based factor over a ``scope``.

    Parameters
    ----------
    scope : list of Variable
        The clique of variables the energy ranges over (>= 1). 
    parametrization : nn.Module or dict[str, nn.Module]
        The energy module. It receives the aggregated scope values, concatenated
        along the last dim in ``scope`` order, and must return **one scalar per
        leading (batch-like) element** (shape ``(*leading,)`` or
        ``(*leading, 1)``). Any unbuilt :class:`~torch_concepts.nn.modules.low.lazy.LazyConstructor`
        entry is instantiated here, sized from ``scope`` with ``out_concepts=1``
        (the energy module always emits a single scalar, not a per-variable
        parameter).
    name : str, optional
        Factor-graph key. Defaults to ``"phi(<scope names>)"``.
    aggregate : callable or dict, optional
        As in :class:`ParametricFactor`; aggregates the inputs before the energy
        module (default: concatenate along the last dim).
    noise_conditioned : bool, default False
        Make this a conditional energy ``E(scope, sigma)``. :meth:`energy` then
        requires a ``sigma=`` keyword, embeds it with
        :class:`GaussianFourierEmbedding`, and concatenates the embedding onto the
        aggregated scope values — so the energy module stays an ordinary
        ``nn.Sequential`` and never sees ``sigma`` as a separate argument. Its input
        width grows by ``noise_embedding_size``.

        This is what a score-matching objective needs: the score at a coarse noise
        level and at a fine one are different functions, and one unconditioned
        network would have to average them.
    noise_embedding_size : int, default 32
        Width of that embedding. Ignored when ``noise_conditioned`` is False.

    Raises
    ------
    ValueError
        If ``scope`` is empty, or ``parametrization`` is a dict without exactly
        the key ``'energy'``.
    TypeError
        If a scope entry is not a :class:`Variable`, or ``parametrization`` is
        neither an ``nn.Module`` nor a dict.
    """

    def __init__(
        self,
        scope: List[Variable],
        parametrization: Union[nn.Module, Dict[str, nn.Module]],
        name: Optional[str] = None,
        aggregate=None,
        noise_conditioned: bool = False,
        noise_embedding_size: int = 32,
    ) -> None:
        if not isinstance(scope, (list, tuple)) or not scope:
            raise ValueError("ParametricPotential: `scope` must be a non-empty list of Variables.")
        for v in scope:
            if not isinstance(v, Variable):
                raise TypeError(
                    f"ParametricPotential: every scope entry must be a Variable, "
                    f"got {type(v).__name__}."
                )
        self._scope: List[Variable] = list(scope)
        # The scope is exactly the aggregation input. (An undirected factor has
        # no parents — the base class asks only for ``inputs``.)
        self.inputs: List[Variable] = list(self._scope)
        self._name: str = name if name is not None else self._default_name()

        parametrization = self._instantiate_lazy(
            self._normalize_parametrization(parametrization),
            self._scope,
            noise_embedding_size if noise_conditioned else 0,
        )
        super().__init__(
            parametrization=parametrization,
            aggregate=aggregate,
        )
        # After ``super().__init__``: this is an nn.Module and cannot register a
        # submodule before it is initialised.
        self.noise_embedding = (
            GaussianFourierEmbedding(noise_embedding_size)
            if noise_conditioned else None
        )

    def _default_name(self) -> str:
        return f"phi({','.join(v.name for v in self._scope)})"

    @staticmethod
    def _instantiate_lazy(
        parametrization: Dict[str, nn.Module],
        scope: List[Variable],
        noise_embedding_size: int = 0,
    ) -> Dict[str, nn.Module]:
        """Build any unbuilt :class:`LazyConstructor` entries into concrete modules.

        Returns ``parametrization`` unchanged when there is nothing to build —
        the common, eagerly-constructed case skips all the work below.

        A :class:`LazyConstructor` defers module creation until the input/output
        sizes are known; those sizes come from this potential's ``scope``:

        * ``in_concepts``   — summed size of the ``"concept"`` scope variables;
        * ``in_embeddings`` — summed size of the ``"embedding"`` scope variables;
        * ``out_concepts``  — always ``1``: unlike a CPD's per-parameter output,
          the energy module produces a single scalar per leading element, not a
          value sized to any particular variable.

        ``noise_embedding_size`` (0 when unconditioned) is added to
        ``in_embeddings``, because a conditional energy sees the noise embedding
        concatenated onto the scope values.
        """
        from ...low.lazy import LazyConstructor

        # Fast path: every module is already a concrete layer — nothing to build.
        if not any(
            isinstance(m, LazyConstructor) and m.module is None
            for m in parametrization.values()
        ):
            return parametrization

        in_concepts = sum(v.size for v in scope if v.variable_type == "concept")
        in_embeddings = sum(v.size for v in scope if v.variable_type == "embedding")
        in_embeddings += noise_embedding_size

        resolved: Dict[str, nn.Module] = {}
        for pname, module in parametrization.items():
            if isinstance(module, LazyConstructor) and module.module is None:
                module = module.build(
                    out_concepts=1,
                    in_concepts=in_concepts or None,
                    in_embeddings=in_embeddings or None,
                )
            resolved[pname] = module
        return resolved

    def _normalize_parametrization(
        self, parametrization: Union[nn.Module, Dict[str, nn.Module]]
    ) -> Dict[str, nn.Module]:
        """Normalize the parametrization to a dict with exactly the key 'energy'."""
        if isinstance(parametrization, nn.Module):
            return {"energy": parametrization}
        if isinstance(parametrization, dict):
            if set(parametrization) != {"energy"}:
                raise ValueError(
                    "ParametricPotential: parametrization dict must have exactly the key "
                    f"'energy', got {sorted(parametrization)}."
                )
            return parametrization
        raise TypeError(
            "ParametricPotential: `parametrization` must be an nn.Module or a "
            "{'energy': nn.Module} dict."
        )

    # ---- unified factor-graph interface (see ParametricFactor) --------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def scope(self) -> List[Variable]:
        return list(self._scope)

    def energy(
        self,
        scope_values: Mapping[Variable, torch.Tensor],
        **layer_kwargs,
    ) -> torch.Tensor:
        """Scalar energy ``E(scope)`` of shape ``(*leading,)``.

        Aggregates the scope values and applies the energy module. The leading
        (batch-like) dimensions are read off the aggregated input, whose last
        axis is the feature axis, so the energy is reduced to one scalar per
        leading element whether the module emits ``(*leading,)`` or
        ``(*leading, 1)``.

        When ``noise_conditioned=True`` a ``sigma=`` keyword is required and is
        **consumed here**: it is embedded and concatenated onto the aggregated
        values, so the energy module receives one ordinary input tensor and never a
        ``sigma`` argument. Otherwise ``layer_kwargs`` pass through untouched, which
        is how a module that wants to handle ``sigma`` itself still can.
        """
        inputs: Dict[Variable, torch.Tensor] = {v: scope_values[v] for v in self._scope}

        mod = self.parametrization["energy"]
        cat = self._aggregators["energy"](inputs)
        if self.noise_embedding is not None:
            sigma = layer_kwargs.pop("sigma", None)
            if sigma is None:
                raise ValueError(
                    f"{type(self).__name__} {self._name!r} is noise-conditioned and "
                    "needs a `sigma=` keyword; call energy(..., sigma=s) or "
                    "MarkovNetwork.compute_score(..., sigma=s)."
                )
            cat = self._append_noise(cat, sigma)
        if isinstance(cat, dict):
            leading = next(iter(cat.values())).shape[:-1]
            out = mod(**cat, **layer_kwargs)
        else:
            leading = cat.shape[:-1]
            out = mod(cat, **layer_kwargs)
        return out.reshape(*leading)

    def _append_noise(self, cat, sigma: torch.Tensor):
        """Concatenate the embedded noise level onto the aggregated scope values.

        ``sigma`` may be a scalar (one level for the whole batch) or one value per
        leading element; both broadcast to the aggregate's leading shape, so a
        training step that draws a different level per example costs nothing extra.
        """
        if isinstance(cat, dict):
            raise TypeError(
                f"{type(self).__name__} {self._name!r}: noise conditioning needs an "
                "aggregator that concatenates into one tensor, but this one emitted "
                "a dict. Fold the noise level in inside the energy module instead."
            )
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, dtype=cat.dtype, device=cat.device)
        embedded = self.noise_embedding(sigma.to(dtype=cat.dtype, device=cat.device))
        return torch.cat([cat, embedded.expand(*cat.shape[:-1], embedded.shape[-1])], dim=-1)

    def log_potential(
        self,
        assignment: Mapping[Variable, torch.Tensor],
    ) -> torch.Tensor:
        """``-E(scope)`` at the given assignment (shape ``(*leading,)``).

        ``assignment`` must cover the whole scope: free variables at the value
        being scored, observed ones (an embedding, say) at their evidence. A
        superset of keys is fine — :meth:`energy` reads only the scope entries.
        """
        return -self.energy(assignment)

    def forward(
        self,
        inputs: Optional[Mapping[str, torch.Tensor]] = None,
        **layer_kwargs,
    ) -> torch.Tensor:
        """Energy for a name-keyed ``inputs`` dict holding every scope value.
        Returns ``(*leading,)``."""
        inputs = inputs or {}
        scope_values = {v: self.resolve_value(v, inputs) for v in self._scope}
        return self.energy(scope_values, **layer_kwargs)
