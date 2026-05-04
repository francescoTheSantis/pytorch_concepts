"""
ELBOInference — ELBO-based training engine for Pyro-backed probabilistic models.

Wraps Pyro's ``ELBOModule`` and exposes a ``query()`` method that:

1. Returns an :class:`~torch_concepts.nn.modules.outputs.InferenceOutput` with
   marginal predictions for the queried variables (for monitoring /
   visualization).
2. Attaches the **negative ELBO** as ``result.loss`` so the training loop is
   as close as possible to the non-latent case::

       # Non-latent (no hidden variables)
       result = det_inference.query(vars, evidence={'input': x}, return_parameters=True)
       loss = loss_fn(result.parameters, targets)
       loss.backward()

       # With latents (ELBO training)
       result = elbo_engine.query(vars, evidence={'input': x},
                                  targets={'task': y})
       result.loss.backward()   # negative ELBO already computed internally

The :meth:`log_prob` / :meth:`forward` convenience methods are retained for
backward compatibility.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Type

import torch
import torch.nn as nn

import pyro
import pyro.infer
from pyro.nn import PyroModule

from ...outputs import InferenceOutput


class ELBOInference(nn.Module):
    """
    ELBO-based training engine for a Pyro-backed :class:`ProbabilisticModel`.

    Constructs a Pyro ``ELBOModule`` internally.  Parameters from both the
    model and the guide are accessible through ``engine.parameters()`` so a
    single optimiser can train everything.

    The primary interface is :meth:`query`, which returns an
    :class:`~torch_concepts.nn.modules.outputs.InferenceOutput` with:

    * ``result.parameters`` / ``result.probs``: posterior-predictive marginals
      from the trained guide (useful for monitoring; **no gradient** attached
      since they come from a ``torch.no_grad()`` forward pass).
    * ``result.loss``: the **negative ELBO** (differentiable).  Call
      ``result.loss.backward()`` in the training loop.

    Parameters
    ----------
    model : ProbabilisticModel
        The Pyro-backed probabilistic model.
    guide : PyroModule or type, optional
        A Pyro guide (variational posterior).  Pass an instance or an
        AutoGuide class (e.g. ``pyro.infer.autoguide.AutoNormal``).  If
        ``None``, an :class:`AmortizedGuide` is built automatically.
    elbo : pyro ELBO class, optional
        The ELBO estimator class.  Defaults to ``pyro.infer.Trace_ELBO``.
    num_particles : int
        Number of Monte-Carlo particles for the ELBO estimator.  Default: 1.

    Example
    -------
    .. code-block:: python

        pyro.settings.set(module_local_params=True)

        model  = ProbabilisticModel(variables, factors)
        engine = ELBOInference(model, num_particles=4)
        optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)

        for x_batch, c_batch, y_batch in dataloader:
            optimizer.zero_grad()
            result = engine.query(
                ['c1', 'c2', 'task'],
                evidence={'input': x_batch},
                targets={'c1': c_batch, 'task': y_batch},
            )
            result.loss.backward()   # negative ELBO
            optimizer.step()
    """

    def __init__(
        self,
        model: 'ProbabilisticModel',  # noqa: F821
        guide=None,
        elbo=None,
        num_particles: int = 1,
    ) -> None:
        super().__init__()

        # Ensure module-local params so .parameters() works correctly
        pyro.settings.set(module_local_params=True)

        if guide is None:
            from .guide import AmortizedGuide
            guide = AmortizedGuide(model)
        elif isinstance(guide, type):
            guide = guide(model)

        if elbo is None:
            elbo = pyro.infer.Trace_ELBO

        self.elbo_module = elbo(num_particles=num_particles)(model, guide)

    # ------------------------------------------------------------------
    # Primary API: query()
    # ------------------------------------------------------------------

    def query(
        self,
        query: List[str],
        evidence: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        return_parameters: bool = False,
        return_probs: bool = True,
    ) -> InferenceOutput:
        """
        Run ELBO forward and return marginal predictions + training loss.

        The returned :class:`~torch_concepts.nn.modules.outputs.InferenceOutput`
        has:

        * ``result.loss``: the **negative ELBO** (differentiable scalar).
          Call ``result.loss.backward()`` in the training loop.
        * ``result.parameters`` / ``result.probs``: posterior-predictive marginals
          from the guide (no gradient — useful for monitoring).

        Parameters
        ----------
        query : list of str
            Variable names whose marginal predictions to return.
        evidence : dict
            Root / always-observed variables.
        targets : dict, optional
            Additional observed tensors (concept/task labels) used for
            conditioning.  Merged with *evidence* before ELBO computation.
        return_parameters : bool
            If ``True``, populate ``result.parameters`` (raw CPD output).
        return_probs : bool
            If ``True`` (default), populate ``result.probs`` (activated).

        Returns
        -------
        InferenceOutput
            ``result.loss`` always set; ``result.parameters`` / ``result.probs``
            set according to the ``return_*`` flags.
        """
        # 1. Compute the negative ELBO (differentiable — used as training loss)
        neg_elbo = self.elbo_module(evidence, targets)

        # 2. Collect posterior-predictive marginals (no gradient needed here)
        parameters_parts: List[torch.Tensor] = []
        probs_parts: List[torch.Tensor] = []

        if return_parameters or return_probs:
            model = self.elbo_module.model   # the ProbabilisticModel
            guide = self.elbo_module.guide   # the fitted guide

            with torch.no_grad():
                # Sample latents from the guide and build a context dict
                guide_ctx = guide(evidence, targets)

            for name in query:
                var = model.concept_to_variable.get(name)
                if var is None:
                    continue
                cpd = model.get_module_of_concept(name)
                if cpd is None:
                    continue

                with torch.no_grad():
                    raw = model._run_cpd(cpd, guide_ctx, {**evidence, **(targets or {})})

                if return_parameters:
                    if isinstance(raw, dict):
                        parameters_parts.append(torch.cat(list(raw.values()), dim=-1))
                    else:
                        parameters_parts.append(raw)
                if return_probs:
                    probs_parts.append(var.make_distribution(raw).mean)

        out_params = torch.cat(parameters_parts, dim=-1) if parameters_parts else None
        out_probs  = torch.cat(probs_parts,  dim=-1) if probs_parts  else None

        return InferenceOutput(parameters=out_params, probs=out_probs, loss=neg_elbo)

    # ------------------------------------------------------------------
    # Convenience: backward-compat log_prob / forward
    # ------------------------------------------------------------------

    def log_prob(
        self,
        evidence: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute the ELBO (Evidence Lower BOund) — positive scalar.

        Prefer :meth:`query` in new code.
        """
        return -self.elbo_module(evidence, targets)

    def forward(
        self,
        evidence: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Backward-compatible alias — returns ``-log_prob`` (negative ELBO)."""
        return -self.log_prob(evidence, targets)


# ---------------------------------------------------------------------------
# Trivial empty guide (kept for internal use / testing)
# ---------------------------------------------------------------------------

class _EmptyGuide(nn.Module):
    """Guide that samples nothing."""

    def forward(self, data):
        pass
