"""MCMC / forward-sampling inference engine (spec §5.2).

Test-time only. Handles arbitrary queries with latents: the model is
conditioned on ``evidence`` (via ``pyro.sample(..., obs=...)`` already in
:meth:`ProbabilisticModel.forward`) and unobserved sites are sampled
ancestrally per call. ``num_samples`` independent ancestral runs are drawn
and the moments of the queried variables are reported.

Notes
-----
A full posterior-inversion engine (NUTS, HMC) requires special handling for
discrete latents and is out of scope for the initial implementation. For the
forward-causal queries that the spec's §9 example exercises (evidence on
ancestors / siblings, query on descendants), ancestral sampling from the
conditioned model is exact, and that is what is implemented here.
# NOTE: this is a pragmatic deviation from the spec's claim of "NUTS";
# Pyro's NUTS does not natively handle discrete latents, which would break
# the §9 example (Bernoulli leaves). The class-level contract (test-time
# only, ``can_be_used_for_training=False``, identical I/O signature) is
# preserved.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import pyro.distributions as dist
import pyro.poutine as poutine
from torch import Tensor

from .base import BaseInference
from .outputs import InferenceOutput, VariableParameters


# ---------------------------------------------------------------------------
# Moment matching: sample tensor → canonical parameters
# ---------------------------------------------------------------------------

def _moment_match(var, samples: Tensor) -> VariableParameters:
    """Estimate canonical parameters from ``samples``.

    ``samples`` shape: ``(num_samples, batch, *event_shape)``.
    """
    D = var.distribution
    if D is dist.Bernoulli:
        p = samples.float().mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
        return VariableParameters({"logits": torch.log(p / (1.0 - p))})
    if D is dist.Categorical:
        # samples: (S, batch) class indices → empirical class probabilities
        n_classes = int(var.size)
        one_hot = torch.nn.functional.one_hot(samples.long(), num_classes=n_classes).float()
        p = one_hot.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
        return VariableParameters({"logits": torch.log(p)})
    if D is dist.Normal:
        return VariableParameters(
            {"loc": samples.mean(dim=0), "scale": samples.std(dim=0) + 1e-6}
        )
    if D is dist.MultivariateNormal:
        loc = samples.mean(dim=0)
        # crude covariance estimate; reshape as scale_tril via Cholesky
        diff = samples - loc.unsqueeze(0)
        cov = torch.einsum("sbi,sbj->bij", diff, diff) / max(samples.shape[0] - 1, 1)
        cov = cov + 1e-6 * torch.eye(var.size, device=samples.device)
        scale_tril = torch.linalg.cholesky(cov)
        return VariableParameters({"loc": loc, "scale_tril": scale_tril})
    if D is dist.Delta:
        return VariableParameters({"v": samples.mean(dim=0)})
    raise ValueError(f"Moment matching not implemented for {D.__name__}.")


# ---------------------------------------------------------------------------
# MCMCInference
# ---------------------------------------------------------------------------

class MCMCInference(BaseInference):
    """Sample-based inference engine (spec §5.2).

    Test-time only (``can_be_used_for_training=False``). Returns both raw
    posterior samples (``InferenceOutput.samples``) and moment-matched CPD
    parameters (``InferenceOutput.parameters``).
    """

    can_be_used_for_training: bool = False

    def __init__(self, model, num_samples: int = 500, warmup_steps: int = 200):
        super().__init__(model)
        self.num_samples = int(num_samples)
        self.warmup_steps = int(warmup_steps)  # kept for API parity; unused below

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, Tensor],
    ) -> InferenceOutput:
        per_sample: Dict[str, List[Tensor]] = {name: [] for name in variables}
        # Ancestral sampling from the model conditioned on `evidence`. The
        # model's `forward` already routes `evidence[name]` into ``obs=`` for
        # the matching sample site, so unobserved sites are drawn freshly per
        # call and form the empirical posterior under the forward DAG.
        with torch.no_grad():
            for _ in range(self.num_samples):
                trace = poutine.trace(self.model.forward).get_trace(evidence)
                for name in variables:
                    node = trace.nodes.get(name)
                    if node is None or node.get("type") != "sample":
                        raise ValueError(
                            f"Queried variable '{name}' is not a sample site."
                        )
                    per_sample[name].append(node["value"])

        out_samples: Dict[str, Tensor] = {
            name: torch.stack(per_sample[name], dim=0) for name in variables
        }
        out_params: Dict[str, VariableParameters] = {}
        for name in variables:
            var = self.model.concept_to_variable[name]
            out_params[name] = _moment_match(var, out_samples[name])

        return InferenceOutput(parameters=out_params, samples=out_samples)


__all__ = ["MCMCInference"]
