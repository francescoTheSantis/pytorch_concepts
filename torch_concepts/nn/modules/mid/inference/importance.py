"""Importance-sampling inference engine.

Wraps :class:`pyro.infer.Importance` to provide a test-time backward-query
engine that genuinely conditions upstream marginals on downstream evidence.

With ``guide=None`` (default) Pyro builds a proposal by blocking observed
sites in the model — i.e. each trace draws every latent ancestrally from
the prior, then the observed sites contribute their data likelihood as the
trace's importance weight.

Under :class:`ProbabilisticModel`'s mandatory data plate (spec §8 rule 11)
the *scalar* log-weight returned by Pyro mixes all batch elements into one
number, which is useless for per-example posterior moments. This engine
therefore re-derives **per-batch-element** log-weights from each trace by
summing site log-probs over the evidence keys, then self-normalises
across traces independently for each batch element.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import pyro.distributions as dist
from pyro.infer import Importance
from torch import Tensor

from .base import BaseInference
from .outputs import InferenceOutput, VariableParameters


# ---------------------------------------------------------------------------
# Weighted moment matching: weighted samples → canonical parameters
# ---------------------------------------------------------------------------

def _weighted_moment_match(
    var, samples: Tensor, weights: Tensor,
) -> VariableParameters:
    """Estimate canonical parameters from ``samples`` weighted by ``weights``.

    Parameters
    ----------
    var:
        :class:`Variable` whose distribution family dictates the canonical
        parameters to return.
    samples:
        Tensor of shape ``(S, batch, *event_shape)``.
    weights:
        Self-normalised weights of shape ``(S, batch)`` (each column sums
        to 1) or ``(S,)`` (broadcast across batch).
    """
    D = var.distribution

    # Broadcast weights to match samples for elementwise multiplication.
    if weights.dim() == 1:
        w = weights.view((-1,) + (1,) * (samples.dim() - 1))
    else:
        w = weights.view(weights.shape + (1,) * (samples.dim() - weights.dim()))
    w = w.to(samples.dtype)

    if D is dist.Bernoulli:
        p = (samples.float() * w).sum(dim=0).clamp(1e-6, 1.0 - 1e-6)
        return VariableParameters({"logits": torch.log(p / (1.0 - p))})
    if D is dist.Categorical:
        n_classes = int(var.size)
        one_hot = torch.nn.functional.one_hot(
            samples.long(), num_classes=n_classes,
        ).float()
        if weights.dim() == 1:
            w_oh = weights.view((-1,) + (1,) * (one_hot.dim() - 1)).to(one_hot.dtype)
        else:
            w_oh = weights.view(
                weights.shape + (1,) * (one_hot.dim() - weights.dim())
            ).to(one_hot.dtype)
        p = (one_hot * w_oh).sum(dim=0).clamp(1e-6, 1.0 - 1e-6)
        return VariableParameters({"logits": torch.log(p)})
    if D is dist.Normal:
        loc = (samples * w).sum(dim=0)
        var_ = (w * (samples - loc.unsqueeze(0)) ** 2).sum(dim=0)
        return VariableParameters({"loc": loc, "scale": var_.sqrt() + 1e-6})
    if D is dist.Delta:
        return VariableParameters({"v": (samples * w).sum(dim=0)})
    if D is dist.MultivariateNormal:
        loc = (samples * w).sum(dim=0)
        diff = samples - loc.unsqueeze(0)
        if weights.dim() == 1:
            cov = torch.einsum(
                "s,sbi,sbj->bij", weights.to(samples.dtype), diff, diff,
            )
        else:
            cov = torch.einsum(
                "sb,sbi,sbj->bij", weights.to(samples.dtype), diff, diff,
            )
        cov = cov + 1e-6 * torch.eye(var.size, device=samples.device)
        scale_tril = torch.linalg.cholesky(cov)
        return VariableParameters({"loc": loc, "scale_tril": scale_tril})
    raise ValueError(
        f"Weighted moment matching not implemented for {D.__name__}."
    )


# ---------------------------------------------------------------------------
# Per-batch log-weight extraction
# ---------------------------------------------------------------------------

def _per_batch_log_weight(
    trace, evidence_keys: List[str], batch_size: int,
) -> Tensor:
    """Sum per-site log-probs over the observed evidence sites of ``trace``.

    Returns a tensor of shape ``(batch_size,)``. Non-batch event axes are
    summed out before accumulation.
    """
    trace.compute_log_prob()
    out = torch.zeros(batch_size)
    for name in evidence_keys:
        node = trace.nodes.get(name)
        if node is None or node.get("type") != "sample":
            continue
        lp = node["log_prob"]
        while lp.dim() > 1:
            lp = lp.sum(dim=-1)
        if lp.dim() == 0:
            lp = lp.expand(batch_size)
        out = out + lp.to(out.dtype)
    return out


# ---------------------------------------------------------------------------
# ImportanceSamplingInference
# ---------------------------------------------------------------------------

class ImportanceSamplingInference(BaseInference):
    """Self-normalised importance sampling via :class:`pyro.infer.Importance`.

    Test-time only (``can_be_used_for_training=False``). Returns:

    * ``InferenceOutput.samples[name]`` — raw samples of shape
      ``(S, batch, *event)`` drawn from the proposal (the prior if
      ``guide=None``).
    * ``InferenceOutput.parameters[name]`` — weighted-moment-matched
      canonical parameters using **per-batch-element** self-normalised
      weights, so upstream marginals genuinely shift in response to
      downstream evidence.

    Parameters
    ----------
    model:
        The :class:`ProbabilisticModel` to query.
    num_samples:
        Number of importance-sampling traces drawn per query.
    guide:
        Optional proposal distribution as a Pyro guide. ``None`` (default)
        uses the prior (model with observed sites blocked) as proposal.
    """

    can_be_used_for_training: bool = False

    def __init__(self, model, num_samples: int = 500, guide=None):
        super().__init__(model)
        self.num_samples = int(num_samples)
        self.guide = guide

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, Tensor],
    ) -> InferenceOutput:
        if not evidence:
            raise ValueError("`evidence` cannot be empty.")
        any_v = next(iter(evidence.values()))
        batch_size = int(any_v.shape[0])

        importance = Importance(
            self.model, guide=self.guide, num_samples=self.num_samples,
        )
        with torch.no_grad():
            posterior = importance.run(evidence)

        traces = list(posterior.exec_traces)
        if not traces:
            raise RuntimeError("Importance sampling produced no traces.")

        # Per-batch-element log-weights: (S, batch).
        evidence_keys = list(evidence.keys())
        log_weights = torch.stack([
            _per_batch_log_weight(tr, evidence_keys, batch_size)
            for tr in traces
        ], dim=0)
        # Self-normalise per batch element (across the S axis).
        log_w_norm = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = log_w_norm.exp()  # (S, batch), each column sums to 1.

        per_sample: Dict[str, List[Tensor]] = {name: [] for name in variables}
        for tr in traces:
            for name in variables:
                node = tr.nodes.get(name)
                if node is None or node.get("type") != "sample":
                    raise ValueError(
                        f"Queried variable '{name}' is not a sample site "
                        f"in the importance trace."
                    )
                per_sample[name].append(node["value"])

        out_samples: Dict[str, Tensor] = {
            name: torch.stack(per_sample[name], dim=0) for name in variables
        }
        out_params: Dict[str, VariableParameters] = {}
        for name in variables:
            var = self.model.concept_to_variable[name]
            out_params[name] = _weighted_moment_match(
                var, out_samples[name], weights,
            )

        return InferenceOutput(parameters=out_params, samples=out_samples)


__all__ = ["ImportanceSamplingInference"]
