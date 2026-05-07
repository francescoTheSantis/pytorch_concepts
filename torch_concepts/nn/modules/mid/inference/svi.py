"""Single-Sample Variational Inference engine.

See ``mid_level_api.md`` §5.2 — including the guide-replay pattern.
"""
from __future__ import annotations

from typing import Dict, List

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.poutine.util import site_is_subsample
from torch import Tensor

from .base import BaseInference
from .outputs import InferenceOutput, VariableParameters


# ---------------------------------------------------------------------------
# Distribution → canonical parameter dict
# ---------------------------------------------------------------------------

def _extract_params(fn) -> VariableParameters:
    """Read canonical parameter tensors from a Pyro distribution.

    Supports the spec §2.4 distribution families. Independent / TransformedDistribution
    wrappers (created by ``.to_event``) are unwrapped to reach the base
    distribution holding the canonical attributes.
    """
    base = fn
    # Unwrap Independent / MaskedDistribution etc. via base_dist
    while hasattr(base, "base_dist"):
        base = base.base_dist

    if isinstance(base, dist.Bernoulli):
        return VariableParameters({"logits": base.logits})
    if isinstance(base, dist.Categorical):
        return VariableParameters({"logits": base.logits})
    if isinstance(base, dist.MultivariateNormal):
        return VariableParameters(
            {"loc": base.loc, "scale_tril": base.scale_tril}
        )
    if isinstance(base, dist.Normal):
        return VariableParameters({"loc": base.loc, "scale": base.scale})
    if isinstance(base, dist.Delta):
        return VariableParameters({"v": base.v})

    raise ValueError(
        f"Cannot extract canonical parameters from distribution "
        f"{type(base).__name__}."
    )


# ---------------------------------------------------------------------------
# SVIInference
# ---------------------------------------------------------------------------

class SVIInference(BaseInference):
    """Single-sample Variational Inference engine (spec §5.2).

    * No latents → guide is empty; ``query()`` reduces to a deterministic
      topological forward pass.
    * Latents present → single-sample Monte Carlo via guide-replay.
      Returns ``latent_params`` (from the guide trace) and ``prior_params``
      (from the replayed model trace) so the user can compute KL externally.
    """

    can_be_used_for_training: bool = True

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, Tensor],
    ) -> InferenceOutput:
        # Step 1 — trace the guide. Latent sites here come from q_phi(z|.).
        guide_trace = poutine.trace(self.model.guide).get_trace(evidence)

        # Step 2 — replay the model so latent sites reuse the guide's draws.
        # Sites that are observed in the model retain their model-side fn.
        model_trace = poutine.trace(
            poutine.replay(self.model.forward, trace=guide_trace)
        ).get_trace(evidence)

        # Step 3 — read canonical parameters from the model trace for each
        # queried variable.
        parameters: Dict[str, VariableParameters] = {}
        for name in variables:
            if name not in model_trace.nodes:
                raise ValueError(f"Queried variable '{name}' is not in the model.")
            node = model_trace.nodes[name]
            if node.get("type") != "sample":
                raise ValueError(f"Site '{name}' is not a sample site.")
            parameters[name] = _extract_params(node["fn"])

        # Step 4 — for every latent site in the model trace (sample site
        # that is not observed), record q-side and p-side params.
        # AutoNormal samples in unconstrained space and exposes the variational
        # distribution at "<name>_unconstrained"; for sites whose support is
        # already real-valued (e.g. Normal) the transform is identity but the
        # naming convention still applies.
        latent_params: Dict[str, VariableParameters] = {}
        prior_params: Dict[str, VariableParameters] = {}
        for site_name, mnode in model_trace.nodes.items():
            if mnode.get("type") != "sample":
                continue
            if site_is_subsample(mnode):
                continue
            if mnode.get("is_observed", False):
                continue
            # Locate the variational distribution in the guide trace.
            unconstrained = f"{site_name}_unconstrained"
            if unconstrained in guide_trace.nodes:
                gnode = guide_trace.nodes[unconstrained]
            elif site_name in guide_trace.nodes:
                gnode = guide_trace.nodes[site_name]
            else:
                continue
            if gnode.get("type") != "sample" or site_is_subsample(gnode):
                continue
            latent_params[site_name] = _extract_params(gnode["fn"])
            prior_params[site_name] = _extract_params(mnode["fn"])

        return InferenceOutput(
            parameters=parameters,
            latent_params=latent_params if latent_params else None,
            prior_params=prior_params if prior_params else None,
        )


__all__ = ["SVIInference", "_extract_params"]
