"""Exact discrete-latent enumeration inference (spec §5.2)."""
from __future__ import annotations

from typing import Dict, List

from torch import Tensor

from .svi import SVIInference
from .outputs import InferenceOutput


class EnumerationInference(SVIInference):
    """Exact marginalization of **discrete** latents via Pyro enumeration.

    Internally this engine reuses :class:`SVIInference`'s guide-replay
    machinery; the actual enumeration happens inside the model's ``forward``
    (which tags discrete latents with ``infer={"enumerate": "parallel"}``)
    and is consumed by Pyro's :class:`TraceEnum_ELBO` when training.

    With no latents, this engine is equivalent to :class:`SVIInference`.

    Notes
    -----
    Spec §5.2 mandates ``TraceEnum_ELBO(max_plate_nesting=1)`` for the
    training-time loss; that constructor is created by the *training*
    consumer, not by ``query()`` itself, which only returns CPD parameters.

    A runtime check rejects continuous latents that lack
    ``has_enumerate_support`` (spec §5.2).
    """

    can_be_used_for_training: bool = True

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, Tensor],
    ) -> InferenceOutput:
        # Enforce: every latent variable in the query must be discrete.
        evidenced = set(evidence.keys())
        queried = set(variables)
        for var in self.model.sorted_variables:
            if var.concept in evidenced or var.concept in queried:
                continue
            # var is latent for this query → must be discrete with finite support
            d = var.distribution
            # Use a class-level probe: instantiate with dummy logits if needed?
            # Simpler: rely on Pyro's distribution.has_enumerate_support
            # attribute, which is a class attribute on most distributions.
            if not getattr(d, "has_enumerate_support", False):
                raise ValueError(
                    f"EnumerationInference requires every latent variable to "
                    f"be discrete with finite support; latent '{var.concept}' "
                    f"has distribution {d.__name__}."
                )
        return super().query(variables, evidence)


__all__ = ["EnumerationInference"]
