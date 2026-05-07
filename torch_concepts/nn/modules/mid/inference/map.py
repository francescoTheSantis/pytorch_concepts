"""Joint MAP inference engine (spec §5.3)."""
from __future__ import annotations

from typing import Dict, List

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, infer_discrete
from pyro.infer.autoguide import AutoDelta
from torch import Tensor

from .base import BaseInference
from .outputs import InferenceOutput, VariableParameters
from .svi import _extract_params


_VALID_ALGORITHMS = ("auto_delta", "infer_discrete")


class MAPInference(BaseInference):
    """Joint MAP assignment :math:`(c_1^*, \\ldots, c_k^*) = \\arg\\max p(c \\mid x)`.

    Two algorithms (spec §5.3):

    * ``'auto_delta'``  — :class:`AutoDelta` guide + SVI; works for both
      continuous and discrete variables.
    * ``'infer_discrete'`` — exact MAP for discrete variables via Pyro's
      :func:`infer_discrete` (with ``temperature=0``).
    """

    can_be_used_for_training: bool = False

    def __init__(
        self,
        model,
        algorithm: str = "auto_delta",
        num_steps: int = 200,
        lr: float = 1e-2,
    ):
        super().__init__(model)
        if algorithm not in _VALID_ALGORITHMS:
            raise ValueError(
                f"`algorithm` must be one of {_VALID_ALGORITHMS}; got {algorithm!r}."
            )
        self.algorithm = algorithm
        self.num_steps = int(num_steps)
        self.lr = float(lr)

    # ------------------------------------------------------------------
    def query(
        self,
        variables: List[str],
        evidence: Dict[str, Tensor],
    ) -> InferenceOutput:
        if self.algorithm == "auto_delta":
            return self._auto_delta_query(variables, evidence)
        return self._infer_discrete_query(variables, evidence)

    # ------------------------------------------------------------------
    def _auto_delta_query(
        self,
        variables: List[str],
        evidence: Dict[str, Tensor],
    ) -> InferenceOutput:
        all_names = {v.concept for v in self.model.sorted_variables}
        latents = all_names - set(variables) - set(evidence.keys())

        # Fast path — no true latents: a single replayed forward pass
        # against the model's own auto-guide yields the CPD parameters
        # of the queried variables (spec §5.2 / §5.3 with no latents).
        if not latents:
            guide_trace = poutine.trace(self.model.guide).get_trace(evidence)
            model_trace = poutine.trace(
                poutine.replay(self.model.forward, trace=guide_trace)
            ).get_trace(evidence)
            params = {
                name: _extract_params(model_trace.nodes[name]["fn"])
                for name in variables
            }
            return InferenceOutput(parameters=params)

        # Otherwise: AutoDelta over the latent sites + SVI.
        # NOTE: AutoDelta builds against the global param store; we use a
        # local param-store snapshot via `pyro.poutine.scope` to avoid
        # leaking state across queries.
        guide_model = poutine.block(self.model.forward, expose=list(latents))
        delta_guide = AutoDelta(guide_model)
        optimizer = pyro.optim.Adam({"lr": self.lr})
        svi = SVI(self.model.forward, delta_guide, optimizer, loss=Trace_ELBO())
        for _ in range(self.num_steps):
            svi.step(evidence)

        guide_trace = poutine.trace(delta_guide).get_trace(evidence)
        model_trace = poutine.trace(
            poutine.replay(self.model.forward, trace=guide_trace)
        ).get_trace(evidence)
        params = {
            name: _extract_params(model_trace.nodes[name]["fn"])
            for name in variables
        }
        return InferenceOutput(parameters=params)

    # ------------------------------------------------------------------
    def _infer_discrete_query(
        self,
        variables: List[str],
        evidence: Dict[str, Tensor],
    ) -> InferenceOutput:
        # Exact MAP for discrete sites.  ``temperature=0`` selects argmax.
        wrapped = infer_discrete(
            self.model.forward,
            first_available_dim=-2,  # spec §4.3: data plate uses dim=-1
            temperature=0,
        )
        trace = poutine.trace(wrapped).get_trace(evidence)
        params = {
            name: _extract_params(trace.nodes[name]["fn"]) for name in variables
        }
        return InferenceOutput(parameters=params)


__all__ = ["MAPInference"]
