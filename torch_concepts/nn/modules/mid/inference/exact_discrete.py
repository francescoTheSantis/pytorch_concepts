"""
PyroExactDiscreteQuery — exact posterior inference for discrete latents.

Uses Pyro's variable-elimination algorithm (``infer_discrete``) to compute
the exact posterior over discrete hidden variables.  Handles any evidence
pattern, including backward and v-structure queries, without approximation
error.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import pyro.poutine as poutine
from pyro.infer import config_enumerate
from pyro.infer.discrete import infer_discrete


class ExactDiscreteQuery:
    """
    Exact posterior query via variable elimination for discrete latents.

    Parameters
    ----------
    pgm : ProbabilisticModel
        The trained Pyro PGM.
    temperature : int
        ``1`` to sample, ``0`` for MAP.  Default: 1.
    """

    def __init__(
        self,
        pgm: 'ProbabilisticModel',  # noqa: F821
        temperature: int = 1,
    ) -> None:
        self._pgm         = pgm
        self._temperature = temperature

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, torch.Tensor],
        num_samples: int = 200,
    ) -> 'InferenceOutput':
        """
        Draw exact posterior samples via variable elimination.

        Parameters
        ----------
        variables : list of str
            Variable names to return.
        evidence : dict
            Observed values.
        num_samples : int
            Number of independent exact-posterior draws.

        Returns
        -------
        InferenceOutput
            ``result.samples`` maps variable name → Tensor
            ``(num_samples, *batch_dims, size)``; ``result.probs`` is the
            empirical mean over the leading sample dimension, concatenated
            in the requested-variable order.
        """
        from ...outputs import InferenceOutput
        from ..models.probabilistic_model import _concat_sample_means

        conditioned = poutine.condition(self._pgm, data=evidence)
        enumerated  = config_enumerate(conditioned, "parallel")
        exact_model = infer_discrete(
            enumerated,
            temperature=self._temperature,
            first_available_dim=-2,
        )

        collected: Dict[str, List[torch.Tensor]] = {var: [] for var in variables}
        for _ in range(num_samples):
            trace = poutine.trace(exact_model).get_trace(evidence)
            for var in variables:
                node = trace.nodes.get(var)
                if node is not None and node.get("type") == "sample":
                    collected[var].append(node["value"])

        samples = {
            var: torch.stack(vals)
            for var, vals in collected.items()
            if vals
        }
        return InferenceOutput(
            samples=samples,
            probs=_concat_sample_means(variables, samples),
        )
