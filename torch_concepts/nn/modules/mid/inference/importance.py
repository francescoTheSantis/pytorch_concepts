"""
PyroImportanceQuery — importance-resampling posterior queries.

Works correctly for backward queries and v-structure conditioning
(e.g. predicting a cause given an observed child) because importance
weights account for the full joint log probability rather than relying
on a factored forward-pass guide.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import pyro


class ImportanceQuery:
    """
    Posterior query via importance resampling.

    Draws ``num_samples`` proposals from *guide* (or from the prior when no
    guide is provided) and resamples them with replacement proportionally to
    the unnormalised importance weights.

    Parameters
    ----------
    pgm : ProbabilisticModel
        The trained Pyro PGM.
    """

    def __init__(self, pgm: 'ProbabilisticModel') -> None:  # noqa: F821
        self._pgm = pgm

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, torch.Tensor],
        num_samples: int = 500,
        guide: Optional[object] = None,
    ) -> 'InferenceOutput':
        """
        Draw posterior samples via importance resampling.

        Parameters
        ----------
        variables : list of str
            Variable names to return.
        evidence : dict
            Observed values.
        num_samples : int
            Number of samples to return after resampling.
        guide : callable, optional
            Proposal distribution.  Defaults to the prior when none is given.

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

        proposal = guide if guide is not None else getattr(self._pgm, 'guide', None)
        posterior = pyro.infer.Importance(
            self._pgm, guide=proposal, num_samples=num_samples,
        )
        posterior.run(evidence)

        weights = posterior.get_normalized_weights()
        indices = torch.multinomial(weights, num_samples, replacement=True)

        samples: Dict[str, torch.Tensor] = {}
        for var in variables:
            vals = torch.stack([
                posterior.exec_traces[i].nodes[var]["value"]
                for i in range(len(posterior.exec_traces))
            ])
            samples[var] = vals[indices]
        return InferenceOutput(
            samples=samples,
            probs=_concat_sample_means(variables, samples),
        )
