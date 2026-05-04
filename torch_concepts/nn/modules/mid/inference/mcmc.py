"""
PyroMCMCQuery — NUTS / HMC posterior queries.

MCMC produces asymptotically exact posterior samples for any evidence
pattern, including backward and v-structure queries.  Works for
continuous latent variables; use :class:`PyroExactDiscreteQuery` for
discrete ones.
"""
from __future__ import annotations

from typing import Dict, List, Literal

import torch
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, HMC


class MCMCQuery:
    """
    Posterior query via NUTS or HMC MCMC.

    Parameters
    ----------
    pgm : ProbabilisticModel
        The trained Pyro PGM.
    kernel : ``"nuts"`` or ``"hmc"``
        MCMC transition kernel.  Default: ``"nuts"``.
    warmup_steps : int
        Number of burn-in steps.  Default: 200.
    num_chains : int
        Number of independent chains.  Default: 1.
    step_size : float
        HMC leapfrog step size.  Default: 0.1.
    num_steps : int
        HMC leapfrog steps per transition.  Default: 10.
    """

    def __init__(
        self,
        pgm: 'ProbabilisticModel',  # noqa: F821
        kernel: Literal["nuts", "hmc"] = "nuts",
        warmup_steps: int = 200,
        num_chains: int = 1,
        step_size: float = 0.1,
        num_steps: int = 10,
    ) -> None:
        self._pgm          = pgm
        self._kernel_name  = kernel
        self._warmup_steps = warmup_steps
        self._num_chains   = num_chains
        self._step_size    = step_size
        self._num_steps    = num_steps

    def query(
        self,
        variables: List[str],
        evidence: Dict[str, torch.Tensor],
        num_samples: int = 200,
    ) -> 'InferenceOutput':
        """
        Draw posterior samples via MCMC.

        Parameters
        ----------
        variables : list of str
            Variable names to return.
        evidence : dict
            Observed values.
        num_samples : int
            Post-warmup samples to collect per chain.

        Returns
        -------
        InferenceOutput
            ``result.samples`` maps variable name → Tensor
            ``(num_samples, *batch_dims, size)``; ``result.probs`` is the
            empirical mean over the leading sample dimension, concatenated
            in the requested-variable order (``None`` when sites have
            inconsistent shapes).
        """
        from ...outputs import InferenceOutput
        from ..models.probabilistic_model import _concat_sample_means

        conditioned = poutine.condition(self._pgm, data=evidence)

        if self._kernel_name == "nuts":
            kernel = NUTS(conditioned)
        else:
            kernel = HMC(
                conditioned,
                step_size=self._step_size,
                num_steps=self._num_steps,
            )

        mcmc = MCMC(
            kernel,
            num_samples=num_samples,
            warmup_steps=self._warmup_steps,
            num_chains=self._num_chains,
        )
        mcmc.run(evidence)

        posterior = mcmc.get_samples()
        samples = {var: posterior[var] for var in variables if var in posterior}
        return InferenceOutput(
            samples=samples,
            probs=_concat_sample_means(variables, samples),
        )
