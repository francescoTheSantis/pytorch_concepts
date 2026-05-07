"""Typed output containers for inference engines.

See ``mid_level_api.md`` §6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

from torch import Tensor


@dataclass
class VariableParameters:
    """Distribution parameters for one PGM variable.

    Canonical parameter names by distribution family
    (see spec §2.4 / §6.1):

      * Bernoulli / Categorical : ``'logits'``
      * Normal(size=k)          : ``'loc'``, ``'scale'``
      * MultivariateNormal(k)   : ``'loc'``, ``'scale_tril'``
      * Delta(size=k)           : ``'v'``
    """
    params: Dict[str, Tensor]

    def __getitem__(self, key: str) -> Tensor:
        return self.params[key]

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def __iter__(self) -> Iterator[str]:
        return iter(self.params)


@dataclass
class InferenceOutput:
    """Container for the result of a single :meth:`BaseInference.query` call.

    Field semantics (spec §6.2):

    * ``parameters`` — CPD parameters of the queried variables.
      Evidence and latent variables are NOT included.
    * ``samples`` — raw posterior samples; populated by sample-based
      engines (e.g. :class:`MCMCInference`).
    * ``latent_params`` — parameters of the variational posterior
      :math:`q_\\phi(z_j \\mid \\cdot)`; populated by
      :class:`SVIInference` whenever the query has latent variables.
    * ``prior_params`` — parameters of the model prior
      :math:`p(z_j \\mid \\mathrm{PA}(z_j))` evaluated on the same trace.
    """
    parameters: Optional[Dict[str, VariableParameters]] = None
    samples: Optional[Dict[str, Tensor]] = None
    latent_params: Optional[Dict[str, VariableParameters]] = None
    prior_params: Optional[Dict[str, VariableParameters]] = None


__all__ = ["VariableParameters", "InferenceOutput"]
