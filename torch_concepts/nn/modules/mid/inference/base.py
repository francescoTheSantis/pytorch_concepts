"""Abstract interface for inference engines.

See ``mid_level_api.md`` §5.1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from torch import Tensor

from .outputs import InferenceOutput


class BaseInference(ABC):
    """Abstract base for all inference engines (spec §5.1).

    Class attribute :attr:`can_be_used_for_training` indicates whether
    gradients flow through :meth:`query` back to ``model.parameters()``.
    Engines with ``can_be_used_for_training=False`` (e.g. MCMC) must not be
    used inside a training loop.
    """

    can_be_used_for_training: bool = False

    def __init__(self, model):
        self.model = model

    # Convenience alias used in spec §5.2 prose.
    @property
    def pgm(self):
        return self.model

    @abstractmethod
    def query(
        self,
        variables: List[str],
        evidence: Dict[str, Tensor],
    ) -> InferenceOutput:
        """Return CPD parameters for each ``variables`` entry given ``evidence``."""
        raise NotImplementedError


__all__ = ["BaseInference"]
