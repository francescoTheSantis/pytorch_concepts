"""
Abstract base class for parametric factors in a PGM.

See ``mid_level_api.md`` §3.1.
"""
from __future__ import annotations

from pyro.nn import PyroModule


class ParametricFactor(PyroModule):
    """Abstract base for all parametric factors.

    Carries no functional code; reserves the class hierarchy for future
    undirected-factor support. Direct instantiation raises
    :class:`NotImplementedError`.
    """

    def __init__(self, *args, **kwargs):
        if type(self) is ParametricFactor:
            raise NotImplementedError(
                "ParametricFactor is abstract and cannot be instantiated "
                "directly. Use ParametricCPD (or another concrete subclass)."
            )
        super().__init__()


__all__ = ["ParametricFactor"]
