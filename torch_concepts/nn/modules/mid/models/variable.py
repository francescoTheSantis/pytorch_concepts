"""
Variable specs for Concept-based Probabilistic Graphical Models (mid-level API).

A :class:`Variable` is a *metadata* object describing one node in a PGM:
its unique name, its (Pyro) distribution family and its event size.

Variables are **not** Pyro primitives — they hold the information that
:meth:`ProbabilisticModel.forward` needs in order to call ``pyro.sample(name,
dist, obs=...)`` with the correct distribution.

See ``mid_level_api.md`` §2 for the full specification.
"""
from __future__ import annotations

import copy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
import pyro.distributions as dist
from pyro.distributions import (
    Bernoulli,
    Categorical,
    Delta,
    Distribution,
    MultivariateNormal,
    Normal,
)


# ---------------------------------------------------------------------------
# Canonical parameter dimensions — spec §2.4
# ---------------------------------------------------------------------------

_PARAM_DIM_TABLE: Dict[Type[Distribution], Callable[[int], int]] = {
    Bernoulli: lambda k: k,                                 # logits  (k,)
    Categorical: lambda k: k,                               # logits  (k,)  k = #classes
    Normal: lambda k: 2 * k,                                # loc (k,)  + scale (k,)
    MultivariateNormal: lambda k: k + k * (k + 1) // 2,     # loc (k,)  + scale_tril ((k(k+1)/2),)
    Delta: lambda k: k,                                     # v (k,)
}


def param_dim(distribution: Type[Distribution], size: int) -> int:
    """Return the output dimension a CPD's parametrization must produce.

    See spec §2.4 for the canonical table.
    """
    fn = _PARAM_DIM_TABLE.get(distribution)
    if fn is None:
        raise ValueError(
            f"Unsupported distribution '{distribution.__name__}' for param_dim. "
            f"Supported: {[d.__name__ for d in _PARAM_DIM_TABLE]}"
        )
    if size is None or size <= 0:
        raise ValueError(f"`size` must be a positive integer; got {size!r}.")
    return int(fn(size))


# ---------------------------------------------------------------------------
# Variable hierarchy — spec §2.1 / §2.2
# ---------------------------------------------------------------------------

class Variable:
    """Abstract base spec for a PGM variable.

    Use one of :class:`ConceptVariable`, :class:`LatentVariable` or
    :class:`ExogenousVariable` in user code; the only difference between
    them is ``metadata['variable_type']``.

    Construction supports two modes (mutually exclusive):

    * Single   — ``Variable(concept='c1', distribution=..., size=...)``
                 returns a single :class:`Variable` instance.
    * Multiple — ``Variable(concepts=['c1','c2'], distribution=..., size=...)``
                 returns ``List[Variable]``, one independently-instantiated
                 per name.

    Passing **neither** or **both** of ``concept=`` / ``concepts=`` raises
    :class:`ValueError`.
    """

    # Subclasses set this to populate ``metadata['variable_type']``.
    _VARIABLE_TYPE: Optional[str] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __new__(
        cls,
        concept: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        distribution: Optional[Type[Distribution]] = None,
        size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dist_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Mutual-exclusion check (spec §2.2)
        if concept is None and concepts is None:
            raise ValueError(
                "Exactly one of `concept=` (single) or `concepts=` (list) "
                "must be provided."
            )
        if concept is not None and concepts is not None:
            raise ValueError(
                "`concept=` and `concepts=` are mutually exclusive."
            )

        # Single-instance path
        if concept is not None:
            if not isinstance(concept, str):
                raise ValueError("`concept` must be a string.")
            return object.__new__(cls)

        # Multi-instance path: returns List[cls]
        if not isinstance(concepts, (list, tuple)) or not all(
            isinstance(c, str) for c in concepts
        ):
            raise ValueError("`concepts` must be a list of strings.")
        return [
            cls(
                concept=name,
                distribution=distribution,
                size=size,
                metadata=copy.deepcopy(metadata) if metadata else None,
                dist_kwargs=copy.deepcopy(dist_kwargs) if dist_kwargs else None,
            )
            for name in concepts
        ]

    def __init__(
        self,
        concept: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        distribution: Optional[Type[Distribution]] = None,
        size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dist_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # When ``__new__`` returned a list, Python skips ``__init__``;
        # this guard handles defensive cases.
        if concept is None:
            return

        if distribution is None:
            distribution = Delta
        if size is None:
            size = 1

        # Validate (param_dim raises if distribution is not supported)
        param_dim(distribution, size)

        self.concept: str = concept
        self.distribution: Type[Distribution] = distribution
        self.size: int = int(size)
        self.dist_kwargs: Dict[str, Any] = dict(dist_kwargs) if dist_kwargs else {}
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        if self._VARIABLE_TYPE is not None:
            self.metadata.setdefault("variable_type", self._VARIABLE_TYPE)

    # ------------------------------------------------------------------
    @property
    def out_features(self) -> int:
        """Output dim a CPD network must produce — alias of :func:`param_dim`."""
        return param_dim(self.distribution, self.size)

    @property
    def param_dim(self) -> int:  # convenience attribute
        return param_dim(self.distribution, self.size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"concept={self.concept!r}, "
            f"distribution={self.distribution.__name__}, "
            f"size={self.size}, "
            f"metadata={self.metadata})"
        )


class ConceptVariable(Variable):
    """Interpretable, supervised variable.

    ``metadata['variable_type'] = 'concept'``.
    """
    _VARIABLE_TYPE = "concept"


class LatentVariable(Variable):
    """Non-interpretable internal representation.

    ``metadata['variable_type'] = 'latent'``.
    """
    _VARIABLE_TYPE = "latent"


class ExogenousVariable(Variable):
    """Non-interpretable input to the PGM.

    ``metadata['variable_type'] = 'exogenous'``. Reserved for future
    semantic distinctions from :class:`LatentVariable`.
    """
    _VARIABLE_TYPE = "exogenous"


# ---------------------------------------------------------------------------
# Backward-compat aliases — kept so ``torch_concepts/__init__.py`` keeps
# importing without modification. NOT part of the spec public API.
# ---------------------------------------------------------------------------

InputVariable = LatentVariable
EndogenousVariable = ConceptVariable


# ---------------------------------------------------------------------------
# Legacy default tables — referenced by ``torch_concepts/utils.py`` for the
# annotation-driven defaults pipeline. Not part of the spec public API.
# ---------------------------------------------------------------------------

_DEFAULT_DISTRIBUTIONS: Dict[str, Type[Distribution]] = {
    "binary": Bernoulli,
    "categorical": Categorical,
    "continuous": Normal,
}

_DEFAULT_DIST_KWARGS: Dict[Type[Distribution], Dict[str, Any]] = {}

_DEFAULT_ACTIVATIONS: Dict[Type[Distribution], Callable[[torch.Tensor], torch.Tensor]] = {
    Bernoulli: torch.sigmoid,
    Categorical: partial(torch.softmax, dim=-1),
    Normal: lambda x: x,
    MultivariateNormal: lambda x: x,
    Delta: lambda x: x,
}


__all__ = [
    "param_dim",
    "Variable",
    "ConceptVariable",
    "LatentVariable",
    "ExogenousVariable",
    "InputVariable",
    "EndogenousVariable",
]
