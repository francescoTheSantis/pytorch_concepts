"""
:class:`ParametricCPD` — directed parametric factor for the mid-level PGM API.

See ``mid_level_api.md`` §3.2.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Union

import torch.nn as nn
from pyro.nn import PyroModule
from pyro.nn.module import to_pyro_module_

from .factor import ParametricFactor
from .variable import Variable


def _ensure_pyro_module(module: nn.Module) -> nn.Module:
    """In-place convert ``module`` to a :class:`PyroModule` if it is not one.

    Spec §8 rule 3 requires parametrizations to be PyroModule-compatible.
    """
    if not isinstance(module, PyroModule):
        to_pyro_module_(module, recurse=True)
    return module


class ParametricCPD(ParametricFactor):
    """Directed factor :math:`p(c_i \\mid \\mathrm{PA}(c_i))`.

    Holds the neural network :math:`f_{\\theta_i}` mapping concatenated
    parent values to canonical parameters of :math:`c_i`'s distribution.

    Construction supports two modes (mutually exclusive):

    * Single — ``ParametricCPD(concept='c1', parametrization=net,
      parents=[...])`` returns one CPD.
    * Multiple — ``ParametricCPD(concepts=['c1','c2'], parametrization=net,
      parents=[...])`` returns ``List[ParametricCPD]``, deep-copying
      ``parametrization`` for each entry.

    ``parametrization=None`` declares the variable as an **evidence-only
    root** (implicitly :math:`\\mathrm{Delta}`-distributed). Such roots
    must always appear in the ``evidence`` dict at inference time.

    Notes
    -----
    The output-dim validation against
    ``param_dim(variable.distribution, variable.size)`` (spec §8 rule 9)
    is performed at :class:`ProbabilisticModel` construction, where the
    binding between CPD and child :class:`Variable` is resolved.
    # NOTE: the spec phrases this as "at ParametricCPD.__init__"; the CPD
    # constructor only receives the child concept name (string), not the
    # variable, so the actual check is deferred to the smallest enclosing
    # object that has full info — i.e. the PGM. User-facing semantics
    # (a ValueError at construction time) are unchanged.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __new__(
        cls,
        concept: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        parametrization: Optional[nn.Module] = None,
        parents: Optional[List[Union[Variable, str]]] = None,
    ):
        # Mutual-exclusion check
        if concept is None and concepts is None:
            raise ValueError(
                "Exactly one of `concept=` (single) or `concepts=` (list) "
                "must be provided."
            )
        if concept is not None and concepts is not None:
            raise ValueError(
                "`concept=` and `concepts=` are mutually exclusive."
            )

        if concept is not None:
            if not isinstance(concept, str):
                raise ValueError("`concept` must be a string.")
            return PyroModule.__new__(cls)

        # multi-name path
        if not isinstance(concepts, (list, tuple)) or not all(
            isinstance(c, str) for c in concepts
        ):
            raise ValueError("`concepts` must be a list of strings.")
        return [
            cls(
                concept=name,
                parametrization=(
                    copy.deepcopy(parametrization)
                    if parametrization is not None
                    else None
                ),
                parents=list(parents) if parents else [],
            )
            for name in concepts
        ]

    def __init__(
        self,
        concept: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        parametrization: Optional[nn.Module] = None,
        parents: Optional[List[Union[Variable, str]]] = None,
    ):
        if concept is None:
            return  # `__new__` returned a list — skip init

        super().__init__()
        if parametrization is not None:
            parametrization = _ensure_pyro_module(parametrization)
        self.concept: str = concept
        # Back-compat alias used by some legacy callers.
        self.concepts = concept
        self.parametrization: Optional[nn.Module] = parametrization
        self.parents: List[Union[Variable, str]] = list(parents) if parents else []

    def __repr__(self) -> str:
        parent_names = [
            p.concept if isinstance(p, Variable) else p for p in self.parents
        ]
        param_repr = (
            self.parametrization.__class__.__name__
            if self.parametrization is not None
            else "None"
        )
        return (
            f"{self.__class__.__name__}("
            f"concept={self.concept!r}, parametrization={param_repr}, "
            f"parents={parent_names})"
        )


__all__ = ["ParametricCPD"]
