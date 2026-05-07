"""Mid-level PGM model classes."""
from .variable import (
    Variable,
    ConceptVariable,
    LatentVariable,
    ExogenousVariable,
    InputVariable,        # back-compat alias
    EndogenousVariable,   # back-compat alias
    param_dim,
)
from .factor import ParametricFactor
from .cpd import ParametricCPD
from .probabilistic_model import ProbabilisticModel

__all__ = [
    "Variable",
    "ConceptVariable",
    "LatentVariable",
    "ExogenousVariable",
    "InputVariable",
    "EndogenousVariable",
    "param_dim",
    "ParametricFactor",
    "ParametricCPD",
    "ProbabilisticModel",
]
