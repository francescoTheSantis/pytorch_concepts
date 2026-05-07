"""Mid-level inference engines (spec §5)."""
from .outputs import InferenceOutput, VariableParameters
from .base import BaseInference
from .svi import SVIInference
from .enumeration import EnumerationInference
from .mcmc import MCMCInference
from .importance import ImportanceSamplingInference
from .map import MAPInference

# Legacy submodules — re-exported so that the existing top-level
# ``torch_concepts.nn`` package import keeps working without modification.
# These are NOT part of the new spec API.
try:
    from .forward import ForwardInference
    from .deterministic import DeterministicInference
    from .ancestral import AncestralSamplingInference
    from .independent import IndependentInference
except Exception:  # pragma: no cover  -- legacy import may fail; tolerate it
    ForwardInference = None  # type: ignore
    DeterministicInference = None  # type: ignore
    AncestralSamplingInference = None  # type: ignore
    IndependentInference = None  # type: ignore

__all__ = [
    "BaseInference",
    "InferenceOutput",
    "VariableParameters",
    "SVIInference",
    "EnumerationInference",
    "MCMCInference",
    "ImportanceSamplingInference",
    "MAPInference",
    # legacy
    "ForwardInference",
    "DeterministicInference",
    "AncestralSamplingInference",
    "IndependentInference",
]
