"""Backward compatibility tests: existing mid-level API must still work."""
import pytest
import torch
import torch.nn as nn
import inspect
from torch.distributions import Bernoulli

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, LatentVariable,
)
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD


def test_probabilistic_model_unchanged():
    from torch_concepts.nn.modules.mid.models.probabilistic_model import BayesianNetwork
    input_var = LatentVariable(concept='input', distribution=Delta, size=10)
    var_A     = ConceptVariable(concept='A', distribution=Bernoulli, size=1)
    cpd_i     = ParametricCPD(concept='input', parametrization=nn.Identity())
    cpd_A     = ParametricCPD(concept='A', parametrization=nn.Linear(10, 1), parents=[input_var])
    pgm = BayesianNetwork(variables=[input_var, var_A], factors=[cpd_i, cpd_A])
    assert pgm is not None


def test_deterministic_inference_unchanged():
    from torch_concepts.nn.modules.mid.models.probabilistic_model import BayesianNetwork
    from torch_concepts.nn.modules.mid.inference.deterministic import DeterministicInference
    input_var = LatentVariable(concept='input', distribution=Delta, size=10)
    var_A     = ConceptVariable(concept='A', distribution=Bernoulli, size=1)
    cpd_i     = ParametricCPD(concept='input', parametrization=nn.Identity())
    cpd_A     = ParametricCPD(concept='A', parametrization=nn.Linear(10, 1), parents=[input_var])
    pgm       = BayesianNetwork(variables=[input_var, var_A], factors=[cpd_i, cpd_A])
    inf       = DeterministicInference(pgm)
    x         = torch.randn(4, 10)
    out       = inf.query(['A'], evidence={'input': x})
    assert out.probs.shape == (4, 1)


def test_ancestral_sampling_inference_unchanged():
    from torch_concepts.nn.modules.mid.models.probabilistic_model import BayesianNetwork
    from torch_concepts.nn.modules.mid.inference.ancestral import AncestralSamplingInference
    input_var = LatentVariable(concept='input', distribution=Delta, size=10)
    var_A     = ConceptVariable(concept='A', distribution=Bernoulli, size=1)
    cpd_i     = ParametricCPD(concept='input', parametrization=nn.Identity())
    cpd_A     = ParametricCPD(concept='A', parametrization=nn.Linear(10, 1), parents=[input_var])
    pgm       = BayesianNetwork(variables=[input_var, var_A], factors=[cpd_i, cpd_A])
    inf       = AncestralSamplingInference(pgm)
    out       = inf.query(['A'], evidence={'input': torch.randn(4, 10)})
    assert out.probs.shape == (4, 1)


def test_cpd_forward_signature_unchanged():
    cpd = ParametricCPD(concept='A', parametrization=nn.Linear(10, 1))
    sig = inspect.signature(cpd.forward)
    assert any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
