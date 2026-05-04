"""Tests for R1: Neural networks parametrize distributions."""
import pytest
import torch
import torch.nn as nn
import pyro.poutine as poutine
from torch.distributions import Bernoulli
from torch.testing import assert_close

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, LatentVariable,
)
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.elbo import ELBOInference


def test_nn_weights_in_pgm_parameters(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    param_set = {id(p) for p in pgm.parameters()}
    for cpd in pgm.factors.values():
        for p in cpd.parametrization.parameters():
            assert id(p) in param_set, \
                f"CPD parametrization weight not found in pgm.parameters()"


def test_nn_weights_affect_distribution(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    data     = {'input': torch.randn(4, 10)}

    with poutine.trace() as tr1:
        pgm(data)
    val1 = tr1.trace.nodes['A']['fn'].base_dist.logits.detach().clone()

    # Perturb weights of cpd_A
    with torch.no_grad():
        for p in pgm.factors['A'].parametrization.parameters():
            p.add_(10.0)

    with poutine.trace() as tr2:
        pgm(data)
    val2 = tr2.trace.nodes['A']['fn'].base_dist.logits.detach()

    assert not torch.allclose(val1, val2), \
        "Perturbing NN weights must change the distribution parameters"


def test_gradients_flow_to_nn_weights(cbm_pgm):
    pgm    = cbm_pgm
    engine = ELBOInference(pgm)
    data   = {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}
    engine(data)  # init

    loss = -engine.log_prob({'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)})
    loss.backward()

    for name, cpd in pgm.factors.items():
        for p in cpd.parametrization.parameters():
            assert p.grad is not None, \
                f"No gradient on CPD '{name}' parametrization weight"


def test_deep_nn_parametrization():
    encoder = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    input_var = LatentVariable(concept='input', distribution=Delta, size=10)
    var_A     = ConceptVariable(concept='A', distribution=Bernoulli, size=1)

    cpd_input = ParametricCPD(concept='input', parametrization=nn.Identity())
    cpd_A     = ParametricCPD(concept='A', parametrization=encoder, parents=[input_var])

    pgm  = BayesianNetwork(variables=[input_var, var_A], factors=[cpd_input, cpd_A])
    data = {'input': torch.randn(4, 10)}

    with poutine.trace() as tr:
        pgm(data)

    assert tr.trace.nodes['A']['value'].shape == (4, 1)
