"""Tests for ParametricCPD as PyroModule + pyro_forward."""
import pytest
import torch
import torch.nn as nn
import pyro
import pyro.poutine as poutine
from pyro.nn import PyroModule
from torch.testing import assert_close

from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD


def test_cpd_is_pyromodule():
    cpd = ParametricCPD(concept='A', parametrization=nn.Linear(10, 1))
    assert isinstance(cpd, PyroModule)


def test_cpd_pyro_forward_returns_tensor(binary_chain_pgm):
    pgm, input_var, var_A, _ = binary_chain_pgm
    cpd_A   = pgm.factors['A']
    context = {'input': torch.randn(4, 10)}
    result  = cpd_A.sample(context)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (4, 1)


def test_cpd_pyro_forward_with_obs(binary_chain_pgm):
    pgm, input_var, var_A, _ = binary_chain_pgm
    cpd_A   = pgm.factors['A']
    context = {'input': torch.randn(4, 10)}
    obs     = torch.zeros(4, 1)
    result  = cpd_A.sample(context, obs=obs)
    assert_close(result, obs)


def test_cpd_pyro_forward_creates_site(binary_chain_pgm):
    pgm, input_var, var_A, _ = binary_chain_pgm
    cpd_A   = pgm.factors['A']
    context = {'input': torch.randn(4, 10)}

    with poutine.trace() as tr:
        with pyro.plate('data', 4):
            cpd_A.sample(context)

    assert 'A' in tr.trace.nodes


def test_cpd_forward_backward_compat():
    cpd = ParametricCPD(concept='A', parametrization=nn.Linear(10, 1))
    x   = torch.randn(4, 10)
    out = cpd(input=x)
    assert out.shape == (4, 1)


def test_cpd_parameters_accessible():
    cpd    = ParametricCPD(concept='A', parametrization=nn.Linear(10, 1))
    params = list(cpd.parameters())
    assert len(params) > 0
    assert all(isinstance(p, torch.Tensor) for p in params)
