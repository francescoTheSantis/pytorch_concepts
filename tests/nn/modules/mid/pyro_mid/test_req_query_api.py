"""Tests for R3: Pluggable query API."""
import pytest
import torch
import pyro

from torch_concepts.nn.modules.mid.inference.elbo import ELBOInference
from torch_concepts.nn.modules.mid.inference.guide import AmortizedGuide
from torch_concepts.nn.modules.outputs import InferenceOutput
from .conftest import pgm_train_and_fit


def _fit(pgm, batch=4, steps=3):
    """Quick helper: a few gradient steps to init params, returns guide."""
    engine = ELBOInference(pgm)
    data   = {'input': torch.zeros(batch, 16), 'task': torch.zeros(batch, 1)}
    engine(data)
    opt = torch.optim.Adam(engine.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        (-engine.log_prob({'input': torch.randn(batch, 16),
                           'task': torch.zeros(batch, 1)})).backward()
        opt.step()
    return engine


def test_query_returns_dict(cbm_pgm):
    pgm    = cbm_pgm
    engine = _fit(pgm)
    guide  = engine.elbo_module.guide
    result = pgm.query(['A', 'B'], evidence={'input': torch.randn(4, 16)},
                       guide=guide)
    assert isinstance(result, InferenceOutput)
    assert 'A' in result.samples
    assert 'B' in result.samples


@pytest.mark.parametrize("num_samples", [1, 10, 50])
def test_query_output_shape(num_samples, cbm_pgm):
    pgm    = cbm_pgm
    engine = _fit(pgm)
    guide  = engine.elbo_module.guide
    batch  = 4
    result = pgm.query(['A'], evidence={'input': torch.randn(batch, 16)},
                       guide=guide, num_samples=num_samples)
    assert result.samples['A'].shape == (num_samples, batch, 1)


def test_query_evidence_conditioning(cbm_pgm):
    pgm    = cbm_pgm
    engine = _fit(pgm)
    guide  = engine.elbo_module.guide

    fixed_A = torch.ones(4, 1)
    result  = pgm.query(
        ['task'],
        evidence={'input': torch.randn(4, 16), 'A': fixed_A},
        guide=guide, num_samples=20,
    )
    assert 'task' in result.samples
    assert result.samples['task'].shape[0] == 20


def test_query_returns_only_requested_variables(cbm_pgm):
    pgm    = cbm_pgm
    engine = _fit(pgm)
    guide  = engine.elbo_module.guide

    result = pgm.query(['task'], evidence={'input': torch.randn(4, 16)},
                       guide=guide, num_samples=5)
    assert set(result.samples.keys()) == {'task'}


def test_query_importance_weighted(cbm_pgm):
    pgm = cbm_pgm
    _fit(pgm)
    result = pgm.query(['A', 'task'], evidence={'input': torch.randn(4, 16)},
                       method='importance', num_samples=20)
    assert result.samples['A'].shape[0] == 20
    assert result.samples['task'].shape[0] == 20


def test_query_ancestral_sampling(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    result  = pgm.query(['A', 'B'], evidence={'input': torch.randn(4, 10)},
                        num_samples=10)
    assert 'A' in result.samples
    assert 'B' in result.samples
    assert result.samples['A'].shape == (10, 4, 1)


def test_query_guide_vs_ancestral(cbm_pgm):
    pgm    = cbm_pgm
    engine = _fit(pgm)
    guide  = engine.elbo_module.guide

    r1 = pgm.query(['A'], evidence={'input': torch.randn(2, 16)},
                   guide=guide, num_samples=5)
    r2 = pgm.query(['A'], evidence={'input': torch.randn(2, 16)},
                   num_samples=5)

    assert r1.samples['A'].shape == r2.samples['A'].shape


def test_query_is_stochastic(cbm_pgm):
    pgm    = cbm_pgm
    engine = _fit(pgm)
    guide  = engine.elbo_module.guide

    x  = {'input': torch.randn(4, 16)}
    r1 = pgm.query(['A'], evidence=x, guide=guide, num_samples=20)
    r2 = pgm.query(['A'], evidence=x, guide=guide, num_samples=20)
    assert not torch.allclose(r1.samples['A'], r2.samples['A']), \
        "Query samples should be stochastic across calls"
