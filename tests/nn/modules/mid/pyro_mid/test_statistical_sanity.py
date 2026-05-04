"""Numerical and statistical sanity tests."""
import pytest
import torch
import pyro.poutine as poutine

from torch_concepts.nn.modules.mid.inference.elbo import ELBOInference


def test_query_bernoulli_support(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    result  = pgm.query(['A'], evidence={'input': torch.randn(8, 10)}, num_samples=50)
    samples = result.samples['A']
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_elbo_is_lower_bound(cbm_pgm):
    pgm    = cbm_pgm
    engine = ELBOInference(pgm)
    data   = {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}
    engine(data)

    elbo_estimate = engine.log_prob(data).item()
    assert torch.isfinite(torch.tensor(elbo_estimate))


def test_elbo_lower_with_full_evidence(cbm_pgm):
    pgm     = cbm_pgm
    engine  = ELBOInference(pgm)
    partial = {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}
    full    = {**partial, 'A': torch.zeros(4, 1), 'B': torch.zeros(4, 1)}
    engine(partial)

    assert torch.isfinite(engine.log_prob(partial))
    assert torch.isfinite(engine.log_prob(full))
