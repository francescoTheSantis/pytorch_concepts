"""Tests for PyroProbabilisticModel structure and model() forward."""
import pytest
import torch
import pyro.poutine as poutine
from torch.testing import assert_close


def test_pyro_pgm_instantiation(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    assert pgm is not None


def test_model_creates_correct_sites(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    data    = {'input': torch.randn(4, 10)}

    with poutine.trace() as tr:
        pgm(data)

    site_names = {n for n, s in tr.trace.nodes.items() if s['type'] == 'sample'}
    assert 'A' in site_names
    assert 'B' in site_names


def test_model_observed_site_has_obs(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    obs_A   = torch.zeros(4, 1)
    data    = {'input': torch.randn(4, 10), 'A': obs_A}

    with poutine.trace() as tr:
        pgm(data)

    node_A = tr.trace.nodes['A']
    assert node_A['type'] == 'sample'
    assert node_A['is_observed'] is True
    assert_close(node_A['value'], obs_A)


def test_model_latent_site_not_observed(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    data    = {'input': torch.randn(4, 10)}

    with poutine.trace() as tr:
        pgm(data)

    node_A = tr.trace.nodes['A']
    assert node_A['is_observed'] is False


def test_model_topological_order(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    execution_order = []

    with poutine.trace() as tr:
        pgm({'input': torch.randn(4, 10)})

    for name, s in tr.trace.nodes.items():
        if s['type'] == 'sample':
            execution_order.append(name)

    assert execution_order.index('A') < execution_order.index('B')


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_model_output_batch_shape(batch_size, binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    data    = {'input': torch.randn(batch_size, 10)}

    with poutine.trace() as tr:
        pgm(data)

    assert tr.trace.nodes['A']['value'].shape[0] == batch_size
    assert tr.trace.nodes['B']['value'].shape[0] == batch_size


def test_model_log_prob_sum_finite(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    data    = {'input': torch.randn(4, 10), 'A': torch.zeros(4, 1), 'B': torch.zeros(4, 1)}

    trace = poutine.trace(pgm).get_trace(data)
    lp    = trace.log_prob_sum()
    assert torch.isfinite(lp)
