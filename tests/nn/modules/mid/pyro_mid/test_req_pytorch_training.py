"""Tests for R2: Standard PyTorch training loop."""
import pytest
import torch
import torch.nn as nn

from torch_concepts.nn.modules.mid.inference.elbo import ELBOInference


def _make_engine(pgm, batch=4):
    """Build and warm-up an ELBOInference engine."""
    engine = ELBOInference(pgm)
    data   = {'input': torch.zeros(batch, 16), 'task': torch.zeros(batch, 1)}
    engine(data)  # lazy parameter initialisation
    return engine


def test_engine_is_nn_module(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    engine = ELBOInference(pgm)
    assert isinstance(engine, nn.Module)


def test_engine_has_parameters(cbm_pgm):
    engine = _make_engine(cbm_pgm)
    params = list(engine.parameters())
    assert len(params) > 0


def test_adam_optimizer_accepts_engine_parameters(cbm_pgm):
    engine    = _make_engine(cbm_pgm)
    optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)
    assert optimizer is not None


def test_log_prob_returns_scalar(cbm_pgm):
    engine = _make_engine(cbm_pgm)
    data   = {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}
    elbo   = engine.log_prob(data)
    assert elbo.ndim == 0
    assert torch.isfinite(elbo)


def test_negative_log_prob_is_differentiable(cbm_pgm):
    engine = _make_engine(cbm_pgm)
    data   = {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}
    loss   = -engine.log_prob(data)
    loss.backward()  # must not raise


def test_training_loss_decreases(cbm_pgm):
    engine    = _make_engine(cbm_pgm)
    data      = {'input': torch.randn(8, 16), 'task': torch.zeros(8, 1)}
    optimizer = torch.optim.Adam(engine.parameters(), lr=1e-2)
    losses    = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = -engine.log_prob(data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first_avg = sum(losses[:5]) / 5
    last_avg  = sum(losses[-5:]) / 5
    assert last_avg < first_avg, \
        f"Loss did not decrease: first={first_avg:.3f}, last={last_avg:.3f}"


def test_sgd_optimizer_works(cbm_pgm):
    engine    = _make_engine(cbm_pgm)
    data      = {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}
    optimizer = torch.optim.SGD(engine.parameters(), lr=1e-3)
    optimizer.zero_grad()
    (-engine.log_prob(data)).backward()
    optimizer.step()


def test_engine_state_dict_round_trip(cbm_pgm):
    engine = _make_engine(cbm_pgm)
    data   = {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}
    sd     = engine.state_dict()
    engine.load_state_dict(sd)
    assert torch.isfinite(engine.log_prob(data))


def test_gradient_clipping(cbm_pgm):
    engine    = _make_engine(cbm_pgm)
    data      = {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}
    optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)
    optimizer.zero_grad()
    (-engine.log_prob(data)).backward()
    torch.nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
    optimizer.step()
