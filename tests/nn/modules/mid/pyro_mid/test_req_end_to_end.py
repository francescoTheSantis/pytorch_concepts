"""Tests for R4: Full end-to-end pipeline."""
import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close
from torch.distributions import Bernoulli

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, LatentVariable,
)
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.elbo import ELBOInference
from torch_concepts.nn.modules.mid.inference.guide import AmortizedGuide


def _train(engine, data, steps=5, lr=1e-2):
    optimizer = torch.optim.Adam(engine.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        (-engine.log_prob(data)).backward()
        optimizer.step()


def test_e2e_binary_cbm():
    input_var = LatentVariable(concept='input', distribution=Delta, size=8)
    var_A     = ConceptVariable(concept='A', distribution=Bernoulli, size=1)
    var_B     = ConceptVariable(concept='B', distribution=Bernoulli, size=1)
    task_var  = ConceptVariable(concept='task', distribution=Bernoulli, size=1)

    pgm = BayesianNetwork(
        variables=[input_var, var_A, var_B, task_var],
        factors=[
            ParametricCPD(concept='input', parametrization=nn.Identity()),
            ParametricCPD(concept='A',    parametrization=nn.Linear(8, 1), parents=[input_var]),
            ParametricCPD(concept='B',    parametrization=nn.Linear(8, 1), parents=[input_var]),
            ParametricCPD(concept='task', parametrization=nn.Linear(2, 1), parents=[var_A, var_B]),
        ],
    )

    engine = ELBOInference(pgm)
    dummy  = {'input': torch.zeros(2, 8), 'task': torch.zeros(2, 1)}
    engine(dummy)  # warm-up

    _train(engine, {'input': torch.randn(8, 8), 'task': torch.zeros(8, 1)})

    result = pgm.query(['task', 'A', 'B'],
                       evidence={'input': torch.randn(4, 8)},
                       num_samples=10)
    assert result.samples['task'].shape == (10, 4, 1)
    assert result.samples['A'].shape    == (10, 4, 1)
    assert result.samples['B'].shape    == (10, 4, 1)


def test_e2e_auto_normal_guide(cbm_pgm):
    from pyro.infer.autoguide import AutoNormal

    pgm    = cbm_pgm
    batch  = 4
    engine = ELBOInference(pgm, guide=AutoNormal)
    init   = {'input': torch.zeros(batch, 16), 'task': torch.zeros(batch, 1)}
    engine(init)  # warm-up — pins AutoNormal to this batch size

    _train(engine, {'input': torch.randn(batch, 16), 'task': torch.zeros(batch, 1)})

    result = pgm.query(['task'], evidence={'input': torch.randn(batch, 16)},
                       guide=engine.elbo_module.guide, num_samples=5)
    assert result.samples['task'].shape == (5, batch, 1)


def test_e2e_amortized_guide(cbm_pgm):
    pgm    = cbm_pgm
    guide  = AmortizedGuide(pgm)
    engine = ELBOInference(pgm, guide=guide)
    init   = {'input': torch.zeros(2, 16), 'task': torch.zeros(2, 1)}
    engine(init)

    _train(engine, {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)})

    result = pgm.query(['A', 'B', 'task'],
                       evidence={'input': torch.randn(3, 16)},
                       guide=guide, num_samples=8)
    assert result.samples['task'].shape == (8, 3, 1)


def test_e2e_test_loss_finite(cbm_pgm):
    pgm    = cbm_pgm
    engine = ELBOInference(pgm)
    init   = {'input': torch.zeros(2, 16), 'task': torch.zeros(2, 1)}
    engine(init)

    _train(engine, {'input': torch.randn(8, 16), 'task': torch.zeros(8, 1)})

    engine.eval()
    with torch.no_grad():
        elbo = engine.log_prob({'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)})
    assert torch.isfinite(elbo)


def test_e2e_map_deterministic_query():
    from torch.distributions import Normal
    from pyro.infer.autoguide import AutoDelta

    input_var = LatentVariable(concept='input', distribution=Delta, size=8)
    var_z     = ConceptVariable(concept='z', distribution=Normal, size=2)

    pgm = BayesianNetwork(
        variables=[input_var, var_z],
        factors=[
            ParametricCPD(concept='input', parametrization=nn.Identity()),
            ParametricCPD(concept='z',     parametrization=nn.Linear(8, 4), parents=[input_var]),
        ],
    )
    batch  = 4
    engine = ELBOInference(pgm, guide=AutoDelta)
    engine({'input': torch.zeros(batch, 8)})  # warm-up

    _train(engine, {'input': torch.randn(batch, 8)}, steps=3)

    guide = engine.elbo_module.guide
    x  = torch.randn(batch, 8)
    r1 = pgm.query(['z'], evidence={'input': x}, guide=guide, num_samples=1)
    r2 = pgm.query(['z'], evidence={'input': x}, guide=guide, num_samples=1)
    assert_close(r1.samples['z'], r2.samples['z'])


def test_e2e_save_reload(cbm_pgm, tmp_path):
    pgm    = cbm_pgm
    engine = ELBOInference(pgm)
    init   = {'input': torch.zeros(2, 16), 'task': torch.zeros(2, 1)}
    engine(init)

    _train(engine, {'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)}, steps=3)

    path      = tmp_path / "model.pt"
    torch.save(engine.state_dict(), path)
    sd_before = {k: v.clone() for k, v in engine.state_dict().items()}

    engine2 = ELBOInference(cbm_pgm)
    engine2(init)
    engine2.load_state_dict(torch.load(path, weights_only=True))

    for k, v in engine2.state_dict().items():
        assert_close(v, sd_before[k], msg=f"Parameter '{k}' differs after reload")

    with torch.no_grad():
        elbo = engine2.log_prob({'input': torch.randn(4, 16), 'task': torch.zeros(4, 1)})
    assert torch.isfinite(elbo)
