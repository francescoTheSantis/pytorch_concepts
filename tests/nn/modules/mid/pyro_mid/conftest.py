"""Shared fixtures for Pyro mid-level API tests."""
import pytest
import pyro
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, RelaxedBernoulli, Normal
from torch.testing import assert_close

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, ExogenousVariable, LatentVariable,
)
from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.elbo import ELBOInference
from torch_concepts.nn.modules.mid.inference.guide import AmortizedGuide


@pytest.fixture(autouse=True)
def reset_pyro():
    """Clear Pyro's param store and reset settings before every test."""
    pyro.clear_param_store()
    pyro.settings.set(module_local_params=True)
    yield
    pyro.clear_param_store()


@pytest.fixture
def binary_chain_pgm():
    """
    Minimal DAG:  input (Delta/10) -> A (Bernoulli/1) -> B (Bernoulli/1)
    Returns (pgm, input_var, var_A, var_B).
    """
    input_var = LatentVariable(concept='input', distribution=Delta, size=10)
    var_A     = ConceptVariable(concept='A', distribution=Bernoulli, size=1)
    var_B     = ConceptVariable(concept='B', distribution=Bernoulli, size=1)

    cpd_input = ParametricCPD(concept='input', parametrization=nn.Identity())
    cpd_A     = ParametricCPD(concept='A', parametrization=nn.Linear(10, 1), parents=[input_var])
    cpd_B     = ParametricCPD(concept='B', parametrization=nn.Linear(1, 1),  parents=[var_A])

    pgm = BayesianNetwork(
        variables=[input_var, var_A, var_B],
        factors=[cpd_input, cpd_A, cpd_B],
    )
    return pgm, input_var, var_A, var_B


@pytest.fixture
def cbm_pgm():
    """
    Classic CBM:  input -> {A, B} (RelaxedBernoulli) -> task (Bernoulli)
    Returns pgm.
    """
    input_var = LatentVariable(concept='input', distribution=Delta, size=16)
    var_A     = ConceptVariable(concept='A', distribution=RelaxedBernoulli, size=1)
    var_B     = ConceptVariable(concept='B', distribution=RelaxedBernoulli, size=1)
    task_var  = ConceptVariable(concept='task', distribution=Bernoulli, size=1)

    cpd_input = ParametricCPD(concept='input', parametrization=nn.Identity())
    cpd_A     = ParametricCPD(concept='A',    parametrization=nn.Linear(16, 1), parents=[input_var])
    cpd_B     = ParametricCPD(concept='B',    parametrization=nn.Linear(16, 1), parents=[input_var])
    cpd_task  = ParametricCPD(concept='task', parametrization=nn.Linear(2,  1), parents=[var_A, var_B])

    pgm = BayesianNetwork(
        variables=[input_var, var_A, var_B, task_var],
        factors=[cpd_input, cpd_A, cpd_B, cpd_task],
    )
    return pgm


def pgm_train_and_fit(pgm, steps: int = 3, batch: int = 4):
    """Helper: run *steps* gradient steps on *pgm* to initialise parameters."""
    engine    = ELBOInference(pgm)
    dummy = _make_dummy_data(pgm, batch)
    # warm-up forward pass to initialise lazy parameters
    engine(dummy)

    optimizer = torch.optim.Adam(engine.parameters(), lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad()
        engine(dummy).backward()
        optimizer.step()
    return engine


def _make_dummy_data(pgm, batch: int = 4) -> dict:
    """Build a dummy data dict that satisfies *pgm*'s root variable."""
    data = {}
    for var in pgm.variables:
        cpd = pgm.get_module_of_concept(var.concept)
        if cpd is not None and not cpd.parents:
            # Root variable → must be in data
            data[var.concept] = torch.zeros(batch, var.size)
        elif var.distribution.__name__ == 'Delta':
            data[var.concept] = torch.zeros(batch, var.size)
    # Add observed concept labels (zeros) for all ConceptVariables that are
    # leaves (no children referencing them as parents — just use all)
    for var in pgm.variables:
        if isinstance(var, ConceptVariable) and var.concept not in data:
            data[var.concept] = torch.zeros(batch, var.size)
    return data
