"""Unit tests for spec §4: ProbabilisticModel."""
import pytest
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, LatentVariable,
)
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import (
    ProbabilisticModel,
)


pyro.settings.set(module_local_params=True)


def _build_pgm_with_latent():
    """Graph with a Normal latent z, used by §10 example."""
    pyro.clear_param_store()
    x = LatentVariable("input", distribution=dist.Delta, size=16)
    c1 = ConceptVariable("c1", distribution=dist.Bernoulli, size=1)
    z = LatentVariable("z", distribution=dist.Normal, size=8)
    task = ConceptVariable("task", distribution=dist.Bernoulli, size=1)
    return ProbabilisticModel(
        variables=[x, c1, z, task],
        factors=[
            ParametricCPD("input", parametrization=None),
            ParametricCPD("c1", parametrization=nn.Linear(16, 1), parents=[x]),
            ParametricCPD("z", parametrization=nn.Linear(16, 16), parents=[x]),
            ParametricCPD(
                "task",
                parametrization=nn.Sequential(
                    nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 1)
                ),
                parents=[c1, z],
            ),
        ],
    )


class TestTopologicalSort:
    def test_sorted_after_unordered_input(self):
        pyro.clear_param_store()
        x = LatentVariable("x", distribution=dist.Delta, size=4)
        a = ConceptVariable("a", distribution=dist.Bernoulli, size=1)
        b = ConceptVariable("b", distribution=dist.Bernoulli, size=1)
        # Declared b before a, but a → b in the DAG.
        pgm = ProbabilisticModel(
            variables=[b, a, x],
            factors=[
                ParametricCPD("x", parametrization=None),
                ParametricCPD("a", parametrization=nn.Linear(4, 1), parents=[x]),
                ParametricCPD("b", parametrization=nn.Linear(1, 1), parents=[a]),
            ],
        )
        names = [v.concept for v in pgm.sorted_variables]
        assert names.index("x") < names.index("a") < names.index("b")


class TestUndirectedNotImplemented:
    def test_raises(self):
        pyro.clear_param_store()
        x = LatentVariable("x", distribution=dist.Delta, size=2)
        with pytest.raises(NotImplementedError):
            ProbabilisticModel(
                variables=[x],
                factors=[ParametricCPD("x", parametrization=None)],
                directed=False,
            )


class TestEagerGuidePriming:
    """Spec §4.1: pgm.parameters() must include φ after construction."""

    def test_phi_present_after_construction(self):
        pgm = _build_pgm_with_latent()
        names = {n for n, _ in pgm.named_parameters()}
        # θ — the Linear layers in the four CPDs.
        theta_present = any(
            n.startswith("factors.c1") for n in names
        ) and any(n.startswith("factors.task") for n in names)
        # φ — AutoNormal locs/scales for the latent ``z`` site.
        phi_present = any("locs.z" in n for n in names) and any(
            "scales.z" in n for n in names
        )
        assert theta_present, f"missing θ; have {names}"
        assert phi_present, f"missing φ; have {names}"

    def test_priming_does_not_leak_to_global_store(self):
        # The priming pass is wrapped in poutine.block so the *global*
        # Pyro param store should remain empty after construction.
        pyro.clear_param_store()
        _build_pgm_with_latent()
        store = pyro.get_param_store()
        assert len(list(store.keys())) == 0


class TestRootEvidenceRequired:
    def test_missing_root_evidence_raises(self):
        pyro.clear_param_store()
        x = LatentVariable("x", distribution=dist.Delta, size=2)
        c1 = ConceptVariable("c1", distribution=dist.Bernoulli, size=1)
        pgm = ProbabilisticModel(
            variables=[x, c1],
            factors=[
                ParametricCPD("x", parametrization=None),
                ParametricCPD("c1", parametrization=nn.Linear(2, 1), parents=[x]),
            ],
        )
        with pytest.raises(ValueError):
            pgm.forward(evidence={"c1": torch.zeros(2, 1)})
