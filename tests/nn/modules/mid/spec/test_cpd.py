"""Unit tests for spec §3: ParametricCPD."""
import pytest
import torch.nn as nn
import pyro
import pyro.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, LatentVariable,
)
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.factor import ParametricFactor
from torch_concepts.nn.modules.mid.models.probabilistic_model import (
    ProbabilisticModel,
)


pyro.settings.set(module_local_params=True)


class TestParametricFactor:
    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(NotImplementedError):
            ParametricFactor()


class TestParametricCPDConstruction:
    def test_evidence_only_root(self):
        cpd = ParametricCPD(concept="x", parametrization=None)
        assert cpd.concept == "x"
        assert cpd.parametrization is None
        assert cpd.parents == []

    def test_single_with_parents(self):
        v = LatentVariable(concept="x", distribution=dist.Delta, size=4)
        cpd = ParametricCPD(
            concept="c1", parametrization=nn.Linear(4, 1), parents=[v]
        )
        assert cpd.concept == "c1"
        assert cpd.parents == [v]

    def test_multi_returns_list_with_independent_modules(self):
        v = LatentVariable(concept="x", distribution=dist.Delta, size=4)
        cpds = ParametricCPD(
            concepts=["c1", "c2"], parametrization=nn.Linear(4, 1), parents=[v]
        )
        assert isinstance(cpds, list) and len(cpds) == 2
        # Deep-copied — the underlying weight tensors must not share storage.
        w1 = cpds[0].parametrization.weight
        w2 = cpds[1].parametrization.weight
        assert w1.data_ptr() != w2.data_ptr()

    def test_neither_or_both_concept_raises(self):
        with pytest.raises(ValueError):
            ParametricCPD(parametrization=None)
        with pytest.raises(ValueError):
            ParametricCPD(
                concept="c1", concepts=["c2"], parametrization=None
            )


class TestOutputDimValidation:
    """Spec §3.2 / §8 rule 9: validation against ``param_dim``."""

    def test_mismatch_raises_value_error(self):
        x = LatentVariable(concept="x", distribution=dist.Delta, size=4)
        c1 = ConceptVariable(concept="c1", distribution=dist.Bernoulli, size=1)
        # param_dim(Bernoulli, 1) = 1 — passing Linear(4, 2) is wrong.
        with pytest.raises(ValueError):
            ProbabilisticModel(
                variables=[x, c1],
                factors=[
                    ParametricCPD("x", parametrization=None),
                    ParametricCPD(
                        "c1", parametrization=nn.Linear(4, 2), parents=[x]
                    ),
                ],
            )

    def test_correct_dim_passes(self):
        x = LatentVariable(concept="x", distribution=dist.Delta, size=4)
        c1 = ConceptVariable(concept="c1", distribution=dist.Normal, size=3)
        # param_dim(Normal, 3) = 6
        ProbabilisticModel(
            variables=[x, c1],
            factors=[
                ParametricCPD("x", parametrization=None),
                ParametricCPD(
                    "c1", parametrization=nn.Linear(4, 6), parents=[x]
                ),
            ],
        )
