"""Unit tests for spec §2: Variable hierarchy and param_dim table."""
import pytest
import pyro.distributions as dist

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, LatentVariable, ExogenousVariable, Variable, param_dim,
)


class TestVariableConstruction:
    def test_single(self):
        v = ConceptVariable(concept="c1", distribution=dist.Bernoulli, size=1)
        assert isinstance(v, ConceptVariable)
        assert v.concept == "c1"
        assert v.distribution is dist.Bernoulli
        assert v.size == 1
        assert v.metadata["variable_type"] == "concept"

    def test_multi_returns_list(self):
        vs = ConceptVariable(
            concepts=["c1", "c2", "c3"], distribution=dist.Bernoulli, size=1
        )
        assert isinstance(vs, list)
        assert len(vs) == 3
        assert [v.concept for v in vs] == ["c1", "c2", "c3"]
        assert all(isinstance(v, ConceptVariable) for v in vs)
        # Independent metadata copies (mutating one must not affect others).
        vs[0].metadata["k"] = "v"
        assert "k" not in vs[1].metadata

    def test_neither_concept_raises(self):
        with pytest.raises(ValueError):
            ConceptVariable(distribution=dist.Bernoulli, size=1)

    def test_both_concept_and_concepts_raises(self):
        with pytest.raises(ValueError):
            ConceptVariable(
                concept="c1", concepts=["c2"],
                distribution=dist.Bernoulli, size=1,
            )

    def test_latent_and_exogenous_types(self):
        z = LatentVariable(concept="z", distribution=dist.Normal, size=4)
        x = ExogenousVariable(concept="x", distribution=dist.Delta, size=8)
        assert z.metadata["variable_type"] == "latent"
        assert x.metadata["variable_type"] == "exogenous"


class TestParamDim:
    def test_bernoulli(self):
        assert param_dim(dist.Bernoulli, 1) == 1
        assert param_dim(dist.Bernoulli, 5) == 5

    def test_categorical(self):
        assert param_dim(dist.Categorical, 4) == 4

    def test_normal(self):
        assert param_dim(dist.Normal, 1) == 2
        assert param_dim(dist.Normal, 8) == 16

    def test_multivariate_normal(self):
        # k + k(k+1)/2
        assert param_dim(dist.MultivariateNormal, 1) == 1 + 1
        assert param_dim(dist.MultivariateNormal, 3) == 3 + 6

    def test_delta(self):
        assert param_dim(dist.Delta, 7) == 7

    def test_unsupported_distribution(self):
        with pytest.raises(ValueError):
            param_dim(dist.Beta, 1)
