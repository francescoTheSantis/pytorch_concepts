"""Tests for Variable.make_distribution and related Pyro extensions."""
import pytest
import torch
import pyro
from torch.distributions import Bernoulli, Normal, OneHotCategorical
from torch.testing import assert_close

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, ExogenousVariable, LatentVariable,
)
from torch_concepts.distributions import Delta


def test_make_distribution_bernoulli_shape():
    var    = ConceptVariable(concept='c', distribution=Bernoulli, size=1)
    params = torch.zeros(4, 1)
    d      = var.make_distribution(params)
    sample = d.sample()
    assert sample.shape == (4, 1)
    # make_distribution wraps Bernoulli in Independent(.to_event(1))
    assert isinstance(d.base_dist, pyro.distributions.Bernoulli)


def test_make_distribution_bernoulli_support():
    var     = ConceptVariable(concept='c', distribution=Bernoulli, size=1)
    params  = torch.randn(8, 1)
    d       = var.make_distribution(params)
    samples = d.sample((10,))
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_make_distribution_normal_loc_scale():
    var    = ConceptVariable(concept='c', distribution=Normal, size=4)
    params = torch.randn(4, 8)   # 4 loc + 4 scale = 8
    d      = var.make_distribution(params)
    assert d.base_dist.loc.shape   == (4, 4)
    assert d.base_dist.scale.shape == (4, 4)
    assert (d.base_dist.scale > 0).all()


def test_make_distribution_categorical_shape():
    var    = ConceptVariable(concept='c', distribution=OneHotCategorical, size=3)
    params = torch.randn(4, 3)
    d      = var.make_distribution(params)
    sample = d.sample()
    assert sample.shape == (4, 3)


def test_make_distribution_to_event():
    var    = ConceptVariable(concept='c', distribution=Normal, size=4)
    params = torch.randn(4, 8)
    d      = var.make_distribution(params)
    assert d.event_dim == 1
    assert d.log_prob(d.sample()).shape == (4,)


def test_make_distribution_bernoulli_log_prob_shape():
    var    = ConceptVariable(concept='c', distribution=Bernoulli, size=1)
    params = torch.randn(4, 1)
    d      = var.make_distribution(params)
    lp     = d.log_prob(d.sample())
    assert lp.shape == (4,)


def test_is_observed_exogenous():
    var = ExogenousVariable(concept='input', distribution=Delta, size=10)
    assert var.is_observed is True


def test_is_observed_concept():
    var = ConceptVariable(concept='A', distribution=Bernoulli, size=1)
    assert var.is_observed is False


def test_is_observed_latent():
    var = LatentVariable(concept='z', distribution=Normal, size=4)
    assert var.is_observed is False


def test_pyro_site_name():
    var = ConceptVariable(concept='my_concept', distribution=Bernoulli, size=1)
    assert var.pyro_site_name == 'my_concept'
