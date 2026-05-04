"""Tests for inference engine construction and configuration."""
import pytest
import torch
import torch.nn as nn
import pyro
from pyro.infer.autoguide import AutoNormal, AutoDelta

from torch_concepts.nn.modules.mid.inference.elbo import ELBOInference
from torch_concepts.nn.modules.mid.inference.guide import AmortizedGuide
from torch_concepts.nn.modules.mid.inference.deterministic import DeterministicInference as BayesianInference


def test_bayesian_inference_is_nn_module(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    engine = BayesianInference(pgm)
    assert isinstance(engine, nn.Module)


def test_bayesian_inference_query_returns_inference_output(binary_chain_pgm):
    from torch_concepts.nn.modules.outputs import InferenceOutput
    pgm, *_ = binary_chain_pgm
    engine = BayesianInference(pgm)
    result = engine.query(['A', 'B'], evidence={'input': torch.randn(4, 10)})
    assert isinstance(result, InferenceOutput)


def test_bayesian_inference_logits_shape(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    engine = BayesianInference(pgm)
    result = engine.query(['A', 'B'], evidence={'input': torch.randn(4, 10)},
                          return_parameters=True, return_probs=False)
    # A(1) + B(1) = 2 features concatenated
    assert result.parameters is not None
    assert result.parameters.shape == (4, 2)


def test_bayesian_inference_probs_in_zero_one(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    engine = BayesianInference(pgm)
    result = engine.query(['A'], evidence={'input': torch.randn(8, 10)},
                          return_probs=True)
    assert result.probs is not None
    assert result.probs.min() >= 0.0
    assert result.probs.max() <= 1.0


def test_elbo_inference_default_guide(cbm_pgm):
    pgm    = cbm_pgm
    engine = ELBOInference(pgm)
    assert isinstance(engine.elbo_module, pyro.infer.elbo.ELBOModule)


def test_elbo_inference_auto_normal_guide(cbm_pgm):
    pgm    = cbm_pgm
    engine = ELBOInference(pgm, guide=AutoNormal)
    assert engine.elbo_module is not None


def test_elbo_inference_explicit_guide(cbm_pgm):
    pgm    = cbm_pgm
    guide  = AmortizedGuide(pgm)
    engine = ELBOInference(pgm, guide=guide)
    assert engine.elbo_module is not None


def test_elbo_inference_unknown_method_on_model_raises(binary_chain_pgm):
    pgm, *_ = binary_chain_pgm
    with pytest.raises((ValueError, AttributeError, TypeError)):
        # Passing a nonsensical object as guide should raise at init or call time
        engine = ELBOInference(pgm, guide="gibbs_sampling")
        engine({'input': torch.randn(2, 10)})
