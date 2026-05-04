"""Tests for PyroAmortizedGuide."""
import pytest
import torch
import pyro.poutine as poutine

from torch_concepts.nn.modules.mid.inference.guide import AmortizedGuide


def test_amortized_guide_instantiation(cbm_pgm):
    guide = AmortizedGuide(cbm_pgm)
    assert guide is not None


def test_guide_sites_match_latent_variables(cbm_pgm):
    guide = AmortizedGuide(cbm_pgm)
    data  = {'input': torch.randn(4, 16)}

    with poutine.trace() as tr:
        guide(data)

    guide_sites = {n for n, s in tr.trace.nodes.items() if s['type'] == 'sample'}
    assert 'A'    in guide_sites
    assert 'B'    in guide_sites
    assert 'task' in guide_sites
    assert 'input' not in guide_sites


def test_guide_site_names_match_model(cbm_pgm):
    guide = AmortizedGuide(cbm_pgm)
    data  = {'input': torch.randn(4, 16)}

    with poutine.trace() as model_tr:
        cbm_pgm(data)
    with poutine.trace() as guide_tr:
        guide(data)

    model_latent = {n for n, s in model_tr.trace.nodes.items()
                    if s['type'] == 'sample' and not s['is_observed']}
    guide_sites  = {n for n, s in guide_tr.trace.nodes.items()
                    if s['type'] == 'sample'}
    assert guide_sites == model_latent


def test_guide_has_parameters(cbm_pgm):
    guide  = AmortizedGuide(cbm_pgm)
    params = list(guide.parameters())
    assert len(params) > 0
