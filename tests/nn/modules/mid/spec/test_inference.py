"""Unit tests for spec §5: SVI / Enumeration / MAP / MCMC inference."""
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
from torch_concepts.nn.modules.mid.inference.svi import SVIInference
from torch_concepts.nn.modules.mid.inference.enumeration import (
    EnumerationInference,
)
from torch_concepts.nn.modules.mid.inference.map import MAPInference


pyro.settings.set(module_local_params=True)


def _supervised_pgm():
    pyro.clear_param_store()
    x = LatentVariable("input", distribution=dist.Delta, size=16)
    c1 = ConceptVariable("c1", distribution=dist.Bernoulli, size=1)
    c2 = ConceptVariable("c2", distribution=dist.Bernoulli, size=1)
    task = ConceptVariable("task", distribution=dist.Bernoulli, size=1)
    return ProbabilisticModel(
        variables=[x, c1, c2, task],
        factors=[
            ParametricCPD("input", parametrization=None),
            ParametricCPD("c1", parametrization=nn.Linear(16, 1), parents=[x]),
            ParametricCPD("c2", parametrization=nn.Linear(16, 1), parents=[x]),
            ParametricCPD(
                "task",
                parametrization=nn.Sequential(
                    nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)
                ),
                parents=[c1, c2],
            ),
        ],
    )


def _latent_pgm():
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


class TestSVINoLatentFastPath:
    def test_returns_param_dict_no_latents(self):
        pgm = _supervised_pgm()
        svi = SVIInference(pgm)
        out = svi.query(
            variables=["c1", "c2", "task"],
            evidence={"input": torch.randn(4, 16)},
        )
        assert set(out.parameters) == {"c1", "c2", "task"}
        assert out.parameters["c1"]["logits"].shape == (4, 1)
        # No latents → no q/p side dicts.
        assert out.latent_params is None
        assert out.prior_params is None

    def test_gradients_flow_back_to_theta(self):
        pgm = _supervised_pgm()
        svi = SVIInference(pgm)
        out = svi.query(["c1"], evidence={"input": torch.randn(4, 16)})
        out.parameters["c1"]["logits"].sum().backward()
        # The Linear(16,1) for c1 must have a non-None grad.
        cpd = pgm.factors["c1"]
        assert cpd.parametrization.weight.grad is not None


class TestSVIWithLatents:
    def test_returns_q_and_p_params(self):
        pgm = _latent_pgm()
        svi = SVIInference(pgm)
        out = svi.query(["c1", "task"], evidence={"input": torch.randn(4, 16)})
        assert "z" in out.latent_params
        assert "z" in out.prior_params
        assert out.latent_params["z"]["loc"].shape == (4, 8)
        assert out.latent_params["z"]["scale"].shape == (4, 8)
        assert out.prior_params["z"]["loc"].shape == (4, 8)
        assert out.prior_params["z"]["scale"].shape == (4, 8)

    def test_kl_is_finite_and_differentiable(self):
        pgm = _latent_pgm()
        svi = SVIInference(pgm)
        out = svi.query(["c1", "task"], evidence={"input": torch.randn(4, 16)})
        q = dist.Normal(
            out.latent_params["z"]["loc"], out.latent_params["z"]["scale"]
        ).to_event(1)
        p = dist.Normal(
            out.prior_params["z"]["loc"], out.prior_params["z"]["scale"]
        ).to_event(1)
        kl = torch.distributions.kl.kl_divergence(q, p).sum()
        assert torch.isfinite(kl)
        kl.backward()  # must not error


class TestEnumerationInference:
    def test_no_latents_equivalent_to_svi(self):
        pgm = _supervised_pgm()
        out = EnumerationInference(pgm).query(
            ["c1"], evidence={"input": torch.randn(2, 16)}
        )
        assert "c1" in out.parameters

    def test_continuous_latent_rejected(self):
        pgm = _latent_pgm()
        with pytest.raises(ValueError):
            # ``z`` is Normal — not enumerable.
            EnumerationInference(pgm).query(
                ["c1"], evidence={"input": torch.randn(2, 16)}
            )


class TestMAPInference:
    def test_auto_delta_no_latents(self):
        pgm = _supervised_pgm()
        m = MAPInference(pgm, algorithm="auto_delta")
        out = m.query(
            ["c1", "c2", "task"], evidence={"input": torch.randn(2, 16)}
        )
        for n in ("c1", "c2", "task"):
            assert out.parameters[n]["logits"].shape == (2, 1)

    def test_infer_discrete_no_latents(self):
        pgm = _supervised_pgm()
        m = MAPInference(pgm, algorithm="infer_discrete")
        out = m.query(
            ["c1", "c2", "task"], evidence={"input": torch.randn(2, 16)}
        )
        # ``infer_discrete`` returns deterministic ``Delta`` sites whose
        # canonical parameter is ``v`` (the argmax assignment).
        assert {"c1", "c2", "task"} <= set(out.parameters)

    def test_invalid_algorithm_raises(self):
        pgm = _supervised_pgm()
        with pytest.raises(ValueError):
            MAPInference(pgm, algorithm="bogus")

    def test_can_be_used_for_training_flag_false(self):
        assert MAPInference.can_be_used_for_training is False
