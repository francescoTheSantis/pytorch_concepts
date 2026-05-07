"""Integration tests: spec §9 and §10 end-to-end examples."""
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from torch.nn.functional import binary_cross_entropy_with_logits

from torch_concepts.nn.modules.mid.models.variable import (
    ConceptVariable, LatentVariable,
)
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import (
    ProbabilisticModel,
)
from torch_concepts.nn.modules.mid.inference.svi import SVIInference
from torch_concepts.nn.modules.mid.inference.map import MAPInference
from torch_concepts.nn.modules.mid.inference.mcmc import MCMCInference


pyro.settings.set(module_local_params=True)


def test_section_9_supervised_cbm_end_to_end():
    torch.manual_seed(42)
    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    input_var = LatentVariable("input", distribution=dist.Delta, size=16)
    c1_var = ConceptVariable("c1", distribution=dist.Bernoulli, size=1)
    c2_var = ConceptVariable("c2", distribution=dist.Bernoulli, size=1)
    task_var = ConceptVariable("task", distribution=dist.Bernoulli, size=1)

    pgm = ProbabilisticModel(
        variables=[input_var, c1_var, c2_var, task_var],
        factors=[
            ParametricCPD("input", parametrization=None),
            ParametricCPD(
                "c1", parametrization=nn.Linear(16, 1), parents=[input_var]
            ),
            ParametricCPD(
                "c2", parametrization=nn.Linear(16, 1), parents=[input_var]
            ),
            ParametricCPD(
                "task",
                parametrization=nn.Sequential(
                    nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)
                ),
                parents=[c1_var, c2_var],
            ),
        ],
    )

    inference = SVIInference(pgm)
    optimizer = torch.optim.Adam(pgm.parameters(), lr=1e-2)

    def loss_fn(parameters, labels):
        loss = torch.tensor(0.0)
        for var_name, label in labels.items():
            logits = parameters[var_name]["logits"]
            loss = loss + binary_cross_entropy_with_logits(
                logits, label.float()
            )
        return loss

    B = 8
    x = torch.randn(B, 16)
    c1l = (torch.rand(B, 1) > 0.5).long()
    c2l = (torch.rand(B, 1) > 0.5).long()
    tl = (torch.rand(B, 1) > 0.5).long()

    pgm.train()
    initial_loss = None
    final_loss = None
    for epoch in range(20):
        optimizer.zero_grad()
        result = inference.query(
            variables=["c1", "c2", "task"], evidence={"input": x}
        )
        labels = {"c1": c1l, "c2": c2l, "task": tl}
        loss = loss_fn(result.parameters, labels)
        loss.backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss, (
        f"loss did not decrease ({initial_loss} → {final_loss})"
    )

    # MAP evaluation
    pgm.eval()
    map_inference = MAPInference(pgm, algorithm="auto_delta")
    x_test = torch.randn(4, 16)
    with torch.no_grad():
        result = map_inference.query(
            variables=["c1", "c2", "task"], evidence={"input": x_test}
        )
        predictions = {
            name: (vp["logits"] > 0).long()
            for name, vp in result.parameters.items()
        }
    for name in ("c1", "c2", "task"):
        assert predictions[name].shape == (4, 1)

    # MCMC test-time query
    mcmc_inference = MCMCInference(pgm, num_samples=20, warmup_steps=0)
    c1_observed = torch.zeros(4, 1)
    with torch.no_grad():
        result = mcmc_inference.query(
            variables=["c2"],
            evidence={"input": x_test, "c1": c1_observed},
        )
    assert result.parameters["c2"]["logits"].shape == (4, 1)
    assert result.samples["c2"].shape == (20, 4, 1)


def test_section_10_partially_observed_training_end_to_end():
    torch.manual_seed(0)
    pyro.set_rng_seed(0)
    pyro.settings.set(module_local_params=True)
    pyro.clear_param_store()

    input_var = LatentVariable("input", distribution=dist.Delta, size=16)
    c1_var = ConceptVariable("c1", distribution=dist.Bernoulli, size=1)
    z_var = LatentVariable("z", distribution=dist.Normal, size=8)
    task_var = ConceptVariable("task", distribution=dist.Bernoulli, size=1)

    pgm = ProbabilisticModel(
        variables=[input_var, c1_var, z_var, task_var],
        factors=[
            ParametricCPD("input", parametrization=None),
            ParametricCPD(
                "c1", parametrization=nn.Linear(16, 1), parents=[input_var]
            ),
            ParametricCPD(
                "z", parametrization=nn.Linear(16, 16), parents=[input_var]
            ),
            ParametricCPD(
                "task",
                parametrization=nn.Sequential(
                    nn.Linear(1 + 8, 16), nn.ReLU(), nn.Linear(16, 1)
                ),
                parents=[c1_var, z_var],
            ),
        ],
    )

    inference = SVIInference(pgm)
    optimizer = torch.optim.Adam(pgm.parameters(), lr=1e-3)

    def loss_fn(parameters, labels, latent_params, prior_params):
        nll = torch.tensor(0.0)
        for var_name, label in labels.items():
            logits = parameters[var_name]["logits"]
            nll = nll + binary_cross_entropy_with_logits(
                logits, label.float(), reduction="sum"
            )
        kl_total = torch.tensor(0.0)
        for var_name in latent_params:
            q = dist.Normal(
                latent_params[var_name]["loc"],
                latent_params[var_name]["scale"],
            ).to_event(1)
            p = dist.Normal(
                prior_params[var_name]["loc"],
                prior_params[var_name]["scale"],
            ).to_event(1)
            kl_total = kl_total + torch.distributions.kl.kl_divergence(
                q, p
            ).sum()
        return nll + kl_total

    B = 8
    x = torch.randn(B, 16)
    c1l = (torch.rand(B, 1) > 0.5).long()
    tl = (torch.rand(B, 1) > 0.5).long()

    pgm.train()
    initial_loss = None
    final_loss = None
    for epoch in range(30):
        optimizer.zero_grad()
        result = inference.query(
            variables=["c1", "task"], evidence={"input": x}
        )
        labels = {"c1": c1l, "task": tl}
        loss = loss_fn(
            result.parameters,
            labels,
            result.latent_params,
            result.prior_params,
        )
        loss.backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss
    assert "z" in result.latent_params
    assert result.latent_params["z"]["loc"].shape == (B, 8)
