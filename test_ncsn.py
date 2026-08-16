import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import Normal

from torch_concepts import seed_everything, EmbeddingVariable
from torch_concepts.nn import (
    AnnealedLangevinDynamics,
    LangevinDynamics,
    MarkovNetwork,
    ParametricPotential
)

def energy_net(in_features: int, hidden: int = 128) -> nn.Module:
    """One clique's energy. Plain, because the noise level is handled elsewhere.

    ``ParametricPotential(noise_conditioned=True)`` embeds sigma and concatenates it
    onto the scope values before this net is called, so nothing here needs to know
    about noise levels — it just needs `in_features` to include the embedding width.

    SiLU and two hidden layers, both load-bearing: with one ReLU layer the same
    benchmark gave 51%/95% mode coverage on two seeds, and with this one, 100%/100%.
    A ReLU energy has a piecewise-constant score, so there is no curvature inside a
    basin for score matching to fit.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, 1),
    )


def generate_data(num_samples: int) -> torch.Tensor:

    data = torch.randn(num_samples, 2)
    data[: num_samples // 3] += torch.tensor([10.0, 6.0])
    data[num_samples // 3: 2 * num_samples // 3] += torch.tensor([-10.0, -6.0])
    data[2 * num_samples // 3:] += torch.tensor([0.0, -6.0])
    plt.scatter(data[:, 0], data[:, 1], label="True Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.savefig("synthetic_dataset.png")
    # plt.show()
    # plt.close()
    return data

seed_everything(0)

generated_data = generate_data(10000)

# --------------------------------------------------------------------------- model
X = EmbeddingVariable("X", distribution=Normal, size=1)
Y = EmbeddingVariable("Y", distribution=Normal, size=1)

# Dimension of the noise embedding vector
NOISE_EMBEDDING = 32

# Energy function that accepts the variables in the scope + the noise embedding vector
factor = ParametricPotential(
    name="XY",
    scope=[X, Y],
    parametrization=energy_net(in_features=2 + NOISE_EMBEDDING),
    noise_conditioned=True,
    noise_embedding_size=NOISE_EMBEDDING,
)

MRF = MarkovNetwork(variables=[X, Y], factors=[factor])

# Enstablish a noise ladder.
N_LEVELS = 10
SIGMAS = torch.exp(torch.linspace(torch.tensor(12.0).log(), torch.tensor(0.01).log(), N_LEVELS))
print(f"noise ladder: {[round(s, 4) for s in SIGMAS.tolist()]}")

# ---------------------------------------------------------------------- training
optimizer = torch.optim.AdamW(MRF.parameters(), lr=1e-3)
n_epochs = 1000
batch_size = 512

MRF.train()
started = time.time()
for epoch in range(n_epochs):

    # Sample true data
    clean = generated_data[torch.randint(0, len(generated_data), (batch_size,))]

    # Sample noise levels
    sigma = SIGMAS[torch.randint(0, N_LEVELS, (batch_size,))]

    # Sample noise
    z = torch.randn_like(clean)

    # Create the noisy sample
    perturbed = clean + sigma.reshape(-1, 1) * z

    # compute the score for each energy function in the PGM: s(x, sigma) = - \sum_c grad_c E(c, sigma). 
    scores = MRF.compute_score(
        {"X": perturbed[:, 0:1], "Y": perturbed[:, 1:2]}, sigma=sigma
    )
    score_value = torch.cat([scores["X"], scores["Y"]], dim=-1)

    # compute the loss: ||sigma * s(x, sigma) + z||^2
    loss = (sigma.reshape(-1, 1) * score_value + z).pow(2).sum(-1).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(MRF.parameters(), 1.0)
    optimizer.step()

    if epoch % 500 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch:5d} | loss {loss.item():8.4f}")

per_step = (time.time() - started) / n_epochs
print(f"\ntrained in {time.time() - started:.1f}s  ({per_step * 1000:.1f} ms/step)")

# ---------------------------------------------------------------------- sampling
# Annealed Langevin: `step_size` is NCSN's base epsilon and `steps` its per-level L.
# `grad_clip` must be None -- the ladder's step sizes span six orders of magnitude
# and one clip value would flatten every level to the same step.
MRF.eval()
sampler = AnnealedLangevinDynamics(
    MRF,
    sigmas=SIGMAS,
    step_size=2e-5,
    steps=100,
    grad_clip=None,
)

# Unconditional generation
out = sampler.query(query=["X", "Y"], n_samples=10000)
samples = torch.cat([out.samples[["X"]].tensor, out.samples[["Y"]].tensor], dim=-1)
plt.scatter(samples[:, 0], samples[:, 1], label="Samples from Trained Model", alpha=0.1)

# Conditional generation on X=5
y_evidence = torch.ones(100, 1) * (-7.0)
out = sampler.query(query=["X"], evidence={"Y": y_evidence})
samples = torch.cat([out.samples[["X"]].tensor, y_evidence], dim=-1)
plt.scatter(samples[:, 0], samples[:, 1], label="Conditional Samples (Y = -7.0)", alpha=0.1)

# Conditional with RePainting: evidence X=5, initial guess Y=-10
y_evidence = torch.ones(100, 1) * (-6.0)
x_initial_guess = torch.ones(100, 1) * (-10.0)
out = sampler.query(query={"X": x_initial_guess}, evidence={"Y": y_evidence}, release={"X": 3})
samples = torch.cat([out.samples[["X"]].tensor, y_evidence], dim=-1)
plt.scatter(samples[:, 0], samples[:, 1], label="Repaint: evidence (Y = -6.0), initial guess (X = -10.0)", alpha=0.1)


plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.savefig("samples_from_trained_model.png")





