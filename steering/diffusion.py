"""DDPM over plain vectors, plus the SDEdit sampler the steering path uses.

Training and sampling are Ho et al. (2020) Algorithms 1 and 2 verbatim, with
``sigma_t^2 = beta_t``. The only addition is :meth:`DDPM.sdedit`, which pins some
coordinates to a reference while the rest evolve.

``python -m steering.diffusion`` runs the whole thing on a 2-D toy and writes a
figure: the data, unconditional samples, and SDEdit with one dimension pinned.
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class DDPM(nn.Module):
    """Denoising diffusion over ``R^dim``.

    The per-dimension standardisation is part of the model rather than the
    caller's job: this runs over ``[z; e]``, where an autoencoder's ``z`` has
    whatever scale training left it at while ``e`` is order 1. DDPM's schedule
    assumes roughly unit-scale data, so mixing the two raw would quietly let one
    block dominate the objective. :meth:`fit_scaler` records the statistics;
    every entry point converts on the way in and back on the way out.
    """

    def __init__(self, dim: int, steps: int = 1000, hidden: int = 512,
                 time_dim: int = 128):
        super().__init__()
        self.dim, self.steps, self.time_dim = dim, steps, time_dim

        # Ho et al.'s linear 1e-4 -> 0.02 schedule is calibrated for T = 1000; its
        # total noise budget is `sum(betas)`, and the forward process only reaches
        # pure noise because that sum is ~10. Reused verbatim at a smaller T it
        # silently stops destroying the signal -- measured alpha_bar_T: 0.017 at
        # T=400, 0.13 at T=200, 0.36 at T=100, against 4e-5 at T=1000. Sampling
        # then starts from a `randn` the model never saw at training time, and
        # every SDEdit noise level is wrong with it. Rescaling by 1000/T holds the
        # budget fixed, so T is purely a compute knob (alpha_bar_T ~ 3e-5 for all
        # of the above).
        betas = (torch.linspace(1e-4, 0.02, steps) * (1000.0 / steps)).clamp(max=0.999)
        alphas = 1.0 - betas
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        # Index 0 is t = 0, where alpha_bar = 1: `q_sample` at t = 0 returns the
        # reference untouched, which is what pins SDEdit's fixed block exactly.
        self.register_buffer('alphas_cumprod',
                             torch.cat([torch.ones(1), alphas.cumprod(0)]))
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))

        self.eps_net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    # -- scaling ---------------------------------------------------------------
    def fit_scaler(self, x0: torch.Tensor) -> None:
        self.mean.copy_(x0.mean(0))
        self.std.copy_(x0.std(0).clamp_min(1e-6))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    # -- the model -------------------------------------------------------------
    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Standard sinusoidal embedding of the integer step."""
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        angles = t.float().unsqueeze(-1) * freqs
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def eps(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.eps_net(torch.cat([x_t, self._time_embedding(t)], dim=-1))

    def q_sample(self, x0: torch.Tensor, t, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: ``sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps``.

        ``x0`` is already normalised. ``t`` may be a scalar or a per-row tensor.
        """
        t = torch.as_tensor(t, device=x0.device).reshape(-1)
        a_bar = self.alphas_cumprod[t].reshape(-1, 1)
        noise = torch.randn_like(x0) if noise is None else noise
        return a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Algorithm 1: predict the noise added at a uniformly drawn step."""
        x0 = self.normalize(x0)
        t = torch.randint(1, self.steps + 1, (len(x0),), device=x0.device)
        noise = torch.randn_like(x0)
        return nn.functional.mse_loss(self.eps(self.q_sample(x0, t, noise), t), noise)

    def _reverse_step(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One step of Algorithm 2: ``x_t -> x_{t-1}``."""
        step = torch.full((len(x_t),), t, device=x_t.device, dtype=torch.long)
        alpha, a_bar = self.alphas[t - 1], self.alphas_cumprod[t]
        mean = (x_t - (1 - alpha) / (1 - a_bar).sqrt() * self.eps(x_t, step)) / alpha.sqrt()
        if t == 1:
            return mean
        return mean + self.betas[t - 1].sqrt() * torch.randn_like(x_t)

    # -- sampling --------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """Algorithm 2, from pure noise."""
        x = torch.randn(n, self.dim, device=self.mean.device)
        for t in range(self.steps, 0, -1):
            x = self._reverse_step(x, t)
        return self.denormalize(x)

    @torch.no_grad()
    def sdedit(
        self,
        reference: torch.Tensor,
        fixed_mask: torch.Tensor,
        release_step: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Reverse diffusion with part of the vector pinned to ``reference``.

        After every reverse step the pinned coordinates are overwritten with the
        reference *re-noised to that level*, ``q_sample(reference, t-1)``, rather
        than with the clean value -- so the network always sees an input with the
        noise statistics it was trained on. At ``t = 0`` the noise is zero, so the
        pinned block lands exactly on its reference value.

        Args:
            reference: ``(B, dim)`` un-normalised, e.g. ``[z ; e~]``.
            fixed_mask: ``(dim,)`` bool. ``True`` coordinates (the concept
                embedding) are pinned for the whole chain.
            release_step: The free coordinates (``z``) are pinned while
                ``t >= release_step`` and evolve on their own below it. ``0``
                leaves the reference untouched, ``steps`` regenerates it from
                noise; in between is the SDEdit strength knob.

        Returns:
            ``(B, dim)`` un-normalised.
        """
        device = self.mean.device
        reference = self.normalize(reference.to(device))
        fixed_mask = fixed_mask.to(device)

        # The literal procedure runs from t = steps with *both* blocks pinned and
        # releases the free one at `release_step`. But while both are pinned every
        # coordinate is overwritten each step, so those iterations cannot
        # influence the result: the state on arrival at `release_step` is exactly
        # `q_sample(reference, release_step)` either way. Starting there is the
        # same chain for `steps / release_step` less work.
        x = self.q_sample(reference, release_step)
        for t in range(release_step, 0, -1):
            x = self._reverse_step(x, t)
            # The fixed block is re-pinned at every step, including the last, so
            # it ends exactly on its reference value. The free block is not
            # touched again -- it is released from here to t = 0.
            x = torch.where(fixed_mask, self.q_sample(reference, t - 1), x)
        return self.denormalize(x)


def train_ddpm(
    model: DDPM,
    x0: torch.Tensor,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True,
) -> DDPM:
    """Fit the scaler and the noise network on ``x0 (N, dim)``."""
    device = device or torch.device('cpu')
    model = model.to(device)
    x0 = x0.to(device)
    model.fit_scaler(x0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x0), device=device)
        total = 0.0
        for start in range(0, len(x0), batch_size):
            batch = x0[order[start:start + batch_size]]
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"  DDPM epoch {epoch:4d} | eps MSE {total / len(x0):.5f}")
    model.eval()
    return model


# --------------------------------------------------------------------- toy test
def _toy_main():
    """2-D test on a curve, mirroring the real pipeline's geometry.

    Dimension 0 plays the role of ``e`` (pinned to a value we choose) and
    dimension 1 the role of ``z`` (released, so it must move onto the curve at
    the new dimension-0 value). If SDEdit works, the third panel sits on the
    curve, shifted along it -- not on the original points and not on the whole
    unconditional cloud.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    from torch_concepts import seed_everything
    from steering import resolve_device

    seed_everything(0)
    device = resolve_device()
    print(f"device: {device}")
    n, shift, release = 2000, 1.2, 160

    x0 = torch.rand(n, 1) * 4 - 2
    data = torch.cat([x0, (3 * x0).sin() + 0.15 * torch.randn(n, 1)], dim=-1)

    model = train_ddpm(DDPM(dim=2), data, epochs=300)

    unconditional = model.sample(n)

    # Pin dimension 0 to a shifted value; dimension 1 has to follow the curve.
    columns = torch.arange(600)
    reference = data[columns].clone()
    reference[:, 0] = (reference[:, 0] + shift).clamp(-2, 2)
    edited = model.sdedit(reference, torch.tensor([True, False]), release)

    curve = torch.linspace(-2, 2, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    panels = [('data', data, 'tab:blue'),
              ('unconditional samples', unconditional, 'tab:orange'),
              (f'SDEdit: dim 0 pinned to +{shift}, release t={release}',
               edited, 'tab:green')]
    for ax, (title, points, color) in zip(axes, panels):
        ax.plot(curve, (3 * curve).sin(), color='k', lw=1, alpha=0.4, zorder=0)
        ax.scatter(points[:, 0], points[:, 1], s=4, alpha=0.4, color=color)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('dim 0  ("e")')
    axes[0].set_ylabel('dim 1  ("z")')

    path = Path(__file__).parent / 'figures' / 'diffusion_toy.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # The claim the figure makes, checked numerically.
    on_curve = (edited[:, 1] - (3 * edited[:, 0]).sin()).abs().mean()
    pinned = (edited[:, 0] - reference[:, 0]).abs().max()
    print(f"unconditional |dim1 - sin(3 dim0)| = "
          f"{(unconditional[:, 1] - (3 * unconditional[:, 0]).sin()).abs().mean():.4f}")
    print(f"SDEdit        |dim1 - sin(3 dim0)| = {on_curve:.4f}  (data noise 0.15)")
    print(f"SDEdit        max |dim0 - pinned|  = {pinned:.2e}")
    print(f"wrote {path}")


if __name__ == '__main__':
    _toy_main()
