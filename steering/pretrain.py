"""The concept-agnostic pre-trained model: ``x -> z -> x_hat``.

Nothing here knows that concepts exist -- that is the point. Everything the
steering pipeline does downstream treats this model as frozen, and touches it
only through the latents ``z`` it produces and the decoder that consumes them.

``python -m steering.pretrain`` trains one and writes a reconstruction grid.
"""
from typing import Tuple

import torch
import torch.nn as nn

IMAGE_SHAPE = (3, 28, 28)


class ConvAE(nn.Module):
    """Deterministic autoencoder over 3x28x28 images."""

    def __init__(self, latent: int = 32):
        super().__init__()
        self.latent = latent
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),    # 14x14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 7x7
            nn.Flatten(),
        )
        self.head = nn.Linear(64 * 7 * 7, latent)
        # No output non-linearity, on purpose. A `Sigmoid` head saturates on
        # ColorMNIST -- the images are ~96% black and `colorize` zeroes two of the
        # three channels outright -- so the decoder is driven to a large negative
        # pre-activation where the sigmoid gradient vanishes, and training sticks
        # at the constant-zero solution (measured: MSE 0.0373 == E[x^2], worse than
        # predicting the mean image). Linear output reaches 0.0084. Values are
        # clamped to [0, 1] at read-out instead; see `reconstruct`.
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),   # 14x14
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),               # 28x28
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The latent used everywhere downstream."""
        return self.head(self.body(x))

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.decoder(self.encode(x)), x)


class ConvVAE(ConvAE):
    """Same shapes, but ``z`` is a Gaussian posterior. ``encode`` returns its mean,
    so the latent dataset the diffusion model sees stays deterministic."""

    def __init__(self, latent: int = 32, beta: float = 1.0):
        super().__init__(latent)
        self.head = nn.Linear(64 * 7 * 7, 2 * latent)
        self.beta = beta

    def posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loc, logvar = self.head(self.body(x)).chunk(2, dim=-1)
        return loc, logvar.clamp(-8, 8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.posterior(x)[0]

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        loc, logvar = self.posterior(x)
        z = loc + torch.randn_like(loc) * (0.5 * logvar).exp()
        recon = nn.functional.mse_loss(self.decoder(z), x)
        kl = 0.5 * (loc.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
        # Scaled to the same per-pixel units as `recon`, so `beta` reads as a
        # ratio rather than as an artefact of the image size.
        return recon + self.beta * kl / x[0].numel()


def pretrain(
    images: torch.Tensor,
    kind: str = 'ae',
    latent: int = 32,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[nn.Module, nn.Module, torch.Tensor]:
    """Train the pre-trained model and read off its latents.

    Args:
        images: ``(N, 3, 28, 28)`` in ``[0, 1]``.
        kind: ``'ae'`` or ``'vae'``.

    Returns:
        ``(model, decoder, z)`` with ``z`` of shape ``(N, latent)``.
    """
    device = device or torch.device('cpu')
    model = (ConvAE(latent) if kind == 'ae' else ConvVAE(latent)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(images))
        total = 0.0
        for start in range(0, len(images), batch_size):
            batch = images[order[start:start + batch_size]].to(device)
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  {kind} epoch {epoch:3d} | loss {total / len(images):.5f}")
    model.eval()

    with torch.no_grad():
        z = torch.cat([model.encode(images[i:i + batch_size].to(device)).cpu()
                       for i in range(0, len(images), batch_size)])
    return model, model.decoder, z


@torch.no_grad()
def reconstruct(decoder: nn.Module, z: torch.Tensor, device=None) -> torch.Tensor:
    """``z (B, D) -> images (B, 3, 28, 28)``, clamped into the displayable range."""
    device = device or next(decoder.parameters()).device
    return decoder(z.to(device)).clamp(0, 1).cpu()


if __name__ == '__main__':
    from pathlib import Path

    import matplotlib.pyplot as plt

    from torch_concepts import seed_everything
    from steering.data import load_colormnist

    seed_everything(0)
    images, digits, color = load_colormnist(n=10000)

    for kind in ('ae', 'vae'):
        model, decoder, z = pretrain(images, kind=kind, epochs=10)
        mse = (reconstruct(decoder, z[:512]) - images[:512]).pow(2).mean()
        print(f"{kind}: z {tuple(z.shape)} | reconstruction MSE {mse:.5f}")

        columns = torch.arange(8)
        fig, axes = plt.subplots(2, len(columns), figsize=(len(columns), 2.3))
        for ax, original, recon in zip(axes.T, images[columns],
                                       reconstruct(decoder, z[columns])):
            ax[0].imshow(original.permute(1, 2, 0))
            ax[1].imshow(recon.permute(1, 2, 0).clamp(0, 1))
            for cell in ax:
                cell.axis('off')
        axes[0, 0].set_title('original', loc='left', fontsize=8)
        axes[1, 0].set_title('reconstruction', loc='left', fontsize=8)
        path = Path(__file__).parent / 'figures' / f'pretrain_{kind}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  wrote {path}")
