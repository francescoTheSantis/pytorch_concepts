"""The headline figure: original / reconstructed / intervened.

Row 3 is the whole pipeline in one line -- flip the colour, propagate it through
the MRF, encode the resulting concept set, SDEdit the pre-trained latent against
it, and decode with the *frozen* decoder.
"""
from pathlib import Path
from typing import Dict, Tuple

import torch

from steering.data import COLOR_NAMES, one_hot
from steering.pretrain import reconstruct

Concepts = Dict[str, torch.Tensor]


@torch.no_grad()
def intervene(
    z: torch.Tensor,
    color: torch.Tensor,
    propagate,
    cvae,
    ddpm,
    release_step: int,
    resample: int = 10,
) -> Tuple[torch.Tensor, Concepts]:
    """Flip the colour and steer ``z`` to match. Returns ``(z_tilde, c_tilde)``.

    The concept embedding occupies the second half of the diffusion vector and is
    pinned for the whole reverse chain; ``z`` occupies the first half and is
    released at ``release_step``.
    """
    c_tilde = propagate({'color': one_hot(1 - color, len(COLOR_NAMES))})
    e_tilde = cvae.encode(c_tilde)

    latent = z.shape[-1]
    fixed_mask = torch.arange(latent + e_tilde.shape[-1]) >= latent
    edited = ddpm.sdedit(torch.cat([z, e_tilde], dim=-1), fixed_mask,
                         release_step, resample)
    return edited[:, :latent], c_tilde


def _value(concepts: Concepts, name: str, index: int):
    value = concepts[name][index].argmax(-1).item()
    return COLOR_NAMES[value] if name == 'color' else value


def _label(concepts: Concepts, index: int) -> str:
    """``4, green`` -- the *known* concepts of one column, in declaration order.

    Driven by the keys rather than hard-coded, so the no-digit dataset simply
    prints ``green``.
    """
    return ', '.join(str(_value(concepts, n, index)) for n in concepts)


def _intervention_label(c_tilde: Concepts, index: int) -> str:
    """What was clamped, and what the MRF made of it.

    ``color`` is the only intervened concept, so it appears on both lines by
    construction; anything else on the second line is what BP re-sampled. When
    colour is the only known concept there is nothing to re-sample, so the second
    line is dropped rather than repeating the first.
    """
    clamped = f"do color={_value(c_tilde, 'color', index)}"
    if set(c_tilde) == {'color'}:
        return clamped
    return f"{clamped}\n→ {_label(c_tilde, index)}"


@torch.no_grad()
def make_figure(
    images: torch.Tensor,
    z: torch.Tensor,
    concepts: Concepts,
    color: torch.Tensor,
    decoder,
    propagate,
    cvae,
    ddpm,
    release_step: int,
    resample: int,
    path: Path,
) -> Tuple[torch.Tensor, Concepts]:
    """Write the 3-row figure. Every argument is already restricted to the
    columns being shown."""
    import matplotlib.pyplot as plt

    z_tilde, c_tilde = intervene(z, color, propagate, cvae, ddpm,
                                 release_step, resample)
    rows = [images, reconstruct(decoder, z), reconstruct(decoder, z_tilde)]
    names = ['original', 'reconstruction', f'intervened (release t={release_step})']

    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(1.05 * n, 3.9))
    for row, (label, panel) in enumerate(zip(names, rows)):
        for column in range(n):
            ax = axes[row, column]
            ax.imshow(panel[column].permute(1, 2, 0).clamp(0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, ha='right', va='center')
            if row == 0:
                ax.set_title(_label(concepts, column), fontsize=7)
            if row == 2:
                ax.set_xlabel(_intervention_label(c_tilde, column), fontsize=6)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return z_tilde, c_tilde
