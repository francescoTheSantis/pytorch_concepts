"""ColorMNIST with a *soft* correlation between the digit and its colour.

``ColorMNISTDataset``'s ``coloring`` argument is either fully random or a
deterministic digit->colour map, and neither gives the 90% coupling this
experiment needs, so the split is built from the two helpers that dataset
itself uses.
"""
from typing import Tuple

import torch
import torch.nn.functional as F

from torch_concepts.data.datasets.mnist import default_root, load_mnist
from torch_concepts.data.utils import colorize

#: Cardinality of every concept the images carry, in concept-vector order.
CARDS = {'digit': 10, 'color': 2}
COLOR_NAMES = ('red', 'green')
#: ``colorize`` wants an RGB channel; the concept value is the palette position.
COLOR_CHANNELS = torch.tensor([0, 1])

#: The two settings, as ``(known concepts, MRF factor scopes, p_agree)``.
#:
#: ``colormnist`` is the full setting: both concepts are annotated, coloured at
#: ``p_agree = 0.9``, and the MRF has a pairwise factor coupling them, so
#: intervening on the colour propagates to the digit.
#:
#: ``colormnist-no-digit`` drops the digit from the annotation **and** decorrelates
#: it from the colour (``p_agree = 0.5``, i.e. an independent fair coin). The digit
#: still exists and still drives the image; it is simply unobserved, so nothing
#: downstream can represent it except the pre-trained model's own latent ``z``.
#: That makes it the sharper test of SDEdit: ``e~`` carries a new colour and
#: nothing else, so a steered image that keeps its digit proves ``z`` kept it.
#:
#: The decorrelation is what makes that inference valid. Leaving ``p_agree`` at
#: 0.9 while hiding the digit would put a colour flip *in conflict with the joint*
#: the DDPM learned -- green co-occurs with even-digit ``z``, so recolouring an odd
#: digit green lands in a low-density region and the sampler is under pressure to
#: move ``z``'s digit content too. Any loss of digit identity would then be
#: ambiguous between "SDEdit failed to preserve ``z``" and "the model correctly
#: followed a correlation we built in". At ``p_agree = 0.5`` colour and digit are
#: independent, so the only consistent response to a colour flip is to leave the
#: digit alone, and the shape correlation measures exactly one thing.
#:
#: The MRF degenerates to a single unary factor fitting ``p(colour)``, and with the
#: only concept clamped BP has nothing left to propagate; the pipeline runs
#: unchanged anyway rather than special-casing it away.
DATASETS = {
    'colormnist': (['digit', 'color'], [('digit', 'color')], 0.9),
    'colormnist-no-digit': (['color'], [('color',)], 0.5),
}


def dataset_spec(name: str):
    """``(cards, scopes, p_agree)`` -- the concepts a run may see, and the colouring."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; pick from {list(DATASETS)}.")
    names, scopes, p_agree = DATASETS[name]
    return {n: CARDS[n] for n in names}, scopes, p_agree


def one_hot(index: torch.Tensor, k: int) -> torch.Tensor:
    """``(N,)`` integer codes -> ``(N, k)`` float one-hot rows."""
    return F.one_hot(index.long(), k).float()


def load_colormnist(
    train: bool = True,
    n: int = 10000,
    p_agree: float = 0.9,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MNIST tinted so that even digits are green ``p_agree`` of the time.

    Args:
        train: Whether to draw from the MNIST train split.
        n: How many images to keep (sampled without replacement).
        p_agree: P(colour agrees with parity). ``0.9`` is the experiment's setting;
            ``0.5`` decouples the two concepts entirely.
        seed: Seeds both the subsample and the colour draw.

    Returns:
        ``(images (n, 3, 28, 28) in [0, 1], digit (n,), color (n,))``, where
        colour 0 is red and 1 is green.
    """
    images, digits = load_mnist(default_root('mnist'), train)
    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(digits), generator=generator)[:n]
    images, digits = images[keep], digits[keep]

    parity = (digits % 2 == 0).long()               # 1 == even
    agrees = torch.rand(len(digits), generator=generator) < p_agree
    color = torch.where(agrees, parity, 1 - parity)  # 1 == green
    return colorize(images, COLOR_CHANNELS[color]), digits, color


def concept_codes(digits: torch.Tensor, color: torch.Tensor, names=None) -> dict:
    """The concept dict every other module speaks: name -> one-hot ``(N, K)``.

    ``names`` restricts the result to the concepts a dataset declares known; the
    omitted ones are still what generated the images, just not annotated.
    """
    everything = {'digit': one_hot(digits, CARDS['digit']),
                  'color': one_hot(color, CARDS['color'])}
    return {n: everything[n] for n in (names or everything)}


if __name__ == '__main__':
    images, digits, color = load_colormnist(n=20000)
    print(f"images {tuple(images.shape)} in [{images.min():.2f}, {images.max():.2f}]")
    even = digits % 2 == 0
    print(f"P(green | even) = {(color[even] == 1).float().mean():.3f}  (target 0.900)")
    print(f"P(green | odd)  = {(color[~even] == 1).float().mean():.3f}  (target 0.100)")
