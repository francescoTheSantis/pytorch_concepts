"""Convolutional layers for image-valued variables.

A PGM keeps every variable on a single flat event axis, so an image variable of
shape ``(C, H, W)`` is ``C * H * W`` scalars to the graph. A dense head to that
many outputs is ruinous — CelebA at native resolution is 116,412 pixels, so the
last layer alone would carry ~60M parameters. :class:`ConvDecoder` spends those
parameters on a stack of transposed convolutions instead, and flattens only at
the very end so the CPD contract is unchanged.

:class:`ConditionalUNet` is the image-to-image counterpart, for a variable whose
CPD maps a picture to another picture of the same shape — a diffusion model's
noise prediction. It flattens at the same boundary and for the same reason.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


#: Normalisations :class:`ConvDecoder` will insert between stages.
NORMS = ("batch", "group", "none")


def _norm_layer(kind: str, channels: int) -> Optional[nn.Module]:
    """A normalisation over ``channels``, or ``None`` for ``'none'``.

    ``'group'`` exists for the generative case. ``BatchNorm`` keeps running
    statistics accumulated from whatever the decoder saw in training — for a VAE
    that is bottlenecks driven by ``q(z | x)`` — and then normalises by them at
    ``eval()``. Decoding a *prior* sample feeds it a differently-distributed
    input, so generations are normalised by statistics that do not describe them
    while reconstructions are fine. ``GroupNorm`` has no running statistics and
    cannot develop that split.
    """
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        # gcd keeps the group count legal for any width; 8 is the usual default
        # and degrades to LayerNorm-per-pixel on a 3-channel output.
        return nn.GroupNorm(math.gcd(8, channels), channels)
    if kind == "none":
        return None
    raise ValueError(f"ConvDecoder: norm must be one of {NORMS}, got {kind!r}.")


def _derive_stages(side: int, base_channels: int, max_base_size: int) -> Tuple[int, ...]:
    """Stage widths for a ``side x side`` target, depth chosen from ``side`` itself.

    Each stage doubles the resolution, so the depth is the number of halvings that
    bring ``side`` down to ``max_base_size`` or below — as far as its factors of two
    allow. Widths double from ``base_channels`` upwards, narrowest last.

    >>> _derive_stages(28, 32, 8), _derive_stages(64, 32, 8)
    ((64, 32), (128, 64, 32))
    """
    n, base = 0, side
    while base > max_base_size and base % 2 == 0:
        base //= 2
        n += 1
    return tuple(base_channels << i for i in reversed(range(n)))


class ConvDecoder(nn.Module):
    """Decode a flat feature vector into a flat image via transposed convolutions.

    Projects ``in_features`` to a small ``base_size x base_size`` spatial grid,
    then doubles the resolution once per entry of ``hidden_channels`` before a
    final convolution to the requested channel count. Input and output are both
    **flat**: the module is a drop-in parametrization for an image variable, and
    the parameter's domain activation is applied outside it, as
    :class:`~torch_concepts.nn.modules.mid.factors.cpd.ParametricCPD` requires
    (compose with :class:`~torch_concepts.nn.DefaultActivation`).

    Parameters
    ----------
    in_features : int
        Width of the incoming feature vector (a concept bottleneck, a latent).
    out_shape : tuple of int
        Image event shape ``(channels, height, width)``. ``height`` and
        ``width`` must be ``base_size * 2 ** len(hidden_channels)``.
    hidden_channels : sequence of int or int, default 32
        Channel count after each upsampling stage; its length is the number of
        stages, so each entry doubles the resolution. An **int** is the narrowest
        stage instead, and the depth is derived from ``out_shape`` — see
        :func:`_derive_stages`. A config can then stay resolution-agnostic, which
        matters when one sweep spans datasets of different sizes.
    base_size : int, optional
        Side of the spatial grid the linear projection produces. Derived from
        ``out_shape`` when omitted — ``side / 2 ** len(hidden_channels)``, which
        is the only value that reaches the target. Defaults to ``None`` (derive).
    activation : type, default ``nn.LeakyReLU``
        Activation class used between stages.
    batch_norm : bool, default True
        Legacy switch for ``BatchNorm2d`` between stages. Superseded by
        ``norm``, which is consulted first whenever it is given.
    norm : str, optional
        ``'batch'``, ``'group'`` or ``'none'``. Defaults to ``None``, meaning
        "follow ``batch_norm``". Prefer ``'group'`` in a generative model: see
        :func:`_norm_layer` for why ``BatchNorm`` and prior sampling interact
        badly.
    refine : bool, default False
        Insert a same-resolution ``Conv2d(3x3) -> norm -> activation`` block
        after the linear projection and after every upsampling stage bar the
        last. Off by default, because a decoder with the default settings is
        otherwise **almost affine** — the stack is one ``Linear``, one
        activation and two transposed convolutions — and an affine generator
        reproduces the data mean near the training codes and diverges away from
        them. On a 28x28 target this takes the stack from one activation to
        three, for ~25% more parameters.
    max_base_size : int, default 8
        Largest starting grid an int ``hidden_channels`` will settle for; ignored
        when the stages are given explicitly.

    Raises
    ------
    ValueError
        If ``out_shape`` is not 3-dimensional or not square, or its spatial size
        is not ``base_size * 2 ** len(hidden_channels)``.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.nn import ConvDecoder
    >>> decoder = ConvDecoder(in_features=48, out_shape=(3, 32, 32),
    ...                       hidden_channels=(64, 32, 16))
    >>> decoder(torch.randn(8, 48)).shape
    torch.Size([8, 3072])

    With an int, one call site serves any resolution — 28 takes two stages
    (``28 = 7 * 2 ** 2``) and 64 takes three (``64 = 8 * 2 ** 3``):

    >>> ConvDecoder(in_features=48, out_shape=(3, 28, 28))(torch.randn(2, 48)).shape
    torch.Size([2, 2352])
    >>> ConvDecoder(in_features=48, out_shape=(3, 64, 64))(torch.randn(2, 48)).shape
    torch.Size([2, 12288])

    An explicit depth the target cannot reach is rejected up front rather than
    silently resized:

    >>> ConvDecoder(in_features=48, out_shape=(3, 28, 28),
    ...             hidden_channels=(64, 32, 16))
    Traceback (most recent call last):
        ...
    ValueError: ConvDecoder: a 28x28 target is not reachable with 3 upsampling ...
    """

    def __init__(
        self,
        in_features: int,
        out_shape: Union[Tuple[int, ...], torch.Size],
        hidden_channels: Union[Sequence[int], int] = 32,
        base_size: Optional[int] = None,
        activation: type = nn.LeakyReLU,
        batch_norm: bool = True,
        norm: Optional[str] = None,
        refine: bool = False,
        max_base_size: int = 8,
    ) -> None:
        super().__init__()
        # `norm` supersedes `batch_norm` but does not break it: existing callers
        # pass only the boolean and keep the behaviour they had.
        if norm is None:
            norm = "batch" if batch_norm else "none"
        if norm not in NORMS:
            raise ValueError(f"ConvDecoder: norm must be one of {NORMS}, got {norm!r}.")
        out_shape = tuple(int(s) for s in out_shape)
        if len(out_shape) != 3:
            raise ValueError(
                f"ConvDecoder: out_shape must be (channels, height, width), "
                f"got {out_shape}."
            )
        channels, height, width = out_shape
        if height != width:
            raise ValueError(
                f"ConvDecoder: out_shape {out_shape} is not square "
                f"({height}x{width}). Resize the images to a square first."
            )
        if isinstance(hidden_channels, int):
            hidden_channels = _derive_stages(height, hidden_channels, max_base_size)
        hidden_channels = tuple(int(c) for c in hidden_channels)
        n_stages = len(hidden_channels)
        if base_size is None:
            # Each stage doubles the resolution, so the starting grid is fixed
            # once the depth is chosen. Deriving it lets a config pick only the
            # depth and stay correct across datasets of different resolutions.
            if height % (2 ** n_stages):
                raise ValueError(
                    f"ConvDecoder: a {height}x{height} target is not reachable "
                    f"with {n_stages} upsampling stages ({height} is not "
                    f"divisible by {2 ** n_stages}). Use a different number of "
                    "hidden_channels, or set base_size explicitly."
                )
            base_size = height // (2 ** n_stages)
        reached = base_size * 2 ** n_stages
        if (height, width) != (reached, reached):
            raise ValueError(
                f"ConvDecoder: out_shape {out_shape} has spatial size "
                f"{height}x{width}, but base_size={base_size} and "
                f"{n_stages} upsampling stages reach "
                f"{reached}x{reached}. Choose a base_size and hidden_channels "
                "whose product matches, or resize the images."
            )

        self.in_features = in_features
        self.out_shape = out_shape
        self.base_size = int(base_size)
        self.out_features = channels * height * width

        # No stages (a side with no factor of two to spend) leaves the projection
        # emitting the image itself — a plain dense head, which is all a target
        # that small warrants.
        first = hidden_channels[0] if hidden_channels else channels
        self.project = nn.Linear(in_features, first * base_size * base_size)
        self.unflatten = nn.Unflatten(-1, (first, base_size, base_size))

        def block(width: int) -> list[nn.Module]:
            """Same-resolution refinement: conv, norm, activation."""
            layers: list[nn.Module] = [
                nn.Conv2d(width, width, kernel_size=3, padding=1)
            ]
            if (layer := _norm_layer(norm, width)) is not None:
                layers.append(layer)
            layers.append(activation())
            return layers

        widths = list(hidden_channels) + [channels]
        stages: list[nn.Module] = []
        # A refinement on the projected grid, before any upsampling: without it
        # the only thing between the bottleneck and the first transposed
        # convolution is `project`, a single affine map.
        if refine and hidden_channels:
            stages += block(first)
        for i, (in_c, out_c) in enumerate(zip(widths, widths[1:])):
            stages.append(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            )
            # The last stage stays bare: it emits the raw parameter, which the
            # DefaultActivation composed downstream maps into its domain.
            if i < len(widths) - 2:
                if (layer := _norm_layer(norm, out_c)) is not None:
                    stages.append(layer)
                stages.append(activation())
                if refine:
                    stages += block(out_c)
        self.stages = nn.Sequential(*stages)
        self.flatten = nn.Flatten(start_dim=-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.unflatten(self.project(x))
        # Conv layers take exactly one batch axis, so fold any extra leading
        # dimensions away and restore them afterwards.
        leading = h.shape[:-3]
        h = h.reshape(-1, *h.shape[-3:])
        h = self.stages(h)
        h = h.reshape(*leading, *h.shape[-3:])
        return self.flatten(h)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_shape={self.out_shape}, "
            f"base_size={self.base_size}"
        )


def sinusoidal_embedding(steps: torch.Tensor, width: int) -> torch.Tensor:
    """Transformer-style positional embedding of an integer timestep.

    ``(*leading, 1) -> (*leading, width)``. A diffusion U-Net has to condition on
    ``t``, and feeding the raw integer works badly: the network sees one scalar
    that has to modulate every layer, and neighbouring timesteps are almost
    indistinguishable at the scale the rest of its inputs live on. A bank of
    sinusoids at geometrically spaced frequencies separates adjacent steps while
    keeping distant ones smoothly related.

    >>> import torch
    >>> from torch_concepts.nn.modules.low.conv import sinusoidal_embedding
    >>> sinusoidal_embedding(torch.tensor([[0.], [500.]]), 64).shape
    torch.Size([2, 64])
    """
    if width % 2:
        raise ValueError(f"sinusoidal_embedding: width must be even, got {width}.")
    half = width // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=steps.device, dtype=torch.float32)
        / half
    )
    angles = steps.reshape(*steps.shape[:-1], 1).float() * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class _ResBlock(nn.Module):
    """Two 3x3 convolutions with the conditioning added between them.

    The conditioning enters as a per-channel *shift* predicted from the embedding
    — the standard way a diffusion U-Net is told the timestep and the class. It
    is applied after the first convolution rather than concatenated at the input
    so that every block, at every resolution, is conditioned rather than only the
    first one.
    """

    def __init__(self, in_channels: int, out_channels: int, embedding_size: int,
                 norm: str, activation) -> None:
        super().__init__()
        self.norm1 = _norm_layer(norm, in_channels) or nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embed = nn.Linear(embedding_size, out_channels)
        self.norm2 = _norm_layer(norm, out_channels) or nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = activation()
        # 1x1 projection only when the widths differ, so an unchanged width keeps
        # a true identity path.
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.embed(emb)[..., None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    """Image-to-image U-Net conditioned on a timestep and a condition vector.

    The network a diffusion model's denoising CPD wraps: it reads a noised image,
    the timestep it was noised to, and the (possibly nulled) condition, and
    predicts the noise that was added. Input and output are the *same* shape,
    which is what distinguishes it from :class:`ConvDecoder`.

    Both conditioning signals are reduced to one embedding — the timestep through
    :func:`sinusoidal_embedding`, the condition through a linear map, summed —
    and that single vector shifts every residual block. Summing rather than
    concatenating is what makes the null condition (``y = 0``) a clean
    "no information" input: it contributes its layer's bias and nothing else.

    Flattening is at the boundary only: the CPD contract is a flat
    ``(*leading, C*H*W)`` in and out, and the convolutions see ``(B, C, H, W)``
    in between.

    Parameters
    ----------
    shape : tuple of int
        ``(C, H, W)`` of the image. Input and output both.
    condition_size : int
        Width of the condition vector. ``0`` for an unconditional model.
    hidden_channels : sequence of int
        Width per resolution level, coarsest last. Each level after the first
        halves the resolution, so ``(64, 128)`` is one downsample.
    embedding_size : int, default 128
        Width of the combined timestep/condition embedding.
    norm : str, default ``'group'``
        See :func:`_norm_layer`. ``'group'`` for the same reason
        :class:`ConvDecoder` prefers it.
    activation : type, default ``nn.SiLU``
        Activation class.

    Attributes
    ----------
    out_features : int
        ``C * H * W`` — the flat width, so a CPD can be sized from the module.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.nn import ConditionalUNet
    >>> unet = ConditionalUNet(shape=(3, 32, 32), condition_size=16,
    ...                        hidden_channels=(32, 64))
    >>> x = torch.randn(2, 3 * 32 * 32)
    >>> t = torch.tensor([[10.], [900.]])
    >>> y = torch.randn(2, 16)
    >>> unet(x, t, y).shape          # same flat width back
    torch.Size([2, 3072])
    """

    def __init__(
        self,
        shape: Union[Tuple[int, ...], torch.Size],
        condition_size: int,
        hidden_channels: Sequence[int] = (64, 128),
        embedding_size: int = 128,
        norm: str = "group",
        activation: type = nn.SiLU,
    ) -> None:
        super().__init__()
        shape = tuple(int(s) for s in shape)
        if len(shape) != 3:
            raise ValueError(
                f"ConditionalUNet: shape must be (channels, height, width), got {shape}."
            )
        if not hidden_channels:
            raise ValueError("ConditionalUNet: hidden_channels must be non-empty.")
        channels, height, width = shape
        downsamples = len(hidden_channels) - 1
        if height % (2 ** downsamples) or width % (2 ** downsamples):
            raise ValueError(
                f"ConditionalUNet: {height}x{width} is not divisible by "
                f"2**{downsamples}, which {len(hidden_channels)} levels require. "
                "Use fewer levels, or resize the images."
            )

        self.shape = shape
        self.embedding_size = int(embedding_size)
        self.condition_size = int(condition_size)
        self.out_features = channels * height * width

        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            activation(),
            nn.Linear(embedding_size, embedding_size),
        )
        # No bias: `y = 0` must contribute exactly nothing, so that the null
        # condition is the *absence* of a signal rather than a learned constant
        # competing with the timestep. The blocks' own biases supply the offset.
        self.condition_mlp = (
            nn.Linear(condition_size, embedding_size, bias=False)
            if condition_size else None
        )

        widths = list(hidden_channels)
        self.stem = nn.Conv2d(channels, widths[0], 3, padding=1)

        block = lambda i, o: _ResBlock(i, o, embedding_size, norm, activation)
        self.down = nn.ModuleList(
            block(a, b) for a, b in zip(widths, widths[1:])
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(w, w, 3, stride=2, padding=1) for w in widths[1:]
        )
        self.middle = block(widths[-1], widths[-1])
        self.upsample = nn.ModuleList(
            nn.ConvTranspose2d(w, w, 4, stride=2, padding=1) for w in widths[1:]
        )
        # Each up block consumes the skip concatenated onto its input, hence the
        # doubled fan-in.
        self.up = nn.ModuleList(
            block(b * 2, a) for a, b in zip(widths, widths[1:])
        )
        self.out_norm = _norm_layer(norm, widths[0]) or nn.Identity()
        self.out_act = activation()
        # Zero-initialised final convolution: the network starts by predicting
        # exactly zero noise, so the first training steps move it off a neutral
        # point instead of off whatever the initialisation happened to emit.
        self.out_conv = nn.Conv2d(widths[0], channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal_embedding(t, self.embedding_size))
        if self.condition_mlp is not None and y is not None:
            emb = emb + self.condition_mlp(y)

        # Convolutions take exactly one batch axis; fold any extra leading dims.
        leading = x.shape[:-1]
        h = x.reshape(-1, *self.shape)
        emb = emb.reshape(-1, self.embedding_size)

        h = self.stem(h)
        skips = []
        for block, down in zip(self.down, self.downsample):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.middle(h, emb)
        for block, up, skip in zip(
            reversed(self.up), reversed(self.upsample), reversed(skips)
        ):
            h = up(h)
            h = block(torch.cat([h, skip], dim=1), emb)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h.reshape(*leading, self.out_features)

    def extra_repr(self) -> str:
        return f"shape={self.shape}, condition_size={self.condition_size}"
