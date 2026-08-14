"""The noise schedule shared by a diffusion model's graph and its sampler.

One object owns the schedule so there is exactly one definition of it in the
codebase: the ``x_t`` CPD noises an image with it during training, and the
sampling engine denoises with it at generation time. They *must* agree — a
sampler running a different schedule from the one the network was trained
against produces noise, and nothing in the shapes or the loss would say so.

Notation
--------
DDPM's symbols collide with the continuous-time flow-matching ones, and mixing
them is the easiest way to get a subtly wrong sampler:

===============================  ==========================================
DDPM (here)                      meaning
===============================  ==========================================
``beta_t``                       per-step *variance* added at step ``t``
``alpha_t = 1 - beta_t``         per-step retention
``alpha_bar_t = prod(alpha_s)``  cumulative retention up to ``t``
===============================  ==========================================

The path coefficients — the things that actually multiply the image and the
noise in :math:`x_t = a_t x_0 + b_t \\epsilon` — are :math:`\\sqrt{\\bar\\alpha_t}`
and :math:`\\sqrt{1-\\bar\\alpha_t}`, **not** ``alpha_t`` and ``beta_t``. To keep
that impossible to misread, this module stores a single buffer (``alpha_bar``)
and exposes everything else under names that say which one they are:
:meth:`sqrt_alpha_bar` and :meth:`sqrt_one_minus_alpha_bar`. There is no
attribute called ``alpha`` or ``beta``.

Timesteps run ``T-1`` (pure noise) down to ``0`` (one step from clean data), and
index ``-1`` denotes the clean-data end itself, where ``alpha_bar == 1``.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class DiffusionSchedule(nn.Module):
    """A linear-``beta`` DDPM schedule, plus the DDIM ``sigma`` family.

    Parameters
    ----------
    n_steps : int, default 1000
        ``T``, the number of training timesteps. The reverse process may be run
        on fewer (see :meth:`timestep_pairs`); this is the resolution the
        network is *trained* at.
    beta_start, beta_end : float
        Endpoints of the linear ``beta`` schedule. The defaults are Ho et al.'s
        for ``T = 1000``. They are tied to that ``T`` — the cumulative product
        over more or fewer steps lands somewhere else — so a much smaller ``T``
        wants a proportionally larger ``beta_end``, or the noisiest timestep is
        not actually noisy and the model never learns to start from ``N(0, I)``.

    Attributes
    ----------
    alpha_bar : torch.Tensor
        ``(T,)`` cumulative retention. The only stored quantity; registered
        non-persistently, since it is fully determined by the three constructor
        arguments and does not belong in a checkpoint.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.nn import DiffusionSchedule
    >>> schedule = DiffusionSchedule(n_steps=1000)
    >>> t = torch.tensor([[0.], [999.]])
    >>> a, b = schedule.sqrt_alpha_bar(t), schedule.sqrt_one_minus_alpha_bar(t)
    >>> bool(a[0] > 0.99), bool(b[1] > 0.99)   # t=0 nearly clean, t=T-1 nearly noise
    (True, True)
    >>> float(a.pow(2)[0] + b.pow(2)[0])       # the path is variance preserving
    1.0
    """

    def __init__(
        self,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError(f"DiffusionSchedule: n_steps must be >= 1, got {n_steps}.")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError(
                "DiffusionSchedule: need 0 < beta_start <= beta_end < 1, got "
                f"beta_start={beta_start}, beta_end={beta_end}."
            )
        betas = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float32)
        self.register_buffer(
            "alpha_bar", torch.cumprod(1.0 - betas, dim=0), persistent=False
        )

    @property
    def n_steps(self) -> int:
        """``T`` — the number of training timesteps."""
        return int(self.alpha_bar.shape[0])

    def extra_repr(self) -> str:
        return f"n_steps={self.n_steps}, alpha_bar_T={float(self.alpha_bar[-1]):.2e}"

    # ------------------------------------------------------------------
    # Index resolution
    # ------------------------------------------------------------------
    def index(self, t: torch.Tensor) -> torch.Tensor:
        """Integer timestep indices from a timestep tensor.

        ``t`` is a *continuous* draw during training — the ``t`` root is a
        ``Uniform(0, T)``, because a width-1 continuous variable is far cheaper
        than a ``T``-way one-hot — and an exact integer at sampling time. Both
        land on the same grid here: floor, then clamp into ``[0, T-1]`` so a
        draw of exactly ``T`` (measure zero, but not impossible in float) cannot
        index off the end.
        """
        return t.floor().long().clamp_(0, self.n_steps - 1)

    def alpha_bar_at(self, index: torch.Tensor) -> torch.Tensor:
        """``alpha_bar`` looked up at integer ``index``, with ``index < 0`` meaning 1.

        A negative index is the clean-data end of the process, past the last
        step, where no noise remains. Returning ``1`` there is what makes the
        final reverse step land on ``x_0`` without a special case — and, through
        :meth:`sigma`, what makes that step automatically noiseless.
        """
        value = self.alpha_bar[index.clamp(min=0)]
        return torch.where(index < 0, torch.ones_like(value), value)

    # ------------------------------------------------------------------
    # Path coefficients: x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * eps
    # ------------------------------------------------------------------
    def sqrt_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """The coefficient on the *image*, shaped like ``t`` for broadcasting."""
        return self.alpha_bar_at(self.index(t)).sqrt()

    def sqrt_one_minus_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """The coefficient on the *noise*, shaped like ``t`` for broadcasting."""
        return (1.0 - self.alpha_bar_at(self.index(t))).clamp_(min=0.0).sqrt()

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------
    def timestep_pairs(self, n_steps: int = None) -> List[Tuple[int, int]]:
        """``(t, t_prev)`` pairs for the reverse loop, noisiest first.

        With ``n_steps`` equal to :attr:`n_steps` this is every timestep,
        ``(T-1, T-2), …, (1, 0), (0, -1)`` — the full DDPM trajectory. With
        fewer, the grid is strided and the pairs skip: the DDIM update is
        defined between *any* two noise levels, not just adjacent ones, which is
        what makes 50-step sampling possible from a network trained at 1000.

        The last pair always ends at ``-1``, the clean-data end.
        """
        n_steps = self.n_steps if n_steps is None else int(n_steps)
        if not 1 <= n_steps <= self.n_steps:
            raise ValueError(
                f"DiffusionSchedule.timestep_pairs: n_steps must be in "
                f"[1, {self.n_steps}], got {n_steps}."
            )
        # Descending from the noisiest step, not ascending-then-reversed: the
        # reverse process starts at `T-1` because that is the timestep whose
        # marginal is the `N(0, I)` it is seeded with. An ascending grid would
        # drop `T-1` whenever the endpoints did not land on it — at `n_steps=1`
        # it selects step 0 alone, and the sampler denoises pure noise as though
        # it were nearly clean data.
        grid = torch.linspace(self.n_steps - 1, 0, n_steps).round().long().tolist()
        grid = sorted(set(grid), reverse=True)
        return list(zip(grid, grid[1:] + [-1]))

    def sigma(self, index: torch.Tensor, index_prev: torch.Tensor, eta: float):
        """The DDIM ``sigma`` family — how much noise the reverse step injects.

        .. math::
           \\sigma = \\eta \\sqrt{\\frac{1-\\bar\\alpha_{prev}}{1-\\bar\\alpha_t}}
                     \\sqrt{1 - \\frac{\\bar\\alpha_t}{\\bar\\alpha_{prev}}}

        ``eta=0`` is the deterministic DDIM sampler. ``eta=1`` on the full grid
        makes :func:`ddim_step` algebraically identical to DDPM's Algorithm 2 —
        this is the :math:`\\tilde\\beta_t` variant of Ho et al.'s posterior
        variance, the one their sampler's ``sigma_t**2`` is usually set to (the
        paper also offers the simpler ``beta_t``, which is *not* this family and
        does not coincide with any ``eta``).

        At the final step ``index_prev == -1``, so ``alpha_bar_prev == 1`` and
        the first factor vanishes: the last reverse step is noiseless whatever
        ``eta`` is. That is Algorithm 2's ``z = 0 if t == 1`` clause, falling out
        of the formula instead of being special-cased.
        """
        if eta == 0.0:
            return None
        alpha_bar = self.alpha_bar_at(index)
        alpha_bar_prev = self.alpha_bar_at(index_prev)
        ratio = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp(min=1e-12)
        return (
            eta
            * ratio.clamp(min=0.0).sqrt()
            * (1.0 - alpha_bar / alpha_bar_prev).clamp(min=0.0).sqrt()
        )

    def ddim_step(
        self,
        x: torch.Tensor,
        eps: torch.Tensor,
        index: torch.Tensor,
        index_prev: torch.Tensor,
        eta: float = 0.0,
        clip: Tuple[float, float] = None,
    ) -> torch.Tensor:
        """One reverse step, from noise level ``index`` to ``index_prev``.

        .. math::
           \\hat x_0 = \\frac{x - \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon}{\\sqrt{\\bar\\alpha_t}}
           \\qquad
           x' = \\sqrt{\\bar\\alpha_{prev}}\\,\\hat x_0
                + \\sqrt{1-\\bar\\alpha_{prev}-\\sigma^2}\\,\\epsilon
                + \\sigma z

        Parameters
        ----------
        x, eps : torch.Tensor
            The current state and the (already guidance-combined) noise
            prediction, both ``(*leading, D)``.
        index, index_prev : torch.Tensor
            Integer timestep indices, broadcastable against ``x``.
        eta : float
            See :meth:`sigma`.
        clip : tuple of float, optional
            Range to clamp the predicted clean image :math:`\\hat x_0` into. At
            the noisiest steps ``alpha_bar`` is ~1e-5, so that division is by a
            number near ``0.006`` and :math:`\\hat x_0` lands far outside the data
            range; left alone it drags the whole trajectory off. Pass the data's
            range (``(0, 1)`` for images) to bound it. ``None`` disables the
            clamp, which is what the equivalence with Algorithm 2 requires.
        """
        alpha_bar = self.alpha_bar_at(index)
        alpha_bar_prev = self.alpha_bar_at(index_prev)

        x0_hat = (x - (1.0 - alpha_bar).clamp(min=0.0).sqrt() * eps) / alpha_bar.sqrt()
        if clip is not None:
            x0_hat = x0_hat.clamp(*clip)

        sigma = self.sigma(index, index_prev, eta)
        variance = 1.0 - alpha_bar_prev
        if sigma is not None:
            variance = (variance - sigma.pow(2)).clamp(min=0.0)

        out = alpha_bar_prev.sqrt() * x0_hat + variance.clamp(min=0.0).sqrt() * eps
        if sigma is not None:
            out = out + sigma * torch.randn_like(out)
        return out
