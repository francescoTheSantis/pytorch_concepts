"""AnnealedLangevinDynamics — the sampler that matches a score-matching-trained energy.

:class:`~torch_concepts.nn.LangevinDynamics` runs one chain at one noise level. That
is enough for a fixed energy, but it cannot sample a *noise-conditioned* one: an
energy fitted by denoising score matching is a different function at every level of
its training ladder, and only the finest level describes the data.

This engine walks that ladder (Song & Ermon, NeurIPS 2019, Algorithm 1). Each rung
gets its own Langevin chain, run at the step the level's scale calls for, and the
chain that ends one rung starts the next:

.. math::

    \\alpha_i = \\epsilon\\,\\sigma_i^2 / \\sigma_T^2, \\qquad
    x \\leftarrow x + \\tfrac{\\alpha_i}{2}\\, s_\\theta(x, \\sigma_i)
        + \\sqrt{\\alpha_i}\\,\\epsilon_t

Starting coarse and tightening is what buys multimodality: at a noise level low
enough to settle inside a mode a chain cannot cross between modes, and at one high
enough to cross it cannot settle. A single fixed level gives collapsed point masses
or a diffuse cloud — never the target's own spread.

References
----------
Song & Ermon. "Generative Modeling by Estimating Gradients of the Data
Distribution", NeurIPS 2019.
Lugmayr et al. "RePaint: Inpainting using Denoising Diffusion Probabilistic Models",
CVPR 2022.
"""
from __future__ import annotations

from typing import Optional

import torch

from ...graph.probabilistic_model import ProbabilisticModel
from .langevin import LangevinDynamics


class AnnealedLangevinDynamics(LangevinDynamics):
    """Annealed Langevin dynamics over a noise ladder.

    Same machinery as :class:`~torch_concepts.nn.LangevinDynamics` — the packed
    layout, the clamping, the per-row ``clamp_mask``, the timed ``release`` — with
    three things replaced: the step schedule becomes the ladder, a held value is
    perturbed to the current rung, and an un-supplied variable starts at the
    coarsest rung's scale.

    Requires an energy that accepts a noise level, i.e. potentials built with
    ``ParametricPotential(noise_conditioned=True)``. Use the plain engine for
    anything else.

    Parameters
    ----------
    pgm : ProbabilisticModel
        Whose ``energy`` accepts ``sigma=``.
    sigmas : torch.Tensor
        The descending noise ladder, **required** — it is the method, not an option.
        Should be the ladder the energy was trained on: the chain visits exactly
        these levels, and one the network never saw is one it has no score for.
    steps : int, default 100
        Iterations per rung (NCSN's ``L``).
    step_size : float, default 2e-5
        The base ``epsilon``. The step at rung ``i`` is
        ``epsilon * (sigma_i / sigma_min)**2 / 2``, and the noise its square root —
        the pairing that makes each rung an exact Langevin discretization.
    grad_clip : float or None, default None
        Must stay ``None``, and defaults to it — the base class's ``0.03`` would
        raise. The ladder's steps span orders of magnitude by construction, and one
        clip value caps every one of them at the same size, flattening it.

    Notes
    -----
    **A stage is a rung here**, so ``release={'x': 3}`` on
    :meth:`~torch_concepts.nn.LangevinDynamics.query` holds ``x`` through
    ``sigma_0..sigma_2`` and frees it from ``sigma_3`` on. That is the natural clock
    for repainting: hold a value while the chain is coarse and cannot use it
    precisely, then let it settle with everything else.

    Examples
    --------
    >>> sampler = AnnealedLangevinDynamics(mrf, sigmas=ladder)
    >>> out = sampler.query(query=["x"], evidence={"y": observed})
    """

    name = "AnnealedLangevinDynamics"

    def __init__(
        self,
        pgm: ProbabilisticModel,
        sigmas: torch.Tensor,
        steps: int = 100,
        step_size: float = 2e-5,
        grad_clip: Optional[float] = None,
        **base_kwargs,
    ):
        sigmas = torch.as_tensor(sigmas, dtype=torch.get_default_dtype())
        if sigmas.ndim != 1 or sigmas.numel() == 0:
            raise ValueError(
                f"{self.name}: `sigmas` must be a non-empty 1-D ladder, got shape "
                f"{tuple(sigmas.shape)}."
            )
        if grad_clip is not None:
            raise ValueError(
                f"{self.name}: `grad_clip` must be None. The annealed step sizes "
                "span several orders of magnitude (here "
                f"{float(sigmas[0] / sigmas[-1]) ** 2:.3g}x), and clipping caps "
                "every one of them at the same value, flattening the ladder."
            )
        self.sigmas = sigmas
        super().__init__(
            pgm, steps=steps, step_size=step_size, grad_clip=None, **base_kwargs
        )

    def __repr__(self) -> str:
        # Not the inherited one: it advertises `noise_scale` and `anneal`, and both
        # are dead here — the ladder supplies the schedule.
        return self._format_repr(
            levels=len(self.sigmas),
            sigma_max=float(self.sigmas[0]),
            sigma_min=float(self.sigmas[-1]),
            steps_per_level=self.steps,
            step_size=self.step_size,
        )

    # --------------------------------------------------- the three replaced hooks
    def _stage_count(self) -> int:
        """One stage per rung, so ``release`` counts noise levels."""
        return len(self.sigmas)

    def _init_scale(self) -> float:
        """``N(0, sigma_1^2 I)`` — the distribution the first rung is sampling."""
        return float(self.sigmas[0])

    @staticmethod
    def _held_value(pinned: torch.Tensor, sigma) -> torch.Tensor:
        """A held coordinate, perturbed to the current rung: ``pinned + sigma * eps``.

        The inpainting correction (Song et al. 2021; RePaint, CVPR 2022), and it is
        not optional. The energy at level ``sigma`` was fitted on data blurred by
        exactly that much, so a *sharp* clamp is an input it has never seen — at the
        coarse end of the ladder, where ``sigma`` is on the order of the data's own
        spread, that is far off-distribution, and the free variables would take their
        conditioning from a score evaluated in the wrong place. Redrawn every step,
        so the clamp does not bias the chain in one fixed direction.
        """
        if sigma is None:
            return pinned
        return pinned + float(sigma) * torch.randn_like(pinned)

    def _step_schedule(self):
        """``(rung index, alpha_i/2, sqrt(alpha_i), sigma_i)`` per step.

        ``sigma_min`` is the *last* rung — the level the ladder's steps are expressed
        relative to. (Worth stating because the paper writes it as ``sigma_L`` while
        also using ``L`` for the inner iteration count, which invites reading it as
        the loop counter and rescaling every step by a constant.)

        ``step_size`` and ``steps`` are reused as NCSN's ``epsilon`` and ``L`` rather
        than duplicated under new names.
        """
        smallest = self.sigmas[-1]
        for stage, sigma in enumerate(self.sigmas):
            alpha = self.step_size * float(sigma / smallest) ** 2
            for _ in range(self.steps):
                yield stage, 0.5 * alpha, alpha ** 0.5, sigma
