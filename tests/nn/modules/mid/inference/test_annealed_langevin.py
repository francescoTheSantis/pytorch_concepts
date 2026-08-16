"""What AnnealedLangevinDynamics adds on top of the plain chain.

Four things, and each one is invisible downstream if it is subtly wrong — an
off-by-a-constant ladder still produces plausible-looking clouds — so they are
checked directly rather than through a trained model.
"""
import pytest
import torch
from torch.distributions import Normal

from torch_concepts.nn import (
    AnnealedLangevinDynamics,
    LangevinDynamics,
    MarkovNetwork,
    ParametricPotential,
)
from torch_concepts.nn.modules.mid.variable import EmbeddingVariable


class FirstColumn(torch.nn.Module):
    """``E = (x_0 - 2)^2``. Reads column 0 only, so a conditional potential's
    appended noise embedding cannot disturb it."""

    def forward(self, x):
        return ((x[..., :1] - 2.0) ** 2).sum(-1)


class Coupling(torch.nn.Module):
    """``E(a, b) = (b - a)^2 / (2 * 0.1^2)`` — the pair, not the members."""

    def forward(self, x):
        return ((x[..., 1] - x[..., 0]) ** 2) / (2 * 0.1 ** 2)


def _conditional(parametrization=None, names=("a",)):
    """A noise-conditioned MarkovNetwork over ``names``."""
    variables = EmbeddingVariable(list(names), distribution=Normal, shape=1)
    if not isinstance(variables, (list, tuple)):
        variables = [variables]
    return MarkovNetwork(variables=list(variables), factors=[ParametricPotential(
        scope=list(variables),
        parametrization=parametrization or FirstColumn(),
        name="phi",
        noise_conditioned=True,
    )])


def test_ladder_visits_every_level_with_the_ncsn_step():
    """The schedule is Algorithm 1: alpha_i = eps*(sigma_i/sigma_min)^2, noise sqrt(alpha).

    Checked on the schedule itself: the ``sigma_L`` / ``sigma_min`` mix-up in
    particular rescales every step by a constant and is invisible in the samples.
    """
    pgm = _conditional()
    sigmas = torch.tensor([4.0, 2.0, 1.0])
    schedule = list(
        AnnealedLangevinDynamics(pgm, sigmas=sigmas, steps=3, step_size=0.1)
        ._step_schedule()
    )

    assert len(schedule) == 3 * 3                      # steps per level, all levels
    for index, sigma in enumerate(sigmas):
        alpha = 0.1 * float(sigma / sigmas[-1]) ** 2
        for stage, step_size, noise, level in schedule[index * 3:(index + 1) * 3]:
            assert stage == index                      # the stage IS the rung
            assert step_size == pytest.approx(alpha / 2)
            assert noise == pytest.approx(alpha ** 0.5)
            assert float(level) == float(sigma)


def test_clamp_is_noised_during_the_chain_but_exact_at_the_end():
    """Inpainting: evidence rides the ladder, then lands on the value asked for.

    A sharp clamp is off-distribution for an energy fitted at noise level sigma, so
    held coordinates are perturbed to the current rung *while the chain runs* — but
    the caller conditioned on a value, so the reported one must be that value.
    """
    pgm = _conditional()
    engine = AnnealedLangevinDynamics(pgm, sigmas=torch.tensor([5.0, 1.0]), steps=4)
    pinned = torch.full((512, 1), 2.0)

    torch.manual_seed(0)
    during = engine._held_value(pinned, torch.tensor(5.0))
    assert not torch.allclose(during, pinned)                # noised at the rung
    assert abs(float((during - pinned).std()) - 5.0) < 0.5   # by roughly sigma
    assert torch.equal(engine._held_value(pinned, None), pinned)

    out = engine.query(
        query={"a": pinned}, clamp_mask={"a": torch.ones(512, 1, dtype=torch.bool)}
    )
    assert torch.equal(out.samples["a"].tensor, pinned)


def test_starts_at_the_coarsest_rung_with_nothing_supplied():
    """An un-supplied variable is drawn from N(0, sigma_1^2 I), not N(0, I)."""
    drawn = AnnealedLangevinDynamics(
        _conditional(), sigmas=torch.tensor([9.0, 1.0]), steps=0
    ).query(query=["a"], n_samples=4096).samples["a"].tensor

    assert drawn.shape == (4096, 1)
    assert abs(float(drawn.std()) - 9.0) < 1.0


def test_requires_a_ladder_and_refuses_a_gradient_clip():
    """`sigmas` mandatory; `grad_clip` defaulted to None because 0.03 would raise."""
    pgm = _conditional()
    sigmas = torch.tensor([2.0, 1.0])

    with pytest.raises(TypeError):
        AnnealedLangevinDynamics(pgm)                    # no ladder, no sampler

    engine = AnnealedLangevinDynamics(pgm, sigmas=sigmas)
    assert engine.grad_clip is None
    assert torch.equal(engine.sigmas, sigmas)
    assert engine._stage_count() == 2                    # a stage is a rung

    with pytest.raises(ValueError, match="grad_clip"):
        AnnealedLangevinDynamics(pgm, sigmas=sigmas, grad_clip=0.03)

    with pytest.raises(ValueError, match="1-D ladder"):
        AnnealedLangevinDynamics(pgm, sigmas=torch.zeros(2, 2))


def test_chain_respects_clamped_evidence():
    """Conditional generation: a clamped variable pulls its neighbour at every rung.

    ``Coupling`` is reused unchanged even though the potential is conditional: the
    noise embedding is concatenated *after* the scope values, so columns 0 and 1 are
    still ``a`` and ``b``.
    """
    torch.manual_seed(0)
    pgm = _conditional(Coupling(), names=("a", "b"))
    held = torch.full((32, 1), 3.0)
    out = AnnealedLangevinDynamics(
        pgm, sigmas=torch.tensor([1.0, 0.5, 0.25]), steps=100, step_size=1e-3
    ).query(query=["b"], evidence={"a": held})

    # `a` is evidence, so it is not reported; what matters is that it stayed put
    # long enough to pull `b` to it through every rung of the ladder.
    assert (out.samples["b"].tensor - 3.0).abs().mean() < 0.5


def test_release_frees_a_variable_at_a_chosen_rung():
    """Repainting on the ladder: hold through the coarse rungs, then let it evolve.

    Held to the end, the value comes back exactly as supplied. Released early, the
    chain is free to move it — and it does, toward the energy's minimum at 2.0. The
    contrast between the two is the whole feature.
    """
    pgm = _conditional()
    sigmas = torch.tensor([1.0, 0.5, 0.25])
    supplied = torch.full((256, 1), -6.0)   # far from the energy's minimum at 2.0
    engine = AnnealedLangevinDynamics(
        pgm, sigmas=sigmas, steps=200, step_size=2e-3
    )

    torch.manual_seed(0)
    held = engine.query(query={"a": supplied}, release={"a": len(sigmas)})
    torch.manual_seed(0)
    freed = engine.query(query={"a": supplied}, release={"a": 1})

    assert torch.equal(held.samples["a"].tensor, supplied)   # never let go
    moved = (freed.samples["a"].tensor - supplied).abs().mean()
    assert moved > 1.0, f"released variable did not evolve: moved {moved}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
