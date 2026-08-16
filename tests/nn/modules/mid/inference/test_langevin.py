"""The five things LangevinDynamics has to get right.

Deliberately small: an exact-Gaussian target pins the sampler's correctness, and
the rest pin the free/clamped contract the CE2BM steering relies on — including
the column layout, which the engine depends on being in model order.
"""
import pytest
import torch

from torch_concepts.nn import (
    AnnealedLangevinDynamics,
    LangevinDynamics,
    MarkovNetwork,
)
from torch_concepts.nn.modules.mid.factors.potential import ParametricPotential
from torch_concepts.nn.modules.mid.variable import EmbeddingVariable
from torch.distributions import Normal


class Quadratic(torch.nn.Module):
    """``E(z) = ||z - mu||^2 / (2 sigma^2)`` — an exact ``N(mu, sigma)``."""

    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu))
        self.register_buffer("sigma", torch.tensor(sigma))

    def forward(self, x):
        return ((x - self.mu) ** 2).sum(-1) / (2 * self.sigma ** 2)


def _model(size=1, members=("a",), mu=2.0, sigma=0.5):
    variables = EmbeddingVariable(list(members), distribution=Normal, shape=size)
    by_name = {v.name: v for v in variables}
    factors = [
        ParametricPotential(
            scope=[by_name[m]], parametrization=Quadratic(mu, sigma), name=f"phi_{m}"
        )
        for m in members
    ]
    return variables, MarkovNetwork(variables=variables, factors=factors)


def test_recovers_an_exact_gaussian():
    """With sigma = sqrt(2*lambda) the chain targets p(x) ∝ exp(-E) exactly."""
    torch.manual_seed(0)
    _, pgm = _model()
    engine = LangevinDynamics(
        pgm, steps=2000, step_size=0.01, noise_scale=None, grad_clip=None
    )
    drawn = engine.query(query={"a": torch.zeros(4096, 1)}).samples["a"].tensor
    assert abs(drawn.mean().item() - 2.0) < 0.05
    assert abs(drawn.std().item() - 0.5) < 0.05


class Coupling(torch.nn.Module):
    """``E(a, b) = (b - a)^2 / (2 sigma^2)`` — the pair, not the members."""

    def forward(self, x):
        return ((x[..., 1] - x[..., 0]) ** 2) / (2 * 0.1 ** 2)


def test_clamped_evidence_propagates_to_a_free_member():
    """A held member pulls its neighbour — the CE2BM mechanism, minimally."""
    torch.manual_seed(0)
    a, b = EmbeddingVariable(["a", "b"], distribution=Normal, shape=1)
    pgm = MarkovNetwork(
        variables=[a, b],
        factors=[ParametricPotential(
            scope=[a, b], parametrization=Coupling(), name="phi",
        )],
    )
    out = LangevinDynamics(pgm, steps=200, noise_scale=0.0).query(
        query={"b": torch.zeros(32, 1)}, evidence={"a": torch.full((32, 1), 3.0)}
    )
    assert (out.samples["b"].tensor - 3.0).abs().mean() < 0.5


def test_zero_steps_returns_the_initial_value():
    _, pgm = _model()
    start = torch.randn(8, 1)
    out = LangevinDynamics(pgm, steps=0).query(query={"a": start})
    assert torch.equal(out.samples["a"].tensor, start)


def test_clamp_mask_pins_only_its_rows():
    """The per-row clamp CE2BM steering depends on: same variable, same batch."""
    torch.manual_seed(0)
    _, pgm = _model()
    start = torch.zeros(64, 1)
    mask = torch.zeros(64, 1, dtype=torch.bool)
    mask[::2] = True  # every other row pinned
    out = LangevinDynamics(pgm, steps=50).query(
        query={"a": start}, clamp_mask={"a": mask}
    )
    drawn = out.samples["a"].tensor
    assert torch.equal(drawn[::2], start[::2])          # pinned rows never moved
    assert (drawn[1::2] - start[1::2]).abs().mean() > 0.1  # free rows did


class Pull(torch.nn.Module):
    """``E = (x - target)^2`` — a different target per member, so the recovered
    values identify which column each member was packed into."""

    def __init__(self, target: float):
        super().__init__()
        self.target = target

    def forward(self, x):
        return ((x - self.target) ** 2).sum(-1)


def test_partially_clamped_model_keeps_declaration_order():
    """A clamped variable must stay in ITS column, not be shuffled to the end.

    The engine packs every member into one tensor and slices a variable's value
    straight out of it, which is only correct if the layout follows model order.
    Clamping the *middle* member of three is what catches a layout that appends
    the clamped ones instead of interleaving them.
    """
    variables = EmbeddingVariable(["a", "b", "c"], distribution=Normal, shape=1)
    by_name = {v.name: v for v in variables}
    pgm = MarkovNetwork(variables=variables, factors=[
        ParametricPotential(scope=[by_name[m]], parametrization=Pull(t),
                            name=f"phi_{m}")
        for m, t in (("a", 1.0), ("b", 2.0), ("c", 3.0))
    ])
    out = LangevinDynamics(
        pgm, steps=300, step_size=0.05, noise_scale=0.0, grad_clip=None
    ).query(query=["a", "c"], evidence={"b": torch.full((8, 1), -9.0)})

    columns = out.samples[["a", "c"]][0]
    assert torch.allclose(columns, torch.tensor([1.0, 3.0]), atol=1e-3), columns


class TwoWells(torch.nn.Module):
    """``E(x) = -log(e^{-(x-3)^2/2} + e^{-(x+3)^2/2})`` — two modes, 6 apart."""

    def forward(self, x):
        wells = torch.stack(
            [-((x - 3.0) ** 2) / 2, -((x + 3.0) ** 2) / 2], dim=-1
        )
        return -torch.logsumexp(wells, dim=-1).sum(-1)


def _two_well_pgm():
    a = EmbeddingVariable("a", distribution=Normal, size=1)
    return MarkovNetwork(
        variables=[a],
        factors=[ParametricPotential(
            scope=[a], parametrization=TwoWells(), name="phi",
        )],
    )


def test_anneal_of_one_changes_nothing():
    """The default must reproduce an unannealed chain exactly, not merely closely.

    ``anneal`` decays the step size and the noise, so getting the decay one step
    out of phase would still look plausible on a plot. Bit-equality against a run
    that never passes the argument is what pins it.
    """
    pgm = _two_well_pgm()
    start = torch.full((16, 1), 3.0)
    kwargs = dict(steps=50, step_size=0.05, noise_scale=0.1, grad_clip=None)

    torch.manual_seed(7)
    without = LangevinDynamics(pgm, **kwargs).query(query={"a": start})
    torch.manual_seed(7)
    with_one = LangevinDynamics(pgm, anneal=1.0, **kwargs).query(query={"a": start})

    assert torch.equal(without.samples["a"].tensor, with_one.samples["a"].tensor)


def test_annealing_crosses_between_modes():
    """Why ``anneal`` exists: a fixed low noise cannot leave the basin it started in.

    Every chain starts in the right-hand well. A fixed small noise settles them
    all there — correct-looking, and wrong, because it reports one mode of a
    two-mode target. Starting the noise wide and decaying it lets roughly half
    cross the barrier before the chain freezes.
    """
    pgm = _two_well_pgm()
    start = torch.full((512, 1), 3.0)
    shared = dict(steps=500, step_size=0.05, grad_clip=None)

    torch.manual_seed(0)
    stuck = LangevinDynamics(pgm, noise_scale=0.05, **shared)
    stuck_left = (stuck.query(query={"a": start}).samples["a"].tensor < 0).float().mean()

    torch.manual_seed(0)
    annealed = LangevinDynamics(pgm, noise_scale=2.0, anneal=0.99, **shared)
    annealed_left = (
        annealed.query(query={"a": start}).samples["a"].tensor < 0
    ).float().mean()

    assert stuck_left < 0.02, f"fixed-noise chain unexpectedly crossed: {stuck_left}"
    assert 0.3 < annealed_left < 0.7, f"annealed chain did not mix: {annealed_left}"


def test_compute_score_matches_the_analytic_gradient():
    """``compute_score`` is ``-grad E``, checked against a closed form.

    ``Quadratic`` is ``E = ||z - mu||^2 / 2 sigma^2``, whose score is exactly
    ``-(z - mu) / sigma^2``. Anything that flips a sign or drops the chain rule
    through the factor aggregation shows up here immediately.
    """
    _, pgm = _model(mu=2.0, sigma=0.5)
    z = torch.randn(16, 1) * 3
    got = pgm.compute_score({"a": z}, create_graph=False)["a"]
    assert torch.allclose(got, -(z - 2.0) / 0.5 ** 2, atol=1e-5)


def test_release_holds_a_query_value_for_the_first_k_steps():
    """Repainting: a query value can be held, then let go — a stage here is a step.

    Three assertions, because the endpoints alone would pass for an implementation
    that ignores the number entirely: releasing at ``steps`` never lets go,
    releasing at 0 is bit-identical to not passing ``release``, and an intermediate
    value must land strictly between the two.
    """
    _, pgm = _model()                                   # pulls toward mu = 2.0
    supplied = torch.full((128, 1), -6.0)
    kwargs = dict(steps=60, step_size=0.02, noise_scale=0.0, grad_clip=None)
    engine = LangevinDynamics(pgm, **kwargs)

    torch.manual_seed(3)
    held = engine.query(query={"a": supplied}, release={"a": 60})
    torch.manual_seed(3)
    never_held = engine.query(query={"a": supplied})
    torch.manual_seed(3)
    released_now = engine.query(query={"a": supplied}, release={"a": 0})
    torch.manual_seed(3)
    halfway = engine.query(query={"a": supplied}, release={"a": 30})

    drift = lambda out: float((out.samples["a"].tensor - supplied).abs().mean())

    assert torch.equal(held.samples["a"].tensor, supplied)      # held all 60 steps
    # `release=0` means "never held", which is exactly today's default behaviour.
    assert torch.equal(
        released_now.samples["a"].tensor, never_held.samples["a"].tensor
    )
    assert 0.0 < drift(halfway) < drift(never_held), (
        f"half-length hold moved {drift(halfway)}, free chain {drift(never_held)}"
    )


def test_release_only_accepts_a_valued_query_variable():
    """The three meaningless spellings are refused, not guessed at.

    Each previously "worked": evidence got released, a valueless query name pinned
    its own random init, and a typo was ignored in silence.
    """
    variables = EmbeddingVariable(["a", "b"], distribution=Normal, shape=1)
    pgm = MarkovNetwork(variables=list(variables), factors=[
        ParametricPotential(scope=[v], parametrization=Pull(1.0), name=f"phi_{v.name}")
        for v in variables
    ])
    engine = LangevinDynamics(pgm, steps=5)
    value = torch.full((8, 1), -6.0)

    with pytest.raises(ValueError, match="evidence"):
        engine.query(query={"b": value}, evidence={"a": value}, release={"a": 2})

    with pytest.raises(ValueError, match="no.*tensor"):
        engine.query(query=["a"], evidence={"b": value}, release={"a": 2})

    with pytest.raises(ValueError, match="zzz"):
        engine.query(query={"a": value, "b": value}, release={"zzz": 2})

    # ...and the one legitimate spelling still goes through.
    engine.query(query={"a": value, "b": value}, release={"a": 2})


def test_release_is_a_no_op_when_absent():
    """No `release` argument must reproduce the chain bit-for-bit."""
    _, pgm = _model()
    start = torch.full((16, 1), 3.0)
    kwargs = dict(steps=40, step_size=0.05, noise_scale=0.1, grad_clip=None)

    torch.manual_seed(11)
    without = LangevinDynamics(pgm, **kwargs).query(query={"a": start})
    torch.manual_seed(11)
    with_empty = LangevinDynamics(pgm, **kwargs).query(query={"a": start}, release={})

    assert torch.equal(without.samples["a"].tensor, with_empty.samples["a"].tensor)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
