"""Tests for classifier-free guided diffusion.

Three of the things this model can get wrong are *silent* — they train, they
converge, and the loss curve looks fine — so they are pinned here rather than
left to a smoke test:

* a soft label-drop mask, which turns guidance into interpolation between the
  real and null conditions;
* the guidance arms swapped, which inverts the direction the condition pushes;
* a sampler that does not actually implement the algorithm it claims to.

The rest is the usual contract: the graph assembles, a training step reduces the
loss, and generation works in every evidence regime the analysis harness uses.
"""
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Uniform

from torch_concepts.annotations import Annotations
from torch_concepts.nn import (
    CFGSamplingEngine,
    CFGTrainingEngine,
    ClassifierFreeGuidedDiffusion,
    CompositeLoss,
    ConceptLoss,
    DiffusionSchedule,
    MSELoss,
    NLLProbLoss,
)
from torch_concepts.nn.modules.low.priors import FixedPrior
from torch_concepts.nn.modules.mid.factors.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.inference.torch.utils import sample_from
from torch_concepts.nn.modules.mid.variable import EmbeddingVariable

pytest.importorskip("pyro", reason="the label-drop mask needs pyro's straight-through Bernoulli")

SHAPE = (1, 8, 8)
IMAGE = 64
T = 40


@pytest.fixture
def annotations():
    return Annotations(
        labels=["digit", "color"], cardinalities=[4, 2],
        types=["categorical", "categorical"],
    )


def build_model(annotations, **kwargs):
    kwargs.setdefault("n_steps", T)
    kwargs.setdefault("sample_steps", 4)
    kwargs.setdefault("hidden_channels", (8, 16))
    kwargs.setdefault("embedding_size", 4)
    return ClassifierFreeGuidedDiffusion(
        input_size=SHAPE, annotations=annotations, plate=False, **kwargs
    )


def concept_evidence(digit, color, n_digit=4, n_color=2):
    return {
        "digit": F.one_hot(torch.tensor(digit), n_digit).float(),
        "color": F.one_hot(torch.tensor(color), n_color).float(),
    }


def raw(tensor):
    return getattr(tensor, "tensor", tensor)


class TestUniformRoot:
    """``t`` is a genuinely sampled root, which needs ``Uniform`` in the registry."""

    def test_the_draw_differs_across_the_batch(self):
        # The failure this guards: a root CPD's parameters are produced once and
        # *broadcast* over the batch. A prior that returned a random number would
        # give every row in the batch the same timestep — training would still
        # run, on one timestep per step instead of a spread of them.
        variable = EmbeddingVariable("t", distribution=Uniform, size=1)
        cpd = ParametricCPD(
            variable, parents=[],
            parametrization={
                "low": FixedPrior(torch.zeros(1)),
                "high": FixedPrior(torch.full((1,), float(T))),
            },
        )
        draw = sample_from(variable, cpd.root_params((64,)), torch.tensor(1.0))

        assert draw.shape == (64, 1)
        assert draw.unique().numel() == 64
        assert bool((draw >= 0).all() and (draw < T).all())

    def test_the_model_draws_a_spread_of_timesteps(self, annotations):
        model = build_model(annotations)
        model.train()
        out = model(
            query=model.default_query(torch.zeros(256, 2, dtype=torch.long)),
            input=torch.rand(256, *SHAPE),
        )
        # Not a fixed number of distinct values — a coarse check that the whole
        # range is being visited, which is what the objective averages over.
        steps = model.schedule.index(raw(out.samples["t"]))
        assert steps.unique().numel() > T // 2


class TestLabelDrop:
    """The mask must be an exact bit, or guidance is not guidance."""

    def test_the_mask_is_hard_and_the_condition_is_all_or_nothing(self, annotations):
        # THE silent failure. The ancestral engine draws discrete variables from
        # their *relaxed* surrogate, so a plain Bernoulli would give values in
        # (0, 1) and `y = m * c` would be a partially-faded condition on every
        # row. The model trains, the loss falls, and the unconditional arm of the
        # guidance formula has never been trained — nothing else would catch it.
        model = build_model(annotations, p_uncond=0.5)
        model.train()
        out = model(
            query=model.default_query(torch.zeros(512, 2, dtype=torch.long)),
            input=torch.rand(512, *SHAPE),
        )
        mask, condition = raw(out.samples["m"]), raw(out.samples["y"])

        assert set(mask.flatten().tolist()) <= {0.0, 1.0}
        # Every row is either fully conditioned or fully dropped, never between.
        dropped = condition.abs().sum(-1) == 0
        assert torch.equal(dropped, mask.flatten() == 0)
        assert 0 < int(dropped.sum()) < 512, "both arms must appear at p_uncond=0.5"

    @pytest.mark.parametrize("p_uncond", [0.0, 0.25])
    def test_the_drop_rate_matches_p_uncond(self, annotations, p_uncond):
        model = build_model(annotations, p_uncond=p_uncond)
        model.train()
        out = model(
            query=model.default_query(torch.zeros(2048, 2, dtype=torch.long)),
            input=torch.rand(2048, *SHAPE),
        )
        rate = 1.0 - float(raw(out.samples["m"]).mean())
        assert rate == pytest.approx(p_uncond, abs=0.05)

    def test_p_uncond_of_one_is_rejected(self, annotations):
        with pytest.raises(ValueError, match="p_uncond"):
            build_model(annotations, p_uncond=1.0)


class TestSchedule:
    """The reverse update, against the algorithm it claims to implement."""

    def test_eta_one_reproduces_ddpm_algorithm_2(self):
        # This is what licenses sampling at eta=0 for the metrics: the update is
        # one family, and the stochastic member of it *is* the reference
        # algorithm, so choosing eta is a choice within a verified update rather
        # than a different sampler.
        schedule = DiffusionSchedule(n_steps=200)
        alpha_bar = schedule.alpha_bar
        torch.manual_seed(0)
        x, eps = torch.randn(4, 16), torch.randn(4, 16)

        def algorithm_2(step, z):
            """Ho et al. 2020, Algorithm 2 line 4, with sigma_t^2 = beta_tilde_t."""
            bar_t = alpha_bar[step]
            bar_prev = alpha_bar[step - 1] if step > 0 else torch.tensor(1.0)
            alpha_t = bar_t / bar_prev
            beta_t = 1.0 - alpha_t
            mean = (x - (beta_t / (1.0 - bar_t).sqrt()) * eps) / alpha_t.sqrt()
            beta_tilde = (1.0 - bar_prev) / (1.0 - bar_t) * beta_t
            return mean + beta_tilde.sqrt() * z

        for step in (199, 100, 37, 1, 0):
            index = torch.full((4, 1), step, dtype=torch.long)
            torch.manual_seed(7)
            ours = schedule.ddim_step(x, eps, index, index - 1, eta=1.0)
            torch.manual_seed(7)
            reference = algorithm_2(step, torch.randn(4, 16))
            assert torch.allclose(ours, reference, atol=1e-5), f"step {step}"

    def test_the_last_step_is_noiseless_at_any_eta(self):
        # Algorithm 2's `z = 0 if t == 1` clause, which here falls out of the
        # sigma formula rather than being special-cased — so it cannot be
        # forgotten when eta changes.
        schedule = DiffusionSchedule(n_steps=50)
        last = torch.full((2, 1), 0, dtype=torch.long)
        assert schedule.sigma(last, last - 1, eta=1.0).abs().max() == 0

    def test_the_grid_always_ends_at_the_clean_end(self):
        schedule = DiffusionSchedule(n_steps=100)
        for n in (100, 25, 7, 1):
            pairs = schedule.timestep_pairs(n)
            assert pairs[0][0] == 99, "starts at the noisiest step"
            assert pairs[-1][1] == -1, "ends past the last step, at clean data"
            assert len(pairs) == n


class TestGuidance:
    """The combination, its conventions, and the batching that computes it."""

    @pytest.fixture
    def primed(self, annotations):
        # The U-Net's output convolution is zero-initialised, so an untrained
        # model predicts exactly zero for both arms and every guidance scale
        # agrees trivially. Break that, or these tests pass vacuously.
        model = build_model(annotations)
        torch.nn.init.normal_(model.unet.out_conv.weight, std=0.1)
        torch.nn.init.normal_(model.unet.out_conv.bias, std=0.1)
        return model.eval()

    def test_the_two_arms_are_the_condition_and_zero(self, primed):
        condition = concept_evidence([1, 2], [0, 1])
        y_cond, y_null = primed.sampler._conditions(condition, torch.Size([2]))

        assert y_null.abs().max() == 0
        assert torch.equal(y_cond, torch.cat([condition["digit"], condition["color"]], -1))

    def test_batching_the_arms_matches_two_separate_calls(self, primed):
        engine = primed.sampler
        x = torch.randn(3, IMAGE)
        index = torch.full((3, 1), 5, dtype=torch.long)
        y_cond, y_null = engine._conditions(concept_evidence([1, 2, 3], [0, 1, 0]), torch.Size([3]))

        cpd = primed.pgm.factors["eps_hat"]
        def arm(y):
            return next(iter(cpd(parent_values={
                "x_t": x, "t": index.float(), "y": y,
            }).values()))

        eps_cond, eps_null = arm(y_cond), arm(y_null)
        assert (eps_cond - eps_null).abs().max() > 1e-4, "arms must actually differ"

        for scale in (0.0, 3.0):
            engine.guidance_scale = scale
            combined = engine._guided_noise(x, index, y_cond, y_null)
            expected = (1.0 + scale) * eps_cond - scale * eps_null
            assert torch.allclose(combined, expected, atol=1e-6)
            if scale == 0.0:
                # s=0 is *conditional* sampling, not unconditional — the other
                # convention (w = 1 + s) reads the same formula differently, and
                # mixing them silently halves or doubles the guidance.
                assert torch.allclose(combined, eps_cond, atol=1e-6)


class TestGeneration:
    """The three evidence regimes the analysis harness uses."""

    @pytest.fixture
    def model(self, annotations):
        return build_model(annotations).eval()

    def test_unconditional_draw_reports_what_the_harness_reads(self, model):
        # What FID runs on: no evidence at all, so the concepts come from the
        # learned marginal p(c) and the noise from N(0, I).
        out = model.sample(
            query=["input", model.latent_variable_name, "digit", "color"],
            n_samples=5,
        )

        assert out.params["value"]["input"].shape == (5, IMAGE)
        assert set(out.samples.annotation.labels) >= {"input", "eps", "digit", "color"}

    def test_conditional_draw(self, model):
        out = model.sample(evidence=concept_evidence([1, 2], [0, 1]))
        assert out.params["value"]["input"].shape == (2, IMAGE)

    def test_intervening_on_one_concept_leaves_the_other_drawn(self, model):
        # What steerability runs on: some concepts held, the rest resolved.
        held = {"color": F.one_hot(torch.tensor([1, 1]), 2).float()}
        out = model.sample(query=["input", "digit", "color"], evidence=held)

        assert torch.equal(raw(out.samples["color"]), held["color"])
        assert out.params["value"]["input"].shape == (2, IMAGE)

    def test_eta_zero_generation_is_reproducible(self, model):
        # The property the steerability metric depends on: replaying the same
        # noise with the same concepts must give the same image, so that a
        # difference between two generations is attributable to the concept.
        evidence = {**concept_evidence([1, 2], [0, 1]), "eps": torch.randn(2, IMAGE)}
        first = raw(model.sample(evidence=evidence).params["value"]["input"])
        second = raw(model.sample(evidence=evidence).params["value"]["input"])
        assert torch.equal(first, second)

    def test_eta_above_zero_is_stochastic(self, annotations):
        model = build_model(annotations, eta=1.0, sample_steps=T).eval()
        evidence = {**concept_evidence([1, 2], [0, 1]), "eps": torch.randn(2, IMAGE)}
        first = raw(model.sample(evidence=evidence).params["value"]["input"])
        second = raw(model.sample(evidence=evidence).params["value"]["input"])
        assert not torch.equal(first, second)

    def test_the_sampler_rejects_a_graph_it_cannot_drive(self, model):
        with pytest.raises(ValueError, match="not in the model's graph"):
            CFGSamplingEngine(model.pgm, schedule=model.schedule, prediction="nope")


class TestModelContract:
    """Wiring the rest of the stack relies on."""

    def test_the_objective_engine_serves_train_and_eval(self, annotations):
        # Validation must measure the objective training minimises, so BOTH
        # engines are the denoising pass. Wiring the sampler as `eval_inference`
        # instead leaves val_loss uncomputable — it reports images, and the loss
        # needs eps_hat and eps_target.
        model = build_model(annotations)
        assert isinstance(model.train_inference, CFGTrainingEngine)
        assert isinstance(model.eval_inference, CFGTrainingEngine)
        assert isinstance(model.sampler, CFGSamplingEngine)
        # One schedule instance, or the sampler denoises on a curve the network
        # was never trained against.
        assert model.sampler.schedule is model.schedule

    def test_the_loss_is_computable_in_eval_mode(self, annotations):
        # The regression this pins: `.eval()` must not swap in an engine that
        # cannot report the two nodes the loss compares.
        model = build_model(annotations).eval()
        out = model(
            query=model.default_query(torch.zeros(4, 2, dtype=torch.long)),
            input=torch.rand(4, *SHAPE),
        )
        assert float(MSELoss(variable="eps_hat", target_variable="eps_target")(out)) > 0

    def test_a_backbone_is_refused_rather_than_silently_carried(self, annotations):
        with pytest.raises(ValueError, match="takes no backbone"):
            build_model(annotations, backbone=torch.nn.Linear(IMAGE, 16))

    def test_image_shaped_and_flat_observations_agree(self, annotations):
        model = build_model(annotations)
        image = torch.rand(3, *SHAPE)
        assert torch.equal(
            model.flatten_observation(image), image.reshape(3, IMAGE)
        )
        flat = torch.rand(3, IMAGE)
        assert torch.equal(model.flatten_observation(flat), flat)

    def test_the_latent_name_is_discoverable(self, annotations):
        # The analysis harness replays generations by this name.
        model = build_model(annotations)
        assert model.latent_variable_name in model.pgm.variables


class TestTraining:
    """The objective, end to end."""

    def test_a_few_steps_reduce_the_denoising_loss(self, annotations):
        torch.manual_seed(0)
        model = build_model(annotations)
        loss_fn = CompositeLoss(
            terms=[
                MSELoss(variable="eps_hat", target_variable="eps_target"),
                ConceptLoss(
                    binary_param="probs", categorical_param="probs",
                    categorical=NLLProbLoss(),
                ),
            ],
            weights=[1.0, 1.0],
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()

        concepts = torch.stack([
            torch.randint(0, 4, (32,)), torch.randint(0, 2, (32,))
        ], dim=-1)
        # One fixed batch: the point is that the objective is minimisable and the
        # gradient reaches the network, not that the model generalises.
        images = torch.rand(32, *SHAPE)
        target = model.prepare_target(concepts)

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            out = model(query=model.default_query(concepts), input=images)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))

        assert losses[-1] < losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}"

    def test_the_denoising_gradient_reaches_the_unet(self, annotations):
        model = build_model(annotations)
        model.train()
        concepts = torch.zeros(8, 2, dtype=torch.long)
        out = model(query=model.default_query(concepts), input=torch.rand(8, *SHAPE))
        MSELoss(variable="eps_hat", target_variable="eps_target")(out).backward()

        grads = [p.grad for p in model.unet.parameters() if p.grad is not None]
        assert grads and max(float(g.abs().max()) for g in grads) > 0
