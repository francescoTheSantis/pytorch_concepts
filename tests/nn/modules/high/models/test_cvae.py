"""Smoke tests for the Conditional Variational Autoencoder.

The CVAE's defining property is that the concepts are *evidence*: they are read
by the decoder rather than predicted from the latent. These tests pin the three
consequences — the decoder actually depends on them, the marginal ``p(c)`` is
what the concept loss trains, and an unconditional draw still works, since that
is what FID and steerability run on.
"""
import math

import pytest
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, MultivariateNormal, Normal

from torch_concepts.annotations import Annotations
from torch_concepts.nn import (
    AncestralSamplingInference,
    ConceptLoss,
    ConditionalVariationalAutoencoder,
    MLP,
    NLLProbLoss,
    ReconstructionLoss,
)
from torch_concepts.nn.modules.high.models.cvae import ConceptEmbedding

pytest.importorskip("pyro", reason="the CVAE's default inference engine needs pyro-ppl")

INPUT_SIZE, LATENT_SIZE, EMBEDDING_SIZE = 24, 8, 4


def build_model(annotations, observation=Bernoulli, embedding_size=EMBEDDING_SIZE, **kwargs):
    condition = len(annotations.labels) * embedding_size
    return ConditionalVariationalAutoencoder(
        input_size=INPUT_SIZE,
        annotations=annotations,
        encoder=MLP(INPUT_SIZE, 16, LATENT_SIZE),
        # Raw: the model composes the observation's activation on top.
        decoder=MLP(LATENT_SIZE + condition, 16, INPUT_SIZE),
        latent_size=LATENT_SIZE,
        embedding_size=embedding_size,
        observation=observation,
        plate=False,
        **kwargs,
    )


@pytest.fixture
def binary_annotations():
    return Annotations(labels=["a", "b"], cardinalities=[1, 1], types=["binary", "binary"])


@pytest.fixture
def mixed_annotations():
    return Annotations(
        labels=["a", "digit"], cardinalities=[1, 4], types=["binary", "categorical"],
    )


def ground_truth(annotations, batch=6):
    """One ground-truth row per concept column, integer-coded."""
    return torch.stack([
        torch.randint(0, max(int(c), 2), (batch,)).float()
        for c in annotations.cardinalities
    ], dim=-1)


class TestConditionSizing:
    def test_condition_width_is_one_embedding_per_concept(self, mixed_annotations):
        """Not the raw 1 + 4 columns: each concept gets its own `m`-wide slot, so
        a binary one and a 4-way categorical reach the decoder equally wide."""
        assert build_model(mixed_annotations).condition_size == 2 * EMBEDDING_SIZE

    def test_one_embedding_layer_per_concept_sized_to_that_concept(
        self, mixed_annotations
    ):
        embedder = build_model(mixed_annotations).condition_embedding
        assert embedder.sizes == [1, 4]  # the binary's column, the one-hot's row
        assert len(embedder.embeddings) == 2
        for layer, size in zip(embedder.embeddings, embedder.sizes):
            assert (layer.in_features, layer.out_features) == (size, EMBEDDING_SIZE)

    def test_the_decoder_reads_latent_plus_embedded_condition(self, mixed_annotations):
        model = build_model(mixed_annotations)
        parents = [p.name for p in model.pgm.factors["input"].parents]
        assert parents == ["z", "a", "digit"]

        seen = {}
        model.decoder.register_forward_hook(
            lambda mod, inputs, out: seen.update(width=inputs[0].shape[-1])
        )
        model(query=model.default_query(ground_truth(mixed_annotations)),
              input=torch.rand(6, INPUT_SIZE))
        assert seen["width"] == LATENT_SIZE + model.condition_size

    @pytest.mark.parametrize("plate", [None, True, False])
    def test_the_condition_width_is_the_same_however_concepts_are_grouped(self, plate):
        """`condition_size` is read off the annotations so the decoder can be
        sized before the model exists — it must match what is actually built,
        whether the concepts end up as plates or as individual variables."""
        annotations = Annotations(
            labels=["a", "b", "d1", "d2"], cardinalities=[1, 1, 3, 3],
            types=["binary", "binary", "categorical", "categorical"],
        )
        condition = len(annotations.labels) * EMBEDDING_SIZE
        model = ConditionalVariationalAutoencoder(
            input_size=INPUT_SIZE,
            annotations=annotations,
            encoder=MLP(INPUT_SIZE, 16, LATENT_SIZE),
            decoder=MLP(LATENT_SIZE + condition, 16, INPUT_SIZE),
            latent_size=LATENT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            observation=Bernoulli,
            plate=plate,
        )
        assert model.condition_size == condition
        # One embedding per concept, whether or not the concepts share a plate.
        assert model.condition_embedding.sizes == [1, 1, 3, 3]
        # Sized for that width at construction, so this only runs if it matches.
        model(query=model.default_query(ground_truth(annotations, batch=3)),
              input=torch.rand(3, INPUT_SIZE))

    def test_a_mis_sized_decoder_fails_loudly(self, mixed_annotations):
        """The embedded condition is concatenated, so the decoder must fit it."""
        model = ConditionalVariationalAutoencoder(
            input_size=INPUT_SIZE,
            annotations=mixed_annotations,
            encoder=MLP(INPUT_SIZE, 16, LATENT_SIZE),
            decoder=MLP(LATENT_SIZE, 16, INPUT_SIZE),  # forgot the condition
            latent_size=LATENT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            observation=Bernoulli,
            plate=False,
        )
        with pytest.raises(RuntimeError):
            model(query=model.default_query(ground_truth(mixed_annotations)),
                  input=torch.rand(6, INPUT_SIZE))


class TestSharedConditionEmbedding:
    """The guide and the decoder read ``c`` through the *same* layers.

    One learned representation of the condition, as in the reference's single
    ``label_embedding`` — and, negatively, no second copy silently trained on
    only one of the two paths.
    """

    def test_exactly_one_embedder_exists_in_the_model(self, mixed_annotations):
        model = build_model(mixed_annotations)
        embedders = {id(m) for m in model.modules() if isinstance(m, ConceptEmbedding)}
        assert len(embedders) == 1
        assert next(iter(embedders)) == id(model.condition_embedding)

    def test_the_guide_and_the_decoder_hold_the_same_object(self, mixed_annotations):
        model = build_model(mixed_annotations)
        assert model.pgm.guides["z"].trunk.embedder is model.condition_embedding

    def test_the_embedding_parameters_are_not_duplicated(self, mixed_annotations):
        model = build_model(mixed_annotations)
        shared = {id(p) for p in model.condition_embedding.parameters()}
        assert sum(1 for p in model.parameters() if id(p) in shared) == len(shared)

    def test_a_second_decoder_copy_still_shares_it(self, binary_annotations):
        """``global_scale=False`` deep-copies the decoder for the scale head;
        the embedding must not be cloned along with it."""
        model = build_model(binary_annotations, observation=Normal, global_scale=False)
        embedders = {id(m) for m in model.modules() if isinstance(m, ConceptEmbedding)}
        assert len(embedders) == 1

    def test_reconstruction_gradient_reaches_the_embedding(self, mixed_annotations):
        model = build_model(mixed_annotations)
        out = model(query=model.default_query(ground_truth(mixed_annotations)),
                    input=torch.rand(6, INPUT_SIZE))
        out.probs["input"].sum().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.condition_embedding.parameters())


class TestConditioning:
    def test_the_generation_depends_on_the_condition(self, binary_annotations):
        """Same z, different concepts -> a different image. Without this the
        model is a plain VAE with some unused inputs."""
        model = build_model(binary_annotations)
        model.eval()
        engine = AncestralSamplingInference(model.pgm, p_int=1.0)
        z = torch.randn(4, LATENT_SIZE)
        on = engine.query(query=["input"], evidence={
            "z": z, "a": torch.ones(4, 1), "b": torch.zeros(4, 1)})
        off = engine.query(query=["input"], evidence={
            "z": z, "a": torch.zeros(4, 1), "b": torch.zeros(4, 1)})
        assert (on.probs["input"] - off.probs["input"]).abs().max() > 1e-5

    def test_the_guide_reads_the_condition(self, binary_annotations):
        """q(z | x, c): the same image with different concepts encodes
        differently. This is what keeps z from having to carry the concepts."""
        model = build_model(binary_annotations)
        x = torch.rand(4, INPUT_SIZE)
        posteriors = [
            model(query=model.default_query(c), input=x).guide_params["loc"]["z"]
            for c in (torch.zeros(4, 2), torch.ones(4, 2))
        ]
        assert not torch.allclose(*posteriors)

    def test_an_unconditioned_guide_ignores_the_concepts(self, binary_annotations):
        model = build_model(binary_annotations, condition_encoder=False)
        assert [p.name for p in model.pgm.guides["z"].parents] == ["input"]
        x = torch.rand(4, INPUT_SIZE)
        posteriors = [
            model(query=model.default_query(c), input=x).guide_params["loc"]["z"]
            for c in (torch.zeros(4, 2), torch.ones(4, 2))
        ]
        assert torch.allclose(*posteriors)

    def test_an_unconditioned_guide_encodes_without_any_concepts(self, binary_annotations):
        """The reason the flag exists: an image whose concepts are unknown."""
        model = build_model(binary_annotations, condition_encoder=False)
        out = model(query=list(model.pgm.variables), input=torch.rand(4, INPUT_SIZE))
        assert out.guide_params["loc"]["z"].shape == (4, LATENT_SIZE)

    def test_a_conditioned_guide_says_which_value_is_missing(self, binary_annotations):
        model = build_model(binary_annotations)
        with pytest.raises(KeyError, match="'a'"):
            model(query=list(model.pgm.variables), input=torch.rand(4, INPUT_SIZE))


class TestConceptMarginal:
    def test_the_concepts_are_roots_with_no_parents(self, mixed_annotations):
        model = build_model(mixed_annotations)
        for name in ("a", "digit"):
            assert model.pgm.factors[name].is_root

    def test_the_reported_probabilities_are_the_marginal(self, mixed_annotations):
        """Every row reports the same p(c): this model does not infer concepts
        from the observation, so the value cannot depend on the batch."""
        model = build_model(mixed_annotations)
        out = model(query=model.default_query(ground_truth(mixed_annotations)),
                    input=torch.rand(6, INPUT_SIZE))
        for name in ("a", "digit"):
            probs = out.probs[name]
            assert torch.allclose(probs, probs[:1].expand_as(probs))
        assert torch.allclose(out.probs["digit"].sum(-1), torch.ones(6), atol=1e-5)

    def test_the_concept_loss_fits_the_empirical_frequencies(self):
        """What the marginal is for: an unconditional draw has to produce a
        plausible condition, or FID measures the sampler rather than the model."""
        annotations = Annotations(labels=["a"], cardinalities=[1], types=["binary"])
        torch.manual_seed(0)
        model = build_model(annotations)
        loss_fn = ConceptLoss(binary=nn.BCELoss(), binary_param="probs",
                              categorical=NLLProbLoss(), categorical_param="probs")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2)

        c = torch.bernoulli(torch.full((256, 1), 0.8))
        x = torch.rand(256, INPUT_SIZE)
        for _ in range(150):
            out = model(query=model.default_query(c), input=x)
            optimizer.zero_grad()
            loss_fn(out, model.prepare_target(c)).backward()
            optimizer.step()

        learned = model(query=model.default_query(c), input=x).probs["a"][0]
        assert float(learned) == pytest.approx(float(c.mean()), abs=0.05)


class TestGeneration:
    """The path ``analysis/run_generative_analysis.py`` runs on: an ancestral
    engine draws every root from its own prior, so an unconditioned query really
    does sample ``c ~ p(c)`` and ``z ~ N(0, I)``."""

    def test_an_unconditional_draw_produces_an_image_and_a_condition(
        self, mixed_annotations
    ):
        model = build_model(mixed_annotations, observation=Normal)
        model.eval()
        engine = AncestralSamplingInference(model.pgm, p_int=1.0)
        out = engine.query(query=["input", "z", "a", "digit"], evidence={}, n_samples=5)
        assert out.loc["input"].shape == (5, INPUT_SIZE)
        assert bool((out.scale["input"] > 0).all())
        assert out.samples["a"].shape == (5, 1)
        assert out.samples["digit"].shape == (5, 4)

    def test_reconstruction_loss_is_finite(self, mixed_annotations):
        model = build_model(mixed_annotations, observation=Normal)
        x = torch.rand(6, INPUT_SIZE)
        out = model(query=model.default_query(ground_truth(mixed_annotations)), input=x)
        out.extra = {"evidence": {"input": x}}
        assert torch.isfinite(ReconstructionLoss(variable="input")(out))

    def test_the_default_gaussian_likelihood_is_mse_on_the_mean(self, binary_annotations):
        """At the default sigma=1 the Gaussian NLL is ``0.5*(x-loc)^2`` plus a
        constant, so training is plain MSE on the predicted mean and the KL
        weight is a true beta. Pinned because it is a claim the run configs'
        loss weights are chosen against.
        """
        model = build_model(binary_annotations, observation=Normal)
        scale_head = model.pgm.factors["input"].parametrization["scale"]
        assert sum(p.numel() for p in scale_head.parameters()) == 0  # fixed

        x = torch.rand(5, INPUT_SIZE)
        query = model.default_query(ground_truth(binary_annotations, batch=5))

        def decoder_grads(use_nll):
            torch.manual_seed(0)
            model.zero_grad()
            out = model(query=query, input=x)
            out.extra = {"evidence": {"input": x}}
            loc = out.loc["input"].tensor
            assert torch.equal(out.scale["input"].tensor,
                               torch.ones_like(out.scale["input"].tensor))
            loss = (ReconstructionLoss(variable="input")(out) if use_nll
                    else 0.5 * ((loc - x) ** 2).sum(-1).mean())
            loss.backward()
            return float(loss), [p.grad.clone() for p in model.decoder.parameters()]

        nll, nll_grads = decoder_grads(use_nll=True)
        mse, mse_grads = decoder_grads(use_nll=False)

        constant = 0.5 * INPUT_SIZE * math.log(2 * math.pi)
        assert nll - mse == pytest.approx(constant, abs=1e-4)
        for from_nll, from_mse in zip(nll_grads, mse_grads):
            assert torch.allclose(from_nll, from_mse, atol=1e-6)

    def test_gradients_reach_the_decoder_and_the_guide(self, binary_annotations):
        model = build_model(binary_annotations)
        out = model(query=model.default_query(ground_truth(binary_annotations)),
                    input=torch.rand(6, INPUT_SIZE))
        out.probs["input"].sum().backward()
        assert any(p.grad is not None for p in model.decoder.parameters())
        assert any(p.grad is not None for p in model.encoder.parameters())


class TestDistributionFamilies:
    """Any registered family may stand in for a concept, via
    ``variable_distributions``. The marginal ``p(c)``'s heads are sized from the
    variable's own ``param_sizes``, which is what makes that true for a family
    whose parameters are *not* one scalar per event element.
    """

    def test_a_continuous_concept_reports_loc_and_scale(self):
        """``param_for_discrete_var`` is 'probs', which a Normal does not have,
        so the discrete activation must only reach discrete variables."""
        annotations = Annotations(
            labels=["a", "h"], cardinalities=[1, 1], types=["binary", "continuous"]
        )
        model = build_model(annotations)
        out = model(query=model.default_query(ground_truth(annotations, batch=4)),
                    input=torch.rand(4, INPUT_SIZE))
        assert sorted(out.params["h"]) == ["loc", "scale"]
        assert sorted(out.params["a"]) == ["probs"]
        assert bool((out.scale["h"] > 0).all())

    def test_a_multivariate_normal_concept_gets_a_cholesky_prior(self):
        """A ``scale_tril`` needs ``size*(size+1)//2`` values, not ``size``.

        Sizing the marginal's head from ``variable.size`` raised
        ``TrilActivation: got 3 values but a 3x3 Cholesky factor needs 6``.
        """
        annotations = Annotations(
            labels=["a", "v"], cardinalities=[1, 3], types=["binary", "continuous"]
        )
        model = build_model(
            annotations, observation=Normal,
            variable_distributions={"continuous": MultivariateNormal},
        )
        ground = torch.tensor([[1.0, 0.5], [0.0, -0.5]])
        out = model(query=model.default_query(ground), input=torch.rand(2, INPUT_SIZE))
        assert sorted(out.params["v"]) == ["loc", "scale_tril"]
        assert out.params["v"]["scale_tril"].shape == (2, 3, 3)

    def test_straight_through_families_sample_exact_realizations(self):
        """The plain families draw *soft* Concrete values for an unobserved
        concept; the straight-through ones draw an exact bit / one-hot row, which
        is what the decoder is fed during training."""
        pyro_dist = pytest.importorskip("pyro.distributions")
        annotations = Annotations(
            labels=["a", "digit"], cardinalities=[1, 4],
            types=["binary", "categorical"],
        )
        binary = pyro_dist.RelaxedBernoulliStraightThrough
        categorical = pyro_dist.RelaxedOneHotCategoricalStraightThrough
        model = build_model(
            annotations, observation=Normal,
            variable_distributions={"binary": binary, "categorical": categorical},
            variable_dist_kwargs={binary: {"temperature": 0.5},
                                  categorical: {"temperature": 0.5}},
        )
        model.eval()
        engine = AncestralSamplingInference(model.pgm, p_int=1.0)
        out = engine.query(query=["input", "a", "digit"], evidence={}, n_samples=4)

        drawn_binary = out.samples["a"].tensor
        assert bool(((drawn_binary == 0) | (drawn_binary == 1)).all())
        one_hot = out.samples["digit"].tensor
        assert bool(((one_hot == 0) | (one_hot == 1)).all())
        assert torch.equal(one_hot.sum(-1), torch.ones(4))


class TestGuideSharesOneBackbonePass:
    """``loc`` and ``scale`` are heads over one shared trunk, so a backbone runs
    once per step rather than once per parameter."""

    class CountingBackbone(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.out_features = out_features
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.linear(x)

    def _model(self, annotations, backbone):
        condition = len(annotations.labels) * EMBEDDING_SIZE
        return ConditionalVariationalAutoencoder(
            input_size=INPUT_SIZE,
            annotations=annotations,
            backbone=backbone,
            decoder=MLP(LATENT_SIZE + condition, 16, INPUT_SIZE),
            latent_size=LATENT_SIZE,
            embedding_size=EMBEDDING_SIZE,
            observation=Bernoulli,
            plate=False,
        )

    def test_the_backbone_runs_once_per_forward(self, binary_annotations):
        backbone = self.CountingBackbone(INPUT_SIZE, 32)
        model = self._model(binary_annotations, backbone)
        model(query=model.default_query(ground_truth(binary_annotations)),
              input=torch.rand(6, INPUT_SIZE))
        assert backbone.calls == 1

    def test_the_backbone_is_not_duplicated_into_the_scale_head(self, binary_annotations):
        backbone = self.CountingBackbone(INPUT_SIZE, 32)
        guide = self._model(binary_annotations, backbone).pgm.guides["z"]
        assert sum(m is backbone for m in guide.modules()) == 1
        assert sum(m is backbone for m in guide.trunk.modules()) == 1
