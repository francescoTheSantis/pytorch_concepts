"""Classifier-free guided diffusion, conditioned on a set of concepts.

The third point of the generative comparison. Where
:class:`~torch_concepts.nn.ConceptBottleneckGenerativeModel` routes an image
through a concept bottleneck and
:class:`~torch_concepts.nn.ConditionalVariationalAutoencoder` hands the concepts
to a decoder, this model hands them to a *denoiser* and generates by running a
reverse diffusion process. Both baselines are single-shot decoders, so both may
be losing to a weak generator rather than to their concept structure; a
diffusion model conditioned on the same concepts, measured with the same FID and
steerability, is what separates those two explanations.

It exists as a **baseline** and is not meant to be extended.

The graph
---------
The PGM below is the *training* computation, not the generative process. A
diffusion model's generative distribution is this graph's ``eps_hat`` CPD
applied many times in sequence, which no fixed DAG expresses — that loop lives
in :class:`~torch_concepts.nn.CFGSamplingEngine`. What the graph buys is that
every stochastic ingredient of a training step is a node the framework already
knows how to draw, clamp and intervene on::

    t ~ U{0..T-1} ─┐
    eps ~ N(0, I) ─┼─→ x_t = sqrt(ab_t)*input + sqrt(1-ab_t)*eps
    input (obs) ───┘        │
                            ├─→ eps_hat = UNet(x_t, t, y)      the network
    m ~ Bernoulli(p) ─┐     │
    concepts (obs) ───┴─→ y = m * c                            the condition

    eps, t ────────────→ eps_target                            the target

Three consequences, and they are the whole design:

1. **The label drop is a node, not a training trick.** Classifier-free guidance
   needs the network to have seen both ``p(x | c)`` and ``p(x)``, which is
   normally a line of code inside the training loop. Here it is ``m``, a
   Bernoulli root, and ``y = m * c``. The payoff is at *sampling* time: the
   unconditional arm of the guidance formula is produced by clamping ``m`` to
   zero and re-running the ``y`` CPD, so there is exactly one definition of what
   "no condition" means, rather than one in the trainer and another in the
   sampler.

   ``m`` is declared with a **straight-through** relaxed family on purpose. The
   ancestral engine draws discrete variables from their relaxed surrogate, so a
   plain ``Bernoulli`` would come out soft and ``y = m * c`` would be a *blend*
   of the real and null conditions on every row — which trains, converges, and
   is not classifier-free guidance.

2. **The null token is learned for free.** Dropping the label sets ``y = 0``, and
   the condition reaches the network through
   :class:`~torch_concepts.nn.modules.high.models.cvae.ConceptEmbedding`'s
   per-concept ``Linear``. ``Linear(0)`` is its bias — already a learned vector,
   trained on exactly the rows where the label was dropped. No separate null
   parameter is needed.

3. **The regression target is a node too.** Under ε-prediction ``eps_target`` is
   just ``eps``, so the node looks redundant; it earns its place by making the
   objective "compare two nodes" (:class:`~torch_concepts.nn.MSELoss` with
   ``target_variable``) rather than something bespoke, and by putting the whole
   choice of parametrisation in one module.

Everything is flat
------------------
Every variable here is declared with ``size``, not ``shape`` — including
``input``. Parent values reach a CPD concatenated on the last axis with **no
reshaping** (``_cat_parents``), so an image-shaped ``(B, C, H, W)`` parent
alongside a ``(B, 1)`` timestep is a rank mismatch, not a concatenation. The
image is therefore ``C*H*W`` scalars throughout the graph and
:class:`~torch_concepts.nn.ConditionalUNet` restores the grid internally. Image
evidence of any shape is accepted: the engine reshapes it on the way in.

References
----------
Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
Ho, Salimans. "Classifier-Free Diffusion Guidance", NeurIPS 2021 Workshop.
Song, Meng, Ermon. "Denoising Diffusion Implicit Models", ICLR 2021.
"""
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from torch.distributions import Bernoulli, Normal, OneHotCategorical, Uniform

from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from .....distributions import Delta
from ...low.conv import ConditionalUNet
from ...low.priors import FixedPrior, LearnablePrior
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ...mid.factors.cpd import ParametricCPD
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.inference.base import BaseInference
from ...mid.inference.torch.diffusion.sampling import CFGSamplingEngine
from ...mid.inference.torch.diffusion.schedule import DiffusionSchedule
from ...mid.inference.torch.diffusion.training import CFGTrainingEngine
from ...mid.variable import EmbeddingVariable
from ..base.graph import DirectedGraphModel
from .cvae import ConceptEmbedding


def _straight_through_bernoulli():
    """Pyro's straight-through Bernoulli, or a clear error saying why it is needed.

    See design note 1 in the module docstring: a soft ``m`` silently degrades
    guidance into interpolation, so this is a hard requirement rather than a
    nice-to-have, and failing loudly at construction beats training a model that
    is quietly not the one asked for.
    """
    try:
        from pyro.distributions import RelaxedBernoulliStraightThrough
    except ImportError as exc:  # pragma: no cover - pyro not installed
        raise ImportError(
            "ClassifierFreeGuidedDiffusion needs `pyro-ppl` for the label-drop "
            "mask: the mask must be an exact 0/1, and the ancestral engine draws "
            "a plain Bernoulli from its *relaxed* surrogate, which would make the "
            "dropped rows a blend of the real and null conditions instead of "
            "unconditional. Install pyro-ppl."
        ) from exc
    return RelaxedBernoulliStraightThrough


class NoiseState(nn.Module):
    """``x_t = sqrt(alpha_bar_t) * input + sqrt(1 - alpha_bar_t) * eps``.

    The forward diffusion step, as a CPD head. Parents arrive concatenated on the
    last axis in declaration order, so they are split back apart by their known
    widths — the same thing
    :class:`~torch_concepts.nn.modules.high.models.cvae.ConceptEmbedding` does.
    """

    def __init__(self, schedule: DiffusionSchedule, image_size: int) -> None:
        super().__init__()
        self.schedule = schedule
        self.widths = [image_size, image_size, 1]  # input, eps, t

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        image, eps, t = torch.split(embeddings, self.widths, dim=-1)
        return (
            self.schedule.sqrt_alpha_bar(t) * image
            + self.schedule.sqrt_one_minus_alpha_bar(t) * eps
        )


class NoiseTarget(nn.Module):
    """The regression target: ``eps`` itself, under ε-prediction.

    Takes ``t`` as a parent it does not read. That is deliberate — the target of
    a Gaussian diffusion path is a function of ``(eps, t)`` in general (the score
    is ``-eps / sqrt(1 - alpha_bar_t)``), so keeping the edge means changing
    parametrisation is a change to this module and nothing else.
    """

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.widths = [image_size, 1]  # eps, t

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        eps, _t = torch.split(embeddings, self.widths, dim=-1)
        return eps


class MaskedCondition(nn.Module):
    """``y = m * c`` — the condition, or all-zeros where the label was dropped.

    One mask column gates the whole concept vector, so a row is conditional or
    unconditional as a unit. Masking per concept would teach the network to fill
    in missing concepts from the present ones, which is a different model.
    """

    def forward(self, concepts: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings * concepts


class DenoisingHead(nn.Module):
    """``eps_hat = UNet(x_t, t, embed(y))`` — splits the parents, embeds, predicts."""

    def __init__(
        self, unet: ConditionalUNet, embedder: ConceptEmbedding,
        image_size: int, condition_size: int,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.embedder = embedder
        self.widths = [image_size, 1, condition_size]  # x_t, t, y

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x_t, t, y = torch.split(embeddings, self.widths, dim=-1)
        return self.unet(x_t, t, self.embedder(y))


class ClassifierFreeGuidedDiffusion(DirectedGraphModel):
    """Concept-conditioned DDPM trained with classifier-free guidance.

    Parameters
    ----------
    input_size : int or tuple
        Shape of one observation. A tuple ``(C, H, W)`` is required when the
        default U-Net is built; the graph flattens it either way.
    annotations : Annotations
        Concept annotations. Every concept is a conditioning variable; there are
        no task variables.
    unet : nn.Module, optional
        The denoiser, called as ``unet(x_t, t, condition)`` with ``x_t`` flat
        ``(B, C*H*W)``, ``t`` a ``(B, 1)`` timestep and ``condition`` the
        *embedded* concepts, and returning a flat noise prediction. Defaults to a
        :class:`~torch_concepts.nn.ConditionalUNet` sized from ``input_size``.
    n_steps : int, default 1000
        ``T``, the number of diffusion timesteps.
    beta_start, beta_end : float
        Endpoints of the linear beta schedule. Tied to ``n_steps`` — see
        :class:`~torch_concepts.nn.DiffusionSchedule`.
    embedding_size : int, default 16
        Width of *one* concept's embedding, as in the CVAE, so the condition the
        U-Net reads is ``n_concepts * embedding_size`` however the concepts are
        distributed.
    p_uncond : float, default 0.1
        Probability of dropping the condition on a given row. The standard value;
        guidance needs enough unconditional rows to learn ``p(x)``, but every
        dropped row is one the conditional model does not see.
    hidden_channels : sequence of int, default ``(64, 128)``
        Passed to the default U-Net.
    guidance_scale : float, default 3.0
        ``s`` in ``(1+s)*eps_cond - s*eps_uncond``, used by the sampling engine.
        ``0`` is plain conditional sampling.
    sample_steps : int, optional
        Reverse steps at generation time. Defaults to ``n_steps``; fewer strides
        the grid (valid for the DDIM update) and is what keeps FID affordable.
    eta : float, default 0.0
        Reverse-process stochasticity. ``0`` is deterministic DDIM; ``1`` on the
        full grid is DDPM's Algorithm 2. Kept at ``0`` so a steerability
        comparison holds everything but the concept fixed — with noise injected
        at every step the shared starting point is swamped.
    clip : tuple of float, optional
        Range to clamp the predicted clean image into each step, e.g. ``(0., 1.)``
        for images in that range. Recommended; see
        :meth:`~torch_concepts.nn.DiffusionSchedule.ddim_step`.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Configuration of the **objective** engine — one denoising pass — which
        both training and validation run, so that ``val_loss`` measures what
        training minimises. Defaults to
        :class:`~torch_concepts.nn.CFGTrainingEngine` for both. Generation is a
        different computation, not an evaluation of the objective, and lives on
        :attr:`sampler` instead.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    plate : bool or None, default None
        ``False`` gives one variable per concept, which is what per-concept
        interventions (and the steerability metric) address.
    **kwargs
        Forwarded to :class:`BaseModel`.

    Attributes
    ----------
    schedule : DiffusionSchedule
        The noise schedule, shared with the sampling engine.
    sampler : CFGSamplingEngine
        The reverse process. Reached through :meth:`sample`.
    condition_size : int
        Width of the *embedded* condition the U-Net reads.
    concept_width : int
        Width of the raw concept vector — the width of ``y``.

    Notes
    -----
    Concept accuracy is at the majority-class rate by construction: like the
    CVAE, this model reads its concepts and never infers them. The concept loss
    fits the *marginal* ``p(c)``, which is what makes an unconditional draw (for
    FID) produce a plausible condition. What is comparable across the three
    models is FID and steerability.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.annotations import Annotations
    >>> from torch_concepts.nn import ClassifierFreeGuidedDiffusion
    >>>
    >>> ann = Annotations(labels=['digit', 'color'], cardinalities=[10, 2],
    ...                   types=['categorical', 'categorical'])
    >>> model = ClassifierFreeGuidedDiffusion(
    ...     input_size=(3, 32, 32), annotations=ann,
    ...     n_steps=100, sample_steps=10, plate=False,
    ... )  # doctest: +SKIP
    >>> c = torch.tensor([[3, 1]])
    >>> out = model(query=model.default_query(c),
    ...             input=torch.rand(1, 3, 32, 32))  # doctest: +SKIP

    See Also
    --------
    torch_concepts.nn.ConditionalVariationalAutoencoder : the other baseline
    torch_concepts.nn.CFGSamplingEngine : the reverse process
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})
    #: This model cannot reconstruct a *given* image. The autoencoders have a
    #: guide that encodes one; here there is no encoder at all, and recovering the
    #: noise behind a particular image means inverting the reverse process — a
    #: different procedure, and not the objective. The analysis reads this and
    #: leaves its reconstruction column blank rather than reporting a number that
    #: would not be comparable.
    reconstructs = False
    # Concept values are read by `y = m * c` and fed to the denoiser, so they must
    # be probabilities / one-hot rows rather than logits — the same convention
    # CVAE and CBGM use, which keeps the three models' loss configs interchangeable.
    param_for_discrete_var = "probs"

    variable_distributions = {
        'binary': Bernoulli,
        'categorical': OneHotCategorical,
        'continuous': Normal,
    }
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

    def __init__(
        self,
        input_size,
        annotations: Annotations,
        unet: Optional[nn.Module] = None,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        embedding_size: int = 16,
        p_uncond: float = 0.1,
        hidden_channels: Sequence[int] = (64, 128),
        guidance_scale: float = 3.0,
        sample_steps: Optional[int] = None,
        eta: float = 0.0,
        clip: Optional[Tuple[float, float]] = None,
        inference: Optional[BaseInference] = CFGTrainingEngine,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = CFGTrainingEngine,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        plate: Optional[bool] = None,
        **kwargs,
    ):
        # The image is flat everywhere in the graph (see the module docstring), and
        # the flat width is also the `latent_size` BaseModel wants: with no
        # backbone the latent *is* the raw input, which for this model is literal
        # — the U-Net reads pixels.
        shape = (input_size,) if isinstance(input_size, int) else tuple(input_size)
        image_size = int(math.prod(shape))
        kwargs.setdefault("latent_size", image_size)

        backbone = kwargs.get("backbone")
        if backbone is not None and not isinstance(backbone, nn.Identity):
            raise ValueError(
                f"{type(self).__name__} takes no backbone: the U-Net denoises raw "
                "pixels, so a feature extractor would sit in the optimiser and the "
                "checkpoint without ever running. Set `dataset.backbone=null` and "
                "`model.latent_encoder=null` (and keep `precompute_embeddings` "
                "false — precomputed features would replace the pixels this model "
                "generates)."
            )

        super().__init__(
            input_size=input_size,
            annotations=annotations,
            lightning=lightning,
            plate=plate,
            **kwargs,
        )
        if not 0.0 <= p_uncond < 1.0:
            raise ValueError(
                f"{type(self).__name__}: p_uncond must be in [0, 1), got {p_uncond}. "
                "At 1.0 the condition is dropped on every row and the model is "
                "unconditional."
            )
        self.embedding_size = embedding_size
        self.p_uncond = float(p_uncond)
        self.schedule = DiffusionSchedule(n_steps, beta_start, beta_end)
        self.image_size = image_size
        self.observation_shape = shape
        # Raw concept width — what `y` carries — against the *embedded* width the
        # U-Net reads. They differ: a 10-way categorical is 10 raw columns and
        # `embedding_size` embedded ones.
        self.concept_width = sum(
            self.concept_annotations.concept(n).cardinality for n in self.concept_names
        )
        self.condition_size = len(self.concept_names) * embedding_size

        if unet is None:
            if len(shape) != 3:
                raise ValueError(
                    f"{type(self).__name__}: the default ConditionalUNet needs "
                    f"input_size as (channels, height, width), got {input_size}. "
                    "Pass `unet=` for a flat or otherwise-shaped observation."
                )
            unet = ConditionalUNet(
                shape=shape,
                condition_size=self.condition_size,
                hidden_channels=hidden_channels,
            )
        self.unet = unet

        self.pgm = self._build_model()

        # Train AND eval run the *objective* engine — one denoising pass — so
        # that validation measures the same thing training minimises. Generation
        # is not an evaluation of the objective, it is a different computation
        # entirely, so the sampler is a separate attribute rather than
        # `eval_inference`. Wiring it as the eval engine instead would leave
        # `val_loss` uncomputable: the sampler reports images, and the loss needs
        # `eps_hat` and `eps_target`.
        self.setup_inference(
            inference,
            dict(inference_kwargs or {}),
            train_inference,
            dict(train_inference_kwargs or {}),
        )
        #: The reverse process. Held here rather than as an inference engine —
        #: see above — and reached through :meth:`sample`.
        self.sampler = CFGSamplingEngine(
            self.pgm,
            schedule=self.schedule,
            guidance_scale=guidance_scale,
            n_steps=sample_steps,
            eta=eta,
            clip=clip,
        )

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def _resolve_graph(self) -> ConceptGraph:
        """The edgeless concept graph: the condition's components are independent.

        Nothing in a diffusion model relates one concept to another, and the
        marginal ``p(c)`` that stands in for their joint is a product of
        per-concept factors — as in the CVAE.
        """
        labels = list(self.concept_names)
        return ConceptGraph(torch.zeros(len(labels), len(labels)), node_names=labels)

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def default_query(self, c):
        """Query the concepts and the two nodes the denoising loss compares.

        Narrow on purpose. ``input`` is evidence (clamped, so it emits no
        parameters) and the intermediate nodes are ancestors that get computed
        whether or not they are reported — asking for them would only widen the
        assembled output tensors. The concepts are here because the concept loss
        fits their marginals; they carry ground-truth values because they are
        *evidence*, the condition the denoiser reads.
        """
        return {
            "eps_hat": None,
            "eps_target": None,
            **self.fully_observed_query(c),
        }

    def flatten_observation(self, x: torch.Tensor) -> torch.Tensor:
        """Fold an image's ``(C, H, W)`` trailing dims into the flat event axis.

        ``input`` is declared flat, so ``(B, 3, 32, 32)`` from a dataloader is a
        rank the engine will not accept — it takes ``(*leading, 3072)`` or the
        declared event shape, and ``(B, 3, 32, 32)`` is neither. Rather than
        declare the variable image-shaped (which breaks parent concatenation, see
        the module docstring), the model flattens at its own boundary. Already
        flat input passes through.
        """
        tail = len(self.observation_shape)
        if tail > 1 and tuple(x.shape[-tail:]) == self.observation_shape:
            return x.reshape(*x.shape[:-tail], self.image_size)
        return x

    def forward(self, query, evidence=None, input=None, **inference_kwargs):
        """As :meth:`BaseModel.forward`, flattening the observation first.

        Both entry points are covered: ``input=`` (the direct call) and an
        ``input`` key in ``evidence`` (what the Lightning step passes).
        """
        if input is not None:
            input = self.flatten_observation(input)
        if evidence is not None and evidence.get("input") is not None:
            evidence = {**evidence, "input": self.flatten_observation(evidence["input"])}
        return super().forward(
            query=query, evidence=evidence, input=input, **inference_kwargs
        )

    def sample(self, query=("input",), evidence=None, n_samples=None, **kwargs):
        """Generate by running the reverse process.

        Distinct from :meth:`forward`, which runs the *objective* — one denoising
        pass at a random timestep. That is what training and validation need and
        it is not a generation; this is.

        Parameters
        ----------
        query : sequence of str
            Variables to report. ``input`` is the generated observation;
            ``eps`` and the concepts are worth asking for when the draw is to be
            replayed (an intervention holding everything but one concept).
        evidence : dict, optional
            Concepts to condition on, and/or ``eps`` to seed the process with.
            Anything absent is drawn.
        n_samples : int, optional
            Batch size for an unconditional draw, where no tensor fixes one.
        """
        return self.sampler.query(
            query=list(query), evidence=evidence or {}, n_samples=n_samples, **kwargs
        )

    @property
    def latent_variable_name(self) -> str:
        """The variable a generation is seeded from — read by the analysis harness.

        For a VAE this is ``z``; here it is the noise the reverse process starts
        at. Replaying it with one concept changed is what the steerability metric
        does, so it has to be nameable from outside.
        """
        return "eps"

    # ------------------------------------------------------------------
    # Model assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _prior_heads(variable) -> dict:
        """``first``/``second`` :class:`LearnablePrior` heads for a marginal ``p(v)``.

        Sized from ``param_sizes`` rather than ``size``: a ``MultivariateNormal``
        needs the free entries of a Cholesky factor, not one scalar per element.
        """
        sizes = variable.param_sizes
        if "loc" not in sizes:
            return {"first": LearnablePrior(variable.size), "second": None}
        scale_param = next(param for param in sizes if param != "loc")
        return {
            "first": LearnablePrior(sizes["loc"]),
            "second": LearnablePrior(sizes[scale_param]),
        }

    def _build_model(self) -> BayesianNetwork:
        """Assemble the diffusion Bayesian network (see the module docstring)."""
        image = self.image_size

        # --- variables ---
        # `input` is Delta: the observation is a value the graph reads, never a
        # likelihood it scores. The denoising objective lives on `eps_hat`.
        observed = EmbeddingVariable("input", distribution=Delta, size=image)
        noise = EmbeddingVariable("eps", distribution=Normal, size=image)
        timestep = EmbeddingVariable("t", distribution=Uniform, size=1)
        mask_family = _straight_through_bernoulli()
        mask = EmbeddingVariable(
            "m", distribution=mask_family, size=1,
            dist_kwargs=dict(DEFAULT_DIST_KWARGS.get(mask_family, {})),
        )
        state = EmbeddingVariable("x_t", distribution=Delta, size=image)
        condition = EmbeddingVariable("y", distribution=Delta, size=self.concept_width)
        prediction = EmbeddingVariable("eps_hat", distribution=Delta, size=image)
        target = EmbeddingVariable("eps_target", distribution=Delta, size=image)
        concepts = self.build_concept_variables(self.concept_names, plate_name="concepts")

        self.condition_embedding = ConceptEmbedding(
            sizes=[cvar.member_size for cvar in concepts for _ in cvar.members],
            embedding_size=self.embedding_size,
        )

        # --- roots ---
        # `input` never runs: it is always evidence, which clamps the variable and
        # skips its CPD. It exists because a BayesianNetwork requires exactly one
        # factor per variable.
        observed_cpd = ParametricCPD(
            observed, parents=[],
            parametrization={"value": FixedPrior(torch.zeros(image))},
        )
        noise_cpd = ParametricCPD(
            noise, parents=[],
            parametrization={
                "loc": FixedPrior(torch.zeros(image)),
                "scale": FixedPrior(torch.ones(image)),
            },
        )
        # t ~ Uniform(0, T), floored to an integer index by the schedule. A width-1
        # continuous draw rather than a T-way one-hot, which would be 1000 columns
        # the U-Net immediately argmaxes back.
        timestep_cpd = ParametricCPD(
            timestep, parents=[],
            parametrization={
                "low": FixedPrior(torch.zeros(1)),
                "high": FixedPrior(torch.full((1,), float(self.schedule.n_steps))),
            },
        )
        # m ~ Bernoulli(1 - p_uncond): 1 KEEPS the condition, matching `y = m * c`.
        mask_cpd = ParametricCPD(
            mask, parents=[],
            parametrization={
                "probs": FixedPrior(torch.full((1,), 1.0 - self.p_uncond))
            },
        )
        # p(c): one parent-less parameter per concept, activated into its own
        # domain. Unused while the concepts are observed; fitted by the concept
        # loss so an *unconditional* draw produces a plausible condition.
        concept_cpds = [
            ParametricCPD(
                variable=cvar, parents=[],
                parametrization=self._flexible_parametrization(
                    variable=cvar, **self._prior_heads(cvar),
                ),
            )
            for cvar in concepts
        ]

        # --- deterministic nodes ---
        # Parent order fixes the concatenation order every head splits on.
        state_cpd = ParametricCPD(
            state, parents=[observed, noise, timestep],
            parametrization=self._flexible_parametrization(
                variable=state, first=NoiseState(self.schedule, image),
            ),
        )
        target_cpd = ParametricCPD(
            target, parents=[noise, timestep],
            parametrization=self._flexible_parametrization(
                variable=target, first=NoiseTarget(image),
            ),
        )
        # `m` is an embedding and the concepts are concepts, so they reach the head
        # already split by type — hence MaskedCondition's (concepts, embeddings).
        condition_cpd = ParametricCPD(
            condition, parents=[mask, *concepts],
            parametrization=self._flexible_parametrization(
                variable=condition, first=MaskedCondition(),
            ),
        )
        prediction_cpd = ParametricCPD(
            prediction, parents=[state, timestep, condition],
            parametrization=self._flexible_parametrization(
                variable=prediction,
                first=DenoisingHead(
                    self.unet, self.condition_embedding, image, self.concept_width,
                ),
            ),
        )

        return BayesianNetwork(
            variables=[
                observed, noise, timestep, mask, *concepts,
                state, condition, prediction, target,
            ],
            factors=[
                observed_cpd, noise_cpd, timestep_cpd, mask_cpd, *concept_cpds,
                state_cpd, condition_cpd, prediction_cpd, target_cpd,
            ],
        )
