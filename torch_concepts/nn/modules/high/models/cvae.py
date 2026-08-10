"""Conditional Variational Autoencoder (CVAE), conditioned on a set of concepts.

The reference CVAE (Sohn et al., NeurIPS 2015) conditions a VAE on a *single*
observed variable ``y``: ``q(z | x, y)`` encodes, ``p(x | z, y)`` decodes, and
changing ``y`` at decoding time steers the generation. This model is that one
with the condition replaced by the framework's concept set — an arbitrary mix of
binary, categorical and continuous concepts, of any cardinality, laid out by the
usual :class:`~torch_concepts.annotations.Annotations`.

It exists as the **baseline** for the concept-based generative models: where
:class:`~torch_concepts.nn.ConceptBottleneckGenerativeModel` *generates* its
concepts from ``z`` and routes the observation through them
(``z → concepts → input``), the CVAE takes its concepts as given and feeds them
to the decoder alongside ``z``::

    CBGM   z ─→ concepts ─→ input          concepts are predicted, then decoded
    CVAE   z ─┐                            concepts are given, never inferred
              ├─→ input
    concepts ─┘

Both are therefore driveable with the same ``do(concept)`` intervention and
measurable with the same steerability / FID machinery — the difference being
that the CVAE has no concept bottleneck to lose information in, and no concept
*predictor* at all.

Three things follow from "the concepts are the condition", and they are the
whole design:

1. **The concepts are roots of the graph, with a learnable marginal ``p(c)``.**
   A CVAE proper is only ever asked for ``p(x | z, c)``, but a generative model
   that cannot be sampled *unconditionally* cannot be compared to one that can
   (FID needs draws, not reconstructions). The marginal is what makes the model
   a full joint ``p(c) p(z) p(x | z, c)``; a
   :class:`~torch_concepts.nn.ConceptLoss` term fits it to the empirical concept
   frequencies. It is deliberately factorised — sampling it can produce concept
   combinations the data never shows.

2. **Each concept is embedded on its own before the decoder sees it**
   (:class:`ConceptEmbedding`): ``k`` independent ``Linear(size_i,
   embedding_size)`` layers, one per concept, their outputs concatenated. Fed
   raw, a binary concept would contribute one column against a 100-way
   categorical's hundred, and the decoder's first layer would give them fan-in
   in that ratio; embedding them to a common width lets it weigh them alike.
   The total is ``n_concepts * embedding_size``, known from the annotations
   before construction (the decoder has to be sized for
   ``latent_size + condition_size``) and available afterwards as
   :attr:`condition_size`. The guide reads the condition through the **same**
   embedding layers, so ``c`` has one learned representation, as in the
   reference's shared ``label_embedding``.

3. **The concepts are observed**: the decoder is handed the condition, it never
   predicts it. There is consequently no RandInt knob — the CVAE's decoder
   always reads a value it did not produce, which is exactly what an
   intervention gives it, so it is steerable by construction.

Deviations from the paper, both in the direction of the simpler formulation the
practitioner's version uses (see the Towards Data Science write-up of the same
model):

* The prior is ``p(z) = N(0, I)``, not the paper's conditional prior
  ``p(z | c)``. A learned conditional prior removes the fixed target the KL
  pulls ``q`` toward and makes an unconditional draw of ``z`` require ``c``
  first; the gain — letting the latent distribution shift per condition — is not
  what this baseline is measuring.
* No GSNN / hybrid objective (paper Sec. 4.2): those address the train/test
  mismatch of a *predictive* CVAE, whose condition is an input image. Here the
  condition is a low-dimensional concept vector supplied identically at training
  and at generation, so the mismatch does not arise.

References
----------
Sohn, Yan, Lee. "Learning Structured Output Representation using Deep
Conditional Generative Models", NeurIPS 2015.
"""
import copy
from typing import List, Optional, Type

import torch
import torch.nn as nn

from torch.distributions import Bernoulli, Normal, OneHotCategorical

import torch_concepts as pyc
from .....annotations import Annotations
from .....concept_graph import ConceptGraph
from ...low.priors import FixedPrior, LearnablePrior
from ...low.scales import GlobalScale
from ...mid.inference.base import BaseInference
from ...mid.inference.pyro.variational import VariationalInference
from ...mid.graph.bayesian_network import BayesianNetwork
from ...mid.factors.cpd import ParametricCPD
from ...mid.variable import EmbeddingVariable
from ...mid.distributions import DEFAULT_DIST_KWARGS
from ..base.graph import DirectedGraphModel


class ConceptEmbedding(nn.Module):
    """One learnable embedding per concept: ``k`` separate ``Linear(size_i, m)``.

    Takes the concepts as they arrive from the graph — concatenated on the last
    axis, a Bernoulli's bit next to a categorical's one-hot row next to a
    continuous value — splits them back apart, and gives each its own linear map
    to a common width ``m``. No layer ever sees more than one concept.

    The point is fan-in. Passed raw to the decoder, a binary concept is one input
    column and a 100-way categorical is a hundred, so the first layer weighs them
    in that ratio before it has learned anything. At a common width the decoder
    can pay them equal attention.

    Parameters
    ----------
    sizes : list of int
        Width of each concept's value, in the order they are concatenated —
        ``1`` for a binary or continuous concept, ``cardinality`` for a
        categorical one. Per *member*, so a plate contributes one entry per
        member rather than one for the whole plate.
    embedding_size : int
        Width ``m`` of every concept's embedding.

    Examples
    --------
    >>> import torch
    >>> from torch_concepts.nn.modules.high.models.cvae import ConceptEmbedding
    >>> # a binary concept, a 10-way categorical, a continuous one
    >>> embedder = ConceptEmbedding(sizes=[1, 10, 1], embedding_size=8)
    >>> len(embedder.embeddings)  # one Linear per concept, not one for all
    3
    >>> embedder(torch.zeros(4, 12)).shape
    torch.Size([4, 24])
    """

    def __init__(self, sizes: List[int], embedding_size: int):
        super().__init__()
        self.sizes = [int(s) for s in sizes]
        self.embeddings = nn.ModuleList(
            nn.Linear(size, embedding_size) for size in self.sizes
        )
        self.out_features = len(self.sizes) * embedding_size

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        weight = self.embeddings[0].weight
        parts = torch.split(concepts.to(weight.dtype), self.sizes, dim=-1)
        return torch.cat(
            [embed(part) for embed, part in zip(self.embeddings, parts)], dim=-1
        )


class ConditionedInput(nn.Module):
    """Join a feature vector with the embedded condition.

    Used on both sides of the model, which is what lets the guide and the decoder
    share one :class:`ConceptEmbedding`:

    * the guide's trunk, with ``encoder`` set — ``q(z | input, concepts)`` reads
      the image through the backbone/encoder, and the concepts join *after* it
      (a raw image and a concept vector cannot be concatenated, and a
      convolutional encoder could not consume the result if they could);
    * the decoder's head, with ``encoder`` ``None`` — the feature vector is then
      ``z`` itself, and the output is the ``[z | embedded c]`` the decoder reads.

    Named arguments, not one tensor: the ``concepts`` / ``embeddings`` signature
    is the PyC calling convention, so
    :class:`~torch_concepts.nn.ParametricCPD` splits the parents by type and
    hands each group over already concatenated. ``concepts`` is absent — hence
    optional — when the guide is built unconditioned.
    """

    def __init__(self, embedder: ConceptEmbedding, encoder: Optional[nn.Module] = None):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder

    def forward(self, embeddings, concepts=None):
        features = embeddings if self.encoder is None else self.encoder(embeddings)
        if concepts is None:
            return features
        return torch.cat([features, self.embedder(concepts)], dim=-1)


class ConditionalVariationalAutoencoder(DirectedGraphModel):
    """Conditional VAE whose condition is the concept set.

    Generative process ``p(c) p(z) p(input | z, c)``, trained as a VAE through a
    variational guide ``q(z | input, c)``. Intervening on a concept changes the
    decoder's input directly, with no bottleneck in between.

    Parameters
    ----------
    input_size : int or tuple
        Shape of one observation (the generated variable).
    annotations : Annotations
        Concept annotations (labels, cardinalities, types). Every concept is a
        conditioning variable; there are no task variables.
    encoder : nn.Module
        The guide's feature extractor, mapping an observation to a vector. It
        runs after ``backbone`` and must declare ``out_features`` (an
        :class:`~torch_concepts.nn.MLP` or ``nn.Linear`` does); the embedded
        concepts are appended to its output and two linear readouts produce
        ``loc`` and ``scale``.
    decoder : nn.Module
        The generative network, mapping ``latent_size + condition_size`` values
        to ``input_size`` **raw** values. The model composes the observation
        parameter's activation on top — identity for a ``Normal``'s ``loc``, a
        sigmoid for a ``Bernoulli``'s ``probs`` — so a decoder that squashes its
        own output is activated twice.
    latent_size : int, default 64
        Dimensionality of ``z``.
    embedding_size : int, default 16
        Width ``m`` of one concept's embedding. Every concept gets its own
        ``Linear(size_i, m)`` (:class:`ConceptEmbedding`), so the condition the
        decoder reads is ``n_concepts * m`` wide however the concepts are
        distributed — one binary and one 100-way categorical contribute ``m``
        each, not 1 and 100.
    condition_encoder : bool, default True
        Whether the guide reads the concepts as well as the observation. ``True``
        is the paper's ``q(z | x, y)``: ``z`` then only has to carry what the
        condition does not, which is what stops the decoder learning to ignore
        ``c``. ``False`` gives ``q(z | input)``, which needs no concept values to
        encode with — the ablation, and the fallback if a caller wants to
        reconstruct an observation whose concepts are unknown.
    observation : type, default ``torch.distributions.Normal``
        Distribution family of the generated variable. ``Bernoulli`` for images
        in ``[0, 1]``; ``Normal`` needs a ``scale`` too — see ``global_scale``.
    global_scale : bool, default True
        Only consulted when ``observation`` has a ``scale``. ``True``: one
        learnable sigma shared by every pixel and sample
        (:class:`~torch_concepts.nn.GlobalScale`). ``False``: ``scale`` gets its
        own copy of ``decoder``.
    scale_init : float, default 1.0
        The ``global_scale`` standard deviation — its starting point when
        ``scale_learnable``, its fixed value otherwise. It sets the weight of the
        reconstruction term: the Gaussian NLL's gradient carries a
        ``1 / scale**2`` factor, so halving it quadruples reconstruction relative
        to the KL. At the default of ``1.0`` that factor is ``1`` and the
        Gaussian NLL is ``0.5 * (x - loc)**2`` plus a constant — i.e. training
        reduces to plain MSE on the predicted mean, and the KL weight is a true
        ``beta``.
    scale_learnable : bool, default False
        Whether ``global_scale`` is trained. Off by default so the likelihood
        stays the fixed-sigma one described above. A learned scale settles at the
        residual RMS, which shrinks as the fit improves and therefore keeps
        *raising* the effective reconstruction weight — annealing the KL away
        without that showing up in any loss weight.
    inference, inference_kwargs, train_inference, train_inference_kwargs
        Inference engine configuration. Defaults to
        :class:`~torch_concepts.nn.VariationalInference`, with the guide on
        ``z`` injected into ``inference_kwargs['latents']``.
    lightning : bool, default False
        If True, adds Lightning training capabilities.
    plate : bool or None, default None
        Per-level plate preference (see :class:`BaseModel`). ``None``/``True``
        group homogeneous concepts into the minimum number of plates; ``False``
        gives one variable per concept, which is what per-concept interventions
        (and the steerability metric) address.
    **kwargs
        Forwarded to :class:`BaseModel`.

    Attributes
    ----------
    condition_size : int
        Width of the condition the decoder reads,
        ``n_concepts * embedding_size``. Known from the annotations alone, so the
        decoder can be sized before construction.
    condition_embedding : ConceptEmbedding
        The per-concept embedding layers, shared by the decoder and the guide.

    Notes
    -----
    The concept loss trains the **marginal** ``p(c)``, not a concept predictor:
    this model reads its concepts and never infers them from the observation.
    Concept accuracy is therefore at the majority-class rate by construction, and
    is not a number to compare against a CBM/CBGM's. What *is* comparable is the
    steerability of the generations and their FID.

    Any registered distribution family works for a concept, via
    ``variable_distributions``: the plain discrete families (the defaults), their
    relaxed and straight-through variants — the latter making a *sampled* concept
    an exact bit / one-hot row rather than a soft Concrete draw — ``Normal``, and
    ``MultivariateNormal`` for a vector-valued continuous concept (which needs
    ``plate=False``, since a Cholesky factor is not a per-element parameter).

    Examples
    --------
    >>> import torch
    >>> from torch.distributions import Bernoulli
    >>> from torch_concepts.annotations import Annotations
    >>> from torch_concepts.nn import ConditionalVariationalAutoencoder, MLP
    >>>
    >>> ann = Annotations(labels=['digit', 'color'], cardinalities=[10, 2],
    ...                   types=['categorical', 'categorical'])
    >>> condition = len(ann.labels) * 8  # n_concepts * embedding_size
    >>> model = ConditionalVariationalAutoencoder(
    ...     input_size=784, annotations=ann,
    ...     encoder=MLP(784, 128, 32),
    ...     # Raw: the observation's activation (a sigmoid for a Bernoulli's
    ...     # `probs`) is composed on top for you.
    ...     decoder=MLP(32 + condition, 128, 784),
    ...     latent_size=32, embedding_size=8, observation=Bernoulli,
    ... )  # doctest: +SKIP
    >>> # Concepts are evidence, so the query supplies them.
    >>> c = torch.tensor([[3, 1]])
    >>> out = model(query=model.default_query(c), input=torch.rand(1, 784))  # doctest: +SKIP

    See Also
    --------
    torch_concepts.nn.ConceptBottleneckGenerativeModel : the model this baselines
    """

    supported_concept_types = frozenset({"binary", "categorical", "continuous"})
    # The concepts are conditioning values, and a value the decoder can read is a
    # probability (or a one-hot row), not a logit — the same convention CBGM uses,
    # which also keeps the two models' loss configs interchangeable.
    param_for_discrete_var = "probs"

    variable_distributions = {
        'binary': Bernoulli,
        'categorical': OneHotCategorical,
        'continuous': Normal,
    }
    variable_dist_kwargs = dict(DEFAULT_DIST_KWARGS)

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        latent_size: int = 64,
        embedding_size: int = 16,
        condition_encoder: bool = True,
        observation: Type = Normal,
        global_scale: bool = True,
        scale_init: float = 1.0,
        scale_learnable: bool = False,
        inference: Optional[BaseInference] = VariationalInference,
        inference_kwargs: Optional[dict] = None,
        train_inference: Optional[BaseInference] = None,
        train_inference_kwargs: Optional[dict] = None,
        lightning: bool = False,
        plate: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            latent_size=latent_size,
            lightning=lightning,
            plate=plate,
            **kwargs,
        )
        self.embedding_size = embedding_size
        self.condition_encoder = bool(condition_encoder)
        self.observation = observation
        self.global_scale = global_scale
        self.scale_init = scale_init
        self.scale_learnable = scale_learnable
        self.encoder = encoder if encoder is not None else nn.Identity()
        self.decoder = decoder if decoder is not None else nn.Identity()
        # Read off the annotations, not off the built variables: a caller has to
        # size the decoder *before* constructing the model, so the two must agree
        # on one definition of the condition's width.
        self.condition_size = len(self.concept_names) * embedding_size

        self.pgm = self._build_model()

        guide = {"latents": {"z": self._build_guide()}}
        self.setup_inference(
            inference,
            {**guide, **(inference_kwargs or {})},
            train_inference,
            {**guide, **(train_inference_kwargs or {})},
        )

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    def _resolve_graph(self) -> ConceptGraph:
        """Build the edgeless concept graph.

        The condition's components are modelled as mutually independent: nothing
        in a CVAE relates one concept to another, and the marginal ``p(c)`` that
        stands in for their joint is a product of per-concept factors.
        """
        labels = list(self.concept_names)
        return ConceptGraph(
            torch.zeros(len(labels), len(labels)),
            node_names=labels,
        )

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def default_query(self, c):
        """Query **every** variable, supplying the concepts' ground truth.

        :class:`~torch_concepts.nn.VariationalInference` requires all variables
        in the query — observed ones with values, latents absent or ``None`` —
        and the generative loss terms need the ones the base learner's
        concept-only query would leave out (``input``, for the reconstruction).

        The concepts carry values because they are *evidence*: they are the
        condition the decoder reads and, with ``condition_encoder``, part of what
        the guide encodes. Unlike a CBGM's query this is not teacher forcing —
        there is no prediction being overridden.
        """
        return {
            **{name: None for name in self.pgm.variables},
            **self.fully_observed_query(c),
        }

    # ------------------------------------------------------------------
    # Model assembly
    # ------------------------------------------------------------------
    def _concept_variables(self) -> List:
        """The condition's variables, in the order they are concatenated."""
        return [v for v in self.pgm.variables.values() if v.variable_type == "concept"]

    @staticmethod
    def _prior_heads(variable) -> dict:
        """``first``/``second`` :class:`LearnablePrior` heads for a marginal ``p(v)``.

        Sized from the variable's own ``param_sizes`` rather than from its
        ``size``, because the two differ: a ``MultivariateNormal``'s
        ``scale_tril`` needs the ``size * (size + 1) // 2`` free entries of a
        Cholesky factor, not ``size``. Everything else here is one scalar per
        event element, so this reduces to ``size`` for the discrete families, a
        ``Delta``'s ``value`` and a ``Normal``'s ``loc``/``scale``.
        """
        sizes = variable.param_sizes
        if "loc" not in sizes:
            # Discrete (probs/logits) or Delta (value) — a single head, and every
            # candidate parameter has the same width.
            return {"first": LearnablePrior(variable.size), "second": None}
        scale_param = next(param for param in sizes if param != "loc")
        return {
            "first": LearnablePrior(sizes["loc"]),
            "second": LearnablePrior(sizes[scale_param]),
        }

    def _build_guide(self) -> ParametricCPD:
        """The variational posterior ``q(z | input, c)``, a Normal CPD on ``z``.

        The feature extractor plus the embedded concepts is the CPD's **trunk**,
        not part of either parameter's head: ``loc`` and ``scale`` are two small
        linear readouts of the same features, so the backbone runs once per step.
        Sharing is safe here because both heads are independently learnable
        ``Linear`` layers — put the *scoring* layer in a trunk instead and the
        scale would collapse to a fixed function of the location.

        The embedding layers are :attr:`condition_embedding` itself, not a copy:
        the guide and the decoder read ``c`` through one learned representation.
        """
        z = self.pgm.variables["z"]
        observed = self.pgm.variables["input"]

        # Width of the trunk's output: the encoder's if it declares one (MLP,
        # nn.Linear), else the backbone's — `encoder` defaults to nn.Identity.
        width = (getattr(self.encoder, "out_features", None)
                 or getattr(self.backbone, "out_features", None))
        if width is None:
            raise ValueError(
                f"{type(self).__name__}: cannot size the guide's readout — neither "
                "`encoder` nor `backbone` declares `out_features`. Set that attribute "
                "on one of them, or pass an encoder that does (e.g. MLP, nn.Linear)."
            )
        conditioning = self._concept_variables() if self.condition_encoder else []
        width = int(width) + (self.condition_size if self.condition_encoder else 0)
        return ParametricCPD(
            variable=z,
            parents=[observed, *conditioning],
            trunk=ConditionedInput(
                self.condition_embedding,
                encoder=nn.Sequential(self.backbone, self.encoder),
            ),
            parametrization=self._flexible_parametrization(
                variable=z,
                first=nn.Linear(width, z.size),
                second=nn.Linear(width, z.size),
            ),
        )

    def _build_model(self) -> BayesianNetwork:
        """Assemble the CVAE Bayesian network.

        ``{z, concepts} → input``: a standard Normal prior on ``z``, one
        learnable marginal per concept (group), and a decoder reading ``z``
        concatenated with the *embedded* concepts. The decoder's input is
        therefore ``latent_size + condition_size`` wide, ``z`` first, then one
        ``embedding_size``-wide block per concept in annotation order.
        """
        # --- variables ---
        observed = EmbeddingVariable("input", distribution=self.observation, shape=self.input_size)
        latent = EmbeddingVariable("z", distribution=Normal, size=self.latent_size)
        concepts = self.build_concept_variables(self.concept_names, plate_name="concepts")

        # One embedding per concept, sized to that concept's own value: 1 column
        # for a binary or continuous one, `cardinality` for a categorical one.
        # Widths are taken per MEMBER so a plate contributes one entry per member,
        # matching how the parent values arrive concatenated.
        self.condition_embedding = ConceptEmbedding(
            sizes=[cvar.member_size for cvar in concepts for _ in cvar.members],
            embedding_size=self.embedding_size,
        )

        # --- factors ---
        # p(z) = N(0, I): fixed, not learned, so the guide has a fixed target.
        latent_cpd = ParametricCPD(
            latent,
            parents=[],
            parametrization={
                "loc": FixedPrior(torch.zeros(self.latent_size)),
                "scale": FixedPrior(torch.ones(self.latent_size)),
            },
        )
        # p(c): one parent-less parameter per concept (per member, for a plate),
        # activated into its own domain — a sigmoid for a Bernoulli, a per-member
        # softmax for a categorical, softplus on a continuous concept's scale.
        # Unused while the concepts are observed; fitted by the concept loss so
        # that an *unconditional* draw produces a plausible condition.
        concept_cpds = [
            ParametricCPD(
                variable=cvar,
                parents=[],
                parametrization=self._flexible_parametrization(
                    variable=cvar,
                    **self._prior_heads(cvar),
                ),
            )
            for cvar in concepts
        ]

        # The parents reach the head as `z` (an embedding) and the raw concepts,
        # split by type — so the head embeds the concepts and concatenates,
        # producing the [z | embedded c] layout `condition_size` documents.
        def decoder_head(decoder: nn.Module) -> nn.Module:
            return pyc.nn.Sequential(
                ConditionedInput(self.condition_embedding), decoder
            )

        # `loc` always comes from the decoder. A `scale` — only allocated when the
        # family has one — is either one sigma shared by every pixel (the default,
        # fixed at `scale_init`) or its OWN copy of the decoder: sharing a trunk
        # with `loc` would make the spread a fixed function of the mean.
        if "loc" not in observed.param_sizes:
            scale_head = None
        elif self.global_scale:
            scale_head = GlobalScale(observed.size, init=self.scale_init,
                                     learnable=self.scale_learnable)
        else:
            # Only the decoder is copied. Deep-copying the wrapper would clone the
            # embedding layers too and silently unshare them from the guide.
            scale_head = decoder_head(copy.deepcopy(self.decoder))
        decoder_cpd = ParametricCPD(
            variable=observed,
            parents=[latent, *concepts],
            parametrization=self._flexible_parametrization(
                variable=observed,
                first=decoder_head(self.decoder),
                second=scale_head,
            ),
        )

        return BayesianNetwork(
            variables=[latent, *concepts, observed],
            factors=[latent_cpd, *concept_cpds, decoder_cpd],
        )
