"""Concept MRF: an energy-based concept model with loopy belief propagation.

Problem setting
---------------
A high-dimensional continuous ``x`` carries information about a set of discrete
concepts ``c_i``, and the ``c_i`` are related to each other through a known
graph. This script builds an energy-based model with

* a **unary** energy ``E(x, c_i)`` per concept — ``x`` is observed at train and
  test time, so these are conditional unary potentials (the CRF case), and
* a **clique** energy ``E(c_l)`` for every maximal clique of the **moralized**
  concept graph.

Inference is differentiable loopy belief propagation, used identically during
training and at test time.

What the figure shows
---------------------
One row of subplots, one per concept (the task node included). x axis: the
probability that the *other* concepts are swapped to their ground-truth values
(0.0 to 1.0, step 0.1). y axis: that concept's accuracy. One line per model:

* ``CBM``       — every concept is predicted from the latent alone, so its curve
                  is flat. Only the task node rises.
* ``GraphCBM``  — a node is predicted from its DAG parents, so root nodes stay
                  flat and downstream nodes rise. Information flows one way.
* ``AR CBM``    — the autoregressive concept predictor of Havasi et al. (NeurIPS
                  2022, §4.2), ``p(c_k | x, c_1..c_{k-1})``, with their
                  importance-sampling intervention scheme (Eq 8-9). Its weights
                  update concepts *earlier* in the order too, so it rises
                  everywhere. The baseline actually worth beating.
* ``ConceptMRF``— belief propagation moves evidence in *every* direction, so
                  every node with a neighbour rises, roots included.
* ``MRF (cliques only)`` — the trained MRF with its unary energies zeroed: what
                  the concept structure contributes with ``x`` removed.

The contrast the experiment is built on: clamping a concept and re-running is
**do**-semantics for the first two models — descendants update, ancestors never
do — while ``AR CBM`` and ``ConceptMRF`` both perform real conditioning. They
differ in how: reweighted samples versus exact message passing, which is why the
AR effective sample size is reported alongside the curves.

The two directed baselines use straight-through (hard) concept samples. With the
plain ``Bernoulli``/``OneHotCategorical`` default they would propagate *soft*
Concrete values, making them soft CBMs in the paper's sense and confounding
"autoregressive" with "hard".

Runtime
-------
Measured on CPU at the defaults (100 epochs, 3 repeats): ``asia`` ~3 min,
``sachs`` ~6 min. ``insurance`` has 27 concepts, which makes both the clique
tables and the leave-one-out sweep several times larger — budget ~30-60 min on
CPU. Belief propagation dominates: the two directed baselines train in seconds.

The device is picked automatically (CUDA when available); override with
``--device`` or the ``CONCEPT_MRF_DEVICE`` environment variable. On a GPU, raise
``--eval-batch`` — the sweep is many short inference passes over the test split,
so it is the part that benefits most. Lower it if a large ``insurance`` clique
runs out of memory: BP materialises a ``(clique states, batch, width)`` tensor
per clique member.

Usage
-----
    python examples/experiments/concept_mrf_intervention.py                  # all three
    python examples/experiments/concept_mrf_intervention.py --datasets asia
    python examples/experiments/concept_mrf_intervention.py --device cuda --eval-batch 4096
    python examples/experiments/concept_mrf_intervention.py --ar-samples 50   # faster sweep
    python examples/experiments/concept_mrf_intervention.py --epochs 5 --repeats 1  # smoke test
"""

import argparse
import copy
import math
import os
import time
from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, OneHotCategorical
from pyro.distributions import (
    RelaxedBernoulliStraightThrough,
    RelaxedOneHotCategoricalStraightThrough,
)

from torch_concepts import seed_everything, ConceptVariable, EmbeddingVariable
from torch_concepts.data import BnLearnDataModule
from torch_concepts.nn import (
    AncestralSamplingInference,
    BeliefPropagation,
    ConceptBottleneckModel,
    GraphConceptBottleneckModel,
    MarkovNetwork,
    MLP,
    ParametricPotential,
)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device(
    os.environ.get("CONCEPT_MRF_DEVICE")
    or ("cuda" if torch.cuda.is_available() else "cpu")
)

DATASETS = ("asia", "sachs", "insurance")

#: Task node per dataset, taken from ``conceptarium/conf/dataset/dag_*.yaml``
#: (``default_task_names``). Only the CBM needs it — it is the one bipartite
#: model here; GraphCBM and the MRF treat every node uniformly.
TASKS = {"asia": ["dysp"], "sachs": ["Akt"], "insurance": ["PropCost"]}

N_GEN = 5000
#: ``x`` is an autoencoder embedding of the concept values themselves, and this
#: is the ONLY thing that makes it a lossy view of them:
#: ``x = (1-noise)*encoded + noise*randn``. At noise 0 an MLP recovers every
#: concept perfectly, all three models saturate, and every curve is flat — the
#: graph would have nothing left to contribute. At 0.9 the opposite happens: the
#: unary signal collapses to the class prior, so every model starts from the
#: majority class and only the balanced concepts have room to move. 0.5 leaves
#: the unaided predictions well above the prior *and* below the ceiling, which is
#: the regime where the concept-concept structure is what closes the gap.
NOISE = 0.5
AE_KWARGS = {
    "noise": NOISE,
    "latent_dim": 32,
    "lr": 5e-4,
    "epochs": 2000,
    "batch_size": 512,
    "patience": 50,
    # Pinned to CPU on purpose. This autoencoder runs once, at dataset build
    # time, and its output is `torch.save`d to the cache the DataLoader then
    # reloads. Left on CUDA it would write CUDA tensors into that cache, which
    # is neither portable nor usable with `workers > 0`. It is a 2-layer MLP
    # over a tabular frame, so there is nothing to gain on a GPU anyway.
    "device": "cpu",
}

BATCH = 256
#: Belief propagation materialises a ``(grid, batch, width)`` tensor per clique
#: member, where ``grid`` is the clique's full state count. Evaluation runs at a
#: smaller batch than training because the leave-one-out sweep pushes the whole
#: test split through many different evidence sets.
EVAL_BATCH = 512

HIDDEN, LATENT, N_LAYERS = 64, 32, 2
#: Hidden width of each clique potential's energy MLP. This is where the
#: concept-concept coupling lives, so it needs real capacity: a clique over
#: members with `prod(states)` joint configurations is being approximated by
#: this one hidden layer.
CLIQUE_HIDDEN = 64
EPOCHS, LR = 100, 1e-3

#: Hidden width of each autoregressive concept head. The paper uses 20 on
#: MIMIC-III and 50 on CUB; 32 matches the scale of everything else here.
AR_HIDDEN = 32
#: Monte-Carlo samples for the autoregressive model's intervention estimator
#: (Havasi et al. Eq 8-9). The paper uses M = 200. Lower it with
#: ``--ar-samples`` if the sweep is too slow; the printed effective sample size
#: is what says whether the budget is actually the binding constraint.
AR_SAMPLES = 200

#: The MRF's counterpart of the baselines' ``p_int=1`` teacher forcing. The
#: directed models are trained with their parents clamped to ground truth, which
#: is exactly what makes them responsive to interventions at test time. BP has no
#: teacher forcing (it passes messages, it never propagates a realised value), so
#: the equivalent is *conditional* training: clamp a random subset of concepts as
#: evidence and fit the marginals of the rest. The subset rate is drawn uniformly
#: each step, so training covers the whole x axis of the figure rather than one
#: point on it. Set to False to train on the fully-free marginals instead.
TRAIN_CONDITIONAL = True

#: The moral graph is genuinely loopy, so BP is an approximation and needs
#: damping to settle. These are the settings that converged for the ColorMNIST
#: MRF; the per-dataset report prints whether ``tol`` was actually reached.
BP_ITERS, BP_DAMPING, BP_TOL = 20, 0.5, 1e-6

#: Refuse to build a clique whose table exceeds this many cells. A large clique
#: is where BP's memory goes, and a silent OOM ten minutes into training is much
#: worse than an upfront error.
MAX_CLIQUE_CELLS = 4096

P_GRID = [round(0.1 * i, 1) for i in range(11)]
#: Monte-Carlo repeats over which subset of concepts gets intervened on.
#: p=0 (nothing intervened) and p=1 (everything else intervened) are
#: deterministic and skip the repeat loop.
REPEATS = 5

EPS = 1e-6

FIGDIR = Path(__file__).parent / "figures"
#: Models that are actually trained.
MODEL_ORDER = ("CBM", "GraphCBM", "AR CBM", "ConceptMRF")
#: Eval-time ablation of ``ConceptMRF``: same weights, unary energies switched
#: off, so only the clique potentials speak. Added after training.
ABLATION = "MRF (cliques only)"
PLOT_ORDER = (*MODEL_ORDER, ABLATION)
#: Row-label column width for the printed tables (the ablation has the longest name).
LABEL_W = max(len(t) for t in (*PLOT_ORDER, "majority class", "concepts only"))
MODEL_STYLE = {
    "CBM": dict(color="#888888", marker="o", ls="--"),
    "GraphCBM": dict(color="#1f77b4", marker="s", ls="-."),
    "AR CBM": dict(color="#9467bd", marker="D", ls="-"),
    "ConceptMRF": dict(color="#d62728", marker="^", ls="-"),
    ABLATION: dict(color="#2ca02c", marker="v", ls=":"),
}


def as_t(x):
    """``AnnotatedTensor`` wraps a tensor rather than subclassing it."""
    return getattr(x, "tensor", x)


# ---------------------------------------------------------------------------
# 1. data + moralization
# ---------------------------------------------------------------------------
def load_data(name: str):
    """Datamodule, concept metadata, and the maximal cliques of the moral graph.

    The autoencoder noise is not part of ``processed_filenames``, so it has to
    live in ``root`` or two noise levels would silently share one cache.
    """
    dm = BnLearnDataModule(
        seed=SEED,
        name=name,
        root=f"data/{name}_ae_noise_{NOISE}",
        generation_seed=42,
        n_gen=N_GEN,
        batch_size=BATCH,
        autoencoder_kwargs=AE_KWARGS,
    )
    dm.setup()

    ann = dm.annotations
    labels = list(ann.labels)
    # A binary concept is annotated with cardinality 1 (Bernoulli); BP encodes
    # it as a scalar 0./1. of width 1 and enumerates 2 states. A categorical of
    # cardinality K is a one-hot of width K with K states.
    width = {n: (1 if k == 1 else int(k)) for n, k in zip(labels, ann.cardinalities)}
    states = {n: (2 if k == 1 else int(k)) for n, k in zip(labels, ann.cardinalities)}

    # Moralization: marry the parents of every node, then drop directions. Each
    # family is then a clique, and the maximal cliques are the MRF's factors.
    moral = nx.moral_graph(dm.graph.to_networkx())
    cliques = sorted(
        (sorted(c) for c in nx.find_cliques(moral) if len(c) > 1),
        key=lambda c: (-len(c), c),
    )
    for clique in cliques:
        cells = math.prod(states[n] for n in clique)
        if cells > MAX_CLIQUE_CELLS:
            raise ValueError(
                f"clique {clique} has {cells} joint states (> MAX_CLIQUE_CELLS="
                f"{MAX_CLIQUE_CELLS}). Raise the cap, or switch to BN families "
                "(node + parents) instead of maximal cliques."
            )
    return dm, labels, width, states, cliques


def report_structure(name, dm, labels, states, cliques):
    print(f"\n{'=' * 72}\n{name}  —  {len(labels)} concepts, task {TASKS[name]}\n{'=' * 72}")
    print(f"concepts : {', '.join(f'{n}({states[n]})' for n in labels)}")
    print(f"DAG edges: {sorted(dm.graph.to_networkx().edges())}")
    total = sum(math.prod(states[n] for n in c) for c in cliques)
    # These counts are BP's enumeration grid — the number of joint states it
    # scores per clique per batch element — not a parameter count: the energies
    # are MLPs, so their size is set by CLIQUE_HIDDEN instead.
    print(f"moral-graph maximal cliques ({len(cliques)}, {total} enumerated states total):")
    for c in cliques:
        print(f"    {math.prod(states[n] for n in c):5d} states  {c}")
    print(f"AR order : {' -> '.join(dm.graph.topological_sort())}")


def gt_event(c: torch.Tensor, labels, width) -> Dict[str, torch.Tensor]:
    """Ground truth in each variable's own event parametrization.

    ``(B, 1)`` float 0/1 for a binary concept, ``(B, K)`` one-hot for a
    categorical one — the same layout ``BaseModel.fully_observed_query``
    produces. Written once here so the library models and the MRF provably feed
    on identical tensors, both for teacher forcing and for intervention clamps.
    """
    raw = as_t(c)
    out = {}
    for i, n in enumerate(labels):
        out[n] = (
            raw[..., i : i + 1].float()
            if width[n] == 1
            else F.one_hot(raw[..., i].long(), width[n]).float()
        )
    return out


# ---------------------------------------------------------------------------
# 2. models
# ---------------------------------------------------------------------------
class UnaryEnergy(nn.Module):
    """``E(c, z) = -(head(z) · c)`` — a CBM logit head read as an energy.

    The scope is ``[concept, latent]``, so the aggregated input is
    ``cat([c, z])``. The dot product handles both encodings BP uses without a
    branch: a binary concept is a scalar in {0, 1} at width 1, so ``head``
    emits one Bernoulli logit and ``-logit*c`` is the Bernoulli energy; a
    categorical is a one-hot of width K, so ``head`` emits K categorical logits
    and the product selects the active one.
    """

    def __init__(self, width: int, latent_size: int):
        super().__init__()
        self.width = width
        #: Set to 0.0 to switch this potential off, leaving a uniform factor that
        #: carries no information. Used by :class:`CliquesOnlyMRF` at eval time —
        #: a plain float, so it costs nothing and needs no device handling.
        self.scale = 1.0
        self.head = nn.Sequential(
            nn.Linear(latent_size, width),
            nn.LeakyReLU(),
            nn.Linear(width, width)
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        c, z = u[..., : self.width], u[..., self.width :]
        return -self.scale * (self.head(z) * c).sum(-1)


class CliqueEnergy(nn.Module):
    """``E(c_l) = mlp(concat of the members' one-hots)`` — the clique potential.

    The hidden layer is load-bearing, not decoration. A bare ``nn.Linear`` over
    concatenated one-hots is purely *additive* — ``w_a[i] + w_b[j]`` — so it
    expresses no coupling whatsoever and the clique would collapse into a sum of
    unaries, which is asserted by
    ``tests/.../test_potential.py::test_linear_energy_is_additive``. The
    nonlinearity is what makes this a genuine energy over the clique.

    The readout is initialised small (not zero), so the clique term starts
    essentially inert and the model begins life as an independent CBM — every
    departure from that baseline is learned coupling. It must not be *exactly*
    zero: the first layer's gradient is proportional to the readout weight, so a
    zero readout would leave the hidden layer with no gradient at all on the
    first step. The bias may be zero, since a constant shifts every state of the
    clique equally and cancels in the normalisation.

    (This replaces an exact ``prod(states)`` lookup table; an MLP approximates
    that table instead of parameterising it cell by cell, trading exactness for
    parameter sharing across states.)
    """

    def __init__(self, widths: Sequence[int], hidden: int = CLIQUE_HIDDEN):
        super().__init__()
        self.widths = list(widths)
        # A binary member arrives from BP as a scalar 0./1.; widening it to
        # [1-c, c] gives every member a one-hot block, so the first layer holds
        # one weight per state instead of folding state 0 into the bias.
        in_features = sum(2 if w == 1 else w for w in self.widths)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        parts, off = [], 0
        for w in self.widths:
            p = u[..., off : off + w]
            off += w
            parts.append(torch.cat([1.0 - p, p], dim=-1) if w == 1 else p)
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class ConceptMRF(nn.Module):
    """Unary energies conditioned on ``x`` + clique energies, solved by loopy BP.

    The trunk runs *outside* the factor graph and its output is handed to BP as
    evidence. Putting it inside a unary energy would re-run it once per
    enumerated state of every concept, because BP folds the enumeration grid
    into a leading axis. Gradients flow back through the evidence tensor, so the
    trunk, the unary heads and the clique energies all train from one backward
    pass through the message loop.
    """

    def __init__(self, input_size, labels, width, cliques):
        super().__init__()
        self.labels = list(labels)
        self.trunk = MLP(input_size=input_size, hidden_size=HIDDEN,
                         output_size=LATENT, n_layers=N_LAYERS)

        latent = EmbeddingVariable("latent", distribution=Normal, size=LATENT)
        variables = {
            n: (
                ConceptVariable(n, distribution=Bernoulli, size=1)
                if width[n] == 1
                else ConceptVariable(n, distribution=OneHotCategorical, size=width[n])
            )
            for n in self.labels
        }

        factors = [
            ParametricPotential(
                scope=[variables[n], latent],
                parametrization=UnaryEnergy(width[n], LATENT),
                name=f"u_{n}",
            )
            for n in self.labels
        ]
        factors += [
            ParametricPotential(
                scope=[variables[n] for n in clique],
                parametrization=CliqueEnergy([width[n] for n in clique]),
                name="phi(" + ",".join(clique) + ")",
            )
            for clique in cliques
        ]

        # A MarkovNetwork does not require every variable to own a factor, so
        # `latent` needs no prior: it appears only in the unary scopes and is
        # always evidence.
        self.mrf = MarkovNetwork(variables=[latent, *variables.values()], factors=factors)
        self.bp = BeliefPropagation(self.mrf, iters=BP_ITERS, damping=BP_DAMPING, tol=BP_TOL)

    def probs(self, x, clamped=None):
        clamped = clamped or {}
        free = [n for n in self.labels if n not in clamped]
        out = self.bp.query(query=free, evidence={"latent": self.trunk(x), **clamped})
        return {n: as_t(out.probs[n]) for n in free}

    def forward_train(self, x, gt):
        """Conditional training: fit the marginals of a random free subset.

        See ``TRAIN_CONDITIONAL``. BP evidence is per-variable across the batch,
        so the subset is drawn once per step rather than per sample.
        """
        if not TRAIN_CONDITIONAL:
            return self.probs(x)
        rate = torch.rand(1).item()
        keep = (torch.rand(len(self.labels)) < rate).tolist()
        clamped = {n: gt[n] for n, k in zip(self.labels, keep) if k}
        if len(clamped) == len(self.labels):
            clamped.pop(self.labels[0])   # always leave something to fit
        return self.probs(x, clamped)

    def forward_eval(self, x, clamped):
        return self.probs(x, clamped)


class CliquesOnlyMRF(nn.Module):
    """The *trained* :class:`ConceptMRF` with its unary energies multiplied by 0.

    An ablation, not a fourth trained model: it shares every parameter with
    ``ConceptMRF`` and is never optimised. With the unaries off, each unary
    factor's table is uniform and carries no information, so belief propagation
    is left with the clique potentials alone. The resulting curve is what the
    concept-concept structure contributes **on its own**, with ``x`` cut out
    entirely — at ``p = 0`` that is the learned marginal over each concept, and
    at ``p = 1`` it is what the neighbours alone determine.
    """

    def __init__(self, mrf_model: "ConceptMRF"):
        super().__init__()
        self.mrf_model = mrf_model
        self.unaries = [
            f.parametrization["energy"]
            for f in mrf_model.mrf.factors.values()
            if isinstance(f.parametrization["energy"], UnaryEnergy)
        ]

    def forward_eval(self, x, clamped):
        for u in self.unaries:
            u.scale = 0.0
        try:
            return self.mrf_model.forward_eval(x, clamped)
        finally:
            # Restored even if BP raises: the trained model is shared, and
            # leaving its unaries off would silently corrupt every later read.
            for u in self.unaries:
                u.scale = 1.0

    def forward_train(self, x, gt):
        raise RuntimeError("CliquesOnlyMRF is an eval-time ablation; it is not trained.")


class AutoregressiveCBM(nn.Module):
    """Hard autoregressive concept predictor (Havasi et al., NeurIPS 2022, §4.2).

    Concept ``k`` is predicted from the input *and* every earlier concept,
    ``p(c_k | x, c_1..c_{k-1})``, so unlike a plain CBM the predictor can express
    correlations between concepts (mutual exclusivity, implication) rather than
    treating them as conditionally independent given ``x``.

    The reason this is the baseline worth beating: its intervention scheme
    (Eq 8-9) updates the beliefs of concepts *earlier* in the order, not just
    later ones. `CBM` and `GraphCBM` can only push a clamped value forward to
    descendants, which is do-semantics; this model and `ConceptMRF` both do real
    conditioning. The difference is how — reweighted samples here, exact message
    passing there.

    Parameters
    ----------
    order : list of str
        Prediction order. Given the DAG's topological order, every concept's
        true parents precede it, so the model has access to the same structural
        information `GraphCBM` and `ConceptMRF` get. (The published model is
        graph-free and uses an arbitrary order; conditioning on the whole
        history subsumes any DAG either way.)
    """

    def __init__(self, input_size, labels, width, order):
        super().__init__()
        self.labels = list(labels)
        self.order = list(order)
        self.width = dict(width)
        self.trunk = MLP(input_size=input_size, hidden_size=HIDDEN,
                         output_size=LATENT, n_layers=N_LAYERS)
        #: Mean effective sample size of the last `forward_eval`, out of
        #: AR_SAMPLES. Read by the ESS diagnostic; see `ar_ess_report`.
        self.last_ess = float("nan")

        # One head per concept over [latent, history]. `MLP(..., n_layers=1)` is
        # Linear -> ReLU -> Linear, the paper's "small, two-layer network".
        self.heads = nn.ModuleDict()
        history = 0
        for n in self.order:
            self.heads[n] = MLP(input_size=LATENT + history, hidden_size=AR_HIDDEN,
                                output_size=self.width[n], n_layers=1)
            history += self.width[n]

    # -- per-concept parametrisation: width 1 is a Bernoulli logit, width K a
    #    categorical logit vector. Same convention as `gt_event`, so the history
    #    a head consumes has exactly the layout the clamps and targets use.
    def _to_probs(self, logits, n):
        return (torch.sigmoid(logits) if self.width[n] == 1
                else torch.softmax(logits, dim=-1))

    def _log_prob(self, logits, value, n):
        """``log p(value | ...)`` per row, shape ``(rows,)``."""
        if self.width[n] == 1:
            return -F.binary_cross_entropy_with_logits(
                logits, value, reduction="none").squeeze(-1)
        return (F.log_softmax(logits, dim=-1) * value).sum(-1)

    def _sample(self, logits, n):
        """An **exact** discrete draw — no Concrete relaxation anywhere."""
        if self.width[n] == 1:
            return torch.bernoulli(torch.sigmoid(logits))
        idx = torch.distributions.Categorical(logits=logits).sample()
        return F.one_hot(idx, self.width[n]).to(logits.dtype)

    def forward_train(self, x, gt):
        """Teacher-forced likelihood ``log p(c|x) = sum_k log p(c_k | x, c_1:k-1)``.

        The history is ground truth, which is the direct analogue of the
        baselines' ``p_int=1``: every model here is trained with its conditioning
        set to the true concept values.
        """
        z = self.trunk(x)
        history, out = [], {}
        for n in self.order:
            logits = self.heads[n](torch.cat([z, *history], dim=-1))
            out[n] = self._to_probs(logits, n)
            history.append(gt[n])
        return out

    @torch.no_grad()
    def forward_eval(self, x, clamped):
        """Normalised importance sampling under interventions (Eq 8-9).

        The proposal forces every clamped concept to its ground-truth value and
        samples the rest from the model, so ``q`` differs from ``p`` exactly by
        the clamped conditionals — hence ``w_m = prod_{k in I} p(chat_k | x,
        c_1:k-1)``. Because a clamped concept late in the order contributes a
        weight that *depends on the earlier sampled values*, reweighting shifts
        the posterior over those earlier concepts. That backward update is the
        whole point of the scheme.

        Samples and batch share one leading axis, so the only Python loop is
        over concepts. Returns a weighted empirical distribution per free
        concept — width 1 holds ``P(c=1)``, width K holds K probabilities.
        """
        M, B = AR_SAMPLES, x.shape[0]
        zr = self.trunk(x).unsqueeze(0).expand(M, B, -1).reshape(M * B, -1)
        logw = torch.zeros(M * B, dtype=zr.dtype, device=zr.device)
        history, samples = [], {}
        for n in self.order:
            logits = self.heads[n](torch.cat([zr, *history], dim=-1))
            if n in clamped:
                v = clamped[n].unsqueeze(0).expand(M, B, -1).reshape(M * B, -1)
                logw = logw + self._log_prob(logits, v, n)
            else:
                v = self._sample(logits, n)
                samples[n] = v
            history.append(v)

        # softmax over the sample axis IS w_m / sum_m w_m, done in log space.
        w = torch.softmax(logw.view(M, B), dim=0)
        self.last_ess = float((1.0 / (w ** 2).sum(0)).mean())
        return {n: (w.unsqueeze(-1) * s.view(M, B, -1)).sum(0)
                for n, s in samples.items()}


class LibraryModel(nn.Module):
    """Adapter giving CBM / GraphCBM the same two-method interface as ConceptMRF."""

    def __init__(self, model, labels, width):
        super().__init__()
        self.model = model
        self.labels = list(labels)
        self.width = width

    def _probs(self, out, names):
        # BP reports `probs`; the directed engines report `logits`
        # (`param_for_discrete_var = "logits"`).
        if "probs" in out.quantities:
            return {n: as_t(out.probs[n]) for n in names}
        return {
            n: (torch.sigmoid(as_t(out.logits[n])) if self.width[n] == 1
                else torch.softmax(as_t(out.logits[n]), dim=-1))
            for n in names
        }

    def forward_train(self, x, gt):
        # `gt` in the query is the teacher-forcing target; the train engine's
        # p_int=1.0 makes every concept propagate its ground-truth value.
        out = self.model(query=dict(gt), evidence={"input": x})
        return self._probs(out, self.labels)

    def forward_eval(self, x, clamped):
        free = [n for n in self.labels if n not in clamped]
        out = self.model(query=free, evidence={"input": x, **clamped})
        return self._probs(out, free)


def build_models(name, dm, labels, width, states, cliques):
    """The four trained models, all with the same backbone shape and latent width."""
    input_size = int(dm.n_features[0]) if hasattr(dm.n_features, "__len__") else int(dm.n_features)
    common = dict(
        input_size=input_size,
        annotations=dm.annotations,
        latent_size=LATENT,
        # HARD concepts. The plain `Bernoulli` / `OneHotCategorical` default is
        # sampled through `RelaxedBernoulli` at temperature 1.0, i.e. a *soft*
        # CBM in the sense of Havasi et al. — the concept values propagated
        # downstream are continuous and can carry information the concept
        # labels do not (leakage). The straight-through families draw an exact
        # bit / one-hot with a soft gradient, so the only thing separating
        # these baselines from `AR CBM` is the autoregressive structure.
        variable_distributions={
            "binary": RelaxedBernoulliStraightThrough,
            "categorical": RelaxedOneHotCategoricalStraightThrough,
        },
        # Ancestral sampling both ways: p_int=1 in training, so every concept
        # propagates its ground-truth value to its children (teacher forcing);
        # p_int=0 at test time, so the model runs unaided and any lift in the
        # figure comes from the `evidence` clamps alone.
        inference=AncestralSamplingInference,
        inference_kwargs={"p_int": 0.0},
        train_inference=AncestralSamplingInference,
        train_inference_kwargs={"p_int": 1.0},
        lightning=False,
    )

    def backbone():
        return MLP(input_size=input_size, hidden_size=HIDDEN,
                   output_size=LATENT, n_layers=N_LAYERS)

    cbm = ConceptBottleneckModel(
        task_names=TASKS[name], backbone=backbone(),
        # plate=False keeps one PGM variable per concept, so variable names are
        # concept names and a single concept can be clamped or read by name.
        # Graph models already force this.
        plate=False, **common,
    )
    graph_cbm = GraphConceptBottleneckModel(graph=dm.graph, backbone=backbone(), **common)
    mrf = ConceptMRF(input_size, labels, width, cliques)
    # Topological order: every concept's true parents precede it, so the AR
    # model is handed the same structure GraphCBM and the MRF get.
    ar = AutoregressiveCBM(input_size, labels, width, dm.graph.topological_sort())

    return {
        "CBM": LibraryModel(cbm, labels, width).to(DEVICE),
        "GraphCBM": LibraryModel(graph_cbm, labels, width).to(DEVICE),
        "AR CBM": ar.to(DEVICE),
        "ConceptMRF": mrf.to(DEVICE),
    }


# ---------------------------------------------------------------------------
# 3. training
# ---------------------------------------------------------------------------
def concept_nll(probs, c, labels, width):
    """One objective for all three models, scored on probabilities.

    BP reports only ``probs`` (``out.logits is None``), so the shared loss has to
    live in probability space. That costs a little numerical stability versus
    ``BCEWithLogitsLoss`` — hence the clamp — and buys the guarantee that no gap
    in the figure can be blamed on the models optimising different things.
    """
    scored = [n for n in labels if n in probs]
    total = 0.0
    for n in scored:
        i = labels.index(n)
        p = probs[n]
        if width[n] == 1:
            total = total + F.binary_cross_entropy(
                p.squeeze(-1).clamp(EPS, 1 - EPS), c[:, i].float()
            )
        else:
            total = total + F.nll_loss(p.clamp_min(EPS).log(), c[:, i].long())
    return total / max(1, len(scored))


@torch.no_grad()
def mean_accuracy(model, loader, labels, width):
    """Unaided (no intervention) mean per-concept accuracy."""
    model.eval()
    hits = torch.zeros(len(labels))
    n = 0
    for batch in loader:
        x, c = batch["inputs"]["x"].to(DEVICE), as_t(batch["concepts"]["c"]).to(DEVICE)
        probs = model.forward_eval(x, {})
        for i, name in enumerate(labels):
            hits[i] += predicted_class(probs[name], width[name]).eq(c[:, i]).sum().cpu()
        n += x.shape[0]
    return (hits / n)


def predicted_class(p, w):
    return (p.squeeze(-1) > 0.5).long() if w == 1 else p.argmax(-1)


def train_model(tag, model, dm, labels, width, epochs, lr):
    """Train and restore the best-validation weights.

    All three models overfit these small splits well before the last epoch (the
    loss keeps falling while validation accuracy turns over), so reporting the
    final weights would mostly compare how fast each one overfits. Selecting on
    validation accuracy is the same rule for every model, so the comparison stays
    like for like.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{tag}] training — {n_params} parameters, {epochs} epochs")
    t0 = time.time()
    best_acc, best_epoch, best_state = -1.0, -1, None
    for epoch in range(epochs):
        model.train()
        running, steps = 0.0, 0
        for batch in dm.train_dataloader():
            x = batch["inputs"]["x"].to(DEVICE)
            c = as_t(batch["concepts"]["c"]).to(DEVICE)
            probs = model.forward_train(x, gt_event(c, labels, width))
            loss = concept_nll(probs, c, labels, width)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss)
            steps += 1
        val = float(mean_accuracy(model, dm.val_dataloader(), labels, width).mean())
        if val > best_acc:
            best_acc, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"[{tag}] epoch {epoch:4d} | loss {running / steps:.4f} "
                  f"| val concept acc {val:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{tag}] done in {time.time() - t0:.1f}s "
          f"| best val {best_acc:.4f} @ epoch {best_epoch}")
    return model


# ---------------------------------------------------------------------------
# 4. the intervention sweep
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict_chunked(model, X, C, clamped_names, labels, width):
    """Predictions for every free concept, with ``clamped_names`` held at ground truth."""
    per_name = {n: [] for n in labels if n not in clamped_names}
    for start in range(0, X.shape[0], EVAL_BATCH):
        x = X[start : start + EVAL_BATCH]
        gt = gt_event(C[start : start + EVAL_BATCH], labels, width)
        probs = model.forward_eval(x, {n: gt[n] for n in clamped_names})
        for n in per_name:
            per_name[n].append(predicted_class(probs[n], width[n]))
    return {n: torch.cat(v) for n, v in per_name.items()}


@torch.no_grad()
def intervention_curves(models, X, C, labels, width, p_grid, repeats):
    """Per-concept accuracy vs. the probability of intervening on the OTHERS.

    Leave-one-out: concept ``j``'s own value is never clamped when ``j`` is being
    scored. Without that, a clamped variable is reported by BP as a point mass
    and would be trivially 100% correct — and ``p = 1.0`` would have no free
    concept left to measure. The same protocol runs on all three models, so the
    comparison is like for like.

    One subset is drawn per ``(p, repeat)`` and shared across concepts *and*
    models: the concepts outside it are all scored by a single pass (their
    evidence set is identical), and only the concepts inside it need their own
    leave-one-out pass. Restricted to ``labels \\ {j}`` the draw is still i.i.d.
    Bernoulli(p), so this is exact, not an approximation.
    """
    n_labels = len(labels)
    hits = {m: torch.zeros(n_labels, len(p_grid)) for m in models}
    totals = torch.zeros(n_labels, len(p_grid))
    truth = {n: C[:, i] for i, n in enumerate(labels)}
    rng = torch.Generator().manual_seed(SEED)

    for pi, p in enumerate(p_grid):
        # p=0 (nothing clamped) and p=1 (every other concept clamped) are
        # deterministic — repeating them would only burn time.
        n_rep = 1 if p in (0.0, 1.0) else repeats
        for _ in range(n_rep):
            if p >= 1.0:
                subset = set(labels)
            elif p <= 0.0:
                subset = set()
            else:
                draw = torch.rand(n_labels, generator=rng) < p
                subset = {n for n, keep in zip(labels, draw.tolist()) if keep}

            # Pass 1: everything outside the subset shares one evidence set.
            evidence_sets = [(sorted(subset), [n for n in labels if n not in subset])]
            # Pass 2..: each clamped concept needs itself released to be scored.
            evidence_sets += [(sorted(subset - {n}), [n]) for n in sorted(subset)]

            for clamped, scored in evidence_sets:
                if not scored:
                    continue
                for tag, model in models.items():
                    # `_predict_chunked` also returns concepts outside `subset`,
                    # but under a different evidence set than pass 1 gave them —
                    # scoring those too would mix two conditions, so only
                    # `scored` is counted.
                    pred = _predict_chunked(model, X, C, clamped, labels, width)
                    for n in scored:
                        hits[tag][labels.index(n), pi] += pred[n].eq(truth[n]).sum().cpu()
                # Model-independent: each scored concept saw the whole test split
                # exactly once under this evidence set.
                for n in scored:
                    totals[labels.index(n), pi] += X.shape[0]

    return {
        tag: {n: (hits[tag][i] / totals[i].clamp(min=1)).tolist()
              for i, n in enumerate(labels)}
        for tag in models
    }


@torch.no_grad()
def ar_ess_report(model, X, C, labels, width, p_grid, seed=SEED):
    """Effective sample size of the AR model's importance weights, per ``p``.

    The weight of a sample is a product over the *clamped* concepts, so as more
    of them are clamped the weights concentrate on fewer samples and the
    estimator degrades. ESS (out of ``AR_SAMPLES``) is what separates "the
    autoregressive model genuinely cannot do this" from "``M`` was too small",
    and without it a sagging AR curve is unreadable.

    Note the two ends behave differently: at ``p = 0`` nothing is clamped, every
    weight is equal and ESS is exactly ``M``; at ``p = 1`` only the scored
    concept varies across samples, so there are just ``states(j)`` distinct
    weights and ESS stays high. The squeeze is in the middle.
    """
    rng = torch.Generator().manual_seed(seed)
    x, gt = X[:EVAL_BATCH], gt_event(C[:EVAL_BATCH], labels, width)
    out = []
    for p in p_grid:
        draw = torch.rand(len(labels), generator=rng) < p
        # Leave-one-out, matching the sweep: the first concept is never clamped.
        clamped = {n: gt[n] for n, k in zip(labels, draw.tolist())
                   if k and n != labels[0]}
        model.forward_eval(x, clamped)
        out.append((p, len(clamped), model.last_ess))
    return out


def graph_only_reference(C_train, C_test, labels):
    """Accuracy of predicting each concept from the exact values of all others.

    Two reference lines per subplot:

    * ``graph_only`` — a majority-class lookup over the other concepts, fitted on
      the train split and applied to the test split with a prior backoff for
      unseen combinations. It says how much the concept-concept structure carries
      **on its own**. It is *not* an upper bound: the models also see ``x``, so a
      curve legitimately sits above it. It is conservative once the joint is
      sparsely sampled (``insurance``), where most test combinations are unseen.
    * ``prior`` — the majority-class accuracy, where a model sits having learned
      nothing at all.
    """
    ceiling, prior = {}, {}
    n_test = C_test.shape[0]
    for j, name in enumerate(labels):
        tr_y = C_train[:, j].tolist()
        te_y = C_test[:, j].tolist()
        tr_k = [tuple(r) for r in torch.cat([C_train[:, :j], C_train[:, j + 1:]], 1).tolist()]
        te_k = [tuple(r) for r in torch.cat([C_test[:, :j], C_test[:, j + 1:]], 1).tolist()]

        counts = {}
        for k, y in zip(tr_k, tr_y):
            counts.setdefault(k, {}).setdefault(y, 0)
            counts[k][y] += 1
        marginal = {}
        for y in tr_y:
            marginal[y] = marginal.get(y, 0) + 1
        fallback = max(marginal, key=marginal.get)
        best = {k: max(v, key=v.get) for k, v in counts.items()}

        ceiling[name] = sum(best.get(k, fallback) == y for k, y in zip(te_k, te_y)) / n_test
        prior[name] = sum(y == fallback for y in te_y) / n_test
    return ceiling, prior


# ---------------------------------------------------------------------------
# 5. figure
# ---------------------------------------------------------------------------
def make_figure(name, labels, curves, p_grid, unaided, ceiling, prior):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"concept_intervention_{name}.png"

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.2), sharey=True)
    axes = [axes] if n == 1 else list(axes)

    for ax, concept in zip(axes, labels):
        for tag in PLOT_ORDER:
            ax.plot(p_grid, curves[tag][concept], label=tag, markersize=4,
                    linewidth=1.6, **MODEL_STYLE[tag])
        title = concept + (" (task)" if concept in TASKS[name] else "")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("p(intervene)", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("concept accuracy", fontsize=9)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"{name} — per-concept accuracy vs. probability of intervening on the other concepts "
        f"(autoencoder noise {NOISE})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save everything the plot consumed, so a layout change never costs a re-run.
    torch.save(
        {"dataset": name, "labels": labels, "p_grid": p_grid, "curves": curves,
         "unaided": unaided, "ceiling": ceiling, "prior": prior,
         "noise": NOISE, "tasks": TASKS[name]},
        path.with_suffix(".pt"),
    )
    print(f"\nsaved {path}\nsaved {path.with_suffix('.pt')}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(name, epochs, repeats):
    dm, labels, width, states, cliques = load_data(name)
    report_structure(name, dm, labels, states, cliques)

    models = build_models(name, dm, labels, width, states, cliques)
    for tag in MODEL_ORDER:
        train_model(tag, models[tag], dm, labels, width, epochs, LR)

    # Shares the trained MRF's weights — an ablation, so it is added after
    # training and never optimised.
    models[ABLATION] = CliquesOnlyMRF(models["ConceptMRF"]).to(DEVICE)

    print(f"\nunaided per-concept accuracy (no intervention), {name}:")
    header = "  ".join(f"{n:>10s}" for n in labels)
    print(f"{'':>{LABEL_W}s}  {header}")
    unaided = {}
    for tag in PLOT_ORDER:
        acc = mean_accuracy(models[tag], dm.test_dataloader(), labels, width)
        unaided[tag] = acc.tolist()
        print(f"{tag:>{LABEL_W}s}  " + "  ".join(f"{a:10.3f}" for a in acc.tolist())
              + f"   | mean {float(acc.mean()):.3f}")

    X = torch.cat([b["inputs"]["x"] for b in dm.test_dataloader()]).to(DEVICE)
    C = torch.cat([as_t(b["concepts"]["c"]) for b in dm.test_dataloader()]).to(DEVICE)
    C_train = torch.cat([as_t(b["concepts"]["c"]) for b in dm.train_dataloader()]).cpu()
    ceiling, prior = graph_only_reference(C_train, C.cpu(), labels)
    # Cross-check on the ablation line, which should reproduce both of these:
    # with the unaries off and nothing clamped it predicts each concept's learned
    # marginal, so p=0 lands on `majority class`; with every neighbour clamped it
    # is a concepts-only predictor, so p=1 lands near `concepts only`.
    print(f"\nreference accuracies, {name}:")
    print(f"{'majority class':>{LABEL_W}s}  " + "  ".join(f"{prior[n]:10.3f}" for n in labels))
    print(f"{'concepts only':>{LABEL_W}s}  " + "  ".join(f"{ceiling[n]:10.3f}" for n in labels))
    for m in models.values():
        m.eval()

    t0 = time.time()
    curves = intervention_curves(models, X, C, labels, width, P_GRID, repeats)
    print(f"\nintervention sweep: {time.time() - t0:.1f}s over {X.shape[0]} test samples")

    ess = ar_ess_report(models["AR CBM"], X, C, labels, width, P_GRID)
    print(f"\nAR importance-weight effective sample size (out of {AR_SAMPLES}):")
    print("           p  " + "  ".join(f"{p:6.1f}" for p, _, _ in ess))
    print("     clamped  " + "  ".join(f"{k:6d}" for _, k, _ in ess))
    print("         ESS  " + "  ".join(f"{e:6.1f}" for _, _, e in ess))

    print(f"\nat p = 1.0 (every other concept known):")
    for tag in PLOT_ORDER:
        print(f"{tag:>{LABEL_W}s}  " + "  ".join(f"{curves[tag][n][-1]:10.3f}" for n in labels))

    make_figure(name, labels, curves, P_GRID, unaided, ceiling, prior)
    return curves


def main():
    # Declared up front: Python rejects a `global` that follows any use of the
    # name in the same scope, and the defaults below read these.
    global DEVICE, BATCH, EVAL_BATCH, AR_SAMPLES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / cuda:0 (default: CONCEPT_MRF_DEVICE, else auto)")
    parser.add_argument("--batch", type=int, default=BATCH, help="training batch size")
    parser.add_argument("--eval-batch", type=int, default=EVAL_BATCH,
                        help="inference batch size for the intervention sweep; "
                             "raise it on a GPU, lower it if a large clique runs out of memory")
    parser.add_argument("--ar-samples", type=int, default=AR_SAMPLES,
                        help="Monte-Carlo samples for the autoregressive model's "
                             "intervention estimator (paper: 200)")
    args = parser.parse_args()

    if args.device:
        DEVICE = torch.device(args.device)
    BATCH, EVAL_BATCH, AR_SAMPLES = args.batch, args.eval_batch, args.ar_samples

    seed_everything(SEED)
    print(f"device: {DEVICE} | train batch {BATCH} | eval batch {EVAL_BATCH}")
    if DEVICE.type == "cuda":
        print(f"        {torch.cuda.get_device_name(DEVICE)}")
    for name in args.datasets:
        run(name, args.epochs, args.repeats)


if __name__ == "__main__":
    main()
