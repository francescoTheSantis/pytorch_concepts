"""Toy shortcut experiment: a spurious correlation the graph does not sanction.

Setting
-------
Three discrete concepts, deliberately mirroring a Color-MNIST-style shortcut:

* ``digit``  — 10-way categorical, **hard** to read from the input,
* ``color``  — 3-way categorical (red / green / blue), **easy** to read,
* ``parity`` — binary, the task; exactly ``digit % 2 == 0``.

The graph handed to every model is the true one and contains a single edge,
``digit -> parity``. There is **no edge to colour**, because colour does not
cause parity and parity does not cause colour.

But in the training population they are strongly correlated anyway:
``P(red | even) = 0.90``. At test time that correlation is **reversed** —
``P(green | even) = 0.90`` — so a model that learned "red means even" is not
merely uninformed at test time, it is actively wrong.

Why the shortcut is attractive here
-----------------------------------
``parity`` is a deterministic function of ``digit``, so if the digit were easy to
read no model would ever need colour and every model would look robust. ``x`` is
therefore built so ``digit`` is *noisy*: the Bayes-optimal decoder caps
parity-via-digit at ~0.75, while the colour is worth 0.90 in training. A model
free to use the colour will.

The graph, and what it does *not* say
-------------------------------------
Every graph-based model is handed exactly::

    x -> digit        x -> color        digit -> parity

Two things follow. There is no ``color -> parity`` edge, which is the spurious
link. And there is no ``x -> parity`` edge either: parity is reachable *only*
through ``digit``. The second one is easy to violate by accident — the concept
MRF gives every concept a unary energy ``E(x, c)`` by default, and a unary is an
``x -> c`` edge, so left alone it would hand parity a direct view of ``x`` that
the graph forbids. `UNARY_ON` below restricts unaries to the graph's roots.

The residual leak, which is real and affects everyone
-----------------------------------------------------
The colour *is* encoded in ``x`` (the graph says ``x -> color``), so any encoder
reading ``x`` whole can pick the shortcut up before the concept graph gets a
say. Measured on this generator with a 2-layer probe:

===========================  ==========  ==========
predictor                    train pop   shifted
===========================  ==========  ==========
digit block -> parity            0.768       0.744
full x      -> parity            0.937       0.405
digit block -> digit             0.587       0.557
full x      -> digit             0.685       0.337
===========================  ==========  ==========

Reading ``x`` whole beats the digit block's own Bayes ceiling (0.685 vs 0.587)
and then collapses below it. That leak reaches ``digit`` for *every* model here,
because every encoder sees all of ``x`` — the graph constrains the
concept-to-concept channel, not what a shared backbone reads. So expect a common
drop on ``digit``, and read the model comparison on ``parity``: that is where
the graph is the only thing standing between a model and the shortcut.

The probe is rerun at startup so the numbers above are never stale.

What to expect
--------------
* ``CBM``       — its task head takes *every* concept as a parent, colour
                  included, so it takes the shortcut and collapses on the shift.
* ``AR CBM``    — conditions on the whole history, so it can and does learn the
                  colour-parity correlation. Same collapse. This is the model
                  from Havasi et al. that buys expressiveness by dropping the
                  graph, and this is what that costs.
* ``GraphCBM``  — predicts parity from its DAG parents only, i.e. ``digit``.
                  Structurally unable to see colour. Should not move.
* ``ConceptMRF``— parity's only clique is with ``digit``; colour is an isolated
                  node with no clique at all, so no message can carry colour
                  into parity. Its unary reads ``x``, which holds nothing but the
                  digit block. Robust for the same structural reason as
                  GraphCBM, while still using ``x`` for every concept.

Usage
-----
    python examples/experiments/toy_shortcut.py
    python examples/experiments/toy_shortcut.py --epochs 30 --device cuda
"""

import argparse
import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
import pandas as pd
import torch

from torch_concepts import seed_everything
from torch_concepts.annotations import Annotations
from torch_concepts.data.base import ConceptDataModule, ConceptDataset
from torch_concepts.data.splitters import FixedIndicesSplitter
from torch_concepts.data.utils import concat_datasets

# The model zoo, the shared loss and the training loop are the ones the bnlearn
# experiment already uses — same models, same objective, different data.
import concept_mrf_intervention as M

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
SEED = 42
N_DIGITS, N_COLORS = 10, 3
COLOR_NAMES = ("red", "green", "blue")
LABELS = ["digit", "parity", "color"]

N_TRAIN, N_TEST = 20000, 5000
VAL_FRACTION = 0.1

#: Width of each concept's block in ``x``. Parity gets **no block**: the only
#: routes to it are through the digit block (hard) or the colour block
#: (spurious), which is the whole point of the design.
D_DIGIT, D_COLOR = 16, 4

#: Per-concept observation noise, added to a fixed random prototype per class.
#: Measured against the Bayes-optimal (nearest-prototype) classifier at these
#: settings: digit is recovered at ~0.57, which caps **parity via digit** at
#: ~0.75 — clearly below the 0.90 colour shortcut, so during training the colour
#: really is the better bet and a model free to use it will. Colour itself reads
#: at ~1.00, so nothing is hidden on that side. `report_setup` reprints these
#: ceilings at run time, since they are what the whole design turns on.
SIGMA_DIGIT, SIGMA_COLOR = 2.2, 0.25

#: ``P(color | parity)``, rows indexed by the parity value (0 = odd, 1 = even),
#: columns by `COLOR_NAMES`. Training pins red to even; the test population
#: swaps that onto green, so the correlation does not merely vanish, it inverts.
TRAIN_BIAS = torch.tensor([[0.050, 0.475, 0.475],    # odd  -> green / blue
                           [0.900, 0.050, 0.050]])   # even -> RED
TEST_BIAS = torch.tensor([[0.475, 0.050, 0.475],     # odd  -> red / blue
                          [0.050, 0.900, 0.050]])    # even -> GREEN

BATCH, EPOCHS, LR = 256, 60, 1e-3

FIGDIR = Path(__file__).parent / "figures"


# ---------------------------------------------------------------------------
# 1. the toy dataset
# ---------------------------------------------------------------------------
def prototypes(seed=SEED):
    """Fixed class prototypes, shared by the train and test populations.

    They must be shared: if the two populations used different embeddings the
    test split would be a different problem entirely, and any drop would be
    unreadable. The *only* thing that differs between them is `P(color|parity)`.
    """
    g = torch.Generator().manual_seed(seed)
    return {
        "digit": torch.randn(N_DIGITS, D_DIGIT, generator=g),
        "color": torch.randn(N_COLORS, D_COLOR, generator=g),
    }


def signal_ceilings(n=20000, seed=SEED):
    """Bayes-optimal accuracy for each block, i.e. what the input alone permits.

    The noise is isotropic Gaussian around a prototype, so the optimal decision
    rule is nearest prototype and this is exact rather than an estimate. Printed
    at startup because the experiment is only meaningful while parity-via-digit
    stays below the 0.90 colour bias.
    """
    proto = prototypes(seed)
    g = torch.Generator().manual_seed(seed + 99)
    digit = torch.randint(N_DIGITS, (n,), generator=g)
    xd = proto["digit"][digit] + SIGMA_DIGIT * torch.randn(n, D_DIGIT, generator=g)
    hat_d = torch.cdist(xd, proto["digit"]).argmin(1)
    color = torch.randint(N_COLORS, (n,), generator=g)
    xc = proto["color"][color] + SIGMA_COLOR * torch.randn(n, D_COLOR, generator=g)
    hat_c = torch.cdist(xc, proto["color"]).argmin(1)
    return (float((hat_d == digit).float().mean()),
            float(((hat_d % 2) == (digit % 2)).float().mean()),
            float((hat_c == color).float().mean()))


class ToyShortcutDataset(ConceptDataset):
    """One population: digits uniform, parity determined, colour biased."""

    def __init__(self, n_samples, bias, seed, proto):
        g = torch.Generator().manual_seed(seed)
        digit = torch.randint(N_DIGITS, (n_samples,), generator=g)
        parity = (digit % 2 == 0).long()
        color = torch.multinomial(bias[parity], 1, replacement=True,
                                  generator=g).squeeze(-1)

        x = torch.cat([
            proto["digit"][digit] + SIGMA_DIGIT * torch.randn(n_samples, D_DIGIT, generator=g),
            proto["color"][color] + SIGMA_COLOR * torch.randn(n_samples, D_COLOR, generator=g),
        ], dim=1)

        # The true graph, and the one every model is given. Colour is isolated:
        # its correlation with parity is real in the data and absent here.
        graph = pd.DataFrame(0, index=LABELS, columns=LABELS)
        graph.loc["digit", "parity"] = 1

        super().__init__(
            input_data=x,
            # Columns follow the declared annotation order (see `set_concepts`).
            concepts=torch.stack([digit, parity, color], dim=1),
            annotations=Annotations(
                labels=LABELS,
                cardinalities=[N_DIGITS, 1, N_COLORS],
                types=["categorical", "binary", "categorical"],
            ),
            graph=graph,
            name="toy_shortcut",
        )


def build_datamodule(seed=SEED):
    """Two populations concatenated, with the split drawn on the boundary.

    Validation is carved out of the *training* population, so it is
    in-distribution; the test split is exactly the shifted population. That is
    what lets one number (val) act as the control for another (test).
    """
    proto = prototypes(seed)
    train_pop = ToyShortcutDataset(N_TRAIN, TRAIN_BIAS, seed, proto)
    test_pop = ToyShortcutDataset(N_TEST, TEST_BIAS, seed + 1, proto)
    dataset = concat_datasets(train_pop, test_pop)

    n_val = int(VAL_FRACTION * N_TRAIN)
    order = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(seed)).tolist()
    splitter = FixedIndicesSplitter(
        train_idxs=order[n_val:],
        val_idxs=order[:n_val],
        test_idxs=range(N_TRAIN, N_TRAIN + N_TEST),
    )
    dm = ConceptDataModule(dataset=dataset, splitter=splitter,
                           batch_size=BATCH, seed=seed)
    dm.setup()
    return dm


def graph_roots(dm):
    """Concepts with no parents — exactly the ones the graph feeds from ``x``.

    These get a unary energy in the MRF. Giving one to a non-root would add an
    ``x -> c`` edge the graph does not have; see the module docstring.
    """
    g = dm.graph.to_networkx()
    return [n for n in dm.graph.node_names if g.in_degree(n) == 0]


@torch.no_grad()
def leak_probe(n=8000, seed=SEED):
    """How much of the shortcut a plain encoder can take before the graph acts.

    A closed-form linear-discriminant stand-in would understate it, so this is a
    small trained probe: `digit` from its own block versus from all of ``x``, on
    the training population and on the shifted one. The gap is the corruption
    every model inherits through its encoder, independent of any graph.
    """
    import torch.nn as nn
    proto = prototypes(seed)

    def pop(bias, s):
        g = torch.Generator().manual_seed(s)
        d = torch.randint(N_DIGITS, (n,), generator=g)
        par = (d % 2 == 0).long()
        c = torch.multinomial(bias[par], 1, replacement=True, generator=g).squeeze(-1)
        x = torch.cat([proto["digit"][d] + SIGMA_DIGIT * torch.randn(n, D_DIGIT, generator=g),
                       proto["color"][c] + SIGMA_COLOR * torch.randn(n, D_COLOR, generator=g)], 1)
        return x, d, par

    xtr, dtr, ptr = pop(TRAIN_BIAS, seed + 7)
    xte, dte, pte = pop(TEST_BIAS, seed + 8)
    out = {}
    for view, sl in (("digit block", slice(0, D_DIGIT)),
                     ("full x", slice(0, D_DIGIT + D_COLOR))):
        for tgt, (ytr, yte), k in (("parity", (ptr, pte), 2), ("digit", (dtr, dte), N_DIGITS)):
            with torch.enable_grad():
                net = nn.Sequential(nn.Linear(xtr[:, sl].shape[1], 64), nn.ReLU(),
                                    nn.Linear(64, k))
                opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
                for _ in range(300):
                    opt.zero_grad()
                    nn.functional.cross_entropy(net(xtr[:, sl]), ytr).backward()
                    opt.step()
            out[(view, tgt)] = (float((net(xtr[:, sl]).argmax(1) == ytr).float().mean()),
                                float((net(xte[:, sl]).argmax(1) == yte).float().mean()))
    return out


def report_setup(dm, labels, width, states, cliques):
    print(f"\n{'=' * 72}\ntoy shortcut  —  {len(labels)} concepts, task ['parity']\n{'=' * 72}")
    print(f"concepts : {', '.join(f'{n}({states[n]})' for n in labels)}")
    print(f"graph    : {sorted(dm.graph.to_networkx().edges())}   (colour is isolated)")
    print(f"cliques  : {cliques}")
    print(f"roots    : {graph_roots(dm)}   <- the only concepts that may read x;"
          f" no model gets an x -> parity edge")
    print(f"x        : {D_DIGIT}-d digit block (sigma {SIGMA_DIGIT}) + "
          f"{D_COLOR}-d colour block (sigma {SIGMA_COLOR}); no parity block")
    d_acc, p_acc, c_acc = signal_ceilings()
    print(f"ceilings : digit {d_acc:.3f} | parity-via-digit {p_acc:.3f} | colour {c_acc:.3f}"
          f"   <- shortcut is worth 0.90, so parity-via-digit must sit below it")

    # The correlation the graph does not sanction, measured in each population.
    print("encoder leak probe (train-pop / shifted):")
    for (view, tgt), (a, b) in leak_probe().items():
        print(f"    {view:>11s} -> {tgt:<6s}  {a:.3f} / {b:.3f}")

    for tag, split in (("train", "train"), ("shifted test", "test")):
        C = torch.cat([M.as_t(b["concepts"]["c"])
                       for b in getattr(dm, f"{split}_dataloader")()])
        par = C[:, labels.index("parity")]
        col = C[:, labels.index("color")]
        row = "  ".join(
            f"P({c}|even)={float((col[par == 1] == i).float().mean()):.2f}"
            for i, c in enumerate(COLOR_NAMES))
        print(f"{tag:>13s}: {row}")


# ---------------------------------------------------------------------------
# 2. evaluation
# ---------------------------------------------------------------------------
def task_blanket(tag, model, dm, labels, task="parity"):
    """The concepts *this model* lets the task depend on.

    The graph's own Markov blanket for ``parity`` is ``{digit}`` — it has one
    parent and no children or co-parents. A model that respects the graph has
    the same blanket; one that does not has a larger one, and the extra members
    are exactly the edges it invented. Clamping a model's own blanket therefore
    tells you whether it *can* be fixed by intervention, while clamping the
    graph's blanket tells you whether it *agrees with the graph*.
    """
    if tag == "AR CBM":
        # Its conditioning set is everything earlier in the order.
        return [n for n in model.order[:model.order.index(task)]]
    if hasattr(model, "mrf"):                       # ConceptMRF
        nb = {v.name for f in model.mrf.factors.values()
              for v in f.scope if any(u.name == task for u in f.scope)}
        return sorted(nb - {task, "latent"})
    if hasattr(model, "mrf_model"):                 # the cliques-only ablation
        return task_blanket("ConceptMRF", model.mrf_model, dm, labels, task)
    parents = model.model.pgm.factors[task].parents  # CBM / GraphCBM
    return [p.name for p in parents if p.name not in ("latent", "input")]


@torch.no_grad()
def accuracy(model, loader, labels, width, observe=()):
    """Per-concept accuracy, optionally with some concepts clamped to truth.

    ``observe`` is the lever that separates the two failure channels. Unaided,
    every model inherits whatever its encoder learned from ``x`` — and since the
    colour lives in ``x``, that includes the shortcut, for all of them. Clamp
    ``digit`` and the encoder is taken out of the question for parity: what is
    left is purely the concept-to-concept channel, which is the only thing the
    graph actually governs. A model whose parity depends on ``digit`` alone
    must recover; one that also reads ``color`` cannot.
    """
    model.eval()
    scored = [n for n in labels if n not in observe]
    hits, n = torch.zeros(len(labels)), 0
    for batch in loader:
        x = batch["inputs"]["x"].to(M.DEVICE)
        c = M.as_t(batch["concepts"]["c"]).to(M.DEVICE)
        gt = M.gt_event(c, labels, width)
        probs = model.forward_eval(x, {k: gt[k] for k in observe})
        for i, name in enumerate(labels):
            if name not in scored:
                continue
            hits[i] += M.predicted_class(probs[name], width[name]).eq(c[:, i]).sum().cpu()
        n += x.shape[0]
    return [float("nan") if name in observe else float(hits[i] / n)
            for i, name in enumerate(labels)]


# ---------------------------------------------------------------------------
# 3. figure
# ---------------------------------------------------------------------------
BLANKETS = {}


def make_figure(labels, results):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / "toy_shortcut.png"
    tags = list(results)
    bars = [("val", "in-distribution", "#4c72b0"),
            ("test", "shifted", "#c44e52"),
            ("test_do_digit", "shifted, graph blanket given", "#55a868"),
            ("test_do_blanket", "shifted, own blanket given", "#8172b2")]

    fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 4.2))
    axes = [axes] if len(labels) == 1 else list(axes)
    xs = list(range(len(tags)))
    for ax, concept in zip(axes, labels):
        j = labels.index(concept)
        for k, (split, caption, color) in enumerate(bars):
            vals = [results[t][split][j] for t in tags]
            # `digit` is clamped in the third condition, so it has no score.
            offs = (k - 1.5) * 0.21
            ax.bar([v + offs for v in xs], [0 if a != a else a for a in vals],
                   0.20, label=caption if concept == labels[0] else None, color=color)
        ax.set_title(concept + (" (task)" if concept == "parity" else ""), fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("accuracy", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Spurious colour-parity correlation, reversed at test "
        "(train P(red|even)=0.90 -> test P(green|even)=0.90); the graph has no colour edge.\n"
        "Green bars remove the encoder from the question: with the true digit supplied, "
        "only the concept-to-concept channel is left.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    torch.save({"labels": labels, "results": results, "blankets": BLANKETS,
                "train_bias": TRAIN_BIAS, "test_bias": TEST_BIAS},
               path.with_suffix(".pt"))
    print(f"\nsaved {path}\nsaved {path.with_suffix('.pt')}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ar-samples", type=int, default=M.AR_SAMPLES)
    args = parser.parse_args()
    if args.device:
        M.DEVICE = torch.device(args.device)
    M.AR_SAMPLES = args.ar_samples

    seed_everything(SEED)
    print(f"device: {M.DEVICE}")

    dm = build_datamodule()
    ann = dm.annotations
    labels = list(ann.labels)
    width = {n: (1 if k == 1 else int(k)) for n, k in zip(labels, ann.cardinalities)}
    states = {n: (2 if k == 1 else int(k)) for n, k in zip(labels, ann.cardinalities)}
    cliques = sorted((sorted(c) for c in nx.find_cliques(nx.moral_graph(dm.graph.to_networkx()))
                      if len(c) > 1), key=lambda c: (-len(c), c))
    report_setup(dm, labels, width, states, cliques)

    # `build_models` reads the task name off this table.
    M.TASKS["toy"] = ["parity"]
    models = M.build_models("toy", dm, labels, width, states, cliques,
                            x_observes=graph_roots(dm))
    for tag in M.MODEL_ORDER:
        M.train_model(tag, models[tag], dm, labels, width, args.epochs, LR)
    models[M.ABLATION] = M.CliquesOnlyMRF(models["ConceptMRF"]).to(M.DEVICE)

    results, blankets = {}, {}
    for tag in M.PLOT_ORDER:
        results[tag] = {
            "val": accuracy(models[tag], dm.val_dataloader(), labels, width),
            "test": accuracy(models[tag], dm.test_dataloader(), labels, width),
            "test_do_digit": accuracy(models[tag], dm.test_dataloader(), labels, width,
                                      observe=("digit",)),
            "test_do_blanket": accuracy(models[tag], dm.test_dataloader(), labels, width,
                                        observe=tuple(task_blanket(tag, models[tag], dm, labels))),
        }
        blankets[tag] = task_blanket(tag, models[tag], dm, labels)

    head = "  ".join(f"{n:>9s}" for n in labels)
    SPLITS = (("val", "in-distribution"), ("test", "SHIFTED"),
              ("test_do_digit", "SHIFTED, graph blanket given {digit}"),
              ("test_do_blanket", "SHIFTED, own blanket given"))
    print(f"\n{'':>{M.LABEL_W}s}  {head}   split")
    j = labels.index("parity")
    for tag in M.PLOT_ORDER:
        for k, (split, caption) in enumerate(SPLITS):
            row = "  ".join("      ---" if a != a else f"{a:9.3f}"
                            for a in results[tag][split])
            print(f"{tag if k == 0 else '':>{M.LABEL_W}s}  {row}   {caption}")
        print(f"{'':>{M.LABEL_W}s}  {'':>9s}  parity {results[tag]['test'][j] - results[tag]['val'][j]:+.3f} "
              f"under shift | own blanket = {blankets[tag]}\n")

    BLANKETS.update(blankets)
    make_figure(labels, results)


if __name__ == "__main__":
    main()
