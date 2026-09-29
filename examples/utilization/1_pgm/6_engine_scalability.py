"""
In this example we measure how long every inference engine takes to answer one
batch of queries as the number of concepts grows.

The model is a synthetic BayesianNetwork:

    X ──► c_1 ─┐
    X ──► c_2 ─┤
       ...     ├──► Y
    X ──► c_n ─┘

- X is a continuous 64-dimensional embedding, always observed (the input).
- c_1 ... c_n are Bernoulli concepts, conditionally independent given X.
- Y is a Bernoulli task with every concept as a parent.

Every engine answers P(c_1, ..., c_n, Y | X) for a batch of ``BATCH`` inputs, and
the only thing recorded is the wall-clock time of that ``query`` call (model
and engine construction are not timed).

The concepts are modelled in two ways:

- **separate variables**: one ConceptVariable and one CPD per concept, so an
  engine walks n variables;
- **a plate**: one ConceptVariable with n members and a single CPD producing all
  of them at once (see 0.2_cbm_plate.py).

Belief propagation and variable elimination do not work on plates (they unpack
them into separate variables internally), so they only run on separate
variables. They also enumerate Y's factor, whose table has 2^(n+1) entries, so
their grid stops at ``max(EXACT_GRID)`` concepts rather than 200k.

A curve stops early when the next size would not fit: ImportanceSampling is
skipped once its estimated peak memory exceeds ``MEMORY_BUDGET_GB``, and any
engine is skipped at larger sizes once a single call took more than
``TIME_BUDGET_S``. The last point of such a curve is marked with ✕.

Run with ``--quick`` for a seconds-long smoke test on a tiny grid.
"""

import argparse
import statistics
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from torch.distributions import Bernoulli

from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.nn import ParametricCPD, BayesianNetwork, LearnablePrior, \
    DeterministicInference, IndependentInference, AncestralSamplingInference, \
    MAPForwardInference, RejectionSampling, ImportanceSampling, MutilatedNetworkProposal, \
    BeliefPropagation, PgmpyVariableElimination


BATCH = 512          # rows answered by one query call
X_DIM = 64           # size of the input embedding X
N_SAMPLES = 10       # samples per row for rejection and importance sampling

# Number of concepts. The exact grid is for the engines that enumerate Y's
# 2^(n+1) table; everything else gets the wide one.
GRID = [20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
EXACT_GRID = [2, 4, 6, 8, 10, 12, 14, 16]
QUICK_GRID, QUICK_EXACT_GRID = [20, 200], [2, 4]

MEMORY_BUDGET_GB = 6.0
TIME_BUDGET_S = 300.0

# A point is timed up to REPEATS times, while the calls so far took less than
# REPEAT_BUDGET_S, and the median is kept: millisecond calls are noisy, and a
# minute-long call does not need repeating.
REPEATS = 3
REPEAT_BUDGET_S = 10.0

# ImportanceSampling packs N_SAMPLES * BATCH rows into one batch, so its peak
# memory is a multiple of one float32 (N_SAMPLES * BATCH, n) tensor. Measured
# with peak RSS on a 5k-20k concept model: ~8-9x with a plate (every stage
# holds a full (rows, n) block), ~2.5x with separate variables (one small
# tensor per variable). Rounded up.
IS_MEMORY_FACTOR = {"plate": 10.0, "separate": 3.0}

# Engines in plotting order: label -> (factory, runs on a plate, grid kind).
ENGINES = {
    "Deterministic": (DeterministicInference, True, "wide"),
    "Independent": (IndependentInference, True, "wide"),
    "Ancestral": (AncestralSamplingInference, True, "wide"),
    "MAP forward": (MAPForwardInference, True, "wide"),
    "Rejection": (lambda pgm: RejectionSampling(pgm, n_samples=N_SAMPLES), True, "wide"),
    "Importance": (lambda pgm: ImportanceSampling(
        pgm, MutilatedNetworkProposal(pgm), n_samples=N_SAMPLES), True, "wide"),
    "Belief prop.": (BeliefPropagation, False, "exact"),
    "Var. elim.": (PgmpyVariableElimination, False, "exact"),
}
# Engines that take a target assignment and return its probability, rather
# than returning the queried variables' distributions.
ESTIMATORS = {"Rejection", "Importance"}
LAYOUTS = ["separate", "plate"]


def build_model(n, layout):
    """The X -> concepts -> Y network with ``n`` concepts.

    Returns the model and the names that address the concepts in a query: one
    per concept for separate variables, the plate's name otherwise. The weights
    are left at their random initialisation; they do not affect timing.
    """
    x = EmbeddingVariable("X", distribution=Delta, size=X_DIM)
    factors = [ParametricCPD(x, parametrization=LearnablePrior(X_DIM), parents=[])]
    if layout == "plate":
        concepts = [ConceptVariable("C", distribution=Bernoulli, size=1,
                                    members=[f"c{i}" for i in range(n)])]
        factors.append(ParametricCPD(concepts[0], parametrization={"logits": nn.Linear(X_DIM, n)},
                                     parents=[x]))
    else:
        concepts = [ConceptVariable(f"c{i}", distribution=Bernoulli, size=1) for i in range(n)]
        factors += [ParametricCPD(c, parametrization={"logits": nn.Linear(X_DIM, 1)}, parents=[x])
                    for c in concepts]
    y = ConceptVariable("Y", distribution=Bernoulli, size=1)
    factors.append(ParametricCPD(y, parametrization={"logits": nn.Linear(n, 1)}, parents=concepts))
    model = BayesianNetwork(variables=[x, *concepts, y], factors=factors)
    return model, [c.name for c in concepts]


def make_query(label, concept_names, n):
    """The query an engine expects for P(c_1, ..., c_n, Y | X).

    The estimators need a joint assignment to score; a random one is as good as
    any for timing (its probability over thousands of bits is ~0 anyway). The
    other engines take the names and return every queried distribution.
    """
    names = concept_names + ["Y"]
    if label not in ESTIMATORS:
        return names
    # A plate's value is (BATCH, n); a separate concept's is (BATCH, 1).
    width = {name: 1 for name in names}
    if concept_names == ["C"]:
        width["C"] = n
    return {name: torch.bernoulli(torch.full((BATCH, width[name]), 0.5)) for name in names}


def estimated_memory_gb(label, layout, n):
    """Peak memory an engine adds at ``n`` concepts, when it can outgrow the budget.

    Only ImportanceSampling does. The forward engines hold (BATCH, n)
    activations (under 1 GB at 200k), rejection sampling draws N_SAMPLES rows at
    a time, and the exact engines stop at EXACT_GRID.
    """
    if label != "Importance":
        return 0.0
    return N_SAMPLES * BATCH * n * 4 * IS_MEMORY_FACTOR[layout] / 1e9


def time_query(engine, query, evidence):
    """Median wall-clock seconds of ``engine.query(query, evidence)``.

    CPU only, so ``perf_counter`` needs no device synchronisation. Warnings are
    silenced: with thousands of queried bits the estimators rightly report a
    vanishing acceptance rate / effective sample size, which is expected here
    and would otherwise flood the output.
    """
    times = []
    while len(times) < REPEATS and sum(times) < REPEAT_BUDGET_S:
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            engine.query(query, evidence)
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="tiny grid, for a smoke test")
    args = parser.parse_args()
    grids = {"wide": QUICK_GRID if args.quick else GRID,
             "exact": QUICK_EXACT_GRID if args.quick else EXACT_GRID}
    seed_everything(42)

    runs = [(label, layout) for label, (_, plate_ok, _) in ENGINES.items()
            for layout in LAYOUTS if layout == "separate" or plate_ok]

    # Warm-up: the first call of an engine pays one-off costs (imports such as
    # pgmpy's, first kernel dispatch) that do not belong to any grid size.
    for layout in LAYOUTS:
        model, concept_names = build_model(2, layout)
        evidence = {"X": torch.randn(BATCH, X_DIM)}
        for label, run_layout in runs:
            if run_layout == layout:
                time_query(ENGINES[label][0](model), make_query(label, concept_names, 2), evidence)

    # {(label, layout): {"n": [...], "seconds": [...], "stopped": None | reason}}
    results = {run: {"n": [], "seconds": [], "stopped": None} for run in runs}
    for layout in LAYOUTS:
        for n in sorted(set(grids["wide"]) | set(grids["exact"])):
            active = [label for label, run_layout in runs if run_layout == layout
                      and n in grids[ENGINES[label][2]] and results[(label, layout)]["stopped"] is None]
            if not active:
                continue
            start = time.perf_counter()
            model, concept_names = build_model(n, layout)
            print(f"\n[{layout}] n={n:,}: model built in {time.perf_counter() - start:.1f}s")
            evidence = {"X": torch.randn(BATCH, X_DIM)}

            for label in active:
                result = results[(label, layout)]
                memory = estimated_memory_gb(label, layout, n)
                if memory > MEMORY_BUDGET_GB:
                    result["stopped"] = f"memory estimate {memory:.1f} GB at n={n:,}"
                    print(f"  {label:>13}: skipped, {result['stopped']}")
                    continue
                seconds = time_query(ENGINES[label][0](model), make_query(label, concept_names, n),
                                     evidence)
                result["n"].append(n)
                result["seconds"].append(seconds)
                print(f"  {label:>13}: {seconds:.4f}s")
                if seconds > TIME_BUDGET_S:
                    result["stopped"] = f"{seconds:.0f}s call at n={n:,}"
            del model

    print("\n=== Stopped curves ===")
    for (label, layout), result in results.items():
        if result["stopped"] is not None:
            print(f"{label:>13} [{layout}]: larger sizes skipped after {result['stopped']}")

    name = "engine_scalability_quick.png" if args.quick else "engine_scalability.png"
    plot(results, Path(__file__).with_name(name))


def plot(results, path):
    """Elapsed time against number of concepts, one line per engine and layout."""
    surface, ink, ink_muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    # Categorical slots in their fixed order, one per engine.
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    colors = dict(zip(ENGINES, palette))
    styles = {"separate": "-", "plate": "--"}

    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=surface)
    ax.set_facecolor(surface)
    for (label, layout), result in results.items():
        if not result["n"]:
            continue
        ax.plot(result["n"], result["seconds"], color=colors[label], linestyle=styles[layout],
                linewidth=1.8, marker="o", markersize=4.5, zorder=3)
        if result["stopped"] is not None:
            ax.plot(result["n"][-1], result["seconds"][-1], color=colors[label], marker="x",
                    markersize=11, markeredgewidth=2.2, linestyle="none", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v * 1000:g} ms" if v < 1 else f"{v:g} s"))
    ax.set_xlabel("Number of concepts", color=ink, fontsize=10)
    ax.set_ylabel(f"Elapsed time per batch of {BATCH}", color=ink, fontsize=10)
    ax.set_title(f"Time to answer P(c₁, …, cₙ, Y | X) — X: {X_DIM}-d embedding, "
                 f"sampling engines: {N_SAMPLES} samples per row, CPU",
                 fontsize=11, color=ink, loc="left")
    ax.grid(True, which="major", color=grid, linewidth=0.8, zorder=0)
    ax.tick_params(colors=ink_muted, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink_muted)

    engine_legend = ax.legend(
        [Line2D([], [], color=colors[label], linewidth=2.5) for label in ENGINES], list(ENGINES),
        title="Engine", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
        fontsize=9, title_fontsize=9, labelcolor=ink, alignment="left")
    ax.add_artist(engine_legend)
    ax.legend(
        [Line2D([], [], color=ink, linestyle=styles["separate"], linewidth=1.8),
         Line2D([], [], color=ink, linestyle=styles["plate"], linewidth=1.8),
         Line2D([], [], color=ink, marker="x", markersize=9, markeredgewidth=2, linestyle="none")],
        ["Separate variables", "Plate", "Larger sizes skipped"],
        title="Concepts as", loc="upper left", bbox_to_anchor=(1.01, 0.45), frameon=False,
        fontsize=9, title_fontsize=9, labelcolor=ink, alignment="left")

    fig.text(0.01, 0.01,
             f"✕: the next size was not run — ImportanceSampling's estimated memory exceeds "
             f"{MEMORY_BUDGET_GB:g} GB, or a call took over {TIME_BUDGET_S:g} s.\n"
             f"Belief propagation and variable elimination enumerate Y's 2ⁿ⁺¹-entry table, "
             f"so they stop at {max(EXACT_GRID)} concepts. Independent coincides with "
             f"Deterministic at test time.",
             fontsize=8.5, color=ink_muted, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, facecolor=surface)
    print(f"\nSaved plot to {path}")
    plt.show()


if __name__ == "__main__":
    main()
