"""
In this example we train a concept model on the Insurance network and compare how
every inference engine answers two conditional queries, next to the empirical
frequency in the training set.

The model is a BayesianNetwork with one categorical concept per node of the
Insurance DAG. Both queries look at the same family:

    Antilock ────┐
    Mileage ─────┼──► Accident
    DrivQuality ─┘

- Q1 conditions a node on all of its parents:
      P(Accident=Severe | Antilock=False, Mileage=FiftyThou, DrivQuality=Poor)
- Q2 conditions a parent on the child and on the other parents (explaining
  away): observing a severe accident in a car with no ABS makes poor driving
  far more likely than its prior suggests.
      P(DrivQuality=Poor | Accident=Severe, Antilock=False, Mileage=FiftyThou)

The forward engines (Deterministic, Independent, Ancestral) only propagate
information from parents to children, so in Q2 they cannot use the evidence on
Accident and estimate the prior of DrivQuality instead. Ancestral does so by
sampling the ancestors, which recovers the prior; Deterministic and Independent
push probabilities rather than samples through the CPDs, which only
approximates it. The posterior engines (rejection sampling, importance
sampling, belief propagation, variable elimination) condition on the child.

Every engine answers each query ``N_RUNS`` times, each run with its own seed.
Bars show the mean; engines whose answer changes from run to run (the sampling
ones) also get an error bar of one standard deviation.
"""

import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pgmpy.readwrite import BIFReader
from torch.distributions import OneHotCategorical

from torch_concepts import seed_everything, ConceptVariable
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import ParametricCPD, BayesianNetwork, LearnablePrior, \
    DeterministicInference, IndependentInference, AncestralSamplingInference, \
    RejectionSampling, ImportanceSampling, MutilatedNetworkProposal, \
    BeliefPropagation, PgmpyVariableElimination


# The two queries: (query variable, query state), {evidence variable: state}.
QUERIES = {
    "Q1": (("Accident", "Severe"),
           {"Antilock": "False", "Mileage": "FiftyThou", "DrivQuality": "Poor"}),
    "Q2": (("DrivQuality", "Poor"),
           {"Accident": "Severe", "Antilock": "False", "Mileage": "FiftyThou"}),
}

# Engines that only propagate parents -> children; the rest condition on
# evidence anywhere in the graph.
FORWARD_ENGINES = {"Deterministic", "Independent", "Ancestral"}

# How many times every engine answers each query. A deterministic engine gives
# the same number every run (standard deviation 0, no error bar); a sampling
# engine does not, and its spread is what the error bar shows.
N_RUNS = 10


def one_hot(state: int, n_states: int, rows: int = 1) -> torch.Tensor:
    """A ``(rows, n_states)`` one-hot block: the value format query/evidence expect."""
    return F.one_hot(torch.full((rows,), state), n_states).float()


def engine_probability(engine, q_var, q_state, evidence_idx, n_states, rows=1):
    """P(q_var = q_state | evidence) as computed by ``engine``.

    The engines come in two families. The estimators (rejection and importance
    sampling) take the query *target* and return ``out.probabilities``. Every
    other engine returns the query variable's distribution, from which the
    state's column is read. An Ancestral engine draws one sample per row, so it
    gets ``rows`` copies of the evidence and its answer is averaged over them.
    """
    evidence = {v: one_hot(s, n_states[v], rows) for v, s in evidence_idx.items()}
    if isinstance(engine, (RejectionSampling, ImportanceSampling)):
        target = {q_var: one_hot(q_state, n_states[q_var])}
        return engine.query(target, evidence).probabilities.item()

    params = engine.query(query=[q_var], evidence=evidence).params[q_var]
    probs = params["probs"] if "probs" in params else torch.softmax(params["logits"], dim=-1)
    return probs[..., q_state].mean().item()


def main():
    seed_everything(42)
    n_epochs = 500

    # ---- Data ----------------------------------------------------------------
    # The cached copy used by conceptarium; it is built (bnlearn download plus an
    # autoencoder for the unused embeddings) only if it is missing.
    dataset = BnLearnDataset(
        name="insurance",
        root=os.path.expanduser("~/.cache/conceptarium/insurance"),
        seed=42,
        n_gen=10000,
    )
    names = dataset.annotations.labels
    # (N, 27) state indices. Columns follow ``names`` (binaries first), so they
    # are always looked up by name, never by position.
    concepts = dataset.concepts.as_subclass(torch.Tensor).long()
    index = {n: concepts[:, i] for i, n in enumerate(names)}

    # State names come from the network file; index i is the i-th BIF state.
    bif = BIFReader(os.path.join(dataset.root_dir, "insurance.bif")).get_model()
    states = {n: list(bif.get_cpds(n).state_names[n]) for n in names}
    n_states = {n: len(states[n]) for n in names}

    # ---- Model -----------------------------------------------------------------
    # Every node is a OneHotCategorical, binaries included (2 states), so there
    # is a single value encoding and a single loss.
    variables = {
        n: ConceptVariable(n, distribution=OneHotCategorical, size=n_states[n])
        for n in names
    }
    factors = []
    for n in names:
        parents = dataset.graph.get_predecessors(n)
        if not parents:
            parametrization = {"logits": LearnablePrior(n_states[n])}
        else:
            # Parents are concatenated one-hot on the last axis; an MLP rather
            # than a linear layer, so interactions between parents are learnable.
            in_features = sum(n_states[p] for p in parents)
            parametrization = {"logits": nn.Sequential(
                nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, n_states[n]),
            )}
        factors.append(ParametricCPD(
            variables[n],
            parametrization=parametrization,
            parents=[variables[p] for p in parents],
        ))
    concept_model = BayesianNetwork(variables=list(variables.values()), factors=factors)

    # ---- Training ----------------------------------------------------------------
    # Teacher forcing with p_int=1.0: every CPD sees the true values of its
    # parents, so this fits each conditional by maximum likelihood.
    train_engine = DeterministicInference(concept_model, p_int=1.0)
    train_engine.train()
    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    targets = {n: F.one_hot(index[n], n_states[n]).float() for n in names}

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = train_engine.query(query=targets, evidence={})
        loss = sum(F.cross_entropy(out.logits[n], index[n]) for n in names)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}: loss {loss.item():.3f}")
    concept_model.eval()

    # ---- Inference ---------------------------------------------------------------
    n_samples = 50_000
    engines = {
        "Deterministic": DeterministicInference(concept_model),
        "Independent": IndependentInference(concept_model),
        "Ancestral": AncestralSamplingInference(concept_model, initial_temperature=0.1),
        "Rejection": RejectionSampling(concept_model, n_samples=n_samples),
        "Importance": ImportanceSampling(
            concept_model, MutilatedNetworkProposal(concept_model),
            n_samples=n_samples, initial_temperature=0.1,
        ),
        "Belief prop.": BeliefPropagation(concept_model, iters=50, damping=0.3),
        "Var. elim.": PgmpyVariableElimination(concept_model),
    }
    # Samples an Ancestral engine draws: one per row of replicated evidence.
    ancestral_rows = 5000

    results = {}
    for q_name, ((q_var, q_label), evidence_labels) in QUERIES.items():
        q_state = states[q_var].index(q_label)
        evidence_idx = {v: states[v].index(s) for v, s in evidence_labels.items()}

        # Empirical: count(Q=q and E=e) / count(E=e) over the training set.
        mask = torch.ones(len(concepts), dtype=torch.bool)
        for v, s in evidence_idx.items():
            mask &= index[v] == s
        support = int(mask.sum())
        empirical = (index[q_var][mask] == q_state).float().mean().item()
        prior = (index[q_var] == q_state).float().mean().item()

        # {estimator: (mean, standard deviation)}. The empirical frequency is a
        # count over a fixed training set, so it has no run-to-run spread.
        row = {"Empirical": (empirical, 0.0)}
        with torch.no_grad():
            for label, engine in engines.items():
                rows = ancestral_rows if label == "Ancestral" else 1
                runs = []
                for run in range(N_RUNS):
                    seed_everything(run)
                    runs.append(engine_probability(
                        engine, q_var, q_state, evidence_idx, n_states, rows,
                    ))
                # ``statistics`` rather than a float32 tensor: identical runs must
                # give a standard deviation of exactly 0, and float32 rounding of
                # the mean alone would leave a spurious ~3e-8.
                row[label] = (statistics.fmean(runs), statistics.stdev(runs))
        results[q_name] = (q_var, q_label, evidence_labels, support, prior, row)

    # ---- Report ------------------------------------------------------------------
    for q_name, (q_var, q_label, evidence_labels, support, prior, row) in results.items():
        evidence_str = ", ".join(f"{v}={s}" for v, s in evidence_labels.items())
        print(f"\n=== {q_name}: P({q_var}={q_label} | {evidence_str}) "
              f"[{support} training rows; prior P({q_var}={q_label}) = {prior:.3f}] ===")
        for label, (mean, std) in row.items():
            print(f"{label:>14} | {mean:.3f} ± {std:.1g}")

    plot(results, Path(__file__).with_name("insurance_query_engines.png"))


def plot(results, path):
    """One panel per query, one bar per estimator."""
    surface, ink, ink_muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"empirical": "#8f8e89", "forward": "#eb6834", "posterior": "#2a78d6"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=surface)
    for ax, (q_name, (q_var, q_label, evidence_labels, support, _, row)) in zip(axes, results.items()):
        labels = list(row)
        means = [mean for mean, _ in row.values()]
        stds = [std for _, std in row.values()]
        kinds = ["empirical" if l == "Empirical"
                 else "forward" if l in FORWARD_ENGINES else "posterior" for l in labels]
        ax.set_facecolor(surface)
        ax.bar(range(len(labels)), means, width=0.72,
               color=[colors[k] for k in kinds], zorder=3)
        ax.axhline(row["Empirical"][0], color=ink_muted, linestyle="--", linewidth=1, zorder=2)
        # Error bars only where the answer varies across runs: a zero-length bar
        # on a deterministic engine would just add a stray cap.
        varying = [i for i, std in enumerate(stds) if std > 0]
        ax.errorbar(varying, [means[i] for i in varying], yerr=[stds[i] for i in varying],
                    fmt="none", ecolor=ink, elinewidth=1.2, capsize=4, zorder=4)
        for i, (mean, std) in enumerate(zip(means, stds)):
            # One significant digit, so a small but real spread (±0.0002) is not
            # rounded away to ±0.000.
            text = f"{mean:.3f}\n±{std:.1g}" if std > 0 else f"{mean:.3f}"
            ax.text(i, mean + std + 0.015, text, ha="center", va="bottom",
                    fontsize=8.5, color=ink, linespacing=1.1)

        evidence_str = ", ".join(f"{v}={s}" for v, s in evidence_labels.items())
        ax.set_title(f"{q_name}: P({q_var}={q_label} | {evidence_str})\n"
                     f"empirical from {support} training rows",
                     fontsize=10, color=ink, loc="left")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", color=ink, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
        ax.tick_params(axis="y", colors=ink_muted, labelsize=9)
        ax.tick_params(axis="x", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(ink_muted)
    axes[0].set_ylabel("Probability", color=ink, fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
    fig.legend(handles, ["Empirical (training set)",
                         "Forward engine (parents → children only)",
                         "Posterior engine (conditions on all evidence)"],
               loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=ink, bbox_to_anchor=(0.5, 1.0))
    q_var, q_label, _, _, prior, _ = results["Q2"]
    fig.text(0.5, 0.005,
             f"Bars: mean over {N_RUNS} runs. Error bars: ±1 standard deviation, shown only "
             "for engines whose answer changes between runs.\n",
             ha="center", fontsize=9, color=ink_muted, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=surface)
    print(f"\nSaved plot to {path}")
    plt.show()


if __name__ == "__main__":
    main()
