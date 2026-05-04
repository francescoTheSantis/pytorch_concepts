"""
Pyro mid-level API — Asia Bayesian Network
==========================================

This example demonstrates building, training, and querying the classic Asia
Bayesian Network using the Pyro-backed mid-level API with real bnlearn data.

Asia DAG (8 binary concept nodes):

    input (32-dim embedding from autoencoder)
        ├─► asia  ─────────────────► tub  ─────► either ─► xray
        └─► smoke ─► lung ──────────────────────► either ─► dysp
                  └─► bronc ──────────────────────────────► dysp

Training (one fresh PGM per engine):
  - ELBOInference         — Pyro ELBO / variational inference
  - DeterministicInference — standard forward pass + BCE loss

  Evidence: input embedding from the dataset.
  Query:    all 8 Asia concept nodes.

Testing (after each training, using all test-time engines):
  A. Forward query        — predict all nodes from the input embedding.
     Engines: DeterministicInference, AncestralSamplingInference,
              ImportanceQuery, ExactDiscreteQuery
  B. Backward query       — infer causes (smoke, bronc) given symptoms
     (dysp=1, xray=1); requires posterior engines.
     Engines: ImportanceQuery, ExactDiscreteQuery
  C. Hidden-variable query — infer intermediaries (either, lung, tub) given
     a positive X-ray result; requires posterior engines.
     Engines: ImportanceQuery, ExactDiscreteQuery

Comparison: predicted marginals vs. empirical marginals from the data.

Note on MCMCQuery: The Asia network is fully discrete (all Bernoulli), so
NUTS/HMC—which requires continuous latents—is not applicable here. Use
ExactDiscreteQuery for exact posterior inference over discrete latents.
"""

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

import pyro

from torch_concepts.distributions import Delta
from torch_concepts.nn.modules.mid.models.variable import ConceptVariable, LatentVariable
from torch_concepts.nn.modules.mid.models.cpd import ParametricCPD
from torch_concepts.nn.modules.mid.models.probabilistic_model import BayesianNetwork
from torch_concepts.nn.modules.mid.inference.deterministic import DeterministicInference
from torch_concepts.nn.modules.mid.inference.ancestral import AncestralSamplingInference
from torch_concepts.nn.modules.mid.inference.elbo import ELBOInference
from torch_concepts.nn.modules.mid.inference.importance import ImportanceQuery
from torch_concepts.nn.modules.mid.inference.exact_discrete import ExactDiscreteQuery
from torch_concepts.data.datamodules.bnlearn import BnLearnDataModule

# ---------------------------------------------------------------------------
# Reproducibility and Pyro setup
# ---------------------------------------------------------------------------
torch.manual_seed(42)
pyro.set_rng_seed(42)
pyro.settings.set(module_local_params=True)

# ---------------------------------------------------------------------------
# 1. Load the Asia Bayesian Network dataset
# ---------------------------------------------------------------------------
print("=" * 70)
print("Loading Asia Bayesian Network dataset ...")

dm = BnLearnDataModule(name="asia", seed=42, n_gen=10_000, batch_size=512)
dm.setup("fit")

concept_names = dm.dataset.concept_names   # e.g. ['asia','smoke','lung',...]
CONCEPT_IDX   = {name: i for i, name in enumerate(concept_names)}
N_CONCEPTS    = len(concept_names)
EMB_DIM       = dm.dataset.input_data.shape[1]   # 32

print(f"  Concepts  : {concept_names}")
print(f"  Splits    : train={dm.train_len}, val={dm.val_len}, test={dm.test_len}")
print(f"  Embed dim : {EMB_DIM}")

# ---------------------------------------------------------------------------
# 2. Empirical marginals from training data
# ---------------------------------------------------------------------------
train_concepts = dm.dataset.concepts[dm.trainset.indices].float()   # (Ntrain, 8)
empirical_marginals = train_concepts.mean(0)                        # P(concept_i = 1)

# Conditional empirical marginals for backward / hidden queries
dysp_idx   = CONCEPT_IDX["dysp"]
xray_idx   = CONCEPT_IDX["xray"]
smoke_idx  = CONCEPT_IDX["smoke"]
bronc_idx  = CONCEPT_IDX["bronc"]
lung_idx   = CONCEPT_IDX["lung"]
tub_idx    = CONCEPT_IDX["tub"]
either_idx = CONCEPT_IDX["either"]

mask_dysp1 = train_concepts[:, dysp_idx] == 1
mask_xray1 = train_concepts[:, xray_idx] == 1
mask_back  = mask_dysp1 & mask_xray1

emp_smoke_given_symptoms  = train_concepts[mask_back,  smoke_idx].mean()
emp_bronc_given_symptoms  = train_concepts[mask_back,  bronc_idx].mean()
emp_lung_given_xray       = train_concepts[mask_xray1, lung_idx].mean()
emp_tub_given_xray        = train_concepts[mask_xray1, tub_idx].mean()
emp_either_given_xray     = train_concepts[mask_xray1, either_idx].mean()

print("\nEmpirical unconditional marginals (training set):")
for name, marg in zip(concept_names, empirical_marginals.tolist()):
    print(f"  P({name:<6}) = {marg:.3f}")

print("\nEmpirical conditional marginals (training set):")
print(f"  P(smoke  | dysp=1, xray=1) = {emp_smoke_given_symptoms:.3f}")
print(f"  P(bronc  | dysp=1, xray=1) = {emp_bronc_given_symptoms:.3f}")
print(f"  P(lung   | xray=1)         = {emp_lung_given_xray:.3f}")
print(f"  P(tub    | xray=1)         = {emp_tub_given_xray:.3f}")
print(f"  P(either | xray=1)         = {emp_either_given_xray:.3f}")

# ---------------------------------------------------------------------------
# 3. PGM factory — fresh BayesianNetwork with Asia DAG topology
# ---------------------------------------------------------------------------

def build_asia_pgm() -> BayesianNetwork:
    """Return a freshly initialised BayesianNetwork with the Asia DAG."""
    pyro.clear_param_store()

    # Variables
    input_var  = LatentVariable("input",  distribution=Delta,    size=EMB_DIM)
    asia_var   = ConceptVariable("asia",   distribution=Bernoulli, size=1)
    smoke_var  = ConceptVariable("smoke",  distribution=Bernoulli, size=1)
    tub_var    = ConceptVariable("tub",    distribution=Bernoulli, size=1)
    lung_var   = ConceptVariable("lung",   distribution=Bernoulli, size=1)
    bronc_var  = ConceptVariable("bronc",  distribution=Bernoulli, size=1)
    either_var = ConceptVariable("either", distribution=Bernoulli, size=1)
    xray_var   = ConceptVariable("xray",   distribution=Bernoulli, size=1)
    dysp_var   = ConceptVariable("dysp",   distribution=Bernoulli, size=1)

    variables = [
        input_var, asia_var, smoke_var, tub_var, lung_var,
        bronc_var, either_var, xray_var, dysp_var,
    ]

    # CPDs — Asia DAG structure.
    # Root nodes (asia, smoke) receive the 32-dim input embedding.
    # Non-root nodes receive their concept-level DAG parents' activated outputs.
    cpd_input  = ParametricCPD(concept="input",  parametrization=nn.Identity())
    cpd_asia   = ParametricCPD(concept="asia",   parametrization=nn.Linear(EMB_DIM, 1),
                               parents=[input_var])
    cpd_smoke  = ParametricCPD(concept="smoke",  parametrization=nn.Linear(EMB_DIM, 1),
                               parents=[input_var])
    cpd_tub    = ParametricCPD(concept="tub",    parametrization=nn.Linear(1, 1),
                               parents=[asia_var])
    cpd_lung   = ParametricCPD(concept="lung",   parametrization=nn.Linear(1, 1),
                               parents=[smoke_var])
    cpd_bronc  = ParametricCPD(concept="bronc",  parametrization=nn.Linear(1, 1),
                               parents=[smoke_var])
    cpd_either = ParametricCPD(concept="either", parametrization=nn.Linear(2, 1),
                               parents=[lung_var, tub_var])
    cpd_xray   = ParametricCPD(concept="xray",   parametrization=nn.Linear(1, 1),
                               parents=[either_var])
    cpd_dysp   = ParametricCPD(concept="dysp",   parametrization=nn.Linear(2, 1),
                               parents=[either_var, bronc_var])

    factors = [
        cpd_input, cpd_asia, cpd_smoke, cpd_tub, cpd_lung,
        cpd_bronc, cpd_either, cpd_xray, cpd_dysp,
    ]
    return BayesianNetwork(variables=variables, factors=factors)


# ---------------------------------------------------------------------------
# 4. Training — ELBOInference (variational / Pyro ELBO)
# ---------------------------------------------------------------------------

def train_with_elbo(n_epochs: int = 200, lr: float = 1e-3) -> BayesianNetwork:
    """Train a fresh Asia PGM using ELBO (variational inference)."""
    pgm    = build_asia_pgm()
    engine = ELBOInference(pgm, num_particles=4)
    optim  = torch.optim.Adam(engine.parameters(), lr=lr)

    pgm.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for batch in dm.train_dataloader():
            x_b = batch["inputs"]["x"].cpu()      # (B, 32) — move to CPU
            c_b = batch["concepts"]["c"].float()  # (B, 8)
            targets = {
                name: c_b[:, CONCEPT_IDX[name], None]
                for name in concept_names
            }

            optim.zero_grad()
            result = engine.query(
                query=concept_names,
                evidence={"input": x_b},
                targets=targets,
            )
            result.loss.backward()   # differentiable negative ELBO
            optim.step()

            epoch_loss += result.loss.item()
            n_batches  += 1

        if (epoch + 1) % 50 == 0:
            avg = epoch_loss / n_batches
            print(f"    epoch {epoch + 1:3d}/{n_epochs}  neg-ELBO = {avg:.4f}")

    pgm.eval()
    return pgm


# ---------------------------------------------------------------------------
# 5. Training — DeterministicInference (forward pass + BCE)
# ---------------------------------------------------------------------------

def train_with_deterministic(n_epochs: int = 200, lr: float = 1e-3) -> BayesianNetwork:
    """Train a fresh Asia PGM via a deterministic forward pass with BCE loss."""
    pgm     = build_asia_pgm()
    engine  = DeterministicInference(pgm)
    optim   = torch.optim.Adam(pgm.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    pgm.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for batch in dm.train_dataloader():
            x_b = batch["inputs"]["x"].cpu()      # (B, 32) — move to CPU
            c_b = batch["concepts"]["c"].float()  # (B, 8)
            optim.zero_grad()
            result = engine.query(
                query=concept_names,
                evidence={"input": x_b},
                return_logits=True,
                return_probs=False,
                device="cpu",
            )
            # result.logits: (B, N_CONCEPTS) in concept_names order
            loss = loss_fn(result.logits, c_b)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if (epoch + 1) % 50 == 0:
            avg = epoch_loss / n_batches
            print(f"    epoch {epoch + 1:3d}/{n_epochs}  BCE = {avg:.4f}")

    pgm.eval()
    return pgm


# ---------------------------------------------------------------------------
# 6. Test queries
# ---------------------------------------------------------------------------

def _fmt(name: str, model_val: float, empirical_val: float) -> str:
    return f"      P({name:<6}) model={model_val:.3f}  empirical={empirical_val:.3f}"


def run_test_queries(pgm: BayesianNetwork, trained_with: str) -> None:
    """Run forward, backward, and hidden-variable queries on a trained PGM."""
    print(f"\n{'─' * 70}")
    print(f"  Test queries — PGM trained with {trained_with}")
    print("─" * 70)

    # Grab a test mini-batch (16 samples for forward queries)
    test_batch = next(iter(dm.test_dataloader()))
    x_test = test_batch["inputs"]["x"][:16].cpu()    # (16, 32) — move to CPU

    # Single sample for posterior engines (ImportanceQuery / ExactDiscreteQuery)
    x_single = x_test[:1]                      # (1, 32)

    # ── A. Forward query ──────────────────────────────────────────────────────
    print("\n  [A] Forward query: predict all nodes from the input embedding")

    # DeterministicInference
    det_engine  = DeterministicInference(pgm)
    det_result  = det_engine.query(
        query=concept_names,
        evidence={"input": x_test},
        return_probs=True,
        device="cpu",
    )
    pred_det = det_result.probs.mean(0)
    print("\n    DeterministicInference:")
    for name, pred, emp in zip(concept_names,
                               pred_det.tolist(),
                               empirical_marginals.tolist()):
        print(_fmt(name, pred, emp))

    # AncestralSamplingInference
    anc_engine = AncestralSamplingInference(pgm)
    anc_result = anc_engine.query(
        query=concept_names,
        evidence={"input": x_test},
        return_probs=True,
        device="cpu",
    )
    pred_anc = anc_result.probs.mean(0)
    print("\n    AncestralSamplingInference (samples used as proxies for probs):")
    for name, pred, emp in zip(concept_names,
                               pred_anc.tolist(),
                               empirical_marginals.tolist()):
        print(_fmt(name, pred, emp))

    # ImportanceQuery — forward direction (evidence = input only)
    imp_query = ImportanceQuery(pgm)
    imp_fwd   = imp_query.query(
        variables=concept_names,
        evidence={"input": x_single},
        num_samples=300,
    )
    print("\n    ImportanceQuery (forward, 300 samples, single test point):")
    for name, emp in zip(concept_names, empirical_marginals.tolist()):
        pred = imp_fwd.samples[name].float().mean().item()
        print(_fmt(name, pred, emp))

    # ExactDiscreteQuery — forward direction
    exact_query = ExactDiscreteQuery(pgm)
    exact_fwd   = exact_query.query(
        variables=concept_names,
        evidence={"input": x_single},
        num_samples=300,
    )
    print("\n    ExactDiscreteQuery (forward, 300 samples, single test point):")
    for name, emp in zip(concept_names, empirical_marginals.tolist()):
        if name in exact_fwd.samples:
            pred = exact_fwd.samples[name].float().mean().item()
            print(_fmt(name, pred, emp))

    # ── B. Backward query ─────────────────────────────────────────────────────
    print("\n  [B] Backward query: P(smoke, bronc | input, dysp=1, xray=1)")
    back_evidence = {
        "input": x_single,
        "dysp":  torch.ones(1, 1),
        "xray":  torch.ones(1, 1),
    }
    back_vars = ["smoke", "bronc", "asia"]

    # ImportanceQuery
    imp_back = imp_query.query(
        variables=back_vars,
        evidence=back_evidence,
        num_samples=500,
    )
    print("\n    ImportanceQuery (backward, 500 samples):")
    print(_fmt("smoke", imp_back.samples["smoke"].float().mean().item(),
               emp_smoke_given_symptoms.item()))
    print(_fmt("bronc", imp_back.samples["bronc"].float().mean().item(),
               emp_bronc_given_symptoms.item()))

    # ExactDiscreteQuery
    exact_back = exact_query.query(
        variables=back_vars,
        evidence=back_evidence,
        num_samples=500,
    )
    if "smoke" in exact_back.samples and "bronc" in exact_back.samples:
        print("\n    ExactDiscreteQuery (backward, 500 samples):")
        print(_fmt("smoke", exact_back.samples["smoke"].float().mean().item(),
                   emp_smoke_given_symptoms.item()))
        print(_fmt("bronc", exact_back.samples["bronc"].float().mean().item(),
                   emp_bronc_given_symptoms.item()))

    # ── C. Hidden-variable query ──────────────────────────────────────────────
    print("\n  [C] Hidden-variable query: P(either, lung, tub | input, xray=1)")
    hidden_evidence = {
        "input": x_single,
        "xray":  torch.ones(1, 1),
    }
    hidden_vars = ["either", "lung", "tub"]

    # ImportanceQuery
    imp_hidden = imp_query.query(
        variables=hidden_vars,
        evidence=hidden_evidence,
        num_samples=500,
    )
    print("\n    ImportanceQuery (hidden, 500 samples):")
    print(_fmt("either", imp_hidden.samples["either"].float().mean().item(),
               emp_either_given_xray.item()))
    print(_fmt("lung",   imp_hidden.samples["lung"].float().mean().item(),
               emp_lung_given_xray.item()))
    print(_fmt("tub",    imp_hidden.samples["tub"].float().mean().item(),
               emp_tub_given_xray.item()))

    # ExactDiscreteQuery
    exact_hidden = exact_query.query(
        variables=hidden_vars,
        evidence=hidden_evidence,
        num_samples=500,
    )
    if "either" in exact_hidden:
        print("\n    ExactDiscreteQuery (hidden, 500 samples):")
        print(_fmt("either", exact_hidden["either"].float().mean().item(),
                   emp_either_given_xray.item()))
        if "lung" in exact_hidden:
            print(_fmt("lung", exact_hidden["lung"].float().mean().item(),
                       emp_lung_given_xray.item()))
        if "tub" in exact_hidden:
            print(_fmt("tub", exact_hidden["tub"].float().mean().item(),
                       emp_tub_given_xray.item()))


# ---------------------------------------------------------------------------
# 7. Main — train with each engine, then test
# ---------------------------------------------------------------------------

TRAINING_ENGINES = {
    "ELBOInference":          train_with_elbo,
    "DeterministicInference": train_with_deterministic,
}

if __name__ == "__main__":
    for engine_name, train_fn in TRAINING_ENGINES.items():
        print(f"\n{'=' * 70}")
        print(f"  Training with {engine_name} ...")
        print("=" * 70)
        pgm = train_fn(n_epochs=200)
        run_test_queries(pgm, engine_name)

    print("\n" + "=" * 70)
    print("Done.")
