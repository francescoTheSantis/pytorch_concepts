"""Concept Embedding Model (CEM) on the ASIA dataset.

Same architecture as ``0.3_concept_embedding_model.py``, on bnlearn's ASIA
(chest-clinic) network, loaded with :class:`~torch_concepts.data.BnLearnDataset`.
"""

import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.distributions import Bernoulli, Normal

import torch_concepts as pyc
from torch_concepts import seed_everything, EmbeddingVariable, ConceptVariable
from torch_concepts.distributions import Delta
from torch_concepts.data import BnLearnDataset
from torch_concepts.nn import MLP, LinearEmbeddingToConcept, \
    MixConceptEmbeddings, ParametricCPD, BayesianNetwork, \
    DeterministicInference, DefaultActivation, LearnablePrior, Sequential, \
    AncestralSamplingInference, AnnealedLangevinDynamics, MarkovNetwork, \
    ParametricPotential

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
AUTOENCODER_NOISE = 0.9

# Override with CE2BM_DEVICE=cpu|cuda|cuda:1|mps. The sampler follows the energy's
# own device (`LangevinDynamics._dtype_device`), so moving the models and the data
# is enough -- nothing else needs a device argument.
DEVICE = torch.device(os.environ.get("CE2BM_DEVICE")
                      or ("cuda" if torch.cuda.is_available() else "cpu"))


def np_(t):
    """Detached CPU numpy, for the sklearn metrics."""
    return torch.as_tensor(t).detach().cpu().numpy()

# ---- EBM over the mix embeddings (NCSN, see test_ncsn.py) -------------------
EBM_EVAL_ROWS = 1000      # rows used for the repaint row; None = full dataset
# None keeps the intervened embeddings clamped for the whole ladder -- per-row
# evidence, since their values are *known*. Set an int to release them at that
# rung instead (test_ncsn.py's repaint). Measured on this data, holding to the
# end reproduces the anchors exactly (RMSE 0.000) and recovers the free
# embeddings better (0.925 vs 1.001 for guessing the mean); releasing at 30% of
# the ladder destroys the anchor (1.290) because sigma there still dwarfs the
# standardized data's own spread.
RELEASE_RUNG = None
# 60 rungs, not 10: consecutive levels must overlap, and the ratio that works in
# test_ncsn.py's 2-D toy (gamma 2.3) leaves the chain stranded in 28-D -- it
# arrives at each rung outside the region where that rung's score was fitted, so
# nothing pulls it in. gamma 1.11 here (Song & Ermon, technique 2).
N_LEVELS, NOISE_EMBEDDING = 60, 32     # noise_embedding_size must be even
EBM_SIGMA_MIN = 0.05      # in standardized units, where the data has std 1
EBM_EPOCHS, EBM_BATCH, EBM_HIDDEN = 3000, 512, 256
LANGEVIN_STEPS = 30
# Step size is derived, not hardcoded: the sampler uses alpha_i = eps*(sigma_i/sigma_min)^2,
# so fixing eps = RATIO * sigma_min^2 gives alpha_i = RATIO * sigma_i^2 -- the step
# always tracks the local noise scale. RATIO 0.2 reproduces test_ncsn.py's tuned
# pair (sigma_min=0.01, eps=2e-5) while porting to any data scale; its absolute
# 2e-5 here would give a step 4x the coarsest rung's variance and diverge.
LANGEVIN_STEP_RATIO = 0.2


def split_mix(flat, mix_names, emb_dims):
    """(batch, n_concepts * emb_dims) -> {mix_name: (batch, emb_dims)}."""
    return {n: flat[:, j * emb_dims:(j + 1) * emb_dims] for j, n in enumerate(mix_names)}


def energy_net(in_features: int, hidden: int = 128) -> torch.nn.Module:
    """One clique's energy. Plain: `noise_conditioned=True` embeds sigma and
    concatenates it onto the scope values before this net is called.

    SiLU is load-bearing -- a ReLU energy has a piecewise-constant score, so
    there is no curvature inside a basin for score matching to fit.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, 1),
    )


def train_ebm(mix_data, mix_names, emb_dims):
    """Fit a noise-conditioned EBM to the CEM's mix embeddings.

    One clique over all 7 mix variables, so the energy sees the whole joint --
    that coupling is exactly what the CEM lacks, since each mix_i is computed
    independently from its own input features.

    Returns ``(repaint, sample)``: ``repaint(m_int, mask)`` resamples the
    un-masked embeddings while holding the masked ones, and ``sample(n)`` draws
    unconditionally. Both speak the CEM's units -- standardization is internal.
    """
    # The raw embeddings are nowhere near standardized (per-dimension std ranges
    # over an order of magnitude, and one dimension is constant), so a single
    # isotropic noise ladder cannot serve every dimension. Fit in z-space and
    # convert back at the boundary. A constant dimension keeps sd 1, mapping it
    # to a constant 0 rather than dividing by ~0.
    device = mix_data.device
    mu = mix_data.mean(0)
    sd = mix_data.std(0)
    sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
    z_data = (mix_data - mu) / sd

    mix_vars = [EmbeddingVariable(n, distribution=Normal, size=emb_dims)
                for n in mix_names]
    factor = ParametricPotential(
        name="mix_clique",
        scope=mix_vars,
        parametrization=energy_net(len(mix_names) * emb_dims + NOISE_EMBEDDING,
                                   hidden=EBM_HIDDEN),
        noise_conditioned=True,
        noise_embedding_size=NOISE_EMBEDDING,
    )
    mrf = MarkovNetwork(variables=mix_vars, factors=[factor]).to(device)

    # sigma_max = the largest pairwise distance, so the coarsest rung can bridge
    # any two modes (Song & Ermon, technique 1).
    sub = z_data[torch.randperm(len(z_data), device=device)[:500]]
    sigma_max = torch.cdist(sub, sub).max()
    sigma_min = torch.tensor(EBM_SIGMA_MIN, device=device)
    sigmas = torch.exp(torch.linspace(float(sigma_max.log()), float(sigma_min.log()),
                                      N_LEVELS, device=device))
    print(f"\nEBM ladder: {N_LEVELS} rungs, sigma {sigma_max:.2f} -> {sigma_min:.3f}"
          f", gamma {float(sigmas[0] / sigmas[1]):.3f}")

    optimizer = torch.optim.AdamW(mrf.parameters(), lr=1e-3)
    mrf.train()
    for epoch in range(EBM_EPOCHS):
        clean = z_data[torch.randint(0, len(z_data), (EBM_BATCH,), device=device)]
        sigma = sigmas[torch.randint(0, N_LEVELS, (EBM_BATCH,), device=device)]
        z = torch.randn_like(clean)
        perturbed = clean + sigma.reshape(-1, 1) * z

        scores = mrf.compute_score(split_mix(perturbed, mix_names, emb_dims),
                                   sigma=sigma)
        score_value = torch.cat([scores[n] for n in mix_names], dim=-1)
        # ||sigma * s(x, sigma) + z||^2
        loss = (sigma.reshape(-1, 1) * score_value + z).pow(2).sum(-1).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mrf.parameters(), 1.0)
        optimizer.step()

        if epoch % 500 == 0 or epoch == EBM_EPOCHS - 1:
            print(f"  EBM epoch {epoch:5d} | loss {loss.item():8.4f}")

    mrf.eval()
    # grad_clip must stay None: the rung step sizes span orders of magnitude and
    # one clip value would flatten every rung to the same step.
    step_size = LANGEVIN_STEP_RATIO * float(sigma_min) ** 2
    sampler = AnnealedLangevinDynamics(mrf, sigmas=sigmas, step_size=step_size,
                                       steps=LANGEVIN_STEPS, grad_clip=None)

    @torch.no_grad()
    def repaint(m_int, mask):
        """Hold the masked embeddings, resample the rest. Per row, per variable."""
        # `release=None` holds for the whole ladder -- the intervened embeddings
        # are known values, so this is conditional sampling p(free | held), and
        # the held columns come back bit-for-bit.
        release = ({} if RELEASE_RUNG is None
                   else {'release': {n: RELEASE_RUNG for n in mix_names}})
        out = sampler.query(
            query=split_mix((m_int - mu) / sd, mix_names, emb_dims),
            clamp_mask={n: mask[:, j:j + 1].bool() for j, n in enumerate(mix_names)},
            **release,
        )
        return torch.as_tensor(out.samples[mix_names]) * sd + mu

    @torch.no_grad()
    def sample(n):
        return torch.as_tensor(
            sampler.query(query=mix_names, n_samples=n).samples[mix_names]) * sd + mu

    return repaint, sample


@torch.no_grad()
def intervention_curves(engine, x, c_gt, y_gt, concept_names, task_name, aux_names,
                        mix_names, p_grid, generator, repaint, emb_dims, ebm_rows):
    """Accuracy of every node against the intervention rate.

    The intervened concepts are clamped as evidence, so they reach the task head
    *and* -- through each ``mix_i`` -- the auxiliary re-predictors. The concept
    curves are linear by construction (an intervened concept is its own ground
    truth); the task and auxiliary curves are real measurements.

    The ``ebm_*`` curves add the repaint procedure: each row's *intervened* mix
    embeddings are held through the coarse noise rungs while the rest are
    resampled from the EBM's learned joint, then the auxiliary and task heads are
    read off those repainted embeddings.
    """
    probs = engine.query(list(concept_names), evidence={'input': x}).probs

    # Safely format the output probabilities into a (batch, 7) tensor
    c_hat = torch.stack([
        probs[name].view(-1) for name in concept_names
    ], dim=-1)

    ebm_keys = [f"ebm_{n}" for n in [*concept_names, task_name]]
    curves = {name: [] for name in [*concept_names, task_name, *aux_names, *ebm_keys]}
    for p_int in p_grid:
        # Drawn on the CPU generator, then moved: the random sequence -- and so
        # the results -- stay identical whichever device the run uses.
        mask = (torch.rand(c_hat.shape, generator=generator) < p_int).float().to(c_hat.device)
        c_eff = mask * c_gt + (1 - mask) * c_hat
        out = engine.query([task_name, *aux_names, *mix_names], evidence={
            'input': x,
            **{name: c_eff[:, i:i + 1] for i, name in enumerate(concept_names)},
        })
        for i, name in enumerate(concept_names):
            curves[name].append(accuracy_score(np_(c_gt[:, i]), np_(c_eff[:, i] > 0.5)))
        curves[task_name].append(
            accuracy_score(np_(y_gt), np_(out.probs[task_name] > 0.5)))
        for i, aux_name in enumerate(aux_names):
            curves[aux_name].append(
                accuracy_score(np_(c_gt[:, i]), np_(out.probs[aux_name] > 0.5)))

        # ---- repaint: hold the intervened embeddings, resample the rest ------
        rows = ebm_rows
        m_int = torch.as_tensor(out.value[mix_names])[rows]
        m_new = repaint(m_int, mask[rows])
        ebm_out = engine.query([task_name, *aux_names],
                               evidence=split_mix(m_new, mix_names, emb_dims))
        for i, (name, aux_name) in enumerate(zip(concept_names, aux_names)):
            curves[f"ebm_{name}"].append(
                accuracy_score(np_(c_gt[rows][:, i]), np_(ebm_out.probs[aux_name] > 0.5)))
        curves[f"ebm_{task_name}"].append(
            accuracy_score(np_(y_gt[rows]), np_(ebm_out.probs[task_name] > 0.5)))
    return curves


def plot_nodes(curves, p_grid, baselines, concept_names, task_name, title, path):
    """One panel per node, all on a single row.

    Every way of predicting that node shares its panel, so the comparison is
    read within a subplot rather than across rows: the node as the CEM predicts
    it, the auxiliary head reading the CEM's own mix embedding, the same head
    reading the EBM-repainted embedding, and the majority-class baseline.
    """
    nodes = [*concept_names, task_name]
    fig, axes = plt.subplots(1, len(nodes), figsize=(2.4 * len(nodes), 3.6),
                             sharey=True, squeeze=False)
    for col, name in enumerate(nodes):
        ax, is_task = axes[0][col], name == task_name
        ax.plot(p_grid, curves[name], marker='o', color='tab:blue',
                label='task predictor' if is_task else 'intervened concept')
        if not is_task:   # the task has no auxiliary counterpart
            ax.plot(p_grid, curves[f"aux_{name}"], marker='s', color='tab:orange',
                    label='auxiliary (CEM mix)')
        ax.plot(p_grid, curves[f"ebm_{name}"], marker='D', color='tab:red',
                label='auxiliary (EBM repaint)')
        ax.axhline(baselines[name], color='gray', linestyle='--', linewidth=1,
                   label='majority baseline')
        ax.set(title=name + (' (task)' if is_task else ''), xlabel='$p_{int}$',
               ylim=(0, 1.02))
    axes[0][0].set_ylabel('accuracy')
    # Collect labels across panels: the task panel carries the only 'task
    # predictor' handle, the concept panels the only 'auxiliary (CEM mix)' one.
    handles, labels = [], []
    for ax in (axes[0][0], axes[0][-1]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(tag, concept_subset=None):
    """The whole CEM -> EBM -> repaint pipeline for one concept set.

    ``concept_subset`` drops nodes from the *concept set* only -- the input
    embeddings are unchanged (same cache), so information about a dropped node
    is still present in the input but no longer has a concept of its own. With
    ``bronc`` removed, a direct parent of ``dysp`` is missing from the
    bottleneck, so the remaining mix embeddings have to carry it: exactly the
    concept-incompleteness regime where embeddings buy something that concept
    scores cannot.
    """
    seed_everything(42)

    emb_dims = 4
    latent_dims = 20
    n_epochs = 500
    n_samples = 5000
    concept_reg = 1
    aux_reg = 1

    dataset = BnLearnDataset(
        name='asia', seed=42, n_gen=n_samples,
        root=str(Path.cwd() / 'data' / f'asia_ae_noise_{AUTOENCODER_NOISE}'),
        autoencoder_kwargs={'noise': AUTOENCODER_NOISE},
        concept_subset=concept_subset,
    )
    *concept_names, task_name = dataset.annotations.labels
    aux_names = [f"aux_{n}" for n in concept_names]
    mix_names = [f"mix_{n}" for n in concept_names]
    print(f"\n{'=' * 70}\n{tag}: concepts {concept_names} -> task {task_name!r}"
          f"   [device: {DEVICE}]\n{'=' * 70}")
    x_train = dataset.input_data.to(DEVICE)
    all_concepts = dataset.concepts.float().to(DEVICE)
    c_train, y_train = all_concepts[:, :-1], all_concepts[:, -1:]
    rates = all_concepts.mean(0)
    baselines = dict(zip([*concept_names, task_name],
                         torch.maximum(rates, 1 - rates).tolist()))
    # An auxiliary node re-predicts its concept's own ground truth, so it
    # inherits that concept's majority-class baseline; likewise every node read
    # off the repainted embeddings.
    baselines.update({aux: baselines[name] for name, aux in zip(concept_names, aux_names)})
    baselines.update({f"ebm_{n}": baselines[n] for n in [*concept_names, task_name]})

    ######### Variables
    input_var = EmbeddingVariable("input", distribution=Delta, size=x_train.shape[1])
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=latent_dims)
    embs = EmbeddingVariable([f"emb_{n}" for n in concept_names],
                             distribution=Delta, shape=(1, emb_dims))
    
    concepts = ConceptVariable(concept_names, distribution=Bernoulli)
    
    mixs = EmbeddingVariable(mix_names, distribution=Delta, size=emb_dims)

    # Auxiliary concepts: the same 7 concepts predicted a second time, from the
    # mixed embeddings instead of the raw ones. They need their own names -- a
    # BayesianNetwork allows exactly one factor per variable -- and are childless
    # leaves, so the graph stays a DAG.
    aux_concepts = ConceptVariable(aux_names, distribution=Bernoulli)

    tasks = ConceptVariable(task_name, distribution=Bernoulli)

    ######### Layers
    backbone = MLP(input_size=x_train.shape[1], hidden_size=latent_dims,
                   n_layers=1, activation='leaky_relu')

    
    # `embs` is a list of 7 independent Variables (one per concept), so
    # ParametricCPD deep-copies this module once per concept below -- each
    # copy must map latent -> its OWN (1, emb_dims), not all 7 concepts at once.
    emb_encoder = pyc.nn.Sequential(
        torch.nn.Linear(latent_dims, emb_dims),
        torch.nn.Unflatten(unflattened_size=(1, emb_dims), dim=1),
    )
    
    concept_encoder = pyc.nn.Sequential(
        LinearEmbeddingToConcept(in_embeddings=emb_dims, out_concepts=1),
        torch.nn.Flatten(),
        DefaultActivation(concepts[0], 'probs'),
    )
    
    # Mixes ONE concept's own predicted probability with its own embeddings
    mix_encoder = pyc.nn.Sequential(
        MixConceptEmbeddings(
            in_concepts=pyc.Annotations.empty(1, types=['binary']),
            in_embeddings=emb_dims,
        ),
        torch.nn.Flatten(start_dim=1),
    )

    # Re-predicts a concept from its OWN mixed embedding. Same shape as
    # `concept_encoder`, but reading `mix_i` rather than the raw `emb_i`:
    # `mix_encoder`'s trailing Flatten already yields (batch, emb_dims).
    aux_encoder = pyc.nn.Sequential(
        LinearEmbeddingToConcept(in_embeddings=emb_dims, out_concepts=1),
        torch.nn.Flatten(),
        DefaultActivation(aux_concepts[0], 'probs'),
    )

    task_predictor = Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(emb_dims * len(concept_names), 1),
        DefaultActivation(tasks, 'probs'),
    )
    
    ######### CPDs
    input_cpd = ParametricCPD(input_var, parametrization=LearnablePrior(input_var.size), parents=[])
    backbone_cpd = ParametricCPD(latent_var, parametrization=backbone, parents=[input_var])
    emb_cpd = ParametricCPD(embs, parametrization=emb_encoder, parents=[latent_var])
    
    concept_cpds = [
        ParametricCPD(
            concept, 
            parametrization={'probs': deepcopy(concept_encoder)},
            parents=[emb]
        )
        for concept, emb in zip(concepts, embs)
    ]
    
    mix_cpds = [
        ParametricCPD(mix, parametrization=deepcopy(mix_encoder), parents=[concept, emb])
        for mix, concept, emb in zip(mixs, concepts, embs)
    ]

    # Second child of each mix_i, alongside the task head.
    aux_cpds = [
        ParametricCPD(aux, parametrization={'probs': deepcopy(aux_encoder)}, parents=[mix])
        for aux, mix in zip(aux_concepts, mixs)
    ]

    y_predictor = ParametricCPD(
        tasks,
        parametrization={'probs': task_predictor},
        parents=mixs,
    )

    ########## Model
    concept_model = BayesianNetwork(
        variables=[input_var, latent_var, *embs, *concepts, *mixs, *aux_concepts, tasks],
        factors=[input_cpd, backbone_cpd, *emb_cpd, *concept_cpds, *mix_cpds, *aux_cpds, y_predictor],
    ).to(DEVICE)
    
    ########## Setting up inference and training query
    inference_engine = AncestralSamplingInference(concept_model, p_int=1)
    evidence = {'input': x_train}
    query_concepts = {name: c_train[:, i] for i, name in enumerate(concept_names)}
    query_concepts[task_name] = y_train
    # The query set is also what the engine evaluates: an aux node is a childless
    # leaf, so without this its CPD would simply never run.
    query_concepts.update({name: c_train[:, i] for i, name in enumerate(aux_names)})
    
    ########## Training
    optimizer = torch.optim.AdamW(concept_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    concept_model.train()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        cy_pred = inference_engine.query(query=query_concepts, evidence=evidence)
        c_pred = cy_pred.probs[tuple(concept_names)]
        y_pred = cy_pred.probs[task_name]
        aux_pred = cy_pred.probs[tuple(aux_names)]
        loss = (
            loss_fn(c_pred, c_train)
            + concept_reg * loss_fn(y_pred, y_train)
            + aux_reg * loss_fn(aux_pred, c_train)
        )

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            task_accuracy = accuracy_score(np_(y_train), np_(y_pred > 0.5))
            concept_accuracy = accuracy_score(np_(c_train), np_(c_pred > 0.5))
            print(f"Epoch {epoch}: Loss {loss.item():.2f} | Task Acc: {task_accuracy:.2f} | Concept Acc: {concept_accuracy:.2f}")

    # ---- Intervention sweep, one figure per input-noise level --------------
    eval_engine = AncestralSamplingInference(concept_model, p_int=0)
    p_grid = np.linspace(0.0, 1.0, 11)
    fig_dir = Path(__file__).parent / "figures" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)

    concept_model.eval()
    generator = torch.Generator().manual_seed(0)   # CPU, so draws are device-independent
    noise = torch.randn(x_train.shape, generator=generator).to(DEVICE)

    # How much of each concept survives the mix? Read at p_int=0, so mix_i is
    # built from the model's own belief -- training runs at p_int=1, where the
    # mix sees the ground-truth concept instead.
    with torch.no_grad():
        aux_check = eval_engine.query(query=[*concept_names, *aux_names],
                                      evidence=evidence).probs
    print("\nConcept accuracy -- predicted from emb_i vs re-predicted from mix_i:")
    for i, (name, aux_name) in enumerate(zip(concept_names, aux_names)):
        emb_acc = accuracy_score(np_(c_train[:, i]), np_(aux_check[name] > 0.5))
        mix_acc = accuracy_score(np_(c_train[:, i]), np_(aux_check[aux_name] > 0.5))
        print(f"  {name:8s} emb {emb_acc:.3f}  mix {mix_acc:.3f}"
              f"  diff {mix_acc - emb_acc:+.3f}")

    # ---- Second dataset: the CEM's mix embeddings, then an EBM over them ----
    # Concepts clamped to ground truth, so this is the manifold of *correct*
    # mix configurations -- the same regime the aux/task heads trained under.
    gt_evidence = {'input': x_train,
                   **{n: c_train[:, i:i + 1] for i, n in enumerate(concept_names)}}
    with torch.no_grad():
        mix_data = torch.as_tensor(
            eval_engine.query(mix_names, evidence=gt_evidence).value[mix_names])
    print(f"\nmix-embedding dataset: {tuple(mix_data.shape)}")

    repaint, sample_ebm = train_ebm(mix_data, mix_names, emb_dims)

    # Sanity check: a ladder or step size mismatched to the data scale shows up
    # here as samples orders of magnitude off, before the sweep gets confusing.
    drawn = sample_ebm(500)
    print(f"  data   mean {mix_data.mean():+.3f}  std {mix_data.std():.3f}"
          f"  min {mix_data.min():+.3f}  max {mix_data.max():+.3f}")
    print(f"  sampled mean {drawn.mean():+.3f}  std {drawn.std():.3f}"
          f"  min {drawn.min():+.3f}  max {drawn.max():+.3f}")

    ebm_rows = torch.arange(len(x_train), device=DEVICE)
    if EBM_EVAL_ROWS is not None:
        ebm_rows = torch.randperm(len(x_train),
                                  generator=generator)[:EBM_EVAL_ROWS].to(DEVICE)
    print(f"  repaint evaluated on {len(ebm_rows)} rows")

    for noise_level in NOISE_LEVELS:
        x_noisy = (1 - noise_level) * x_train + noise_level * noise
        curves = intervention_curves(eval_engine, x_noisy, c_train, y_train,
                                     concept_names, task_name, aux_names,
                                     mix_names, p_grid, generator,
                                     repaint, emb_dims, ebm_rows)

        stem = f"cem_asia_interventions_noise_{noise_level}.png"
        title = (rf'{tag} -- concept interventions, input noise $\lambda$'
                 rf' = {noise_level}')
        plot_nodes(curves, p_grid, baselines, concept_names, task_name,
                   title, fig_dir / stem)
        # Everything the plots need, so a layout change never costs a re-run.
        torch.save({'curves': curves, 'p_grid': p_grid, 'baselines': baselines,
                    'concept_names': concept_names, 'task_name': task_name,
                    'aux_names': aux_names, 'title': title},
                   fig_dir / stem.replace('.png', '.pt'))

        # Per node only. Averaging accuracy across concepts would pool variables
        # with baselines from 0.51 (smoke) to 0.99 (asia), so the mean says more
        # about which concepts are in the set than about the method.
        print(f"\nlambda = {noise_level}  ->  {stem}   (p_int = 0 / 0.5 / 1)")
        for name, acc in curves.items():
            print(f"  {name:12s} {acc[0]:.3f} / {acc[5]:.3f} / {acc[-1]:.3f}"
                  f"   baseline {baselines[name]:.3f}")


def main():
    # Full ASIA, then the concept-incomplete variant: `tub` and `bronc` are
    # dropped from the concept set (not from the input). `bronc` is a direct
    # parent of the task, so its information can only reach `dysp` through the
    # remaining concepts' embeddings -- the regime where an embedding carries
    # more than its own concept score.
    run_experiment("asia")
    run_experiment("asia_incomplete",
                   concept_subset=['asia', 'smoke', 'lung', 'either', 'xray', 'dysp'])


if __name__ == "__main__":
    main()