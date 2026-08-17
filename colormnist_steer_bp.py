"""ColorMNIST: a discrete concept bottleneck steered post-hoc by loopy belief propagation.

The image counterpart of ``CE2BM.py``. There, a CEM's *continuous* mix embeddings are
modelled post-hoc by an NCSN-style ``MarkovNetwork`` and repainted with Langevin after an
intervention; the result is only readable as accuracy curves. Here the bottleneck is
*discrete* -- three categorical variables -- the post-hoc model is a pairwise Markov
random field over exactly those variables, and the update after an intervention is
:class:`BeliefPropagation`. Steering is then visible as pixels.

Bottleneck (8 dims, one-hot throughout):

    parity(2)  supervised    color(2)  supervised    residual(4)  free

The decoder sees nothing but those 8 dims, so it has exactly 2*2*4 = 16 possible inputs
and its outputs are class prototypes. That is the point: no information can leak past the
discrete code, so intervening on ``parity`` is guaranteed to change the image.

The figure has four rows -- the original, its reconstruction, the naive parity swap
(colour and residual left stale), and the swap propagated through the MRF by loopy BP.

Run with ``python colormnist_steer_bp.py`` (~3 min on CPU). On a GPU, run
``COLORMNIST_DEVICE=cuda python colormnist_steer_bp.py`` and raise ``N_TRAIN`` (60_000 is
the whole MNIST train split) and ``N_EPOCHS``; nothing else is sized to the CPU budget.
"""

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

import pyro.distributions as pyro_dist

from torch_concepts import seed_everything, ConceptVariable, EmbeddingVariable
from torch_concepts.data.datasets.mnist import default_root, load_mnist
from torch_concepts.data.utils import colorize
from torch_concepts.distributions import Delta
from torch_concepts.nn import (AncestralSamplingInference, BayesianNetwork,
                               BeliefPropagation, DefaultActivation, LearnablePrior,
                               MarkovNetwork, ParametricCPD, ParametricPotential)

# Override with COLORMNIST_DEVICE=cpu|cuda|mps.
DEVICE = torch.device(os.environ.get("COLORMNIST_DEVICE")
                      or ("cuda" if torch.cuda.is_available() else "cpu"))

SEED = 42
N_TRAIN, N_TEST = 12_000, 2_000
# Colour agrees with parity 80% of the time. Not the deterministic map
# `ColorMNISTDataset(coloring={...})` offers -- a hard coupling would make the MRF's
# parity-colour factor a lookup and BP's answer trivial. At 0.8 the field has to learn a
# *soft* preference, which is what the marginals then reflect.
P_COLOR_AGREES = 0.8

K_PARITY, K_COLOR, K_RESIDUAL = 2, 2, 4
CODE_WIDTH = K_PARITY + K_COLOR + K_RESIDUAL          # 8
IMAGE_SHAPE = (3, 28, 28)
INPUT_SIZE = 3 * 28 * 28                              # 2352
LATENT_SIZE = 128

N_EPOCHS, BATCH_SIZE, LR = 15, 256, 1e-3
RECON_REG = 1.0
# Entropy bonus on the *batch-averaged* residual posterior. Without it the residual
# collapses onto two of its four states: most of the reconstruction BCE is background
# pixels, so the gain from splitting a prototype further is far too small to escape the
# collapse on its own. Measured here, 0.1 restores all four states at *identical*
# reconstruction loss -- the collapse is an optimisation artifact, not a capacity
# trade-off. It shapes only the marginal; which images land in which state is still
# reconstruction's decision.
USAGE_REG = 0.1

MRF_STEPS, MRF_LR, MRF_HIDDEN = 400, 0.05, 64
BP_ITERS, BP_DAMPING, BP_TOL = 50, 0.5, 1e-8

N_COLUMNS = 10

# The field carries no unary potentials and no image conditioning (by design), so
# p(color, residual | parity) does not depend on the image: BP returns the same answer
# for every row with the same clamped parity, and the last figure row holds at most two
# distinct reconstructions. Set this True to multiply the encoder's own per-image beliefs
# into the BP marginals at readout (p ∝ p_encoder * p_BP). That restores per-image
# variation without adding any factor to the field -- it is a readout choice, not a model
# change -- but it is no longer the pure "BP over the MRF" answer, so it defaults off.
MIX_ENCODER_BELIEFS = False

# Straight-through: an exact one-hot forward, a soft gradient backward. The figure feeds
# the decoder argmax one-hots, so anything softer during training would be a mismatch.
StraightThroughCategorical = pyro_dist.RelaxedOneHotCategoricalStraightThrough


def as_t(x):
    """Plain tensor out of an AnnotatedTensor (which wraps rather than subclasses one)."""
    return getattr(x, 'tensor', x)


# ---------------------------------------------------------------- 1. data
def load_split(train, n_samples, seed):
    """ColorMNIST with colour correlated to parity at ``P_COLOR_AGREES``.

    Built from the two helpers ``ColorMNISTDataset`` itself uses rather than from the
    dataset class: its ``coloring`` argument is either fully random or a deterministic
    digit->colour map, and neither gives a *soft* coupling.

    Returns ``(images (n,3,28,28), parity (n,), color (n,), digits (n,))``; colour
    0 = red, 1 = green, parity 1 = even. The digits are not a concept -- they are kept
    only so the diagnostics can report what the free residual ended up encoding.
    """
    images, digits = load_mnist(default_root('mnist'), train)
    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(digits), generator=generator)[:n_samples]
    images, digits = images[keep], digits[keep]

    parity = (digits % 2 == 0).long()
    agrees = torch.rand(len(digits), generator=generator) < P_COLOR_AGREES
    color = torch.where(agrees, parity, 1 - parity)
    # `colorize` wants the RGB channel; the concept value is the palette position.
    channels = torch.tensor([0, 1])[color]          # red, green
    return colorize(images, channels), parity, color, digits


# ------------------------------------------------- 2. autoencoder as a BayesianNetwork
def build_autoencoder():
    """The bottleneck model, plus a direct handle on the decoder.

    ``recon``'s CPD aggregates its three parents by concatenation, so the decoder sees
    the 8-dim code and nothing else. The handle is what the figure calls: building a code
    by hand and decoding it sidesteps every sampling decision the engines would otherwise
    make on our behalf.
    """
    input_var = EmbeddingVariable("input", distribution=Delta, size=INPUT_SIZE)
    latent_var = EmbeddingVariable("latent", distribution=Delta, size=LATENT_SIZE)
    parity = ConceptVariable("parity", distribution=StraightThroughCategorical,
                             size=K_PARITY)
    color = ConceptVariable("color", distribution=StraightThroughCategorical,
                            size=K_COLOR)
    residual = ConceptVariable("residual", distribution=StraightThroughCategorical,
                               size=K_RESIDUAL)
    recon_var = EmbeddingVariable("recon", distribution=Delta, size=INPUT_SIZE)

    encoder = nn.Sequential(
        nn.Unflatten(1, IMAGE_SHAPE),
        nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),     # 14x14
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),    # 7x7
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, LATENT_SIZE), nn.ReLU(),
    )

    def head(variable):
        return nn.Sequential(nn.Linear(LATENT_SIZE, variable.size),
                             DefaultActivation(variable, 'probs'))

    decoder = nn.Sequential(
        nn.Linear(CODE_WIDTH, 256), nn.ReLU(),
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, INPUT_SIZE), nn.Sigmoid(),
    )

    model = BayesianNetwork(
        variables=[input_var, latent_var, parity, color, residual, recon_var],
        factors=[
            # `input` is always evidence, so this prior never actually runs -- a
            # BayesianNetwork just requires one factor per variable.
            ParametricCPD(input_var, parametrization=LearnablePrior(input_var.size),
                          parents=[]),
            ParametricCPD(latent_var, parametrization=encoder, parents=[input_var]),
            ParametricCPD(parity, parametrization={'probs': head(parity)},
                          parents=[latent_var]),
            ParametricCPD(color, parametrization={'probs': head(color)},
                          parents=[latent_var]),
            ParametricCPD(residual, parametrization={'probs': head(residual)},
                          parents=[latent_var]),
            ParametricCPD(recon_var, parametrization=decoder,
                          parents=[parity, color, residual]),
        ],
    ).to(DEVICE)
    return model, decoder


def cross_entropy(probs, target_one_hot):
    """CE against a one-hot target, from probabilities rather than logits."""
    return -(target_one_hot * probs.clamp_min(1e-8).log()).sum(-1).mean()


def train_autoencoder(model, x_flat, parity_oh, color_oh):
    """Supervise parity and colour, reconstruct from the 8-dim code.

    ``p_int=1`` teacher-forces the two supervised concepts on the way into the decoder,
    which is what makes the decoder actually *obey* them -- and therefore what makes the
    swap change the image at all. ``residual`` is queried with ``None``: never
    teacher-forced, drawn straight-through, free to encode whatever is left.
    """
    engine = AncestralSamplingInference(model, p_int=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    n = len(x_flat)

    model.train()
    for epoch in range(N_EPOCHS):
        order = torch.randperm(n, device=DEVICE)
        totals = torch.zeros(4)
        for start in range(0, n, BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            out = engine.query(
                query={'parity': parity_oh[rows], 'color': color_oh[rows],
                       'residual': None, 'recon': None},
                evidence={'input': x_flat[rows]},
            )
            parity_loss = cross_entropy(out.probs['parity'], parity_oh[rows])
            color_loss = cross_entropy(out.probs['color'], color_oh[rows])
            recon_loss = F.binary_cross_entropy(
                out.value['recon'].clamp(1e-6, 1 - 1e-6), x_flat[rows])
            # -H(mean residual posterior): minimised when all four states are used.
            marginal = out.probs['residual'].mean(0)
            usage_loss = (marginal * marginal.clamp_min(1e-8).log()).sum()
            loss = (parity_loss + color_loss + RECON_REG * recon_loss
                    + USAGE_REG * usage_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += torch.tensor([parity_loss.item(), color_loss.item(),
                                    recon_loss.item(), -usage_loss.item()]) * len(rows)

        p_loss, c_loss, r_loss, entropy = (totals / n).tolist()
        print(f"  epoch {epoch:3d} | parity CE {p_loss:.4f} | color CE {c_loss:.4f}"
              f" | recon BCE {r_loss:.4f} | residual H {entropy:.3f}/{torch.tensor(4.).log():.3f}")
    return engine


@torch.no_grad()
def encode(model, x_flat, batch=1024):
    """Per-image head probabilities, read at ``p_int=0`` (no teacher forcing)."""
    engine = AncestralSamplingInference(model, p_int=0)
    model.eval()
    chunks = {'parity': [], 'color': [], 'residual': []}
    for start in range(0, len(x_flat), batch):
        out = engine.query(query=list(chunks),
                           evidence={'input': x_flat[start:start + batch]})
        for name in chunks:
            chunks[name].append(as_t(out.probs[name]))
    return {name: torch.cat(parts) for name, parts in chunks.items()}


@torch.no_grad()
def describe_residual(decoder, x_flat, parity_oh, color_oh, residual, digits):
    """What the free variable ended up carrying, and whether it carries anything.

    Two things quietly decide whether this experiment says anything. If the residual is
    noise, reconstructing from a shuffled one costs nothing and BP has nothing to
    propagate into. And if the residual is uncorrelated with parity, clamping the swapped
    parity leaves its marginal untouched. The digit histogram shows both at a glance --
    a state that is 90% odd digits is a state parity can move.
    """
    def recon_bce(states):
        code = torch.cat([parity_oh, color_oh, one_hot_rows(states, K_RESIDUAL)], dim=-1)
        return F.binary_cross_entropy(decoder(code).clamp(1e-6, 1 - 1e-6), x_flat).item()

    shuffled = residual[torch.randperm(len(residual), device=residual.device)]
    print(f"\nresidual informativeness: recon BCE {recon_bce(residual):.5f} as encoded"
          f" vs {recon_bce(shuffled):.5f} shuffled")
    print("digit mix per residual state (entries above 8%):")
    for state in range(K_RESIDUAL):
        rows = residual == state
        if not rows.any():
            print(f"  r={state}  unused")
            continue
        histogram = torch.bincount(digits[rows], minlength=10).float()
        histogram = histogram / histogram.sum()
        even = histogram[::2].sum()
        top = " ".join(f"{d}:{v:.2f}" for d, v in enumerate(histogram.tolist()) if v > 0.08)
        print(f"  r={state} (n={int(rows.sum()):5d}, {even:.0%} even)  {top}")


# --------------------------------------------- 3. pairwise MRF + loopy belief propagation
def energy_net(in_features, hidden=MRF_HIDDEN):
    """One pairwise clique's energy, same shape as ``CE2BM.py``'s.

    The hidden layer is load-bearing. A bare ``Linear`` over concatenated one-hots is
    purely additive -- ``w_a[i] + w_b[j]`` -- so it expresses no coupling whatsoever and
    would reduce every "pairwise" factor to a pair of unaries.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, 1),
    )


def build_mrf():
    """Three variables, three pairwise potentials: a triangle, hence genuinely loopy.

    Plain ``OneHotCategorical`` rather than the straight-through family the autoencoder
    uses -- BP enumerates states, it never samples.
    """
    parity = ConceptVariable("parity", distribution=OneHotCategorical, size=K_PARITY)
    color = ConceptVariable("color", distribution=OneHotCategorical, size=K_COLOR)
    residual = ConceptVariable("residual", distribution=OneHotCategorical,
                               size=K_RESIDUAL)
    variables = {'parity': parity, 'color': color, 'residual': residual}

    factors = {
        'phi_pc': ParametricPotential(scope=[parity, color], name='phi_pc',
                                      parametrization=energy_net(K_PARITY + K_COLOR)),
        'phi_pr': ParametricPotential(scope=[parity, residual], name='phi_pr',
                                      parametrization=energy_net(K_PARITY + K_RESIDUAL)),
        'phi_cr': ParametricPotential(scope=[color, residual], name='phi_cr',
                                      parametrization=energy_net(K_COLOR + K_RESIDUAL)),
    }
    mrf = MarkovNetwork(variables=list(variables.values()),
                        factors=list(factors.values())).to(DEVICE)
    return mrf, variables, factors


def state_grid():
    """All 16 joint assignments, one-hot encoded, in C order (residual varies fastest)."""
    grid = torch.cartesian_prod(torch.arange(K_PARITY), torch.arange(K_COLOR),
                                torch.arange(K_RESIDUAL)).to(DEVICE)
    states = {'parity': F.one_hot(grid[:, 0], K_PARITY).float(),
              'color': F.one_hot(grid[:, 1], K_COLOR).float(),
              'residual': F.one_hot(grid[:, 2], K_RESIDUAL).float()}
    return grid, states


def train_mrf(mrf, codes):
    """Exact maximum likelihood over the 16 joint states -- no CD, no sampling.

    The joint has 16 configurations, so ``log Z`` is one ``logsumexp`` and the exact
    likelihood is affordable. What keeps this interesting is the *restriction*: a
    pairwise-only field cannot represent an arbitrary three-way joint, so MLE fits its
    best pairwise approximation -- which is what makes loopy BP on the triangle a real
    approximation rather than a formality.

    ``codes`` is ``(n, 3)`` of integer states, in ``(parity, color, residual)`` order.
    """
    _, states = state_grid()
    index = (codes[:, 0] * K_COLOR * K_RESIDUAL
             + codes[:, 1] * K_RESIDUAL
             + codes[:, 2])
    empirical = torch.bincount(index, minlength=16).float()
    empirical = empirical / empirical.sum()

    optimizer = torch.optim.AdamW(mrf.parameters(), lr=MRF_LR)
    mrf.train()
    for step in range(MRF_STEPS):
        log_joint = (-mrf.energy(states)).log_softmax(0)
        # Cross-entropy against the empirical joint == NLL of the codes, up to a constant.
        loss = -(empirical * log_joint).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == MRF_STEPS - 1:
            kl = F.kl_div(log_joint, empirical, reduction='sum')
            print(f"  MRF step {step:4d} | NLL {loss.item():.4f} | KL to empirical {kl:.4f}")
    mrf.eval()
    return empirical


@torch.no_grad()
def exact_conditional(mrf, parity_state):
    """``p(color, residual | parity)`` by enumeration -- the reference BP is checked against."""
    _, states = state_grid()
    log_joint = (-mrf.energy(states)).log_softmax(0).reshape(K_PARITY, K_COLOR, K_RESIDUAL)
    slab = log_joint[parity_state].reshape(-1).softmax(0).reshape(K_COLOR, K_RESIDUAL)
    return slab.sum(-1), slab.sum(0)          # marginal colour, marginal residual


@torch.no_grad()
def induced_table(factor, scope_vars, cards):
    """A potential's log-potential over every state pair of its scope.

    The factors are MLPs, so their tables are implicit; evaluating one on the full grid is
    how you actually see what coupling it learned.
    """
    grid = torch.cartesian_prod(*[torch.arange(c) for c in cards]).to(DEVICE)
    assignment = {var: F.one_hot(grid[:, axis], card).float()
                  for axis, (var, card) in enumerate(zip(scope_vars, cards))}
    table = -factor.energy(assignment).reshape(*cards)
    return table - table.max()               # a log-potential is defined up to a constant


# ------------------------------------------------------------------- 4. figure
def one_hot_rows(index, k):
    return F.one_hot(index, k).float()


@torch.no_grad()
def build_rows(decoder, x_flat, codes, bp_codes):
    """The four image rows, each already reshaped for plotting."""
    def decode(parity, color, residual):
        code = torch.cat([one_hot_rows(parity, K_PARITY),
                          one_hot_rows(color, K_COLOR),
                          one_hot_rows(residual, K_RESIDUAL)], dim=-1)
        return decoder(code).reshape(-1, *IMAGE_SHAPE)

    parity, color, residual = codes.T
    swapped = 1 - parity
    return [
        x_flat.reshape(-1, *IMAGE_SHAPE),
        decode(parity, color, residual),
        decode(swapped, color, residual),
        decode(swapped, bp_codes[:, 0], bp_codes[:, 1]),
    ]


def save_figure(rows, codes, bp_codes, path):
    """One row per condition, ``N_COLUMNS`` images across; each cell labelled with its code."""
    labels = ["original", "reconstruction", "do(parity)", "do(parity) + loopy BP"]
    parity, color, residual = codes.T.tolist()
    swapped = [1 - p for p in parity]
    bp_color, bp_residual = bp_codes.T.tolist()
    captions = [
        [""] * len(parity),
        [f"p{p} c{c} r{r}" for p, c, r in zip(parity, color, residual)],
        [f"p{p} c{c} r{r}" for p, c, r in zip(swapped, color, residual)],
        [f"p{p} c{c} r{r}" for p, c, r in zip(swapped, bp_color, bp_residual)],
    ]

    n_cols = rows[0].shape[0]
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(1.05 * n_cols, 1.28 * len(rows)), squeeze=False)
    for r, (images, label) in enumerate(zip(rows, labels)):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.imshow(images[c].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if captions[r][c]:
                ax.set_title(captions[r][c], fontsize=6, pad=2)
            if c == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, ha='right', va='center',
                              labelpad=6)
    fig.suptitle("ColorMNIST: swapping parity in a discrete bottleneck, "
                 "before and after loopy BP over the pairwise MRF", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    seed_everything(SEED)
    started = time.time()
    print(f"{'=' * 72}\nColorMNIST discrete bottleneck + pairwise MRF   [device: {DEVICE}]"
          f"\n{'=' * 72}")

    # ---- data ----------------------------------------------------------
    x_train, parity_train, color_train, digits_train = load_split(True, N_TRAIN, SEED)
    x_test, parity_test, color_test, _ = load_split(False, N_TEST, SEED + 1)
    x_train, x_test = x_train.to(DEVICE), x_test.to(DEVICE)
    parity_train, color_train = parity_train.to(DEVICE), color_train.to(DEVICE)
    parity_test, color_test = parity_test.to(DEVICE), color_test.to(DEVICE)
    digits_train = digits_train.to(DEVICE)

    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    agreement = (parity_train == color_train).float().mean()
    print(f"train {tuple(x_train.shape)}  test {tuple(x_test.shape)}"
          f"  |  P(color == parity) = {agreement:.3f}  (target {P_COLOR_AGREES})")

    # ---- autoencoder ---------------------------------------------------
    print("\n-- training the bottleneck autoencoder --")
    parity_oh = one_hot_rows(parity_train, K_PARITY)
    color_oh = one_hot_rows(color_train, K_COLOR)
    model, decoder = build_autoencoder()
    train_autoencoder(model, x_train_flat, parity_oh, color_oh)

    train_probs = encode(model, x_train_flat)
    test_probs = encode(model, x_test_flat)
    parity_accuracy = (test_probs['parity'].argmax(-1) == parity_test).float().mean()
    color_accuracy = (test_probs['color'].argmax(-1) == color_test).float().mean()
    print(f"\ntest accuracy -- parity {parity_accuracy:.3f}  color {color_accuracy:.3f}")

    # ---- the code dataset the MRF is fitted to --------------------------
    # Ground truth for the supervised pair (the regime the decoder trained under), the
    # model's own belief for the residual -- it has no ground truth to have.
    train_codes = torch.stack([parity_train, color_train,
                               train_probs['residual'].argmax(-1)], dim=-1)
    usage = torch.bincount(train_codes[:, 2], minlength=K_RESIDUAL).float()
    print(f"residual state usage: {[f'{v:.3f}' for v in (usage / usage.sum()).tolist()]}")
    describe_residual(decoder, x_train_flat, parity_oh, color_oh,
                      train_codes[:, 2], digits_train)

    print("\n-- fitting the pairwise MRF (exact MLE over 16 states) --")
    mrf, mrf_vars, mrf_factors = build_mrf()
    empirical = train_mrf(mrf, train_codes)

    print("\nempirical joint p(parity, color) and p(residual | parity):")
    joint = empirical.reshape(K_PARITY, K_COLOR, K_RESIDUAL)
    for p in range(K_PARITY):
        pc = joint[p].sum(-1)
        conditional = joint[p].sum(0) / joint[p].sum()
        print(f"  parity={p}  p(color)={[f'{v:.3f}' for v in (pc / pc.sum()).tolist()]}"
              f"  p(residual|parity)={[f'{v:.3f}' for v in conditional.tolist()]}")

    print("\ninduced log-potential tables (max-normalised):")
    for name, scope, cards in [
        ('phi_pc', ('parity', 'color'), (K_PARITY, K_COLOR)),
        ('phi_pr', ('parity', 'residual'), (K_PARITY, K_RESIDUAL)),
        ('phi_cr', ('color', 'residual'), (K_COLOR, K_RESIDUAL)),
    ]:
        table = induced_table(mrf_factors[name], [mrf_vars[s] for s in scope], cards)
        print(f"  {name}  rows={scope[0]}  cols={scope[1]}")
        for row in table.tolist():
            print("    " + "  ".join(f"{v:+7.3f}" for v in row))

    # ---- steering: clamp the swapped parity, let BP update the rest -----
    print("\n-- loopy BP after do(parity) --")
    bp = BeliefPropagation(mrf, iters=BP_ITERS, damping=BP_DAMPING, tol=BP_TOL)

    columns = torch.arange(N_COLUMNS, device=DEVICE)
    test_codes = torch.stack([test_probs['parity'].argmax(-1),
                              test_probs['color'].argmax(-1),
                              test_probs['residual'].argmax(-1)], dim=-1)[columns]
    swapped_parity = one_hot_rows(1 - test_codes[:, 0], K_PARITY)

    with torch.no_grad():
        bp_out = bp.query(query=['color', 'residual'],
                          evidence={'parity': swapped_parity})
    bp_color = as_t(bp_out.probs['color'])
    bp_residual = as_t(bp_out.probs['residual'])

    # Agreement with the enumerated conditional is the check that loopy BP converged on
    # the triangle rather than oscillating.
    worst = 0.0
    for state in range(K_PARITY):
        rows = swapped_parity.argmax(-1) == state
        if not rows.any():
            continue
        exact_color, exact_residual = exact_conditional(mrf, state)
        worst = max(worst,
                    (bp_color[rows][0] - exact_color).abs().max().item(),
                    (bp_residual[rows][0] - exact_residual).abs().max().item())
        print(f"  parity={state}  exact color {[f'{v:.3f}' for v in exact_color.tolist()]}"
              f"  bp {[f'{v:.3f}' for v in bp_color[rows][0].tolist()]}")
        print(f"  parity={state}  exact resid {[f'{v:.3f}' for v in exact_residual.tolist()]}"
              f"  bp {[f'{v:.3f}' for v in bp_residual[rows][0].tolist()]}")
    print(f"  max |BP - exact| = {worst:.2e}")

    if MIX_ENCODER_BELIEFS:
        # Readout-only product of experts, so row 4 varies image by image again.
        bp_color = bp_color * test_probs['color'][columns]
        bp_residual = bp_residual * test_probs['residual'][columns]
        print("  (encoder beliefs mixed into the BP marginals at readout)")

    bp_codes = torch.stack([bp_color.argmax(-1), bp_residual.argmax(-1)], dim=-1)
    distinct = len({tuple(row) for row in
                    torch.cat([1 - test_codes[:, :1], bp_codes], -1).tolist()})
    print(f"  distinct codes in the BP row: {distinct}"
          + ("" if MIX_ENCODER_BELIEFS else
             "  (a unary-free field cannot depend on the image)"))

    # ---- figure --------------------------------------------------------
    rows = build_rows(decoder, x_test_flat[columns], test_codes, bp_codes)
    figure_dir = Path(__file__).parent / "figures" / "colormnist_bp"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "parity_swap.png"
    save_figure(rows, test_codes, bp_codes, path)
    # Everything the plot needs, so a layout change never costs a re-run.
    torch.save({'rows': [r.cpu() for r in rows], 'codes': test_codes.cpu(),
                'bp_codes': bp_codes.cpu(), 'bp_color': bp_color.cpu(),
                'bp_residual': bp_residual.cpu(), 'empirical_joint': empirical.cpu()},
               path.with_suffix('.pt'))
    print(f"\nwrote {path}   ({time.time() - started:.0f}s total)")


if __name__ == "__main__":
    main()
