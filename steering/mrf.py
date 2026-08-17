"""A Markov random field over the concepts, queried with loopy belief propagation.

This is the machinery that makes an intervention *propagate*: clamp the concepts
you want to control, and the rest are re-sampled from the field's conditional.

Run ``python -m steering.mrf`` for a self-contained test on three correlated
binary variables arranged in a triangle -- a genuinely loopy graph, where BP is
an approximation rather than a formality, checked against exact enumeration.
"""
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, OneHotCategorical

from torch_concepts.nn import (BeliefPropagation, ConceptVariable, MarkovNetwork,
                               ParametricPotential)

Concepts = Dict[str, torch.Tensor]


def energy_net(in_features: int, hidden: int = 64) -> nn.Module:
    """A pairwise energy. An MLP, not a linear map: a linear energy over one-hot
    inputs is additive, i.e. no coupling at all, which is the one thing a pairwise
    potential exists to provide."""
    return nn.Sequential(
        nn.Linear(in_features, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, 1),
    )


def build_mrf(
    cards: Dict[str, int],
    edges: Sequence[Tuple[str, str]],
    hidden: int = 64,
) -> Tuple[MarkovNetwork, Dict[str, ConceptVariable]]:
    """One categorical variable per concept, one pairwise potential per edge.

    ``OneHotCategorical`` rather than a relaxed family: BP enumerates states, it
    never samples, so the straight-through machinery would be dead weight.
    """
    variables = {name: ConceptVariable(name, distribution=OneHotCategorical, size=k)
                 for name, k in cards.items()}
    factors = [
        ParametricPotential(
            scope=[variables[a], variables[b]], name=f'phi_{a}_{b}',
            parametrization=energy_net(cards[a] + cards[b], hidden),
        )
        for a, b in edges
    ]
    return MarkovNetwork(variables=list(variables.values()), factors=factors), variables


def state_grid(cards: Dict[str, int], device=None) -> Tuple[torch.Tensor, Concepts]:
    """Every joint assignment, one-hot encoded. ``(grid (S, n_vars), {name: (S, K)})``."""
    names = list(cards)
    grid = torch.cartesian_prod(*[torch.arange(cards[n]) for n in names])
    grid = grid.reshape(-1, len(names)).to(device)
    return grid, {n: F.one_hot(grid[:, i], cards[n]).float()
                  for i, n in enumerate(names)}


def log_joint(mrf: MarkovNetwork, states: Concepts) -> torch.Tensor:
    """Normalised ``log p(assignment)`` over a full enumeration of the joint.

    ``MarkovNetwork`` exposes no ``energy`` of its own, so the field's total
    energy is the sum over its factors. Normalising by ``log_softmax`` over the
    enumerated states is exactly ``-log Z`` -- affordable only because the joint
    here has ``prod(cards)`` states (20 for digit x colour).
    """
    total = sum(factor(states) for factor in mrf.factors.values())
    return (-total).log_softmax(0)


def train_mrf(
    mrf: MarkovNetwork,
    cards: Dict[str, int],
    codes: torch.Tensor,
    steps: int = 600,
    lr: float = 0.05,
    verbose: bool = True,
) -> torch.Tensor:
    """Exact maximum likelihood -- no contrastive divergence, no sampling.

    ``codes`` is ``(N, n_vars)`` of integer states in ``cards`` order. The loss is
    the cross-entropy of the field's joint against the empirical one, which is the
    NLL of the data up to a constant. Returns the empirical joint.
    """
    device = next(mrf.parameters()).device
    _, states = state_grid(cards, device)

    sizes = list(cards.values())
    index = torch.zeros(len(codes), dtype=torch.long, device=codes.device)
    for axis, k in enumerate(sizes):                      # row-major flattening
        index = index * k + codes[:, axis]
    empirical = torch.bincount(index, minlength=int(torch.tensor(sizes).prod()))
    empirical = (empirical.float() / len(codes)).to(device)

    optimizer = torch.optim.AdamW(mrf.parameters(), lr=lr)
    mrf.train()
    for step in range(steps):
        loss = -(empirical * log_joint(mrf, states)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and (step % 200 == 0 or step == steps - 1):
            kl = F.kl_div(log_joint(mrf, states), empirical, reduction='sum')
            print(f"  MRF step {step:4d} | NLL {loss.item():.4f} | KL to empirical {kl:.5f}")
    mrf.eval()
    return empirical


def make_propagator(
    mrf: MarkovNetwork,
    cards: Dict[str, int],
    iters: int = 20,
    damping: float = 0.0,
    generator: torch.Generator = None,
) -> Callable[[Concepts], Concepts]:
    """The intervention operator: clamped concepts in, a full concept set out.

    Passing a variable in ``evidence`` is how this API *clamps* it -- every
    incident factor is reduced to the observed value. BP then reports the free
    variables' **marginals**, which is not a joint sample when several are free,
    so they are drawn one at a time and each draw is clamped in turn before the
    next query. The draws therefore compose by the chain rule

        p(c_free | c_clamped) = prod_i p(c_i | c_1..c_{i-1}, c_clamped)

    into a genuine joint sample -- exact whenever BP is exact, which it is on a
    graph without loops.
    """
    bp = BeliefPropagation(mrf, iters=iters, damping=damping)

    @torch.no_grad()
    def propagate(evidence: Concepts) -> Concepts:
        out = dict(evidence)                                  # the clamped concepts
        for name in [n for n in cards if n not in evidence]:
            probs = bp.query(query=[name], evidence=out).probs[name]
            draw = Categorical(probs=torch.as_tensor(probs)).sample()
            out[name] = F.one_hot(draw, cards[name]).float()
            # `name` is clamped from here on, so the next draw conditions on it.
        return out

    return propagate


@torch.no_grad()
def exact_conditional(
    mrf: MarkovNetwork,
    cards: Dict[str, int],
    clamped: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """``p(free | clamped)`` by enumeration -- the reference BP is checked against."""
    device = next(mrf.parameters()).device
    grid, states = state_grid(cards, device)
    names = list(cards)

    logp = log_joint(mrf, states)
    keep = torch.ones(len(grid), dtype=torch.bool, device=device)
    for name, value in clamped.items():
        keep &= grid[:, names.index(name)] == value
    slab = logp[keep].softmax(0)

    rows = grid[keep]
    return {name: torch.zeros(cards[name], device=device).index_add_(
                0, rows[:, names.index(name)], slab)
            for name in names if name not in clamped}


# --------------------------------------------------------------------- toy test
def _toy_main():
    """Three correlated binary variables on an Ising triangle.

    The triangle is the point: a loopy graph, so BP is doing approximate
    inference and agreeing with exact enumeration is a real result.
    """
    torch.manual_seed(0)
    cards = {'a': 2, 'b': 2, 'c': 2}
    coupling = 0.9

    # Ground truth: p(a,b,c) proportional to exp(J * (ab + bc + ca)) on +/-1 spins.
    grid, _ = state_grid(cards)
    spins = 2.0 * grid.float() - 1.0
    truth = (coupling * (spins[:, 0] * spins[:, 1]
                         + spins[:, 1] * spins[:, 2]
                         + spins[:, 2] * spins[:, 0])).softmax(0)
    draws = Categorical(probs=truth).sample((20000,))
    codes = grid[draws]

    print("Ising triangle over three binary variables (J = %.1f)" % coupling)
    print("  true joint:", " ".join(f"{p:.3f}" for p in truth))

    mrf, _ = build_mrf(cards, edges=[('a', 'b'), ('b', 'c'), ('c', 'a')])
    empirical = train_mrf(mrf, cards, codes)
    print("  empirical :", " ".join(f"{p:.3f}" for p in empirical))
    print("  learned   :", " ".join(f"{p:.3f}" for p in log_joint(
        mrf, state_grid(cards)[1]).exp().detach()))

    # Clamp `a`, let BP update the rest, and check it against exact enumeration.
    propagate = make_propagator(mrf, cards)
    bp = BeliefPropagation(mrf, iters=20)
    worst = 0.0
    for value in range(2):
        clamped = {'a': F.one_hot(torch.tensor([value]), 2).float()}
        marginals = bp.query(query=['b', 'c'], evidence=clamped).probs
        exact = exact_conditional(mrf, cards, {'a': value})
        for name in ('b', 'c'):
            got = torch.as_tensor(marginals[name])[0]
            worst = max(worst, (got - exact[name]).abs().max().item())
            print(f"  a={value}  p({name}|a)  bp {got.tolist()}  exact {exact[name].tolist()}")
    print(f"  max |BP - exact| = {worst:.2e}")
    assert worst < 1e-3, "loopy BP disagrees with exact enumeration"

    # The propagator itself: clamped variables survive, free ones are sampled.
    out = propagate({'a': F.one_hot(torch.zeros(8, dtype=torch.long), 2).float()})
    assert out['a'].argmax(-1).eq(0).all()
    print("  propagator draws (a,b,c):",
          torch.stack([out[n].argmax(-1) for n in ('a', 'b', 'c')], -1).tolist())
    print("OK")


if __name__ == '__main__':
    _toy_main()
