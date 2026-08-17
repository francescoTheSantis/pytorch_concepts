"""A VAE over the concept vector, giving the continuous concept embedding ``e``.

    e ~ N(0, I)             latent, dim(e) == dim(z)
    c_i ~ Cat(f_i(e))       one categorical head per concept
    q(e | c_1..c_k)         diagonal Gaussian encoder

Plain torch: the whole thing is a two-headed MLP and an analytic Gaussian KL, so
there is nothing a graphical-model layer would buy here.

**beta defaults to 0.5, not 1.** The reason is measured, not stylistic. ``e``
exists to carry the concept set into the diffusion model, so a latent that
ignores ``c`` makes the whole steering pipeline a no-op -- and at beta = 1 that
is exactly what the objective asks for. For a discrete source the ELBO is flat
between the two extremes: encoding the concepts costs ``KL ~ H(c)`` and buys
``recon ~ 0``, encoding nothing costs ``KL = 0`` and pays ``recon ~ H(c)``, both
totalling ``H(c)`` (~2.63 nats here). Gaussian overhead breaks the tie in favour
of collapse, so it is not an optimisation artefact a KL warm-up can fix -- warm-up
reached KL 2.46 mid-training and then drifted back to 0.85. Measured round-trip
accuracy on the digit, 60 epochs each::

    beta   1.0    1.0(warm)  0.5    0.2    0.1    0.05   0.01
    digit  0.205  0.282      1.000  1.000  1.000  1.000  1.000

0.5 is the largest value that still encodes, and it is the principled one: its
KL settles at ~2.65 nats, i.e. essentially exactly ``H(c)``, so the latent carries
the concept information and nothing more. It needs no warm-up (checked over three
seeds), so there is none.

``python -m steering.concept_vae`` trains one and checks the concept round-trip.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

Concepts = Dict[str, torch.Tensor]


def _mlp(in_features: int, out_features: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_features),
    )


class ConceptVAE(nn.Module):
    """Encoder ``q(e|c)``, decoder ``p(c|e)``, unit Gaussian prior."""

    def __init__(self, cards: Dict[str, int], emb_size: int, hidden: int = 128):
        super().__init__()
        self.cards, self.emb_size = dict(cards), emb_size
        self.names: List[str] = list(cards)
        self.encoder = _mlp(sum(cards.values()), 2 * emb_size, hidden)
        self.heads = nn.ModuleDict({n: _mlp(emb_size, k, hidden)
                                    for n, k in cards.items()})

    def _flat(self, concepts: Concepts) -> torch.Tensor:
        """Concatenate in `self.names` order -- the encoder's input layout."""
        return torch.cat([concepts[n] for n in self.names], dim=-1)

    def posterior(self, concepts: Concepts) -> Tuple[torch.Tensor, torch.Tensor]:
        loc, logvar = self.encoder(self._flat(concepts)).chunk(2, dim=-1)
        return loc, logvar.clamp(-8, 8)

    def encode(self, concepts: Concepts) -> torch.Tensor:
        """``{name: (B, K)} -> e (B, D)``, the posterior **mean**.

        Deterministic on purpose: it is what fills the latent dataset ``D_z`` and
        what produces ``e~`` at intervention time, so the diffusion model is never
        handed a point from a distribution it was not trained on.
        """
        return self.posterior(concepts)[0]

    def decode(self, e: torch.Tensor) -> Concepts:
        """``e (B, D) -> {name: logits (B, K)}``."""
        return {n: self.heads[n](e) for n in self.names}

    def elbo(self, concepts: Concepts) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(reconstruction, KL)`` in nats, averaged over the batch."""
        loc, logvar = self.posterior(concepts)
        e = loc + torch.randn_like(loc) * (0.5 * logvar).exp()
        logits = self.decode(e)
        recon = sum(-(concepts[n] * logits[n].log_softmax(-1)).sum(-1).mean()
                    for n in self.names)
        kl = 0.5 * (loc.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
        return recon, kl


def train_concept_vae(
    concepts: Concepts,
    cards: Dict[str, int],
    emb_size: int,
    epochs: int = 60,
    beta: float = 0.5,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[ConceptVAE, torch.Tensor]:
    """Fit the concept VAE and return it with ``e`` for the whole dataset.

    Args:
        beta: KL multiplier. See the module docstring for why the default is
            ``0.5`` and why anything near ``1`` collapses the latent.

    Returns:
        ``(model, e_all (N, emb_size))``.
    """
    device = device or torch.device('cpu')
    model = ConceptVAE(cards, emb_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(next(iter(concepts.values())))

    model.train()
    for epoch in range(epochs):
        order = torch.randperm(n)
        totals = torch.zeros(2)
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            batch = {k: v[index].to(device) for k, v in concepts.items()}
            recon, kl = model.elbo(batch)
            loss = recon + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += torch.tensor([recon.item(), kl.item()]) * len(index)
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            recon, kl = (totals / n).tolist()
            print(f"  cVAE epoch {epoch:3d} | recon {recon:.4f} + beta*KL "
                  f"{beta * kl:.4f} | KL {kl:.4f} nats")
    model.eval()

    with torch.no_grad():
        e_all = torch.cat([
            model.encode({k: v[i:i + batch_size].to(device)
                          for k, v in concepts.items()}).cpu()
            for i in range(0, n, batch_size)
        ])
    return model, e_all


@torch.no_grad()
def round_trip_accuracy(model: ConceptVAE, concepts: Concepts) -> Dict[str, float]:
    """Per-concept ``argmax decode(encode(c)) == c``. The check that ``e`` carries
    the concepts at all -- if this is at chance, steering cannot work."""
    logits = model.decode(model.encode(concepts))
    return {n: (logits[n].argmax(-1) == concepts[n].argmax(-1)).float().mean().item()
            for n in model.names}


if __name__ == '__main__':
    from torch_concepts import seed_everything
    from steering.data import CARDS, concept_codes, load_colormnist

    seed_everything(0)
    _, digits, color = load_colormnist(n=10000)
    concepts = concept_codes(digits, color)

    # beta=1 is shown alongside the default so the collapse is visible, not just
    # asserted in the docstring.
    for beta in (0.5, 1.0):
        seed_everything(0)
        print(f"\n-- beta = {beta} --")
        model, e_all = train_concept_vae(concepts, CARDS, emb_size=32, beta=beta)
        print(f"e {tuple(e_all.shape)}  mean {e_all.mean():.3f}  std {e_all.std():.3f}")
        for name, accuracy in round_trip_accuracy(
                model, {k: v[:2000] for k, v in concepts.items()}).items():
            print(f"  round-trip accuracy {name:6s} {accuracy:.3f}")
