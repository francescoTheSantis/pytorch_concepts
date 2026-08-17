"""The full experiment: steer a frozen pre-trained model by intervening on concepts.

    data -> pre-trained model (z)      concept-agnostic, frozen afterwards
         -> MRF over concepts          propagates an intervention
         -> concept VAE (e)            concepts as a vector, dim(e) == dim(z)
         -> DDPM over [z ; e]          the joint
         -> SDEdit                     new e~, partially-kept z -> z~

Run ``python -m steering.main``. Trained pieces are cached under
``steering/artifacts/``; pass ``--fresh`` to retrain.
"""
import argparse
from pathlib import Path
from typing import Tuple

import torch

from torch_concepts import seed_everything
from steering import resolve_device
from steering.concept_vae import ConceptVAE, round_trip_accuracy, train_concept_vae
from steering.data import CARDS, COLOR_NAMES, concept_codes, load_colormnist, one_hot
from steering.diffusion import DDPM, train_ddpm
from steering.figure import make_figure
from steering.mrf import (build_mrf, exact_conditional, log_joint, make_propagator,
                          state_grid, train_mrf)
from steering.pretrain import ConvAE, ConvVAE, pretrain, reconstruct

ARTIFACTS = Path(__file__).parent / 'artifacts'
FIGURES = Path(__file__).parent / 'figures'
EDGES = [('digit', 'color')]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', choices=('ae', 'vae'), default='ae',
                        help='the concept-agnostic model to steer')
    parser.add_argument('--latent', type=int, default=32,
                        help='dim(z); dim(e) matches it')
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--columns', type=int, default=8,
                        help='images shown in the figure')
    parser.add_argument('--diffusion-steps', type=int, default=400)
    parser.add_argument('--release-step', type=int, default=160,
                        help='z is pinned above this step and free below it; '
                             '0 leaves z untouched, --diffusion-steps regenerates it')
    parser.add_argument('--resample', type=int, default=10,
                        help='RePaint harmonisation iterations per SDEdit step; '
                             '1 is naive replacement and under-conditions')
    parser.add_argument('--epochs-pretrain', type=int, default=15)
    parser.add_argument('--epochs-cvae', type=int, default=60)
    parser.add_argument('--epochs-ddpm', type=int, default=600)
    parser.add_argument('--beta', type=float, default=0.5,
                        help='concept-VAE KL weight; see steering.concept_vae')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto',
                        help="'auto' picks CUDA, then MPS, then CPU")
    parser.add_argument('--fresh', action='store_true', help='ignore the cache')
    return parser.parse_args(argv)


@torch.no_grad()
def encode_all(fn, data, device, batch_size: int = 512) -> torch.Tensor:
    """Apply an encoder over the whole dataset in batches, on ``device``.

    Batched rather than one big call: the dataset is 10k images and a single
    forward pass over all of them is an easy way to exhaust a small GPU.
    """
    n = len(data) if torch.is_tensor(data) else len(next(iter(data.values())))
    chunks = []
    for start in range(0, n, batch_size):
        stop = start + batch_size
        batch = (data[start:stop].to(device) if torch.is_tensor(data)
                 else {k: v[start:stop].to(device) for k, v in data.items()})
        chunks.append(fn(batch))
    return torch.cat(chunks)


def build(args, device) -> Tuple:
    """Train (or restore) every component. Returns everything the figure needs."""
    images, digits, color = load_colormnist(n=args.n_train, seed=args.seed)
    concepts = concept_codes(digits, color)
    codes = torch.stack([digits, color], dim=-1)

    pretrained = (ConvAE if args.pretrained == 'ae' else ConvVAE)(args.latent).to(device)
    cvae = ConceptVAE(CARDS, args.latent).to(device)
    mrf, _ = build_mrf(CARDS, EDGES)
    mrf = mrf.to(device)
    ddpm = DDPM(dim=2 * args.latent, steps=args.diffusion_steps).to(device)

    tag = f'{args.pretrained}_d{args.latent}_n{args.n_train}_t{args.diffusion_steps}_s{args.seed}'
    checkpoint = ARTIFACTS / f'{tag}.pt'

    if checkpoint.exists() and not args.fresh:
        print(f"restoring {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        for module, key in ((pretrained, 'pretrained'), (cvae, 'cvae'),
                            (mrf, 'mrf'), (ddpm, 'ddpm')):
            module.load_state_dict(state[key])
        empirical = state['empirical']
    else:
        print(f"\n[1/4] pre-training the concept-agnostic {args.pretrained}")
        pretrained, _, _ = pretrain(images, kind=args.pretrained, latent=args.latent,
                                    epochs=args.epochs_pretrain, device=device)
        print("\n[2/4] fitting the MRF over the concepts")
        empirical = train_mrf(mrf, CARDS, codes.to(device))
        print("\n[3/4] fitting the concept VAE")
        cvae, _ = train_concept_vae(concepts, CARDS, emb_size=args.latent,
                                    epochs=args.epochs_cvae, beta=args.beta,
                                    device=device)
        print("\n[4/4] fitting the DDPM over [z ; e]")
        z = encode_all(pretrained.encode, images, device)
        e = encode_all(cvae.encode, concepts, device)
        ddpm = train_ddpm(ddpm, torch.cat([z, e], dim=-1), epochs=args.epochs_ddpm,
                          device=device)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        torch.save({'pretrained': pretrained.state_dict(), 'cvae': cvae.state_dict(),
                    'mrf': mrf.state_dict(), 'ddpm': ddpm.state_dict(),
                    'empirical': empirical}, checkpoint)
        print(f"\nsaved {checkpoint}")

    for module in (pretrained, cvae, mrf, ddpm):
        module.eval()
    z = encode_all(pretrained.encode, images, device)
    return images, digits, color, concepts, z, pretrained, cvae, mrf, ddpm, empirical


@torch.no_grad()
def diagnostics(args, images, concepts, z, pretrained, cvae, mrf, empirical, device):
    """Everything that has to be true for the figure to mean anything."""
    print("\n-- diagnostics --")
    recon = reconstruct(pretrained.decoder, z[:512], device)
    print(f"pre-trained reconstruction MSE      {(recon - images[:512]).pow(2).mean():.5f}")

    accuracy = round_trip_accuracy(cvae, {k: v[:2000].to(device)
                                          for k, v in concepts.items()})
    for name, value in accuracy.items():
        print(f"concept round-trip accuracy {name:6s}  {value:.3f}")

    learned = log_joint(mrf, state_grid(CARDS, device)[1]).exp()
    print(f"MRF joint max |learned - empirical|  "
          f"{(learned - empirical).abs().max():.2e}")

    # BP must agree with exact enumeration -- on this graph it should be exact.
    from torch_concepts.nn import BeliefPropagation
    bp = BeliefPropagation(mrf, iters=20)
    worst = 0.0
    for value in range(CARDS['color']):
        marginal = bp.query(query=['digit'],
                            evidence={'color': one_hot(torch.tensor([value]),
                                                       CARDS['color']).to(device)}).probs
        exact = exact_conditional(mrf, CARDS, {'color': value})['digit']
        got = torch.as_tensor(marginal['digit'])[0]
        worst = max(worst, (got - exact).abs().max().item())
        even = got[::2].sum()
        print(f"p(even digit | color={COLOR_NAMES[value]:5s}) = {even:.3f}")
    print(f"BP vs exact enumeration              {worst:.2e}")


@torch.no_grad()
def report_steering(images, z_tilde, c_tilde, pretrained, device):
    """Did the decoded image actually take the intended colour?

    ``colorize`` puts all the intensity in one RGB channel, so the brighter of
    the red and green channels is an exact read-out of the colour concept.
    """
    steered = reconstruct(pretrained.decoder, z_tilde, device)
    got = steered.flatten(2).sum(-1)[:, :2].argmax(-1)      # 0 = red, 1 = green
    want = c_tilde['color'].argmax(-1).cpu()
    print(f"\nintervened images showing the intended colour: "
          f"{(got == want).float().mean():.2f} ({int((got == want).sum())}/{len(want)})")


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"device: {device}")
    seed_everything(args.seed)

    (images, digits, color, concepts, z,
     pretrained, cvae, mrf, ddpm, empirical) = build(args, device)

    diagnostics(args, images, concepts, z, pretrained, cvae, mrf, empirical, device)

    columns = torch.arange(args.columns)
    propagate = make_propagator(mrf, CARDS)
    path = FIGURES / f'steering_{args.pretrained}.png'
    z_tilde, c_tilde = make_figure(
        images=images[columns],
        z=z[columns],
        concepts={k: v[columns] for k, v in concepts.items()},
        color=color[columns].to(device),
        decoder=pretrained.decoder,
        propagate=propagate,
        cvae=cvae,
        ddpm=ddpm,
        release_step=args.release_step,
        resample=args.resample,
        path=path,
    )
    report_steering(images[columns], z_tilde, c_tilde, pretrained, device)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
