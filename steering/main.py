"""The full experiment: steer a frozen pre-trained model by intervening on concepts.

    data -> pre-trained model (z)      concept-agnostic, frozen afterwards
         -> MRF over concepts          propagates an intervention
         -> concept VAE (e)            concepts as a vector, dim(e) == dim(z)
         -> DDPM over [z ; e]          the joint
         -> SDEdit                     new e~, partially-kept z -> z~

Two settings, same code path (``--dataset``):

``colormnist``
    Both concepts annotated. Intervening on the colour propagates to the digit
    through the MRF, so ``c~`` differs from the observed concept set in *both*
    coordinates.

``colormnist-no-digit``
    Only the colour is annotated, and the colour is drawn independently of the
    digit. The digit is unobserved, so it exists nowhere in the pipeline except
    the pre-trained ``z``. The MRF degenerates to a unary factor over the colour
    and has nothing to propagate, which is the point: ``e~`` carries a new colour
    and nothing else, so a steered image that keeps its digit proves ``z`` carried
    it through SDEdit. The grey-scale shape correlation in the run's report is
    that claim as a number, and the decorrelation is what stops it being
    confounded -- see ``steering.data.DATASETS``.

Run ``python -m steering.main``. Trained pieces are cached under
``steering/artifacts/`` per dataset; pass ``--fresh`` to retrain.
"""
import argparse
from pathlib import Path
from typing import Tuple

import torch

from torch_concepts import seed_everything
from torch_concepts.nn import BeliefPropagation
from steering import resolve_device
from steering.concept_vae import ConceptVAE, round_trip_accuracy, train_concept_vae
from steering.data import (COLOR_NAMES, DATASETS, concept_codes, dataset_spec,
                           load_colormnist, one_hot)
from steering.diffusion import DDPM, train_ddpm
from steering.figure import make_figure
from steering.mrf import (build_mrf, exact_conditional, log_joint, make_propagator,
                          state_grid, train_mrf)
from steering.pretrain import ConvAE, ConvVAE, pretrain, reconstruct

ARTIFACTS = Path(__file__).parent / 'artifacts'
FIGURES = Path(__file__).parent / 'figures'


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(DATASETS), default='colormnist',
                        help="'colormnist' annotates digit and colour; "
                             "'colormnist-no-digit' annotates only the colour, "
                             'leaving the digit to live entirely in z')
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
    parser.add_argument('--sampler', choices=('ddpm', 'ddim'), default='ddpm',
                        help="'ddpm' is Algorithm 2; 'ddim' drops the per-step "
                             'noise from the free block, keeping z more faithful. '
                             'Sampling-time only, so it reuses the same checkpoint')
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
    cards, scopes, p_agree = dataset_spec(args.dataset)
    images, digits, color = load_colormnist(n=args.n_train, p_agree=p_agree,
                                            seed=args.seed)
    concepts = concept_codes(digits, color, names=cards)
    # Integer codes for the MRF's histogram, in `cards` order. The unannotated
    # concepts are dropped here and nowhere else: the images are identical.
    per_name = {'digit': digits, 'color': color}
    codes = torch.stack([per_name[n] for n in cards], dim=-1)

    pretrained = (ConvAE if args.pretrained == 'ae' else ConvVAE)(args.latent).to(device)
    cvae = ConceptVAE(cards, args.latent).to(device)
    mrf, _ = build_mrf(cards, scopes)
    mrf = mrf.to(device)
    ddpm = DDPM(dim=2 * args.latent, steps=args.diffusion_steps).to(device)

    tag = (f'{args.dataset}_{args.pretrained}_d{args.latent}_n{args.n_train}'
           f'_t{args.diffusion_steps}_s{args.seed}')
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
        empirical = train_mrf(mrf, cards, codes.to(device))
        print("\n[3/4] fitting the concept VAE")
        cvae, _ = train_concept_vae(concepts, cards, emb_size=args.latent,
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
    return (images, digits, color, concepts, cards, z,
            pretrained, cvae, mrf, ddpm, empirical)


@torch.no_grad()
def diagnostics(images, concepts, cards, z, pretrained, cvae, mrf, empirical, device):
    """Everything that has to be true for the figure to mean anything.

    Returns the numbers as a dict as well as printing them, so a sweep can
    tabulate them without re-deriving anything.
    """
    print("\n-- diagnostics --")
    recon = reconstruct(pretrained.decoder, z[:512], device)
    metrics = {'recon_mse': (recon - images[:512]).pow(2).mean().item()}
    print(f"pre-trained reconstruction MSE      {metrics['recon_mse']:.5f}")

    accuracy = round_trip_accuracy(cvae, {k: v[:2000].to(device)
                                          for k, v in concepts.items()})
    for name, value in accuracy.items():
        metrics[f'roundtrip_{name}'] = value
        print(f"concept round-trip accuracy {name:6s}  {value:.3f}")

    learned = log_joint(mrf, state_grid(cards, device)[1]).exp()
    metrics['mrf_joint_err'] = (learned - empirical).abs().max().item()
    print(f"MRF joint max |learned - empirical|  {metrics['mrf_joint_err']:.2e}")

    # Propagation only means something when clamping the colour leaves something
    # free. With colour the only annotated concept the field has nothing to
    # propagate to, which is the point of that dataset, not a failure.
    free = [n for n in cards if n != 'color']
    if not free:
        print("MRF has no free concept given colour -- nothing to propagate")
        return metrics

    # BP must agree with exact enumeration -- on this graph it should be exact.
    bp = BeliefPropagation(mrf, iters=20)
    worst = 0.0
    for value in range(cards['color']):
        evidence = {'color': one_hot(torch.tensor([value]), cards['color']).to(device)}
        marginal = bp.query(query=free, evidence=evidence).probs
        exact = exact_conditional(mrf, cards, {'color': value})
        for name in free:
            got = torch.as_tensor(marginal[name])[0]
            worst = max(worst, (got - exact[name]).abs().max().item())
        if 'digit' in free:
            even = torch.as_tensor(marginal['digit'])[0][::2].sum().item()
            metrics[f'p_even_given_{COLOR_NAMES[value]}'] = even
            print(f"p(even digit | color={COLOR_NAMES[value]:5s}) = {even:.3f}")
    metrics['bp_vs_exact'] = worst
    print(f"BP vs exact enumeration              {worst:.2e}")
    return metrics


@torch.no_grad()
def steering_metrics(z, z_tilde, c_tilde, pretrained, device):
    """The two things steering has to get right, measured without a classifier.

    *Colour changed*: ``colorize`` puts all the intensity in one RGB channel, so
    the brighter of the red and green channels reads the colour concept exactly.

    *Shape kept*: the correlation between the grey-scale (channel-summed)
    reconstruction and the steered image. Summing over RGB discards precisely the
    thing that was supposed to change, leaving the stroke pattern -- so a high
    value means the digit survived the recolouring. This is the headline number
    for ``colormnist-no-digit``, where the digit is nowhere in ``e~`` and can only
    have been carried by ``z``.

    The two pull against each other as ``release_step`` moves, which is what
    ``python -m steering.sweep`` traces out.
    """
    before = reconstruct(pretrained.decoder, z, device)
    steered = reconstruct(pretrained.decoder, z_tilde, device)

    got = steered.flatten(2).sum(-1)[:, :2].argmax(-1)      # 0 = red, 1 = green
    want = c_tilde['color'].argmax(-1).cpu()

    a, b = before.sum(1).flatten(1), steered.sum(1).flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    shape = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-8)

    return {'color_accuracy': (got == want).float().mean().item(),
            'shape_correlation': shape.mean().item(),
            'n': len(want)}


def report_steering(z, z_tilde, c_tilde, pretrained, device):
    metrics = steering_metrics(z, z_tilde, c_tilde, pretrained, device)
    hits = round(metrics['color_accuracy'] * metrics['n'])
    print(f"\nintervened images showing the intended colour: "
          f"{metrics['color_accuracy']:.2f} ({hits}/{metrics['n']})")
    print(f"grey-scale shape correlation before/after:     "
          f"{metrics['shape_correlation']:.3f} (1.0 = digit perfectly preserved)")
    return metrics


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    print(f"device: {device}")
    seed_everything(args.seed)

    (images, digits, color, concepts, cards, z,
     pretrained, cvae, mrf, ddpm, empirical) = build(args, device)

    metrics = diagnostics(images, concepts, cards, z, pretrained, cvae, mrf,
                          empirical, device)

    columns = torch.arange(args.columns)
    propagate = make_propagator(mrf, cards)
    # `release_step` is in the filename because it is the one swept knob that
    # does *not* change the trained models, so a sweep over it reuses one
    # checkpoint and would otherwise overwrite a single figure repeatedly.
    path = FIGURES / (f'steering_{args.dataset}_{args.pretrained}'
                      f'_{args.sampler}_r{args.release_step}.png')
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
        sampler=args.sampler,
        path=path,
    )
    metrics.update(report_steering(z[columns], z_tilde, c_tilde, pretrained, device))
    print(f"wrote {path}")
    return {**metrics, 'figure': path.name}


if __name__ == '__main__':
    main()
