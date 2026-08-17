"""Run the grid in ``conf/sweep.yaml`` and tabulate what came out.

    python -m steering.sweep                    # the whole grid
    python -m steering.sweep --dry-run          # just list the runs
    python -m steering.sweep --conf other.yaml

Results land in ``steering/results/sweep.csv``, one row per run, rewritten after
*every* run rather than once at the end -- these grids are long enough that
losing the lot to a crash in the last run would be annoying.

The grid is ordered so the *sampling-time* knobs vary fastest. That is not
cosmetic: ``release_step`` and ``sampler`` are applied when the figure is drawn
and neither enters the checkpoint tag, so all of their values after the first
reuse one set of trained models. The number of trainings is therefore
``len(dataset) * len(pretrained)`` however long those two lists get.

``release_step: 0`` is worth keeping in any grid as a control: it leaves ``z``
untouched, so the steered image *is* the reconstruction and the run must report
a shape correlation of exactly 1.000 and a colour accuracy of 0.00 (the colour
was flipped, and nothing acted on it). Anything else means the pipeline is
wired wrong.
"""
import argparse
import csv
import itertools
import time
from pathlib import Path

import yaml

from steering import main as main_module

CONF = Path(__file__).parent / 'conf' / 'sweep.yaml'
RESULTS = Path(__file__).parent / 'results'

#: Swept keys, slowest-varying first. Anything in `params` not listed here is
#: appended after them, so adding a knob to the YAML needs no change in here.
ORDER = ['dataset', 'pretrained', 'sampler', 'resample', 'release_step']


def load_grid(path: Path):
    """``(list of run configs, common config)`` from a sweep YAML."""
    conf = yaml.safe_load(path.read_text())
    params, common = conf.get('params', {}), conf.get('common', {})

    # An empty item in a block list (a bare `-`, or an explicit `null`) parses to
    # None and would reach argparse as the string "None", failing mid-sweep. A
    # doubled comma in a *flow* list is a PyYAML parse error already, with a
    # better message than anything here, so this only covers the silent case.
    for key, values in params.items():
        if any(value is None for value in values):
            raise ValueError(
                f"{path}: `params.{key}` has an empty entry ({values!r}) — "
                "an empty list item or an explicit null."
            )

    keys = [k for k in ORDER if k in params] + [k for k in params if k not in ORDER]
    runs = [dict(zip(keys, values))
            for values in itertools.product(*(params[k] for k in keys))]
    return runs, common


def to_argv(config: dict) -> list:
    """A config dict as ``steering.main`` command-line arguments."""
    argv = []
    for key, value in config.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def write_csv(rows: list, path: Path) -> None:
    """Rewrite the whole file; the union of every row's keys is the header.

    Rewritten rather than appended because the columns are not the same across
    runs -- ``colormnist`` reports the digit round-trip and the BP check, which
    ``colormnist-no-digit`` has no equivalent of.
    """
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_tradeoff(rows: list, path: Path) -> None:
    """Both metrics against release step, as a panel matrix.

    Rows are ``(dataset, pretrained)`` -- what was trained. Columns are
    ``(sampler, resample)`` -- what was chosen at sampling time. Every knob that
    changes the *shape* of the curve gets its own panel rather than being pooled:
    ``ddpm`` and ``ddim`` are different reverse processes and ``resample`` moves
    the trade-off bodily, so averaging any of them together would draw a curve no
    run actually produced.

    Within a panel, ``release_step`` buys conditioning strength by *destroying*
    ``z``: colour accuracy climbs, digit shape correlation falls, and the useful
    operating point is wherever the worse of the two peaks -- marked. A run that
    wins one metric by abandoning the other is not steering.

    Across panels, the thing to read is that ``resample`` is *not* a free way out
    of a bad trade-off. It tightens the conditioning but its jump-back
    re-randomises the free block, costing shape correlation faster than
    ``release_step`` does (see ``DDPM.sdedit``). ``sampler='ddim'`` is the one
    knob measured to buy faithfulness back without weakening the conditioning,
    which is why both are in the grid.
    """
    import matplotlib.pyplot as plt

    cells = {}
    for row in rows:
        if 'color_accuracy' in row and 'shape_correlation' in row:
            key = ((row['dataset'], row['pretrained']),
                   (row.get('sampler', 'ddpm'), row.get('resample', '')))
            cells.setdefault(key, []).append(row)
    if not cells:
        return

    y_keys = sorted({k[0] for k in cells})
    x_keys = sorted({k[1] for k in cells})
    fig, axes = plt.subplots(len(y_keys), len(x_keys), squeeze=False, sharey=True,
                             sharex=True,
                             figsize=(3.4 * len(x_keys), 2.9 * len(y_keys)))
    for r, y_key in enumerate(y_keys):
        for c, x_key in enumerate(x_keys):
            ax = axes[r][c]
            group = sorted(cells.get((y_key, x_key), []),
                           key=lambda row: row['release_step'])
            if not group:
                ax.set_axis_off()
                continue
            steps = [row['release_step'] for row in group]
            ax.plot(steps, [row['color_accuracy'] for row in group], 'o-',
                    label='colour accuracy')
            ax.plot(steps, [row['shape_correlation'] for row in group], 's-',
                    label='shape correlation')
            best = max(group, key=lambda row: min(row['color_accuracy'],
                                                  row['shape_correlation']))
            ax.axvline(best['release_step'], color='k', ls='--', lw=1, alpha=0.6,
                       label=f"best: {best['release_step']}")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=6)
            if r == 0:
                ax.set_title(f'{x_key[0]}, resample={x_key[1]}', fontsize=8)
            if r == len(y_keys) - 1:
                ax.set_xlabel('release step')
            if c == 0:
                ax.set_ylabel(f'{y_key[0]}\n{y_key[1]}', fontsize=7)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conf', type=Path, default=CONF)
    parser.add_argument('--out', type=Path, default=RESULTS / 'sweep.csv')
    parser.add_argument('--dry-run', action='store_true',
                        help='list the runs and exit')
    parser.add_argument('--fresh', action='store_true',
                        help='retrain every run, ignoring the cache')
    args = parser.parse_args(argv)

    runs, common = load_grid(args.conf)
    print(f"{len(runs)} runs from {args.conf}")
    if args.dry_run:
        for index, run in enumerate(runs, 1):
            print(f"  {index:3d}. " + '  '.join(f'{k}={v}' for k, v in run.items()))
        return

    rows = []
    started = time.time()
    for index, run in enumerate(runs, 1):
        config = {**common, **run}
        if args.fresh:
            config['fresh'] = True
        print(f"\n{'=' * 70}\n[{index}/{len(runs)}] "
              + '  '.join(f'{k}={v}' for k, v in run.items())
              + f"\n{'=' * 70}")
        run_started = time.time()
        metrics = main_module.main(to_argv(config))
        rows.append({**run, **metrics, 'seconds': round(time.time() - run_started, 1)})
        write_csv(rows, args.out)

    print(f"\nwrote {args.out}  ({len(rows)} runs, {time.time() - started:.0f}s total)")
    plot_tradeoff(rows, args.out.with_suffix('.png'))

    # A compact read of the two numbers the sweep exists to trade off.
    print(f"\n{'dataset':22s} {'pre':4s} {'sampler':7s} {'resamp':>6s} "
          f"{'release':>7s} {'colour':>7s} {'shape':>7s}")
    for row in rows:
        print(f"{row['dataset']:22s} {row['pretrained']:4s} "
              f"{row.get('sampler', 'ddpm'):7s} {row.get('resample', ''):>6} "
              f"{row['release_step']:7d} {row.get('color_accuracy', float('nan')):7.2f} "
              f"{row.get('shape_correlation', float('nan')):7.3f}")


if __name__ == '__main__':
    main()
