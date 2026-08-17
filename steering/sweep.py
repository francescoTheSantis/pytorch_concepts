"""Run the grid in ``conf/sweep.yaml`` and tabulate what came out.

    python -m steering.sweep                    # the whole grid
    python -m steering.sweep --dry-run          # just list the runs
    python -m steering.sweep --conf other.yaml

Results land in ``steering/results/sweep.csv``, one row per run, rewritten after
*every* run rather than once at the end -- these grids are long enough that
losing the lot to a crash in the last run would be annoying.

The grid is ordered so ``release_step`` varies fastest. That is not cosmetic:
``release_step`` is applied at sampling time and does not enter the checkpoint
tag, so all of its values after the first reuse one set of trained models. The
number of trainings is therefore ``len(dataset) * len(pretrained)`` no matter how
long the ``release_step`` list is.

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
ORDER = ['dataset', 'pretrained', 'release_step']


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

    # A compact read of the two numbers the sweep exists to trade off.
    print(f"\n{'dataset':22s} {'pre':4s} {'release':>7s} "
          f"{'colour':>7s} {'shape':>7s}")
    for row in rows:
        print(f"{row['dataset']:22s} {row['pretrained']:4s} "
              f"{row['release_step']:7d} {row.get('color_accuracy', float('nan')):7.2f} "
              f"{row.get('shape_correlation', float('nan')):7.3f}")


if __name__ == '__main__':
    main()
