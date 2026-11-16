"""Export one finished run_direction.py run into the package layout (configs/, results/, metadata/) and write the hash manifest.
Other seeds already present in the all-seed CSV files are kept. CPU only: numpy.
  python code/collect_results.py --run generated/direction/seed0/full --gate generated/direction/seed0/gate \
      --validation generated/direction/seed0/official_validation/validation.json
  python code/collect_results.py --manifest-only        # after editing README or figures
"""
import argparse
import csv
import hashlib
import json
from importlib import metadata as pkg
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
EXPERIMENT = 'direction'
TRACKED = ['README.md', 'run_a100.sh', 'requirements-analysis.txt', 'requirements-train.txt', 'code', 'configs', 'figures', 'metadata', 'results']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def merge_csv(path, header, rows, seed):
    """Replace this seed's rows, keep the others, sort by seed (stable within a seed)."""
    kept = []
    if path.exists():
        with path.open(encoding='utf-8', newline='') as f:
            old = list(csv.reader(f))
        if old and old[0] == header:
            kept = [r for r in old[1:] if int(r[0]) != seed]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(sorted(kept + rows, key=lambda r: int(r[0])))


def manifest():
    files = {}
    for name in TRACKED:
        target = PACKAGE / name
        for path in ([target] if target.is_file() else sorted(target.rglob('*')) if target.is_dir() else []):
            if path.is_file() and '__pycache__' not in path.parts:
                files[path.relative_to(PACKAGE).as_posix()] = sha256(path)
    write_json(PACKAGE / 'manifest_sha256.json', dict(sorted(files.items())))
    print('manifest: %d files' % len(files))


def export(run, gate, validation):
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    environment = json.loads((run / 'environment.json').read_text(encoding='utf-8'))
    h = np.load(run / 'history.npz')
    seed, groups, cs = config['seed'], [str(g) for g in h['groups']], h['cs']
    fork, ev, m, ps = int(h['fork_step']), h['eval_steps'], h['metrics'], h['per_step']
    norm, layer = ps[:, :, 0], np.sqrt(h['layer_norms_squared'])
    with np.errstate(invalid='ignore', divide='ignore'):
        norm_error = np.abs(norm - norm[:, :1]) / norm[:, :1]
    cum_angle = np.nancumsum(ps[:, :, 4], 0)
    results = PACKAGE / 'results' / EXPERIMENT
    g = lambda v: '%.12g' % v

    write_json(PACKAGE / 'configs' / EXPERIMENT / ('seed%d.json' % seed), config)

    rows = []
    for j, name in enumerate(groups):
        for e, s in enumerate(ev):
            rows.append([seed, name, g(cs[j]), int(s), fork + int(s), g(m[e, j, 0]), g(m[e, j, 1]), g(m[e, j, 2]), g(m[e, j, 3]), g(m[e, j, 4]),
                         g(m[e, j, 5]), g(m[e, j, 3] / 10), '%.17g' % norm[s, j], '%.17g' % norm[s, 0], g(norm_error[s, j]), g(cum_angle[s, j]),
                         g(layer[s, j, 0]), g(layer[s, j, 1]), g(layer[s, j, 2])])
    merge_csv(results / 'all_seeds_curves.csv',
              ['seed', 'group', 'c', 'steps_since_fork', 'step', 'train_accuracy', 'val_accuracy', 'test_accuracy', 'train_loss', 'val_loss',
               'test_loss', 'train_mse', 'global_norm', 'reference_norm', 'norm_relative_error', 'cumulative_angle_rad',
               'norm_layer1', 'norm_layer2', 'norm_layer3'], rows, seed)

    by_group = {x['group']: x for x in summary['groups']}
    thresholds = ['%d' % (100 * t) for t in config['thresholds']]
    rows = []
    for j, name in enumerate(groups):
        x = by_group[name]
        blank = lambda v: '' if v is None else v
        rows.append([seed, name, g(cs[j]), fork, g(x['final']['train_acc']), g(x['final']['val_acc']), g(x['final']['test_acc']),
                     g(np.nanmax(m[:, j, 2])), g(norm[0, j]), g(norm[-1, j]), *[blank(x['steps_to_test'][t]) for t in thresholds],
                     blank(x['steps_to_val']['60']), g(x['max_norm_relative_error']), g(x['total_angle_rad']), g(x['total_angle_vs_reference']),
                     g(x['final_relative_distance_to_reference']), blank(x['first_non_finite_step'])])
    merge_csv(results / 'time_to_generalize_all_seeds.csv',
              ['seed', 'group', 'c', 'fork_step', 'final_train_accuracy', 'final_val_accuracy', 'final_test_accuracy', 'max_test_accuracy',
               'fork_norm', 'final_norm', *['steps_to_test_%s' % t for t in thresholds], 'steps_to_val_60', 'max_norm_relative_error',
               'total_angle_rad', 'total_angle_vs_reference', 'final_relative_distance_to_reference', 'first_non_finite_step'], rows, seed)

    # Per-step record: the reference norm trajectory R_t plus each group's stored norm, angle and minibatch MSE.
    with (results / ('seed%d_per_step.csv' % seed)).open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps_since_fork', 'step', 'reference_norm'] + [k + '_' + name for name in groups[1:] for k in ('global_norm', 'norm_relative_error')]
                        + ['angle_rad_' + name for name in groups] + ['batch_mse_' + name for name in groups])
        for s in range(norm.shape[0]):
            writer.writerow([s, fork + s, '%.17g' % norm[s, 0]] + [v for j in range(1, len(groups)) for v in ('%.17g' % norm[s, j], '%.3g' % norm_error[s, j])]
                            + ['' if s == 0 else '%.8g' % ps[s, j, 4] for j in range(len(groups))]
                            + ['' if s == 0 else '%.8g' % ps[s, j, 6] for j in range(len(groups))])

    meta = PACKAGE / 'metadata'
    qa = dict(complete=True, experiment=EXPERIMENT, seed=seed, groups=groups, evaluation_points=int(len(ev)), steps_after_fork=config['steps_after_fork'],
              fork_step=fork, weight_decay=config['weight_decay'], interpretable=summary['interpretable'],
              resume_check=summary['resume_check'], replay_control=summary['replay_control'], norms_matched=summary['norms_matched'],
              max_norm_relative_error={x['group']: x['max_norm_relative_error'] for x in summary['groups']},
              all_metrics_finite=bool(np.isfinite(m).all()), all_per_step_finite=bool(np.isfinite(ps[1:]).all()),
              first_non_finite_step={x['group']: x['first_non_finite_step'] for x in summary['groups']},
              training_wall_seconds_after_fork=summary['timing']['seconds_after_fork'], steps_per_second=summary['timing']['steps_per_second'],
              history_bytes=(run / 'history.npz').stat().st_size, code_sha256=config['code_sha256'],
              curve_export='all_seeds_curves.csv: scheduled full-split metrics with norms at the same steps; seed%d_per_step.csv: every step' % seed,
              loss_units='*_loss is squared error summed over 10 outputs, averaged over samples; train_mse = train_loss / 10 is the training objective.',
              cross_seed='This file covers one seed; cross-seed statistics are in results/direction/summary_by_condition.csv.')
    write_json(meta / EXPERIMENT / ('seed%d_qa.json' % seed), qa)
    if gate:
        gs = json.loads((gate / 'summary.json').read_text(encoding='utf-8'))
        gc = json.loads((gate / 'config.json').read_text(encoding='utf-8'))
        write_json(meta / EXPERIMENT / ('seed%d_gate.json' % seed),
                   dict(purpose='short c=1 replay run executed before the full run', seed=seed, cs=gc['cs'], steps_after_fork=gc['steps_after_fork'],
                        fork_step=gs['fork']['step'], replay_control=gs['replay_control'], resume_check=gs['resume_check'],
                        norms_matched=gs['norms_matched'], passed=gs['interpretable'], code_sha256=gc['code_sha256']))
    if validation:
        write_json(meta / EXPERIMENT / 'validation.json', json.loads(validation.read_text(encoding='utf-8')))
    write_json(meta / 'environment.json', environment)
    export = {}
    for name in ('numpy', 'matplotlib'):
        try: export[name] = pkg.version(name)
        except pkg.PackageNotFoundError: pass
    write_json(meta / 'package_versions.json', dict(training={k: environment[k] for k in ('python', 'torch', 'numpy')}, export_and_figures=export))
    with (results / 'time_to_generalize_all_seeds.csv').open(encoding='utf-8', newline='') as f:
        seeds = sorted({int(r['seed']) for r in csv.DictReader(f)})
    seconds = {}
    for s in seeds:
        q = meta / EXPERIMENT / ('seed%d_qa.json' % s)
        if q.exists(): seconds[str(s)] = json.loads(q.read_text(encoding='utf-8'))['training_wall_seconds_after_fork']
    write_json(meta / 'provenance.json', dict(
        paper='Omnigrok: Grokking Beyond Algorithmic Data (Liu, Michaud, Tegmark, ICLR 2023)',
        official_repository='https://github.com/KindXiaoming/Omnigrok', official_commit=config['official_commit'],
        official_files=['mnist/grokking/mnist-grokking.ipynb'],
        scope='Alternative-hypothesis experiment: tangential-update multiplier c under a per-step shared global-norm trajectory; not a figure of the paper.',
        base_setting='official notebook defaults: N=%d, alpha=%g, weight decay=%g, lr=%g, batch %d, float64'
                     % (config['n'], config['alpha'], config['weight_decay'], config['lr'], config['batch_size']),
        implementation='Model, sampler, AdamW step and data conversion imported from mnist_dynamics/code/run_dynamics.py; '
                       'the intervention is applied to the parameters after each optimizer step.',
        seeds=seeds, groups=groups, code_sha256=config['code_sha256'],
        training_seconds_after_fork_by_seed=seconds, last_exported_seed=seed,
        environment_note='metadata/environment.json and package_versions.json describe the run of the last exported seed.',
        not_published='history.npz, checkpoints and logs stay in generated/ (not in Git).'))
    print('exported seed %d from %s' % (seed, run))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    if not args.manifest_only:
        if not args.run: parser.error('--run is required unless --manifest-only')
        export(args.run, args.gate, args.validation)
    manifest()


if __name__ == '__main__':
    main()
