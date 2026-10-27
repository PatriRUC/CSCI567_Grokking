"""Single-seed QA, portable curve export and Fig.3d/3e plots for one run_dynamics.py root."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
CURVE_FIELDS = ['seed', 'n', 'alpha', 'step', 'train_accuracy', 'test_accuracy', 'train_loss', 'test_loss',
                'global_norm', 'norm_layer1', 'norm_layer2', 'norm_layer3']
TRAINING_CODE = Path(__file__).resolve().parent / 'run_dynamics.py'


def first_reach(steps, values, threshold):
    hit = np.nonzero(values >= threshold)[0]
    return int(steps[hit[0]]) if len(hit) else None


def main(root):
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    seed = int(config['seed'])
    status = json.loads((root/'status.json').read_text(encoding='utf-8'))
    validation = json.loads((root/'validation.json').read_text(encoding='utf-8'))
    if status['state'] != 'complete': raise RuntimeError('Training is not complete; figures cannot be marked final.')
    if not validation['passed']: raise RuntimeError('Validation did not pass.')
    h = np.load(root/'history.npz')
    models = [(int(n), float(a)) for n, a in h['models']]
    if models != [tuple(m) for m in config['models']]: raise RuntimeError('History models differ from config.')
    steps, metrics = h['eval_steps'], h['metrics']
    layer_norms = np.sqrt(h['layer_norms_squared'])  # (steps+1, models, 3)
    norms = np.sqrt(h['layer_norms_squared'].sum(-1))
    budget = config['steps']
    if layer_norms.shape[0] != budget+1 or steps[0] != 0 or steps[-1] != budget: raise RuntimeError('Incomplete history.')
    if not (np.isfinite(metrics).all() and np.isfinite(layer_norms).all()): raise RuntimeError('Non-finite history values.')
    if not ((metrics[..., :2] >= 0).all() and (metrics[..., :2] <= 1).all()): raise RuntimeError('Accuracy outside [0, 1].')
    fit = [first_reach(steps, metrics[:, j, 0], 0.99) for j in range(len(models))]
    if [(-1 if f is None else f) for f in fit] != h['fit_step'].tolist(): raise RuntimeError('Fit steps disagree with run record.')
    checkpoints = sorted((root/'checkpoints').glob('*.pt'))
    if len(checkpoints) != len(models) + sum(f is not None for f in fit): raise RuntimeError('Missing fit/final checkpoints.')

    with (root/'curves.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle); writer.writerow(CURVE_FIELDS)
        for j, (n, alpha) in enumerate(models):
            for e, step in enumerate(steps):
                writer.writerow([seed, n, '%g' % alpha, int(step)] + ['%.12g' % v for v in metrics[e, j]]
                                + ['%.12g' % norms[step, j]] + ['%.12g' % v for v in layer_norms[step, j]])
    rows = []
    for j, (n, alpha) in enumerate(models):
        row = dict(seed=seed, n=n, alpha=alpha, fit_step='' if fit[j] is None else fit[j],
                   final_train_accuracy=float(metrics[-1, j, 0]), final_test_accuracy=float(metrics[-1, j, 1]),
                   max_test_accuracy=float(metrics[:, j, 1].max()), initial_norm=float(norms[0, j]), final_norm=float(norms[-1, j]))
        for t in THRESHOLDS:
            s = first_reach(steps, metrics[:, j, 1], t)
            row['steps_to_test_%d' % round(100*t)] = '' if s is None else s  # empty: not reached within budget
        rows.append(row)
    with (root/'time_to_generalize.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    figs = root/'figures'; figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    experiment = config.get('experiment', 'notebook')  # pre-refactor notebook configs have no key
    x = np.maximum(steps, 1)
    if experiment == 'fig3d': plot_fig3d(figs, seed, models, x, metrics, budget, config['weight_decay'])
    elif experiment == 'fig3e': plot_fig3e(figs, seed, models, rows, budget, config['weight_decay'])
    else: plot_notebook(figs, seed, models, x, metrics, norms, rows, budget)

    qa = dict(complete=True, experiment=experiment, seed=seed, models=len(models), evaluation_points=len(steps), steps=budget,
              weight_decay=config['weight_decay'],
              validation_passed=True, validation_max_parameter_abs_error=max(c['max_parameter_abs_error'] for c in validation['serial_comparisons']),
              all_values_finite=True, fit_step_matches_run_record=True,
              fit_steps={'N%d_alpha%g' % m: fit[j] for j, m in enumerate(models)},
              training_wall_seconds=status['elapsed_seconds'], checkpoint_files=len(checkpoints),
              total_checkpoint_bytes=sum(p.stat().st_size for p in checkpoints),
              history_bytes=(root/'history.npz').stat().st_size,
              training_code_sha256=hashlib.sha256(TRAINING_CODE.read_bytes()).hexdigest(),
              curve_export='curves.csv: scheduled full-split metrics; norms at the same steps (per-step norms only in history.npz)',
              loss_units='Reported loss is squared error summed over 10 outputs, averaged over samples; 10x element-mean MSE.',
              no_confidence_intervals='One seed only; no cross-seed uncertainty estimated.')
    (root/'qa_report.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa), flush=True)


def plot_fig3d(figs, seed, models, x, metrics, budget, wd):
    alphas = [a for _, a in models]
    norm = matplotlib.colors.Normalize(min(alphas), max(alphas))
    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout='constrained')
    for j, a in enumerate(alphas):
        ax.plot(x, metrics[:, j, 0], color=plt.cm.Reds(.25 + .75*norm(a)), lw=1)
        ax.plot(x, metrics[:, j, 1], color=plt.cm.Greens(.25 + .75*norm(a)), lw=1)
    ax.plot([], [], color='tab:red', label='train'); ax.plot([], [], color='tab:green', label='test')
    ax.set_xscale('log'); ax.set_xlim(1, budget); ax.set_ylim(0, 1.02); ax.legend(loc='lower right')
    ax.set_xlabel('Optimization steps'); ax.set_ylabel('Accuracy (full split)')
    fig.colorbar(plt.cm.ScalarMappable(norm, 'Greens'), ax=ax, label='Initialization scale alpha')
    ax.set_title('Figure 3d reproduction | N=1000, gamma=%g | seed %d' % (wd, seed))
    for ext in ('png', 'pdf'): fig.savefig(figs/('figure3d_seed%d.%s' % (seed, ext)), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_fig3e(figs, seed, models, rows, budget, wd):
    pts = [(n, r['steps_to_test_60']) for (n, _), r in zip(models, rows)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout='constrained')
    reached = [(n, s) for n, s in pts if s != '']
    if reached: ax.plot(*zip(*reached), '-o', color='tab:blue', ms=4, label='reached 60% test acc')
    missed = [n for n, s in pts if s == '']
    if missed: ax.plot(missed, [budget]*len(missed), '^', color='tab:red', alpha=.6, label='not reached in %d steps' % budget)
    ax.set_yscale('log'); ax.set_xlabel('Training set size N'); ax.set_ylabel('Steps to test accuracy >= 60%')
    ax.set_title('Figure 3e reproduction | alpha=8, gamma=%g | seed %d' % (wd, seed)); ax.legend(fontsize=8)
    for ext in ('png', 'pdf'): fig.savefig(figs/('figure3e_seed%d.%s' % (seed, ext)), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_notebook(figs, seed, models, x, metrics, norms, rows, budget):
    main_j, ctrl_j = models.index((1000, 8.0)), models.index((1000, 1.0))
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout='constrained')
    ax.plot(x, metrics[:, main_j, 0], color='red', label='train (alpha=8)')
    ax.plot(x, metrics[:, main_j, 1], color='green', label='test (alpha=8)')
    ax.plot(x, metrics[:, ctrl_j, 0], color='red', ls=':', alpha=.6, label='train (alpha=1 control)')
    ax.plot(x, metrics[:, ctrl_j, 1], color='green', ls=':', alpha=.6, label='test (alpha=1 control)')
    ax.set_xscale('log'); ax.set_xlim(1, budget); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Optimization steps'); ax.set_ylabel('Accuracy (full split)'); ax.legend(loc='center left', fontsize=8)
    ax2 = ax.twinx(); ax2.plot(np.arange(1, budget+1), norms[1:, main_j], color='purple'); ax2.set_ylabel('Weight norm (alpha=8)', color='purple')
    ax.set_title('Official notebook setting | N=1000, gamma=0.01 | seed %d' % seed)
    for ext in ('png', 'pdf'): fig.savefig(figs/('notebook_grokking_curve_seed%d.%s' % (seed, ext)), dpi=180, bbox_inches='tight')
    plt.close(fig)

    grok = [(j, n) for j, (n, a) in enumerate(models) if a == 8.0]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout='constrained')
    for k, (t, color) in enumerate(zip(THRESHOLDS, plt.cm.viridis(np.linspace(0, .9, len(THRESHOLDS))))):
        key = 'steps_to_test_%d' % round(100*t)
        pts = [(n, rows[j][key]) for j, n in grok]
        reached = [(n, s) for n, s in pts if s != '']
        if reached: ax.plot(*zip(*reached), '-o', color=color, ms=4, label='test acc >= %.0f%%' % (100*t))
        missed = [n for n, s in pts if s == '']
        if missed: ax.plot(missed, [budget*1.15*1.12**k]*len(missed), 'v', mfc='none', color=color, ms=6)
    ax.axhline(budget, color='gray', lw=.7, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Training set size N'); ax.set_ylabel('Steps to reach threshold')
    ax.set_title('Official notebook setting | alpha=8, gamma=0.01 | seed %d\ntriangles: not reached by %d steps' % (seed, budget), fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    for ext in ('png', 'pdf'): fig.savefig(figs/('notebook_time_to_generalize_seed%d.%s' % (seed, ext)), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True); main(p.parse_args().root)
