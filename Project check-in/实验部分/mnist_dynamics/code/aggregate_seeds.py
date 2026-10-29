"""CPU aggregation and plots for one experiment from portable curve CSV and configs; no torch or server paths."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
FIT_ACCURACY = 0.99
SEEDS = (0, 1, 2)
METRICS = ['train_accuracy', 'test_accuracy', 'train_loss', 'test_loss', 'global_norm']


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def first_reach(steps, values, threshold):
    hit = np.nonzero(values >= threshold)[0]
    return int(steps[hit[0]]) if len(hit) else None


def save(fig, figdir, name, pdf=True):
    for ext in (('png', 'pdf') if pdf else ('png',)): fig.savefig(figdir/(name+'.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def aggregate(root, experiment, input_csv, configs_dir):
    root.mkdir(parents=True, exist_ok=True)
    configs = [json.loads((configs_dir/('seed%d.json' % s)).read_text(encoding='utf-8')) for s in SEEDS]
    for key in configs[0]:
        if key != 'seed': assert all(c[key] == configs[0][key] for c in configs), key
    assert [c['seed'] for c in configs] == list(SEEDS)
    assert configs[0].get('experiment', 'notebook') == experiment
    models = [(int(n), float(a)) for n, a in configs[0]['models']]
    budget, wd = configs[0]['steps'], configs[0]['weight_decay']
    rows = read_csv(input_csv)
    groups = {}
    for r in rows: groups.setdefault((int(r['seed']), int(r['n']), float(r['alpha'])), []).append(r)
    assert set(groups) == {(s, n, a) for s in SEEDS for n, a in models}
    steps = np.array([int(r['step']) for r in groups[(0,) + models[0]]])
    assert steps[0] == 0 and steps[-1] == budget and (np.diff(steps) > 0).all()
    arrays = {}  # metric -> (seed, eval step, model)
    for m in METRICS:
        arrays[m] = np.empty((len(SEEDS), len(steps), len(models)))
        for s in SEEDS:
            for j, key in enumerate(models):
                g = groups[(s,) + key]
                assert [int(r['step']) for r in g] == steps.tolist()
                arrays[m][s, :, j] = [float(r[m]) for r in g]
    assert all(np.isfinite(v).all() for v in arrays.values())
    train, test, norm = arrays['train_accuracy'], arrays['test_accuracy'], arrays['global_norm']

    per_seed, summary = [], []
    for j, (n, alpha) in enumerate(models):
        fit = [first_reach(steps, train[s, :, j], FIT_ACCURACY) for s in SEEDS]
        reach = {t: [first_reach(steps, test[s, :, j], t) for s in SEEDS] for t in THRESHOLDS}
        for s in SEEDS:
            row = dict(seed=s, n=n, alpha=alpha, fit_step='' if fit[s] is None else fit[s],
                       final_train_accuracy=train[s, -1, j], final_test_accuracy=test[s, -1, j],
                       max_test_accuracy=test[s, :, j].max(), initial_norm=norm[s, 0, j], final_norm=norm[s, -1, j])
            for t in THRESHOLDS: row['steps_to_test_%d' % round(100*t)] = '' if reach[t][s] is None else reach[t][s]
            per_seed.append({k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()})
        row = dict(n=n, alpha=alpha, seed_count=len(SEEDS))
        got_fit = [f for f in fit if f is not None]
        row['fit_step_reached'] = len(got_fit)
        row['fit_step_mean'] = float(np.mean(got_fit)) if len(got_fit) == len(SEEDS) else ''
        row['fit_step_std'] = float(np.std(got_fit, ddof=1)) if len(got_fit) == len(SEEDS) else ''
        for name, values in [('final_train_accuracy', train[:, -1, j]), ('final_test_accuracy', test[:, -1, j]),
                             ('max_test_accuracy', test[:, :, j].max(1)), ('final_norm', norm[:, -1, j])]:
            v = np.asarray(values, dtype=float); row[name+'_mean'] = float(v.mean()); row[name+'_std'] = float(v.std(ddof=1))
        for t in THRESHOLDS:
            key = 'steps_to_test_%d' % round(100*t)
            got = [v for v in reach[t] if v is not None]
            row[key+'_reached'] = len(got)
            # Mean/SD only when every seed reached; the budget is never substituted for a missing time.
            row[key+'_mean'] = float(np.mean(got)) if len(got) == len(SEEDS) else ''
            row[key+'_std'] = float(np.std(got, ddof=1)) if len(got) == len(SEEDS) else ''
        summary.append(row)
    write_csv(root/'time_to_generalize_all_seeds.csv', per_seed)
    write_csv(root/'summary_by_condition.csv', summary)

    figdir = root/'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    x = np.maximum(steps, 1)
    if experiment == 'fig3d': plot_fig3d(figdir, models, x, train, test, norm, budget, wd)
    elif experiment == 'fig3e': plot_fig3e(figdir, models, per_seed, x, test, budget, wd)
    else: plot_notebook(figdir, models, per_seed, x, train, test, norm, budget)

    qa = dict(experiment=experiment, figure_source=configs[0].get('figure_source', 'official mnist-grokking.ipynb defaults'),
              weight_decay=wd, seeds=list(SEEDS), models=len(models), evaluation_points=len(steps), total_curve_rows=len(rows),
              same_config_except_seed=True, all_numeric_fields_finite=True, steps=budget,
              fit_definition='first scheduled step with full train accuracy >= %.2f' % FIT_ACCURACY,
              threshold_definition='first scheduled step with full test accuracy >= threshold; empty if not reached within budget',
              standard_deviation_definition='sample SD, ddof=1, across seeds; reported only when all seeds reached; not confidence interval',
              steps_to_test_60_mean={'N%d_alpha%g' % (r['n'], r['alpha']): r['steps_to_test_60_mean'] for r in summary},
              thresholds_not_reached={'N%d_alpha%g' % (r['n'], r['alpha']): {str(t): len(SEEDS) - r['steps_to_test_%d_reached' % round(100*t)]
                                      for t in THRESHOLDS if r['steps_to_test_%d_reached' % round(100*t)] < len(SEEDS)} for r in summary})
    (root/'aggregate_qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print('Verified %d curve rows / %d models / seeds 0,1,2 (%s); output: %s' % (len(rows), len(models), experiment, root))


def plot_fig3d(figdir, models, x, train, test, norm, budget, wd):
    alphas = [a for _, a in models]
    cn = matplotlib.colors.Normalize(min(alphas), max(alphas))
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout='constrained')
    for j, a in enumerate(alphas):
        for arr, cmap in [(train, plt.cm.Reds), (test, plt.cm.Greens)]:
            c = cmap(.25 + .75*cn(a))
            ax.plot(x, arr[:, :, j].mean(0), color=c, lw=1.1)
            ax.fill_between(x, arr[:, :, j].min(0), arr[:, :, j].max(0), color=c, alpha=.25, lw=0)
    ax.plot([], [], color='tab:red', label='train'); ax.plot([], [], color='tab:green', label='test')
    ax.set_xscale('log'); ax.set_xlim(1, budget); ax.set_ylim(0, 1.02); ax.legend(loc='lower right')
    ax.set_xlabel('Optimization steps'); ax.set_ylabel('Accuracy (full split)')
    fig.colorbar(plt.cm.ScalarMappable(cn, 'Reds'), ax=ax, pad=.01, label='')
    fig.colorbar(plt.cm.ScalarMappable(cn, 'Greens'), ax=ax, pad=.01, label='Initialization scale alpha')
    ax.set_title('Figure 3d reproduction | N=1000, gamma=%g | seeds 0, 1, 2\nmean; band = min-max across seeds' % wd, fontsize=10)
    save(fig, figdir, 'figure3d_seeds012')
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout='constrained')
    for j, a in enumerate(alphas):
        ax.plot(x, norm[:, :, j].mean(0), color=plt.cm.viridis(cn(a)*.9), lw=1.1, label='alpha=%g' % a)
    ax.set_xscale('log'); ax.set_xlim(1, budget); ax.set_xlabel('Optimization steps'); ax.set_ylabel('Global L2 weight norm (mean of seeds)')
    ax.set_title('Figure 3d runs: weight norm | N=1000, gamma=%g' % wd, fontsize=10); ax.legend(fontsize=7, ncol=2)
    save(fig, figdir, 'figure3d_weight_norm_seeds012', pdf=False)


def plot_fig3e(figdir, models, per_seed, x, test, budget, wd):
    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout='constrained')
    means, label_dot, label_tri = [], 'reached 60%% test acc in %d steps' % budget, "didn't reach 60%% test acc in %d steps" % budget
    line = []  # (N, y, censored): mean where all seeds reached, budget where none did
    for n, _ in models:
        vals = [r['steps_to_test_60'] for r in per_seed if r['n'] == n]
        got = [v for v in vals if v != '']
        if got: ax.plot([n]*len(got), got, 'o', color='tab:blue', alpha=.35, ms=5, label=label_dot); label_dot = None
        if len(got) < len(vals):
            ax.plot([n]*(len(vals)-len(got)), [budget]*(len(vals)-len(got)), '^', color='tab:red', alpha=.5, ms=6, label=label_tri); label_tri = None
            ax.annotate('%d/%d seeds' % (len(vals)-len(got), len(vals)), (n, budget), textcoords='offset points', xytext=(7, -3), fontsize=7, color='tab:red')
        if len(got) == len(vals): means.append((n, np.mean(got))); line.append((n, np.mean(got), False))
        elif not got: line.append((n, budget, True))
    if means: ax.plot(*zip(*means), '-', color='tab:blue', lw=1.6, label='mean (all 3 seeds reached)')
    # Dashed links to the budget: no seed reached there, so the true time is only bounded below.
    for (n0, y0, c0), (n1, y1, c1) in zip(line, line[1:]):
        if c0 or c1: ax.plot([n0, n1], [y0, y1], '--', color='tab:blue', lw=1.2)
    ax.set_yscale('log'); ax.set_xlabel('Train points N'); ax.set_ylabel('Steps to test accuracy > 60%')
    ax.set_title('Figure 3e reproduction | alpha=8, gamma=%g | seeds 0, 1, 2' % wd, fontsize=10); ax.legend(fontsize=8)
    save(fig, figdir, 'figure3e_seeds012')
    test_curves_by_seed(figdir, models, x, test, budget, 'figure3e_test_curves_by_seed', lambda n, a: 'N=%d' % n)


def test_curves_by_seed(figdir, models, x, test, budget, name, label, extra=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True, layout='constrained')
    colors = plt.cm.plasma(np.linspace(0, .85, len(models)))
    for s, ax in enumerate(axes):
        for j, ((n, a), color) in enumerate(zip(models, colors)):
            if extra and extra(n, a): ax.plot(x, test[s, :, j], color='gray', ls=':', lw=1, label=label(n, a))
            else: ax.plot(x, test[s, :, j], color=color, lw=1, label=label(n, a))
        ax.set_xscale('log'); ax.set_xlim(1, budget); ax.set_ylim(0, 1); ax.set_title('Seed %d' % s); ax.set_xlabel('Optimization steps')
    axes[0].set_ylabel('Test accuracy (full split)'); axes[-1].legend(fontsize=7, loc='upper left')
    save(fig, figdir, name, pdf=False)


def plot_notebook(figdir, models, per_seed, x, train, test, norm, budget):
    main_j, ctrl_j = models.index((1000, 8.0)), models.index((1000, 1.0))
    fig, ax = plt.subplots(figsize=(6.6, 4.5), layout='constrained')
    for j, ls, tag in [(main_j, '-', 'alpha=8'), (ctrl_j, ':', 'alpha=1 control')]:
        for arr, color, split in [(train, 'red', 'train'), (test, 'green', 'test')]:
            v = arr[:, :, j]
            ax.plot(x, v.mean(0), ls=ls, color=color, alpha=1 if ls == '-' else .6, label='%s (%s)' % (split, tag))
            if ls == '-': ax.fill_between(x, v.min(0), v.max(0), color=color, alpha=.2, lw=0)
    ax.set_xscale('log'); ax.set_xlim(1, budget); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Optimization steps'); ax.set_ylabel('Accuracy (full split)'); ax.legend(loc='center left', fontsize=8)
    ax2 = ax.twinx(); nv = norm[:, :, main_j]
    ax2.plot(x, nv.mean(0), color='purple'); ax2.fill_between(x, nv.min(0), nv.max(0), color='purple', alpha=.2, lw=0)
    ax2.set_ylabel('Weight norm (alpha=8)', color='purple')
    ax.set_title('Official notebook setting (supplement) | N=1000, gamma=0.01 | seeds 0, 1, 2\nmean; band = min-max across seeds', fontsize=10)
    save(fig, figdir, 'notebook_grokking_curve_seeds012')

    grok = [(j, n) for j, (n, a) in enumerate(models) if a == 8.0]
    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout='constrained')
    for k, (thr, color) in enumerate(zip(THRESHOLDS, plt.cm.viridis(np.linspace(0, .9, len(THRESHOLDS))))):
        key = 'steps_to_test_%d' % round(100*thr)
        mean_pts, missed = [], []
        for j, n in grok:
            got = [r[key] for r in per_seed if r['n'] == n and r['alpha'] == 8.0 and r[key] != '']
            ax.plot([n]*len(got), got, 'o', color=color, ms=3, alpha=.45)
            if len(got) == len(SEEDS): mean_pts.append((n, np.mean(got)))
            else: missed.append((n, len(SEEDS) - len(got)))
        if mean_pts: ax.plot(*zip(*mean_pts), '-', color=color, lw=1.6, label='test acc >= %.0f%%' % (100*thr))
        y = budget*1.15*1.12**k  # stagger thresholds so markers do not overlap
        for n, m in missed:
            ax.plot(n, y, 'v', mfc='none', color=color, ms=6)
            ax.annotate(str(m), (n, y), textcoords='offset points', xytext=(5, -3), fontsize=6, color=color)
    ax.axhline(budget, color='gray', lw=.7, ls='--')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Training set size N'); ax.set_ylabel('Steps to reach threshold')
    ax.set_title('Official notebook setting (supplement) | alpha=8, gamma=0.01 | seeds 0, 1, 2\n'
                 'dots = seeds, line = mean if 3/3 reached; triangles = seeds not reaching by %d steps' % budget, fontsize=9)
    ax.legend(fontsize=8, loc='lower left')
    save(fig, figdir, 'notebook_time_to_generalize_seeds012')
    test_curves_by_seed(figdir, models, x, test, budget, 'notebook_test_curves_by_seed',
                        lambda n, a: 'N=%d' % n if a == 8.0 else 'N=1000, alpha=1', extra=lambda n, a: a != 8.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['fig3d', 'fig3e', 'notebook'], required=True)
    parser.add_argument('--input', type=Path, help='default: results/<experiment>/all_seeds_curves.csv')
    parser.add_argument('--configs-dir', type=Path, help='default: configs/<experiment>')
    parser.add_argument('--output', type=Path, help='default: generated/summary/<experiment>')
    a = parser.parse_args()
    aggregate(a.output or PACKAGE/'generated/summary'/a.experiment, a.experiment,
              a.input or PACKAGE/'results'/a.experiment/'all_seeds_curves.csv', a.configs_dir or PACKAGE/'configs'/a.experiment)
