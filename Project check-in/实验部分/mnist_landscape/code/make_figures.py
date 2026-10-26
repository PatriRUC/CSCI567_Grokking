"""Render raw and smoothed single-seed landscape figures and verify completeness."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def edges(values):
    logs = np.log10(values)
    mids = .5*(logs[:-1]+logs[1:])
    return 10**np.r_[logs[0]-(mids[0]-logs[0]), mids, logs[-1]+(logs[-1]-mids[-1])]


def main(root):
    config = json.loads((root/'config.json').read_text())
    seed = int(config['seed'])
    status = json.loads((root/'status.json').read_text())
    if status['state'] != 'complete': raise RuntimeError('Training is not complete; figures cannot be marked final.')
    with (root/'results.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    alphas = np.array(config['alphas'])
    ns = np.array(config['all_data_sizes'])
    original_ns = np.array(config['original_data_sizes'])
    lookup = {(int(r['n']), int(r['alpha_index'])): r for r in rows}
    expected = {(int(n), k) for n in ns for k in range(len(alphas))}
    if len(rows) != len(expected) or set(lookup) != expected: raise RuntimeError('Missing or duplicate grid points.')
    numeric_columns = [c for c in rows[0] if c not in ('original_grid',)]
    if not all(np.isfinite(float(r[c])) for r in rows for c in numeric_columns): raise RuntimeError('Non-finite result field.')
    max_drift = max(float(r['max_norm_relative_error']) for r in rows)
    if max_drift > 2e-6: raise RuntimeError('Norm drift exceeds tolerance.')
    figs = root/'figures'; figs.mkdir(exist_ok=True)
    tables = root/'tables'; tables.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42, 'ps.fonttype':42})
    arrays = {}
    for field in ('paper_best_train_accuracy', 'paper_best_test_accuracy', 'paper_min_train_loss', 'paper_min_test_loss',
                  'selected_full_train_accuracy', 'selected_full_test_accuracy', 'selected_full_train_loss', 'selected_full_test_loss',
                  'final_full_train_accuracy', 'final_full_test_accuracy'):
        arrays[field] = np.array([[float(lookup[(int(n),k)][field]) for k in range(len(alphas))] for n in ns])
    np.savez_compressed(tables/('grid_seed%d.npz'%seed), ns=ns, alphas=alphas, original_ns=original_ns, **arrays)
    for n in (200, 8000):
        chosen = [lookup[(n,k)] for k in range(len(alphas))]
        with (tables/('slice_N%d_seed%d.csv'%(n,seed))).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(chosen)
    orig_idx = [int(np.where(ns == n)[0][0]) for n in original_ns]

    def grids(mode, smooth):
        prefix = 'selected_full' if mode == 'train_selected' else 'paper_best'
        a = 1-arrays[prefix+'_train_accuracy']
        b = 1-arrays[prefix+'_test_accuracy']
        a,b = a[orig_idx], b[orig_idx]
        if smooth: a,b = gaussian_filter(a, 1), gaussian_filter(b, 1)
        return a,b

    def heatmap(ax, z, title, fig):
        mesh = ax.pcolormesh(edges(alphas), edges(original_ns), z, cmap='Reds', vmin=0, vmax=1, shading='flat', rasterized=True)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(alphas[0],alphas[-1]); ax.set_ylim(original_ns[0],original_ns[-1])
        ax.set_xlabel('Relative weight norm alpha'); ax.set_ylabel('Training samples N'); ax.set_title(title)
        ax.set_xticks([.1,1,10]); ax.set_xticklabels(['0.1','1','10'])
        ax.set_yticks([10,100,1000,10000,60000]); ax.set_yticklabels(['10','100','1,000','10,000','60,000'])
        levels = [v for v in (.1,.2,.3,.4) if z.min()<v<z.max()]
        if levels:
            cs = ax.contour(alphas, original_ns, z, levels=levels, colors='#716500', linewidths=.7, linestyles='dashed')
            ax.clabel(cs, fontsize=7, fmt='%.1f')
        fig.colorbar(mesh, ax=ax, label='Classification error', fraction=.046, pad=.04)

    def slice_plot(ax, n, mode, title):
        prefix = 'selected_full' if mode == 'train_selected' else 'paper_best'
        row = int(np.where(ns == n)[0][0])
        for label, color, key in [('Train','#1f77b4',prefix+'_train_accuracy'),('Test','#ff7f0e',prefix+'_test_accuracy')]:
            ax.plot(alphas, 1-arrays[key][row], '-o', color=color, markersize=3, linewidth=1.4, label=label)
        ax.set_xscale('log'); ax.set_xlim(.1,10); ax.set_ylim(-.02,1.02)
        ax.set_xticks([.1,1,10]); ax.set_xticklabels(['0.1','1','10'])
        ax.set_ylabel('Classification error'); ax.set_title(title)
        ax.grid(alpha=.2); ax.legend(loc='best')

    for mode,smooth in [('paper_style',False),('paper_style',True),('train_selected',False)]:
        suffix = 'train_selected_full' if mode == 'train_selected' else ('smoothed' if smooth else 'raw')
        a,b = grids(mode,smooth)
        fig = plt.figure(figsize=(16,6.2), layout='constrained')
        spec = fig.add_gridspec(2,3,width_ratios=[1.1,1.1,1])
        heatmap(fig.add_subplot(spec[:,0]), a, '(a) Training error', fig)
        heatmap(fig.add_subplot(spec[:,1]), b, '(b) Test error', fig)
        slice_plot(fig.add_subplot(spec[0,2]), 200, mode, '(c) N = 200 (exact samples)')
        ax = fig.add_subplot(spec[1,2]); slice_plot(ax,8000,mode,'N = 8,000 (exact samples)'); ax.set_xlabel('Relative weight norm alpha')
        description = 'Checkpoint selected by training loss; full-split evaluation' if mode=='train_selected' else 'Independent best accuracies over scheduled evaluations'
        if smooth: description += '; heatmaps Gaussian-smoothed (sigma = 1)'
        fig.suptitle('MNIST fixed-norm landscape | seed %d\n'%seed+description, fontsize=12)
        for ext in ('png','pdf'): fig.savefig(figs/('figure3abc_seed%d_'%seed+suffix+'.'+ext), dpi=180, bbox_inches='tight')
        plt.close(fig)
        if mode=='paper_style' and not smooth:
            for label,z,title in [('a',a,'Training error'),('b',b,'Test error')]:
                fig,ax = plt.subplots(figsize=(6,5),layout='constrained'); heatmap(ax,z,'Figure 3'+label+' reproduction | seed '+str(seed),fig)
                for ext in ('png','pdf'): fig.savefig(figs/('figure3'+label+'_seed%d.'%seed+ext),dpi=180,bbox_inches='tight')
                plt.close(fig)
            fig,axes = plt.subplots(2,1,figsize=(6,6),sharex=True,layout='constrained')
            slice_plot(axes[0],200,mode,'N = 200'); slice_plot(axes[1],8000,mode,'N = 8,000'); axes[1].set_xlabel('Relative weight norm alpha')
            fig.suptitle('Figure 3c reproduction | exact N | seed '+str(seed))
            for ext in ('png','pdf'): fig.savefig(figs/('figure3c_seed%d.'%seed+ext),dpi=180,bbox_inches='tight')
            plt.close(fig)
    group_times = [float(lookup[(int(n),0)]['group_seconds']) for n in ns]
    checkpoints = sorted((root/'groups').glob('N*.pt'))
    if len(checkpoints) != len(ns): raise RuntimeError('Missing group checkpoints.')
    qa = dict(complete=True, seed_values=sorted(set(int(r['seed']) for r in rows)),
              expected_conditions=len(expected), actual_conditions=len(rows), original_grid_conditions=len(original_ns)*len(alphas),
              added_exact_slice_conditions=2*len(alphas), completed_group_checkpoints=len(checkpoints),
              maximum_norm_relative_error=max_drift, norm_error_checked_at_scheduled_evaluations=True, all_numeric_fields_finite=True,
              total_group_wall_seconds=sum(group_times), figure_files=len(list(figs.glob('*'))),
              total_checkpoint_bytes=sum(p.stat().st_size for p in checkpoints),
              no_confidence_intervals='One seed only; no cross-seed uncertainty estimated.',
              loss_units='Reported loss is squared error summed over 10 outputs, averaged over samples; 10x element-mean MSE.',
              selected_full_evaluation='Each model selected using sampled training loss, then evaluated on all N train / 10000 test samples.')
    (root/'qa_report.json').write_text(json.dumps(qa,indent=2))
    manifest = {}
    for path in [root/'results.csv',root/'config.json',root/'validation.json',*figs.glob('*'),*tables.glob('*')]:
        manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(qa),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); main(p.parse_args().root)