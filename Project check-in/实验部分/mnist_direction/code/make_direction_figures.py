"""Figures and a first-pass written analysis for one run of run_direction.py. CPU only: numpy + matplotlib.
Reads <root>/history.npz, summary.json, config.json; writes <root>/figures/*.png|pdf and <root>/PRELIMINARY_ANALYSIS.md.
With --export <package dir> it writes figures/direction/*_seed<seed>.png|pdf and results/direction/seed<seed>_analysis.md there instead.
The cross-seed summary (results/SUMMARY.md) is written by summarize_seeds.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_COLORS = {1.0: '#2a78d6', 2.0: '#eb6834', 0.5: '#1baf7a'}   # colour follows the c value, not the run order
SPARE = ['#eda100', '#e87ba4', '#008300', '#6250d6']


def styles(groups, cs):
    spare, out = iter(SPARE), []
    for j, g in enumerate(groups):
        if j == 0:
            out.append(dict(color=INK, lw=1.3, ls=(0, (4, 2)), zorder=5, label='reference (plain AdamW)'))
        else:
            out.append(dict(color=C_COLORS.get(float(cs[j])) or next(spare), lw=2.0, ls='-', zorder=3, label='%s, norm matched' % g))
    return out


def panel(ax, title, xlabel, ylabel, logx=True, logy=False):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=SECONDARY)
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=SECONDARY, labelsize=8, length=3, color=AXIS)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(AXIS)


def finish(fig, axes, path, title):
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=len(names), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + ext), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def windowed_max(values, idx):
    """Max of a per-step series over each window that ends at idx[i]; keeps spikes visible after downsampling."""
    edges = np.concatenate([[0], idx])
    return np.array([np.nanmax(values[edges[i] + 1:edges[i + 1] + 1]) for i in range(len(idx))])


def figures(out, suffix, h, config):
    groups, cs = [str(g) for g in h['groups']], h['cs']
    st = styles(groups, cs)
    order = list(range(1, len(groups))) + [0]            # reference drawn last, dashed, so overlaps stay visible
    ev, m, ps = h['eval_steps'], h['metrics'], h['per_step']
    steps = ps.shape[0] - 1
    idx = np.unique(np.round(np.logspace(0, np.log10(steps), 1500)).astype(int))
    norm = ps[:, :, 0]
    with np.errstate(invalid='ignore', divide='ignore'):
        norm_error = np.abs(norm - norm[:, :1]) / norm[:, :1]
        rel_dist = ps[:, :, 5] / norm[:, :1]
    cum_angle = np.nancumsum(ps[:, :, 4], 0)
    layer = np.sqrt(h['layer_norms_squared'])
    x_label = 'updates after the fork (fork = step %d)' % int(h['fork_step'])
    head = 'MNIST, N=%d, alpha=%g, weight decay=%g, seed %d' % (config['n'], config['alpha'], config['weight_decay'], config['seed'])
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), facecolor=SURFACE)
    panel(axes[0, 0], 'Test accuracy', x_label, 'accuracy')
    panel(axes[0, 1], 'Train accuracy', x_label, 'accuracy')
    panel(axes[0, 2], 'Train MSE (full training subset)', x_label, 'MSE', logy=True)
    panel(axes[1, 0], 'Global parameter norm', x_label, 'L2 norm')
    panel(axes[1, 1], 'Norm mismatch vs reference', x_label, 'max |norm - R| / R per window', logy=True)
    panel(axes[1, 2], 'Cumulative direction change', x_label, 'sum of per-step angles (rad)')
    for j in order:
        s = st[j]; kw = dict(color=s['color'], lw=s['lw'], ls=s['ls'], zorder=s['zorder'])
        axes[0, 0].plot(ev[1:], m[1:, j, 2], label=s['label'], **kw)
        axes[0, 1].plot(ev[1:], m[1:, j, 0], **kw)
        axes[0, 2].plot(ev[1:], m[1:, j, 3] / 10, **kw)
        axes[1, 0].plot(idx, norm[idx, j], **kw)
        if j: axes[1, 1].plot(idx, np.maximum(windowed_max(norm_error[:, j], idx), 1e-17), **kw)
        axes[1, 2].plot(idx, cum_angle[idx, j], **kw)
    axes[0, 0].axhline(0.6, color=MUTED, lw=0.8, zorder=1)
    axes[0, 0].annotate('60% threshold', (1, 0.6), xytext=(2, 3), textcoords='offset points', fontsize=8, color=MUTED)
    axes[1, 1].axhline(config['tolerances']['norm_relative'], color=MUTED, lw=0.8, zorder=1)
    axes[1, 1].annotate('tolerance', (1, config['tolerances']['norm_relative']), xytext=(2, 3), textcoords='offset points', fontsize=8, color=MUTED)
    finish(fig, axes, out / ('direction_curves' + suffix), 'Tangential-update intervention under a shared norm trajectory: ' + head)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), facecolor=SURFACE)
    panel(axes[0, 0], 'Test accuracy vs direction travelled', 'cumulative angle (rad)', 'accuracy', logx=False)
    panel(axes[0, 1], 'Per-step angle (mean per window)', x_label, 'angle per update (rad)', logy=True)
    panel(axes[0, 2], 'Distance to the reference parameters', x_label, '||theta - theta_ref|| / R', logy=True)
    for l in range(3):
        panel(axes[1, l], 'Layer %d norm (weight + bias)' % (l + 1), x_label, 'L2 norm')
    angle_sum = np.nancumsum(ps[:, :, 4], 0)
    edges = np.concatenate([[0], idx])
    for j in order:
        s = st[j]; kw = dict(color=s['color'], lw=s['lw'], ls=s['ls'], zorder=s['zorder'])
        axes[0, 0].plot(cum_angle[ev, j], m[:, j, 2], label=s['label'], **kw)
        axes[0, 1].plot(idx, (angle_sum[idx, j] - angle_sum[edges[:-1], j]) / np.diff(edges), **kw)
        if j: axes[0, 2].plot(idx, np.maximum(windowed_max(rel_dist[:, j], idx), 1e-17), **kw)
        for l in range(3):
            axes[1, l].plot(idx, layer[idx, j, l], **kw)
    finish(fig, axes, out / ('direction_diagnostics' + suffix), 'Direction and layer-norm diagnostics: ' + head)


def fmt_step(v):
    return '未达到' if v is None else '{:,}'.format(v)


def analysis(path, figure_dir, suffix, summary, config):
    g = {x['group']: x for x in summary['groups']}
    names = [x['group'] for x in summary['groups']]
    fork, rep, res = summary['fork'], summary['replay_control'], summary['resume_check']
    steps = config['steps_after_fork']
    L = []
    L.append('# 方向干预实验初步分析（seed %d，单 seed）' % config['seed'])
    L.append('')
    L.append('本文件由 `make_direction_figures.py` 根据 `summary.json` 自动生成，只描述这一个 seed；下面的比较是单 seed 观察，跨 seed 的汇总由 `summarize_seeds.py` 另行生成。')
    L.append('')
    L.append('## 1. 设置')
    L.append('')
    L.append('- 模型与数据：784→200→200→10 ReLU MLP，前 %d 个 MNIST 训练样本，alpha=%g，AdamW lr=%g，weight decay=%g，batch=%d，FP64。'
             % (config['n'], config['alpha'], config['lr'], config['weight_decay'], config['batch_size']))
    L.append('- 分叉点：第 **{:,}** 步（训练准确率首次 ≥ 99% 的评估步）。此时训练准确率 {:.1%}，训练 MSE {:.4g}，测试准确率 {:.1%}，全局范数 {:.2f}。'
             .format(fork['step'], fork['train_acc'], fork['train_loss'] / 10, fork['test_acc'], fork['global_norm']))
    L.append('- 分叉后每组 {:,} 步，各组使用同一批次序列，并在每一步把全局范数缩放到参考运行的 R_t。'.format(steps))
    rn = summary['reference_norm']
    L.append('- 参考运行的范数：分叉时 {:.2f}，最小 {:.2f}，末步 {:.2f}。'.format(rn['at_fork'], rn['minimum'], rn['final']))
    L.append('')
    L.append('## 2. 控制是否成立')
    L.append('')
    L.append('| 检查 | 结果 | 判定 |')
    L.append('|---|---|---|')
    L.append('| 检查点恢复：参考切片对照不中断的原模型（前 %d 步） | 参数最大绝对差 %.2e | %s |'
             % (res['steps'], res['max_parameter_abs_error'], '通过' if res['passed'] else '**未通过**'))
    L.append('| c=1 重放：前 %d 步与参考运行的相对距离 | 最大 %.2e（容差 %.0e） | %s |'
             % (rep['window_steps'], rep['max_relative_distance_in_window'], config['tolerances']['replay_relative'], '通过' if rep['passed'] else '**未通过**'))
    L.append('| 各组范数匹配：max \\|范数 − R_t\\| / R_t | %s | %s |'
             % ('；'.join('%s %.1e' % (n, g[n]['max_norm_relative_error']) for n in names[1:]), '通过' if summary['norms_matched'] else '**未通过**'))
    L.append('')
    d = rep['max_abs_metric_difference_full_run']
    L.append('c=1 在整个预算内与参考运行的相对距离最大为 %.2e，测试准确率最大相差 %.2f 个百分点，训练 MSE 最大相差 %.2e。'
             % (rep['max_relative_distance_full_run'], 100 * d['test_acc'], d['train_loss'] / 10))
    L.append('c=1 与参考运行只差投影带来的舍入误差（约 1e-16），这个差距会在训练中被逐步放大；它给出本实验对微小扰动的敏感度，下文把它当作噪声底。')
    L.append('')
    if not summary['interpretable']:
        L.append('**控制未全部通过，以下各组结果不应解释。**')
        L.append('')
    L.append('## 3. 泛化等待步数（自分叉点起）')
    L.append('')
    ths = ['%d' % (100 * t) for t in config['thresholds']]
    L.append('| 组 | ' + ' | '.join('测试 ≥%s%%' % t for t in ths) + ' | 验证 ≥60% | 末步测试准确率 |')
    L.append('|---|' + '---:|' * (len(ths) + 2))
    for n in names:
        L.append('| %s | ' % n + ' | '.join(fmt_step(g[n]['steps_to_test'][t]) for t in ths)
                 + ' | %s | %.1f%% |' % (fmt_step(g[n]['steps_to_val']['60']), 100 * g[n]['final']['test_acc']))
    L.append('')
    L.append('"未达到"表示在 {:,} 步预算内没有达到该阈值，不能用预算上限代替。1,000 步之后评估间隔为 100 步。'.format(steps))
    L.append('')
    L.append('## 4. 相同步数处的训练 MSE 与测试准确率')
    L.append('')
    marks = list(g[names[0]]['train_mse_at'])
    L.append('| 分叉后步数 | ' + ' | '.join('%s 训练 MSE' % n for n in names) + ' | ' + ' | '.join('%s 测试' % n for n in names) + ' |')
    L.append('|---:|' + '---:|' * (2 * len(names)))
    for s in marks:
        L.append('| {:,} | '.format(int(s)) + ' | '.join('%.3e' % g[n]['train_mse_at'][s] for n in names) + ' | '
                 + ' | '.join('%.1f%%' % (100 * g[n]['test_acc_at'][s]) for n in names) + ' |')
    L.append('')
    L.append('## 5. 实际方向变化')
    L.append('')
    L.append('| 组 | 累计角度 (rad) | 相对参考运行 | AdamW 候选更新的平均切向范数（乘 c 之前） | 末步与参考参数的相对距离 | 首次出现非有限损失 |')
    L.append('|---|---:|---:|---:|---:|---:|')
    for n in names:
        x = g[n]
        L.append('| %s | %.3f | %.2f× | %.3e | %.2e | %s |' % (n, x['total_angle_rad'], x['total_angle_vs_reference'], x['mean_tangential_norm'],
                                                         x['final_relative_distance_to_reference'],
                                                         '无' if x['first_non_finite_step'] is None else '{:,}'.format(x['first_non_finite_step'])))
    L.append('')
    L.append('c 控制的是切向步幅，不是有效优化速度：累计角度不一定按 c 成比例变化，需要以上表的实测值为准。')
    L.append('')
    L.append('## 6. 初步观察')
    L.append('')
    ref60, c1 = g['ref']['steps_to_test']['60'], g[rep['group']]['steps_to_test']['60']
    floor = None if ref60 is None or c1 is None else max(abs(c1 - ref60), 100)
    if ref60 is None:
        L.append('- 参考运行在预算内没有达到 60% 测试准确率，无法比较等待步数。需要更长预算，或改用其他已预先固定的指标。')
    else:
        L.append('- 参考运行在分叉后 {:,} 步达到 60% 测试准确率；c=1 为 {}。两者之差与 100 步的评估分辨率取较大者，作为噪声底：**{} 步**。'
                 .format(ref60, fmt_step(c1), '无法确定' if floor is None else '{:,}'.format(floor)))
        for n in names[1:]:
            if n == rep['group']: continue
            x = g[n]
            if x['first_non_finite_step'] is not None:
                L.append('- %s 在第 %d 步出现非有限损失，属于干预步幅过大导致的不稳定，不作时间比较。' % (n, x['first_non_finite_step']))
                continue
            t = x['steps_to_test']['60']
            if t is None:
                L.append('- %s 在预算内未达到 60%%（参考运行为 %s 步），末步测试准确率 %.1f%%。' % (n, fmt_step(ref60), 100 * x['final']['test_acc']))
                continue
            diff = t - ref60
            if floor is not None and abs(diff) <= floor:
                verdict = '差值不超过噪声底，本 seed 下看不出差别'
            else:
                verdict = '比参考运行**%s** %.0f%%，超过噪声底' % ('早' if diff < 0 else '晚', 100 * abs(diff) / ref60)
            L.append('- {}：{:,} 步（相差 {:+,} 步），{}。累计角度为参考运行的 {:.2f} 倍。'.format(n, t, diff, verdict, x['total_angle_vs_reference']))
    L.append('- 比较等待步数前先看第 4 节：若某组训练 MSE 明显高于参考运行，测试表现的差别可能来自训练拟合变差，而不是方向更新更快或更慢。')
    L.append('')
    L.append('## 7. 结论边界')
    L.append('')
    L.append('- 单 seed 结果不能区分干预效应与 seed 间波动；正式比较需要至少 3 个配对 seed。')
    L.append('- 匹配的是全局范数。各层范数（见诊断图下排）与内部表征并未匹配。')
    L.append('- 放大切向更新还会改变之后的梯度与 AdamW 状态；即使等待步数改变，也只能说明"相同全局范数轨迹不足以确定泛化时间"，不能说明方向优化是唯一或主要瓶颈。')
    L.append('- 测试集只按固定评估点记录，没有用于选择分叉点、c 或阈值。')
    L.append('')
    L.append('图：[三组曲线、范数与范数误差](%s/direction_curves%s.png)、[角度、与参考的距离、逐层范数](%s/direction_diagnostics%s.png)。' % (figure_dir, suffix, figure_dir, suffix))
    L.append('')
    L.append('![三组曲线、范数与范数误差](%s/direction_curves%s.png)' % (figure_dir, suffix))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--export', type=Path, help='package directory; writes figures/direction and results/direction/seed<seed>_analysis.md')
    args = parser.parse_args()
    config = json.loads((args.root / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((args.root / 'summary.json').read_text(encoding='utf-8'))
    if args.export:
        suffix = '_seed%d' % config['seed']
        fig_dir, link = args.export / 'figures' / 'direction', '../../figures/direction'
        text = args.export / 'results' / 'direction' / ('seed%d_analysis.md' % config['seed'])
    else:
        suffix, fig_dir, text, link = '', args.root / 'figures', args.root / 'PRELIMINARY_ANALYSIS.md', 'figures'
    figures(fig_dir, suffix, np.load(args.root / 'history.npz'), config)
    analysis(text, link, suffix, summary, config)
    print('wrote %s and %s' % (fig_dir, text))


if __name__ == '__main__':
    main()
