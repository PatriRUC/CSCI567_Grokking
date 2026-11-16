"""Cross-seed summary of the exported runs. CPU only: numpy + matplotlib.
Reads results/direction/all_seeds_curves.csv, time_to_generalize_all_seeds.csv and metadata/direction/seed*_qa.json|seed*_gate.json;
writes results/direction/summary_by_condition.csv, figures/direction/direction_seeds<ids>.png|pdf and results/SUMMARY.md.
  python code/summarize_seeds.py          # run after collect_results.py for every seed, then collect_results.py --manifest-only
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_direction_figures import C_COLORS, INK, MUTED, SURFACE, panel

PACKAGE = Path(__file__).resolve().parents[1]
RESULTS, FIGURES, META = PACKAGE / 'results' / 'direction', PACKAGE / 'figures' / 'direction', PACKAGE / 'metadata' / 'direction'
THRESHOLDS = [30, 40, 50, 60, 70, 80, 85, 90]      # 60 is the threshold fixed before the runs; the others show sensitivity
SAME_NORM = [85, 80, 75, 70, 65, 60, 55]
SAME_ANGLE = [5, 10, 15, 20, 30]
COLUMNS = ['steps_since_fork', 'train_accuracy', 'val_accuracy', 'test_accuracy', 'train_mse', 'global_norm', 'cumulative_angle_rad',
           'norm_layer1', 'norm_layer2', 'norm_layer3']


def load():
    data = {}
    with (RESULTS / 'all_seeds_curves.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            data.setdefault((int(r['seed']), r['group']), []).append([float(r[k]) for k in COLUMNS])
    data = {k: dict(zip(COLUMNS, np.array(v).T)) for k, v in data.items()}
    seeds = sorted({s for s, _ in data})
    groups = [g for s, g in data if s == seeds[0]]
    with (RESULTS / 'time_to_generalize_all_seeds.csv').open(encoding='utf-8', newline='') as f:
        table = {(int(r['seed']), r['group']): r for r in csv.DictReader(f)}
    return data, seeds, groups, table


def first_index(values, threshold, falling=False):
    hit = np.nonzero(values <= threshold if falling else values >= threshold)[0]
    return int(hit[0]) if len(hit) else None


def stats(values):
    """mean, sd (n-1), min, max; blank unless every seed has a value."""
    if any(v is None for v in values): return ['', '', '', '']
    a = np.array(values, float)
    return ['%.6g' % a.mean(), '%.6g' % (a.std(ddof=1) if len(a) > 1 else 0.0), '%.6g' % a.min(), '%.6g' % a.max()]


def condition_table(data, seeds, groups, table):
    """Per group and metric: values per seed, plus the same-seed comparison with the reference (ratio for steps, difference for accuracies)."""
    def reach(d, key, threshold):
        i = first_index(d[key], threshold)
        return None if i is None else int(d['steps_since_fork'][i])
    per_seed = {}
    for g in groups:
        for t in THRESHOLDS:
            per_seed[g, 'steps_to_test_%d' % t] = [reach(data[s, g], 'test_accuracy', t / 100) for s in seeds]
        per_seed[g, 'steps_to_val_60'] = [reach(data[s, g], 'val_accuracy', 0.6) for s in seeds]
        per_seed[g, 'final_test_accuracy'] = [float(data[s, g]['test_accuracy'][-1]) for s in seeds]
        per_seed[g, 'max_test_accuracy'] = [float(data[s, g]['test_accuracy'].max()) for s in seeds]
        per_seed[g, 'final_val_accuracy'] = [float(data[s, g]['val_accuracy'][-1]) for s in seeds]
        per_seed[g, 'total_angle_rad'] = [float(data[s, g]['cumulative_angle_rad'][-1]) for s in seeds]
    rows = []
    for (g, metric), values in per_seed.items():
        ref = per_seed[groups[0], metric]
        kind = 'difference' if metric.endswith('accuracy') else 'ratio'
        paired = [None if v is None or r is None else (v - r if kind == 'difference' else v / r) for v, r in zip(values, ref)]
        rows.append([g, table[seeds[0], g]['c'], metric, len(seeds), sum(v is not None for v in values),
                     ' '.join('NA' if v is None else '%.6g' % v for v in values),
                     *stats(values), kind, *stats(paired)])
    with (RESULTS / 'summary_by_condition.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'c', 'metric', 'seeds', 'seeds_reached', 'values_by_seed', 'mean', 'sd', 'min', 'max',
                         'vs_reference', 'vs_reference_mean', 'vs_reference_sd', 'vs_reference_min', 'vs_reference_max'])
        writer.writerows(rows)
    return per_seed


def figure(data, seeds, groups, table, per_seed, name):
    def style(g):
        if g == groups[0]: return dict(color=INK, lw=1.0, ls=(0, (4, 2)), zorder=5)
        return dict(color=C_COLORS[float(table[seeds[0], g]['c'])], lw=1.5, ls='-', zorder=3)
    order = groups[1:] + groups[:1]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), facecolor=SURFACE)
    panel(axes[0, 0], 'Test accuracy', 'updates after the fork', 'accuracy')
    panel(axes[0, 1], 'Steps to reach a test accuracy, relative to the same-seed reference', 'test accuracy threshold (%)', 'steps / reference steps', logx=False, logy=True)
    panel(axes[1, 0], 'Test accuracy at the same global norm', 'global parameter norm (decreases during training)', 'accuracy', logx=False)
    panel(axes[1, 1], 'Test accuracy at the same last-layer norm', 'layer 3 norm, weight + bias (decreases during training)', 'accuracy')
    for g in order:
        for k, s in enumerate(seeds):
            d = data[s, g]
            label = ('reference (plain AdamW)' if g == groups[0] else '%s, norm matched' % g) if k == 0 else None
            axes[0, 0].plot(d['steps_since_fork'][1:], d['test_accuracy'][1:], label=label, **style(g))
            axes[1, 0].plot(d['global_norm'], d['test_accuracy'], **style(g))
            axes[1, 1].plot(d['norm_layer3'], d['test_accuracy'], **style(g))
    axes[0, 0].axhline(0.6, color=MUTED, lw=0.8, zorder=1)
    axes[0, 0].annotate('60% threshold', (1, 0.6), xytext=(2, 3), textcoords='offset points', fontsize=8, color=MUTED)
    axes[0, 0].annotate('one line per seed', (0.02, 0.93), xycoords='axes fraction', fontsize=8, color=MUTED)
    for ax in axes[1]: ax.invert_xaxis()
    ax = axes[0, 1]
    for level, text in ((2.0, '2 = steps proportional to 1/c for c=0.5'), (1.0, None), (0.5, '0.5 = steps proportional to 1/c for c=2')):
        ax.axhline(level, color=MUTED, lw=0.8, ls=':' if text else '-', zorder=1)
        if text: ax.annotate(text, (THRESHOLDS[0], level), xytext=(0, -10), textcoords='offset points', fontsize=8, color=MUTED)
    for g in groups[1:]:
        color = C_COLORS[float(table[seeds[0], g]['c'])]
        mean_x, mean_y = [], []
        for t in THRESHOLDS:
            values, ref = per_seed[g, 'steps_to_test_%d' % t], per_seed[groups[0], 'steps_to_test_%d' % t]
            ratios = [v / r for v, r in zip(values, ref) if v is not None and r is not None]
            ax.plot([t] * len(ratios), ratios, 'o', ms=4, color=color, alpha=0.75, zorder=3)
            if len(ratios) == len(seeds): mean_x.append(t); mean_y.append(np.mean(ratios))
        ax.plot(mean_x, mean_y, '-', lw=1.5, color=color, zorder=2)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 1, 1.25, 1.5, 2]); ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '1', '1.25', '1.5', '2'])
    ax.minorticks_off(); ax.set_ylim(0.4, 2.7); ax.set_xticks(THRESHOLDS)
    ax.annotate('dots: seeds; line: mean where every seed reached the threshold', (0.02, 0.95), xycoords='axes fraction', fontsize=8, color=MUTED)
    ticks = [14, 10, 7, 5, 3, 2, 1.5]
    axes[1, 1].set_xticks(ticks); axes[1, 1].set_xticklabels([str(t) for t in ticks]); axes[1, 1].minorticks_off()
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=len(names), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle('Tangential-update intervention under a shared norm trajectory: MNIST, N=1000, alpha=8, seeds %s' % ', '.join(map(str, seeds)),
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES / (name + '.' + ext), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def cell(values, fmt='%.1f', scale=1.0):
    """'mean (min–max)' over seeds; '—' unless every seed has a value."""
    if any(v is None for v in values): return '—'
    a = scale * np.array(values, float)
    return (fmt + '（' + fmt + '–' + fmt + '）') % (a.mean(), a.min(), a.max())


def text(data, seeds, groups, table, per_seed, name):
    qa = {s: json.loads((META / ('seed%d_qa.json' % s)).read_text(encoding='utf-8')) for s in seeds}
    gate = {s: json.loads(p.read_text(encoding='utf-8')) if (p := META / ('seed%d_gate.json' % s)).exists() else None for s in seeds}
    env = json.loads((PACKAGE / 'metadata' / 'environment.json').read_text(encoding='utf-8'))
    ref = groups[0]
    step = lambda v: '未达到' if v is None else '{:,}'.format(v)
    L = ['# 方向干预实验结果摘要（seeds %s）' % '、'.join(map(str, seeds)), '',
         '本文件由 `code/summarize_seeds.py` 根据 `results/direction/` 下的 CSV 与 `metadata/direction/` 下的校验记录自动生成。'
         '每个 seed 的单独分析见 ' + '、'.join('[seed %d](direction/seed%d_analysis.md)' % (s, s) for s in seeds) + '。', '',
         '![跨 seed 汇总](../figures/direction/%s.png)' % name, '',
         '## 1. 运行与校验', '',
         '运行环境（最后导出的 seed）：%s，PyTorch %s，%s。' % (env['gpu'], env['torch'], env['dtype'].replace('torch.', '')), '',
         '| seed | 分叉步 | 分叉时测试准确率 | 分叉时范数 | 参考末步范数 | 检查点恢复：参数最大绝对差 | c=1 重放：前 1,000 步相对距离 | 范数匹配：最大相对误差 | c=1 短程校验 | 分叉后耗时 |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for s in seeds:
        q, r = qa[s], table[s, ref]
        L.append('| %d | %s | %.1f%% | %.2f | %.2f | %.1e | %.1e | %.1e | %s | %.1f 分钟 |' % (
            s, step(q['fork_step']), 100 * data[s, ref]['test_accuracy'][0], float(r['fork_norm']), float(r['final_norm']),
            q['resume_check']['max_parameter_abs_error'], q['replay_control']['max_relative_distance_in_window'],
            max(q['max_norm_relative_error'].values()), '—' if gate[s] is None else ('通过' if gate[s]['passed'] else '**未通过**'),
            q['training_wall_seconds_after_fork'] / 60))
    L += ['', '判据：检查点恢复 ≤ 1e-9 × 范数，c=1 重放 ≤ 1e-9，范数匹配 ≤ 1e-10。%s' % (
        '全部 seed 的三项校验均通过。' if all(q['interpretable'] for q in qa.values()) else '**有 seed 未通过校验，其结果不应解释。**'), '',
        '## 2. 泛化等待步数（自分叉点起，测试准确率首次达到阈值）', '',
        '| 组 | seed | ' + ' | '.join('≥%d%%' % t for t in THRESHOLDS if t >= 60) + ' | 验证 ≥60% | 末步测试 | 最高测试 |', '|---|---:|' + '---:|' * (sum(t >= 60 for t in THRESHOLDS) + 3)]
    for g in groups:
        for k, s in enumerate(seeds):
            L.append('| %s | %d | ' % (g, s) + ' | '.join(step(per_seed[g, 'steps_to_test_%d' % t][k]) for t in THRESHOLDS if t >= 60)
                     + ' | %s | %.1f%% | %.1f%% |' % (step(per_seed[g, 'steps_to_val_60'][k]), 100 * per_seed[g, 'final_test_accuracy'][k], 100 * per_seed[g, 'max_test_accuracy'][k]))
        L.append('| %s | **均值** | ' % g + ' | '.join('—' if None in per_seed[g, 'steps_to_test_%d' % t] else '**{:,.0f}**'.format(np.mean(per_seed[g, 'steps_to_test_%d' % t])) for t in THRESHOLDS if t >= 60)
                 + ' | **{:,.0f}** | **{:.1f}%** | **{:.1f}%** |'.format(np.mean(per_seed[g, 'steps_to_val_60']), 100 * np.mean(per_seed[g, 'final_test_accuracy']), 100 * np.mean(per_seed[g, 'max_test_accuracy'])))
    L += ['', '"未达到"表示在预算内没有达到该阈值；有 seed 未达到时不计算均值。1,000 步之后评估间隔为 100 步。', '',
          '## 3. 与同 seed 参考运行的步数之比', '',
          '60% 是运行前固定的主阈值；其余阈值用于检查结论是否依赖阈值，30%–50% 是事后补充的。', '',
          '| 阈值 | ' + ' | '.join('%s：各 seed | %s：均值 ± 标准差' % (g, g) for g in groups[1:]) + ' |', '|---:|' + '---|---:|' * (len(groups) - 1)]
    for t in THRESHOLDS:
        row = '| %d%% | ' % t
        for g in groups[1:]:
            values, base = per_seed[g, 'steps_to_test_%d' % t], per_seed[ref, 'steps_to_test_%d' % t]
            ratios = [None if v is None or b is None else v / b for v, b in zip(values, base)]
            row += '、'.join('—' if r is None else '%.3f' % r for r in ratios) + ' | '
            row += ('—' if None in ratios else '%.3f ± %.3f' % (np.mean(ratios), np.std(ratios, ddof=1) if len(ratios) > 1 else 0)) + ' | '
        L.append(row.rstrip())
    L += ['', '若等待步数与 1/c 成正比，c=2 的比值应为 0.5，c=0.5 应为 2。标准差按 n−1 计算，只有 %d 个 seed。' % len(seeds), '',
          '## 4. 相同全局范数处的测试准确率', '',
          '每个 seed 取参考范数首次降到该值以下的评估步（各组在这一步的范数相同）。单元格为各 seed 均值（最小–最大），单位 %。', '',
          '| 全局范数 ≤ | 分叉后步数 | ' + ' | '.join(groups) + ' |', '|---:|---:|' + '---:|' * len(groups)]
    for R in SAME_NORM:
        idx = [first_index(data[s, ref]['global_norm'], R, falling=True) for s in seeds]
        if None in idx: continue
        L.append('| %d | %s | ' % (R, cell([data[s, ref]['steps_since_fork'][i] for s, i in zip(seeds, idx)], '%.0f'))
                 + ' | '.join(cell([data[s, g]['test_accuracy'][i] for s, i in zip(seeds, idx)], scale=100) for g in groups) + ' |')
    L += ['', '## 5. 达到同一测试准确率时的状态', '',
          '单元格为各 seed 均值（最小–最大）。', '',
          '| 阈值 | 组 | 全局范数 | 累计角度 (rad) | 第 1 层范数 | 第 2 层范数 | 第 3 层范数 |', '|---:|---|---:|---:|---:|---:|---:|']
    for t in (40, 60, 80, 85):
        for g in groups:
            idx = [first_index(data[s, g]['test_accuracy'], t / 100) for s in seeds]
            pick = lambda key: [None if i is None else data[s, g][key][i] for s, i in zip(seeds, idx)]
            L.append('| %d%% | %s | %s | %s | %s | %s | %s |' % (t, g, cell(pick('global_norm')), cell(pick('cumulative_angle_rad')), cell(pick('norm_layer1')),
                                                              cell(pick('norm_layer2')), cell(pick('norm_layer3'), '%.2f')))
    L += ['', '## 6. 相同累计角度处的测试准确率', '',
          '每组取累计角度首次达到该值的评估步。单元格为各 seed 均值（最小–最大），单位 %。', '',
          '| 累计角度 ≥ (rad) | ' + ' | '.join(groups) + ' |', '|---:|' + '---:|' * len(groups)]
    for A in SAME_ANGLE:
        row = []
        for g in groups:
            idx = [first_index(data[s, g]['cumulative_angle_rad'], A) for s in seeds]
            row.append(cell([None if i is None else data[s, g]['test_accuracy'][i] for s, i in zip(seeds, idx)], scale=100))
        L.append('| %d | ' % A + ' | '.join(row) + ' |')
    L += ['', '## 7. 末段表现与实际方向变化', '',
          '| 组 | 最高测试准确率 (%) | 出现在分叉后步数 | 末步测试准确率 (%) | 末步验证准确率 (%) | 末步训练 MSE | 分叉后最低训练准确率 (%) | 累计角度相对参考 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        peak = [int(data[s, g]['steps_since_fork'][data[s, g]['test_accuracy'].argmax()]) for s in seeds]
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            g, cell(per_seed[g, 'max_test_accuracy'], scale=100), cell(peak, '%.0f'), cell(per_seed[g, 'final_test_accuracy'], scale=100),
            cell(per_seed[g, 'final_val_accuracy'], scale=100), cell([data[s, g]['train_mse'][-1] for s in seeds], '%.1e'),
            cell([data[s, g]['train_accuracy'].min() for s in seeds], scale=100),
            cell([a / b for a, b in zip(per_seed[g, 'total_angle_rad'], per_seed[ref, 'total_angle_rad'])], '%.2f')))
    L += ['', '## 8. 读表时的限制', '',
          '- 只有 %d 个 seed；seed 只改变初始化和批次顺序，训练子集固定为前 1,000 个 MNIST 训练样本。' % len(seeds),
          '- 匹配的是全局范数。逐层范数（第 5 节）和内部表征没有匹配。',
          '- 干预同时改变之后的梯度与 AdamW 状态；等待步数的变化说明"相同的全局范数轨迹不足以确定泛化时间"，不说明方向优化是唯一或主要瓶颈。',
          '- 测试集只按固定评估点记录，没有用于选择分叉点、c 或主阈值。', '']
    (PACKAGE / 'results' / 'SUMMARY.md').write_text('\n'.join(L), encoding='utf-8')


def main():
    data, seeds, groups, table = load()
    name = 'direction_seeds' + ''.join(map(str, seeds))
    per_seed = condition_table(data, seeds, groups, table)
    figure(data, seeds, groups, table, per_seed, name)
    text(data, seeds, groups, table, per_seed, name)
    print('seeds %s: wrote summary_by_condition.csv, %s.png|pdf, SUMMARY.md' % (seeds, name))


if __name__ == '__main__':
    main()
