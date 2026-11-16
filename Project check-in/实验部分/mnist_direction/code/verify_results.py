"""Check file hashes and independently recompute the published numbers from the CSV files; stdlib only."""
import csv, hashlib, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'direction'
NORM_TOL = 1e-10


def read_csv(name):
    with (ROOT/'results'/EXPERIMENT/name).open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def first_reach(points, threshold):
    return next((step for step, value in points if value >= threshold), None)


def main():
    manifest = json.loads((ROOT/'manifest_sha256.json').read_text(encoding='utf-8'))
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, 'hash mismatch: ' + name
    curves, table = read_csv('all_seeds_curves.csv'), read_csv('time_to_generalize_all_seeds.csv')
    seeds = sorted({int(r['seed']) for r in table})
    for seed in seeds:
        config = json.loads((ROOT/'configs'/EXPERIMENT/('seed%d.json' % seed)).read_text(encoding='utf-8'))
        qa = json.loads((ROOT/'metadata'/EXPERIMENT/('seed%d_qa.json' % seed)).read_text(encoding='utf-8'))
        budget, groups = config['steps_after_fork'], config['groups']
        schedule = [s for s in sorted(set(range(0, 100)) | set(range(100, 1000, 10)) | set(range(1000, budget+1, 100)) | {budget}) if s <= budget]
        assert qa['complete'] and qa['interpretable'] and qa['resume_check']['passed'] and qa['replay_control']['passed'] and qa['norms_matched']
        by_group = {}
        for r in curves:
            if int(r['seed']) == seed: by_group.setdefault(r['group'], []).append(r)
        assert list(by_group) == groups
        for name, rows in by_group.items():
            assert [int(r['steps_since_fork']) for r in rows] == schedule, name
            assert int(rows[0]['step']) == qa['fork_step']
            row = next(t for t in table if int(t['seed']) == seed and t['group'] == name)
            test = [(int(r['steps_since_fork']), float(r['test_accuracy'])) for r in rows]
            val = [(int(r['steps_since_fork']), float(r['val_accuracy'])) for r in rows]
            for t in config['thresholds']:
                reached = first_reach(test, t)
                assert row['steps_to_test_%d' % round(100*t)] == ('' if reached is None else str(reached)), (name, t)
            reached = first_reach(val, 0.6)
            assert row['steps_to_val_60'] == ('' if reached is None else str(reached)), name
            assert math.isclose(float(row['final_test_accuracy']), test[-1][1], rel_tol=1e-9)
            assert math.isclose(float(row['max_test_accuracy']), max(v for _, v in test), rel_tol=1e-9)
            for r in rows:  # norm match at every evaluation point, recomputed from the stored norms
                assert abs(float(r['global_norm']) - float(r['reference_norm'])) <= NORM_TOL * float(r['reference_norm']), name
                assert math.isclose(float(r['train_mse']), float(r['train_loss']) / 10, rel_tol=1e-9)
        # Every step: each intervention group's stored norm against the reference norm R_t.
        worst = {name: 0.0 for name in groups[1:]}
        steps = 0
        with (ROOT/'results'/EXPERIMENT/('seed%d_per_step.csv' % seed)).open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                assert int(r['steps_since_fork']) == steps
                steps += 1
                reference = float(r['reference_norm'])
                for name in worst:
                    worst[name] = max(worst[name], abs(float(r['global_norm_' + name]) - reference) / reference)
        assert steps == budget + 1
        for name, value in worst.items():
            assert value <= NORM_TOL, (name, value)
            assert math.isclose(value, qa['max_norm_relative_error'][name], rel_tol=1e-6, abs_tol=1e-18), (name, value)
        print('seed %d: %d groups x %d evaluation points, %d per-step rows, max norm mismatch %.1e' % (seed, len(groups), len(schedule), steps, max(worst.values())))
    # Cross-seed table: recompute every steps_to_test row and its same-seed ratio to the reference from the curves.
    reach = {}
    for r in curves:
        reach.setdefault((r['group'], int(r['seed'])), []).append((int(r['steps_since_fork']), float(r['test_accuracy'])))
    checked = 0
    for row in read_csv('summary_by_condition.csv'):
        if not row['metric'].startswith('steps_to_test_'): continue
        threshold = int(row['metric'].rsplit('_', 1)[1]) / 100
        values = [first_reach(reach[row['group'], s], threshold) for s in seeds]
        ref = [first_reach(reach['ref', s], threshold) for s in seeds]
        assert int(row['seeds']) == len(seeds) and int(row['seeds_reached']) == sum(v is not None for v in values), row['metric']
        assert row['values_by_seed'].split() == ['NA' if v is None else str(v) for v in values], (row['group'], row['metric'])
        if None in values: assert row['mean'] == '' and row['vs_reference_mean'] == ''
        else:
            ratios = [v / b for v, b in zip(values, ref)]
            assert math.isclose(float(row['mean']), sum(values) / len(values), rel_tol=1e-5), (row['group'], row['metric'])
            assert math.isclose(float(row['vs_reference_mean']), sum(ratios) / len(ratios), rel_tol=1e-5), (row['group'], row['metric'])
        checked += 1
    assert checked, 'summary_by_condition.csv has no steps_to_test rows'
    print('cross-seed table: %d steps_to_test rows recomputed' % checked)
    print('VERIFY_OK: %d files hashed, seeds %s' % (len(manifest), seeds))


if __name__ == '__main__': main()
