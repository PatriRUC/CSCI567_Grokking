"""Check file hashes and independently recompute published tables from the curve CSVs; stdlib only."""
import csv, hashlib, json, math, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ('fig3d', 'fig3e')
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def read_csv(experiment, name):
    with (ROOT/'results'/experiment/name).open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))


def first_reach(points, threshold):
    return next((step for step, value in points if value >= threshold), None)


def same(a, b):
    if a == '' or b == '': return a == b
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def check(experiment):
    configs = [json.loads((ROOT/'configs'/experiment/('seed%d.json' % s)).read_text(encoding='utf-8')) for s in range(3)]
    assert all(c['experiment'] == experiment for c in configs)
    budget = configs[0]['steps']
    schedule = sorted(set(range(0, 100)) | set(range(100, 1000, 10)) | set(range(1000, budget+1, 100)) | {budget})
    models = [(int(n), float(a)) for n, a in configs[0]['models']]
    for c in configs:
        assert [(int(n), float(a)) for n, a in c['models']] == models and c['steps'] == budget == 100000
        assert c['weight_decay'] == configs[0]['weight_decay']
    rows = read_csv(experiment, 'all_seeds_curves.csv')
    assert len(rows) == 3*len(models)*len(schedule)
    curves = {}
    for r in rows:
        for k, v in r.items(): assert math.isfinite(float(v)), k
        for k in ('train_accuracy', 'test_accuracy'): assert 0 <= float(r[k]) <= 1
        curves.setdefault((int(r['seed']), int(r['n']), float(r['alpha'])), []).append(r)
    assert set(curves) == {(s, n, a) for s in range(3) for n, a in models}
    for g in curves.values(): assert [int(r['step']) for r in g] == schedule

    per_seed = read_csv(experiment, 'time_to_generalize_all_seeds.csv')
    assert len(per_seed) == 3*len(models)
    lookup = {}
    for r in per_seed:
        key = (int(r['seed']), int(r['n']), float(r['alpha']))
        g = curves[key]
        train = [(int(x['step']), float(x['train_accuracy'])) for x in g]
        test = [(int(x['step']), float(x['test_accuracy'])) for x in g]
        expected = dict(fit_step=first_reach(train, 0.99), final_train_accuracy=train[-1][1], final_test_accuracy=test[-1][1],
                        max_test_accuracy=max(v for _, v in test), initial_norm=float(g[0]['global_norm']), final_norm=float(g[-1]['global_norm']))
        for t in THRESHOLDS: expected['steps_to_test_%d' % round(100*t)] = first_reach(test, t)
        for k, v in expected.items(): assert same(r[k], '' if v is None else v), (experiment, key, k)
        lookup[key] = r
    summary = read_csv(experiment, 'summary_by_condition.csv')
    assert len(summary) == len(models)
    for r in summary:
        n, a = int(r['n']), float(r['alpha'])
        assert int(r['seed_count']) == 3
        for key in r:
            if not key.endswith('_mean'): continue
            metric = key[:-5]
            got = [float(v) for v in (lookup[(s, n, a)][metric] for s in range(3)) if v != '']
            if metric+'_reached' in r:
                assert int(r[metric+'_reached']) == len(got)
                if len(got) < 3:
                    assert r[key] == '' and r[metric+'_std'] == ''; continue
            assert len(got) == 3, (experiment, n, a, metric)
            assert math.isclose(float(r[key]), statistics.mean(got), rel_tol=1e-12, abs_tol=1e-9), key
            assert math.isclose(float(r[metric+'_std']), statistics.stdev(got), rel_tol=1e-9, abs_tol=1e-9), metric
    return len(rows)


def main():
    manifest = json.loads((ROOT/'manifest_sha256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, 'Hash mismatch: '+name
    counts = {e: check(e) for e in EXPERIMENTS}
    print('PASS: %d file hashes; curve rows %s; time-to-generalize and mean/SD tables recomputed.'
          % (len(manifest), ', '.join('%s %s' % (e, format(c, ',')) for e, c in counts.items())))


if __name__ == '__main__': main()
