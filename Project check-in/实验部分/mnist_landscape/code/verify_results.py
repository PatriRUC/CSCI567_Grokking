"""Check file hashes and independently recompute published statistics; stdlib only."""
import csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_csv(name):
    with (ROOT/'results'/name).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def main():
    manifest=json.loads((ROOT/'manifest_sha256.json').read_text(encoding='utf-8'))
    for name,expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Hash mismatch: '+name
    rows=read_csv('all_seeds_results.csv')
    lookup={(int(r['seed']),int(r['n']),int(r['alpha_index'])):r for r in rows}
    assert len(rows)==len(lookup)==1890
    configs=[json.loads((ROOT/'configs'/('seed%d.json'%s)).read_text(encoding='utf-8')) for s in range(3)]
    expected={(s,n,i) for s in range(3) for n in configs[s]['all_data_sizes'] for i in range(21)}
    assert set(lookup)==expected
    for (s,n,i),r in lookup.items():
        assert math.isclose(float(r['alpha']),configs[s]['alphas'][i],abs_tol=1e-12)
        assert (r['original_grid']=='True')==(n in configs[s]['original_data_sizes'])
        assert all(math.isfinite(float(v)) for k,v in r.items() if k!='original_grid')
        assert configs[s]['steps']==10000
        assert float(r['max_norm_relative_error'])<2e-6
        for k,v in r.items():
            if k.endswith('_accuracy'): assert 0<=float(v)<=1
    summary=read_csv('summary_by_condition.csv')
    assert len(summary)==630
    for r in summary:
        n,i=int(r['n']),int(r['alpha_index'])
        assert int(r['seed_count'])==3
        for key in r:
            if not key.endswith('_mean'): continue
            metric=key[:-5]
            if metric.endswith('_generalization_gap'):
                variant=metric[:-len('_generalization_gap')]
                prefix='paper_best' if variant=='paper' else variant
                values=[float(lookup[(s,n,i)][prefix+'_train_accuracy'])-float(lookup[(s,n,i)][prefix+'_test_accuracy']) for s in range(3)]
            else: values=[float(lookup[(s,n,i)][metric]) for s in range(3)]
            assert math.isclose(float(r[key]),statistics.mean(values),rel_tol=1e-12,abs_tol=1e-12),key
            assert math.isclose(float(r[metric+'_std']),statistics.stdev(values),rel_tol=1e-12,abs_tol=1e-12),metric
    for n in (200,8000):
        assert read_csv('slice_N%d_mean_std.csv'%n)==[r for r in summary if int(r['n'])==n]
    contrasts=read_csv('paired_contrasts_by_seed.csv')
    assert len(contrasts)==24
    for r in contrasts:
        s=int(r['seed']); prefix='paper_best' if r['variant']=='paper' else 'selected_full'
        def err(n,i): return 1-float(lookup[(s,n,i)][prefix+'_test_accuracy'])
        small=err(200,20)-err(200,10); large=err(8000,20)-err(8000,10)
        values=dict(smallN_test_error_increase_alpha1_to10=small,largeN_test_error_increase_alpha1_to10=large,
                    test_error_reduction_N200_to8000_at_alpha10=err(200,20)-err(8000,20),smallN_minus_largeN_alpha_effect=small-large)
        assert math.isclose(float(r['difference']),values[r['contrast']],abs_tol=1e-12)
        assert math.isclose(float(r['difference_percentage_points']),100*values[r['contrast']],abs_tol=1e-10)
    print('PASS:',len(manifest),'file hashes; 1,890 rows; all mean/SD, slice and paired-contrast tables.')

if __name__=='__main__': main()
