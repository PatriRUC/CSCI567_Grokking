"""Independently reconstruct threshold events and validate local run artifacts."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np
from project_scope import load_scope,group_key

ROOT=Path(__file__).resolve().parents[1]


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def analyze(root,output,allow_partial=False):
    output.mkdir(parents=True,exist_ok=True)
    manifest,scope,planned_groups,roles=load_scope()
    results=[];all_metrics=[];missing=[]
    for entry in manifest:
        folder=root/entry['run_id']
        if not (folder/'status.json').exists():
            missing.append(dict(run_id=entry['run_id'],state='not_started'));continue
        status=json.loads((folder/'status.json').read_text())
        if status['state']!='complete':
            missing.append(dict(run_id=entry['run_id'],**status));continue
        config=json.loads((folder/'resolved_config.json').read_text())
        expected=json.loads((ROOT/entry['config']).read_text())
        assert config==expected,entry['run_id']
        digest=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert digest==status['config_hash']
        with (folder/'metrics.csv').open(encoding='utf-8',newline='') as f:
            rows=[{k:float(v) for k,v in r.items() if v!=''} for r in csv.DictReader(f)]
        assert all(math.isfinite(v) for r in rows for v in r.values())
        assert [r['step'] for r in rows]==sorted(set(r['step'] for r in rows))
        assert rows[0]['step']==0 and rows[-1]['step']==config['steps']
        testrows=[r for r in rows if 'test_count' in r]
        # Reconstruct the frozen schedule independently of the training module.
        budget=config['steps']
        if config['phase']=='fig3d':
            interval=math.ceil(config['schedule_budget']/150)
            expected_steps=set(range(30)) | set(range(30,150,10)) | set(range(0,budget+1,interval)) | {budget}
        else:
            expected_steps={0,10,50,100,200,budget} | set(range(500,budget+1,500))
        assert [int(r['step']) for r in testrows]==sorted(expected_steps),entry['run_id']+' evaluation schedule'
        assert all(r['test_count']==10000 for r in testrows)
        assert all(r['train_count']==config['n'] for r in rows)
        for row in rows:
            for split in ('train','test'):
                if split+'_count' not in row:continue
                assert 0<=row[split+'_accuracy']<=1
                assert math.isclose(row[split+'_accuracy'],row[split+'_correct']/row[split+'_count'],abs_tol=1e-14)
                assert math.isclose(row[split+'_mse_element_mean']*10,row[split+'_mse_per_sample_sum_outputs'],rel_tol=1e-12)
        norms=np.load(folder/'norms.npy',mmap_mode='r')
        assert norms.shape==(config['steps']+1,4) and np.isfinite(norms).all()
        assert np.allclose(norms[:,0]**2,np.square(norms[:,1:]).sum(1),rtol=1e-12,atol=1e-12)
        assert all(np.isclose(r['global_norm'],norms[int(r['step']),0],rtol=1e-12) for r in rows)
        hits=[r for r in testrows if int(r['test_correct'])*5>int(r['test_count'])*3]
        t60=int(hits[0]['step']) if hits else None
        events=json.loads((folder/'events.json').read_text())
        assert events['t60']==t60
        fits=[r for r in rows if r['train_accuracy']>=config['fit_accuracy'] and r['train_mse_element_mean']<=config['fit_mse']]
        fit=int(fits[0]['step']) if fits else None
        assert events['fit_step']==fit
        if fit is not None:assert (folder/'checkpoints/fit.pt').is_file()
        for name in ['init.pt','latest.pt','final.pt']:assert (folder/'checkpoints'/name).is_file()
        previous=[r['step'] for r in testrows if t60 is not None and r['step']<t60]
        assert events['previous_eval_step']==(int(previous[-1]) if previous else None)
        assert events['event_observed']==bool(hits)
        assert events['last_observed_step']==budget
        last=rows[-1]
        peak=max(testrows,key=lambda r:r['test_accuracy'])
        results.append(dict(run_id=entry['run_id'],phase=config['phase'],n=config['n'],alpha=config['alpha'],
                            scope_role=roles[entry['run_id']],
                            seed=config['seed'],weight_decay=config['weight_decay'],steps=config['steps'],
                            event_observed=bool(hits),t60=t60,previous_eval_step=int(previous[-1]) if previous else None,
                            threshold_status='reached' if hits else 'threshold_not_reached',fit_step=fit,
                            delay_from_fit=(t60-fit) if t60 is not None and fit is not None else None,
                            final_train_accuracy=last['train_accuracy'],final_test_accuracy=last['test_accuracy'],
                            peak_test_accuracy=peak['test_accuracy'],peak_test_step=int(peak['step']),
                            final_minus_peak_test_accuracy=last['test_accuracy']-peak['test_accuracy'],
                            final_train_mse=last['train_mse_element_mean'],final_test_mse=last['test_mse_element_mean'],
                            initial_norm=float(norms[0,0]),final_norm=float(norms[-1,0]),
                            elapsed_seconds=status['elapsed_seconds'],config_hash=digest))
        results[-1]['full_test_evaluations']=len(testrows)
        all_metrics.extend(dict(run_id=entry['run_id'],phase=config['phase'],n=config['n'],alpha=config['alpha'],seed=config['seed'],**r) for r in rows)
    if missing and not allow_partial:raise RuntimeError(f'{len(missing)} runs incomplete; use --allow-partial for labeled progress only.')
    if results:write_csv(output/'runs.csv',results)
    if all_metrics:write_csv(output/'metrics.csv',all_metrics)
    summary=[]
    for phase,n,alpha in sorted(set((r['phase'],r['n'],r['alpha']) for r in results)):
        group=[r for r in results if (r['phase'],r['n'],r['alpha'])==(phase,n,alpha)]
        hits=[r['t60'] for r in group if r['event_observed']]
        count=len(group)
        planned=planned_groups[(phase,int(n),float(alpha))]
        row=dict(phase=phase,n=n,alpha=alpha,completed=count,planned_seeds=planned,original_planned_seeds=3,
                 scope_role='primary' if planned==3 else 'supplementary_single_seed',reached=len(hits),
                 censored=count-len(hits),all_planned_complete=count==planned,
                 mean_t60=statistics.mean(hits) if len(hits)==count==3 else None,
                 sample_sd_t60=statistics.stdev(hits) if len(hits)==count==3 else None)
        for key in ('final_train_accuracy','final_test_accuracy','peak_test_accuracy','final_norm'):
            vals=[r[key] for r in group]
            row[key+'_mean']=statistics.mean(vals)
            row[key+'_sample_sd']=statistics.stdev(vals) if count>1 else None
        summary.append(row)
    if summary:write_csv(output/'summary.csv',summary)
    qa=dict(complete=len(results)==scope['expected_runs'],completed_runs=len(results),missing=missing,**scope,
            all_complete_runs_passed=True,threshold='strict > 0.60 on full test split',
            norm_rows_per_formal_run=100001,standard_deviation='sample SD, ddof=1',
            full_test_schedule_verified=True,
            mean_policy='only all three planned seeds reached; never substitute budget for missing t60')
    (output/'qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    (ROOT/'metadata/approved_scope.json').write_text(json.dumps(scope,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(qa),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=ROOT/'runs')
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();analyze(a.runs,a.output,a.allow_partial)
