"""CPU audit of saved model states against independent trajectory artifacts."""
import hashlib
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from project_scope import load_scope

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if not args.allow_partial:assert json.loads((ROOT/'results/qa.json').read_text())['complete']
    manifest,scope,_,_=load_scope()
    if args.allow_partial:
        manifest=[entry for entry in manifest if (ROOT/entry['output']/'status.json').exists()
                  and json.loads((ROOT/entry['output']/'status.json').read_text())['state']=='complete']
    source_hash=hashlib.sha256((ROOT/'code/train_dynamics.py').read_bytes()).hexdigest()
    checked=[]
    for entry in manifest:
        folder=ROOT/entry['output']
        config=json.loads((folder/'resolved_config.json').read_text())
        assert np.array_equal(np.load(folder/'train_indices.npy'),np.arange(config['n']))
        norms=np.load(folder/'norms.npy',mmap_mode='r')
        for file in sorted((folder/'checkpoints').glob('*.pt')):
            saved=torch.load(file,map_location='cpu',weights_only=False)
            step=saved['step_completed']
            assert saved['config']==config and saved['code_sha256']==source_hash
            if file.stem in ('final','latest'):assert step==config['steps']
            if file.stem=='init':assert step==0
            if file.stem=='fit':assert step==saved['fit_step']
            assert all(t.dtype==torch.float64 and torch.isfinite(t).all() for t in saved['model'].values())
            square=[sum(float(v.square().sum()) for k,v in saved['model'].items() if k.startswith(str(layer)+'.')) for layer in (1,3,5)]
            actual=np.sqrt([sum(square),*square])
            assert np.allclose(actual,norms[step],rtol=1e-12,atol=1e-12)
            sampler=saved['sampler']
            assert sampler['n']==config['n'] and sampler['batch']==config['batch_size']
            if step:
                assert torch.equal(sampler['permutation'].sort().values,torch.arange(config['n']))
                assert 0<sampler['cursor']<=config['n']
                batches=math.ceil(config['n']/config['batch_size'])
                assert sampler['epoch']==(step-1)//batches
                assert sampler['cursor']==min(((step-1)%batches+1)*config['batch_size'],config['n'])
                assert len(saved['optimizer']['state'])==6
                assert all(int(v['step'])==step for v in saved['optimizer']['state'].values())
            checked.append(dict(run_id=entry['run_id'],checkpoint=file.name,step=step,
                                norm_max_abs_error=float(np.abs(actual-norms[step]).max())))
    result=dict(passed=True,complete=len(manifest)==scope['expected_runs'],runs=len(manifest),checkpoints=len(checked),training_code_sha256=source_hash,
                original_expected_runs=scope['original_expected_runs'],cancelled_by_user=scope['cancelled_by_user'],
                largest_norm_error=max(r['norm_max_abs_error'] for r in checked),details=checked)
    target='checkpoint_audit_progress.json' if args.allow_partial else 'checkpoint_audit.json'
    (ROOT/'metadata'/target).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'}))


if __name__=='__main__':main()
