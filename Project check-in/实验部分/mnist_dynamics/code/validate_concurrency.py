"""Check independent local processes reproduce serial smoke before concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from train_dynamics import ROOT, atomic_json
from validate_dynamics import tree_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    root=ROOT/f'metadata/concurrency{a.workers}'
    root.mkdir(exist_ok=True)
    def task(i):
        out=root/f'worker{i}'
        command=[sys.executable,str(ROOT/'code/train_dynamics.py'),'--config',str(ROOT/'configs/smoke_alpha8.json'),'--output',str(out)]
        with (root/f'worker{i}.log').open('w',encoding='utf-8') as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        return out
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:outputs=list(pool.map(task,range(a.workers)))
    wall=time.perf_counter()-start
    reference=ROOT/'metadata/validation/continuous'
    state=torch.load(reference/'checkpoints/final.pt',map_location='cpu',weights_only=False)
    norm=np.load(reference/'norms.npy')
    checks=[]
    for out in outputs:
        other=torch.load(out/'checkpoints/final.pt',map_location='cpu',weights_only=False)
        err=max(tree_error(state['model'],other['model']),tree_error(state['optimizer'],other['optimizer']))
        norm_error=float(np.abs(norm-np.load(out/'norms.npy')).max())
        assert err<=1e-10 and norm_error<=1e-10
        checks.append(dict(worker=out.name,parameter_optimizer_error=err,norm_error=norm_error,
                           train_seconds=json.loads((out/'status.json').read_text())['elapsed_seconds']))
    result=dict(passed=True,workers=a.workers,total_wall_seconds=wall,checks=checks)
    atomic_json(root/'validation.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
