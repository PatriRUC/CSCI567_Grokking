"""Numerical, input, event, resume and local-cost checks before formal runs."""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torchvision
from train_dynamics import (ROOT, Sampler, atomic_json, capture_rng, environment, evaluate,
                            event_summary, load_data, make_model, make_optimizer, norms, setup, update)


def tree_error(a, b):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype
        return float((a.cpu()-b.cpu()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert set(a) == set(b)
        return max([tree_error(a[k], b[k]) for k in a] or [0.])
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max([tree_error(x, y) for x, y in zip(a, b)] or [0.])
    assert a == b, (a, b)
    return 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'metadata/validation')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/smoke_alpha8.json').read_text())
    setup(0)
    train, test, raw_train = load_data(ROOT/'data', 'cuda')
    direct = train[0][:10].reshape(-1, 1, 28, 28).cpu()
    reference_input = torch.stack([torchvision.transforms.ToTensor()(raw_train[i][0]) for i in range(10)])
    assert torch.equal(direct, reference_input)
    model = make_model(0, 8, 'cuda')
    official = copy.deepcopy(model)
    optimizer = make_optimizer(model, config)
    ref_optimizer = torch.optim.AdamW(official.parameters(), lr=.001, weight_decay=.0001,
                                      betas=(.9,.999), eps=1e-8, foreach=False, fused=False)
    source_loss = nn.MSELoss()
    one_hots = torch.eye(10, device='cuda')
    sampler = Sampler(1000, 200, 101000, 'cuda')
    for step in range(50):
        idx = sampler.next()
        xx, yy = train[0][idx], train[1][idx]
        update(model, optimizer, xx, yy)
        ref_optimizer.zero_grad()
        loss = source_loss(official(xx), one_hots[yy])
        loss.backward()
        ref_optimizer.step()
    source_error = tree_error(model.state_dict(), official.state_dict())
    assert source_error <= 1e-10
    before_model = copy.deepcopy(model.state_dict())
    before_opt = copy.deepcopy(optimizer.state_dict())
    before_sampler = sampler.state_dict()
    before_rng = torch.get_rng_state().clone()
    metrics = evaluate(model, train[0][:503], train[1][:503], chunk=200)
    assert metrics['count'] == 503
    assert tree_error(before_model, model.state_dict()) == 0
    assert tree_error(before_opt, optimizer.state_dict()) == 0
    assert tree_error(before_sampler, sampler.state_dict()) == 0
    assert torch.equal(before_rng, torch.get_rng_state())
    v = norms(model).cpu().numpy()
    assert np.isclose(v[0]**2, np.square(v[1:]).sum(), rtol=1e-14)
    short = Sampler(503, 200, 7, 'cpu')
    first_epoch = [short.next() for _ in range(3)]
    assert [len(i) for i in first_epoch] == [200, 200, 103]
    assert len(set(torch.cat(first_epoch).tolist())) == 503
    saved_sampler = short.state_dict()
    next_expected = short.next()
    restored = Sampler(503, 200, 7, 'cpu')
    restored.load_state_dict(saved_sampler)
    assert torch.equal(restored.next(), next_expected)
    threshold_rows = [dict(step=0, test_correct=1000, test_count=10000),
                      dict(step=500, test_correct=6000, test_count=10000),
                      dict(step=1000, test_correct=6001, test_count=10000)]
    assert event_summary(threshold_rows, 100000, False)['t60'] == 1000
    assert event_summary(threshold_rows[:2], 500, True)['threshold_status'] == 'threshold_not_reached'
    assert event_summary(threshold_rows[:2], 500, False)['threshold_status'] == 'incomplete'
    assert event_summary([dict(step=0,test_correct=6001,test_count=10000)],100000,False)['t60'] == 0
    # Separate OS processes exercise serialization and RNG reconstruction.
    train_script = ROOT/'code/train_dynamics.py'
    continuous, split = out/'continuous', out/'split'
    commands = [
        [sys.executable,str(train_script),'--config',str(ROOT/'configs/smoke_alpha8.json'),'--output',str(continuous)],
        [sys.executable,str(train_script),'--config',str(ROOT/'configs/smoke_alpha8.json'),'--output',str(split),'--stop-after','100'],
        [sys.executable,str(train_script),'--resume',str(split/'checkpoints/latest.pt')]]
    for i, command in enumerate(commands):
        with (out/f'process{i}.log').open('w',encoding='utf-8') as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    a = torch.load(continuous/'checkpoints/final.pt', map_location='cpu', weights_only=False)
    b = torch.load(split/'checkpoints/final.pt', map_location='cpu', weights_only=False)
    resume_error = max(tree_error(a['model'], b['model']), tree_error(a['optimizer'], b['optimizer']))
    assert resume_error <= 1e-10
    assert tree_error(a['sampler'], b['sampler']) == 0
    norm_error = float(np.abs(np.load(continuous/'norms.npy')-np.load(split/'norms.npy')).max())
    assert norm_error <= 1e-10
    assert [{k:v for k,v in r.items() if k!='elapsed_seconds'} for r in a['metrics']] == [{k:v for k,v in r.items() if k!='elapsed_seconds'} for r in b['metrics']]
    atomic_json(out/'validation.json',dict(passed=True,source_update_max_abs_error=source_error,
                input_conversion_bitwise_equal=True,eval_side_effect_free=True,tail_batch_checked=True,
                resume_parameter_optimizer_max_abs_error=resume_error,resume_norm_max_abs_error=norm_error,
                resumed_sampler_identical=True,strict_threshold_and_censoring_checked=True,environment=environment()))
    # Measure the complete update+norm cost, excluding warmup.
    costs={}
    for n in (1000,30000):
        benchmark_model=make_model(0,8,'cuda')
        opt=make_optimizer(benchmark_model,config)
        smp=Sampler(n,200,101000,'cuda')
        buffer=torch.empty((500,4),device='cuda')
        for _ in range(30):
            idx=smp.next(); update(benchmark_model,opt,train[0][idx],train[1][idx])
        torch.cuda.synchronize(); start=time.perf_counter()
        for i in range(500):
            idx=smp.next(); update(benchmark_model,opt,train[0][idx],train[1][idx]); buffer[i]=norms(benchmark_model)
        torch.cuda.synchronize(); update_seconds=(time.perf_counter()-start)/500
        start=time.perf_counter()
        evaluate(benchmark_model,train[0][:n],train[1][:n]); evaluate(benchmark_model,*test)
        torch.cuda.synchronize(); eval_seconds=time.perf_counter()-start
        costs[str(n)]=dict(update_and_norm_seconds=update_seconds,full_train_test_eval_seconds=eval_seconds,
                           estimated_100k_with_205_evals_seconds=100000*update_seconds+205*eval_seconds)
    atomic_json(out/'benchmark.json',costs)
    print(json.dumps(dict(validation='PASS',costs=costs)),flush=True)


if __name__=='__main__':
    main()
