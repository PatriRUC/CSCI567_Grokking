"""Restore the actual training-fit checkpoint and compare to recorded future norms."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from train_dynamics import ROOT, atomic_json, evaluate, load_data, make_model, run, setup


def main():
    reference=ROOT/'runs/fig3d_n1000_a8_s0'
    checkpoint=reference/'checkpoints/fit.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=saved['config'];step=saved['step_completed']
    setup(config['seed'])
    model=make_model(config['seed'],config['alpha'],'cuda');model.load_state_dict(saved['model'])
    train,_,_=load_data(ROOT/'data','cuda')
    fit=evaluate(model,train[0][:config['n']],train[1][:config['n']])
    assert fit['accuracy']>=config['fit_accuracy'] and fit['mse_element_mean']<=config['fit_mse']
    out=ROOT/'metadata/reference_checkpoint_replay'
    out.mkdir(exist_ok=False)
    shutil.copy2(reference/'norms.npy',out/'norms.npy')
    run(config,out,ROOT/'data',resume=checkpoint,stop_after=step+200)
    original=np.load(reference/'norms.npy',mmap_mode='r')
    replay=np.load(out/'norms.npy',mmap_mode='r')
    error=float(np.abs(original[step:step+201]-replay[step:step+201]).max())
    assert error<=1e-10
    atomic_json(out/'verification.json',dict(passed=True,fit_step=step,fit_train_metrics=fit,
                replay_steps=200,maximum_norm_error=error,
                note='Replayed actual fit checkpoint against the original subsequent per-step norms; full state equality separately checked in smoke.'))
    print(json.dumps(dict(passed=True,fit_step=step,maximum_norm_error=error)))


if __name__=='__main__':main()
