"""Tangential-update intervention under one shared global-norm trajectory (single seed, closed loop).

Phase A  trains the official MNIST model (N, alpha, weight decay) until the training set is fitted and forks there.
Phase B  continues from that fork with K stacked copies that see identical minibatches:
           slice 0      plain AdamW                       -> reference run, defines R_t = ||theta_t||
           slice j>=1   AdamW update split into radial + tangential parts, tangential part times c_j,
                        result rescaled to the reference norm R_{t+1} at every step.
         c=1 is the replay control: it must reproduce the reference before any other c is interpreted.

The model, sampler, AdamW step and data conversion are imported unchanged from mnist_dynamics/code/run_dynamics.py.
"""
import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
DYNAMICS_CODE = HERE.parents[1] / 'mnist_dynamics' / 'code'
sys.path.insert(0, str(DYNAMICS_CODE))
import run_dynamics as rd  # noqa: E402

LR = rd.LR
VAL_START = 50000  # validation split: MNIST training images 50000..59999, never used for training (N <= 50000)
THRESHOLDS = [0.60, 0.70, 0.80, 0.85, 0.90]
METRICS = ['train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss']
PER_STEP = ['global_norm', 'radial_coef', 'tangential_norm', 'pre_projection_norm', 'angle', 'dist_to_ref', 'batch_mse']
MATCHED_STEPS = [0, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
REPLAY_WINDOW, REPLAY_TOL, NORM_TOL, RESUME_TOL = 1000, 1e-9, 1e-10, 1e-9


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(device):
    cuda = device.type == 'cuda'
    return dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), python=sys.version, platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, device=device.type, cuda=torch.version.cuda if cuda else None,
                gpu=torch.cuda.get_device_name() if cuda else None,
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if cuda else None,
                cpu_threads=torch.get_num_threads(), dtype=str(torch.get_default_dtype()),
                tf32=torch.backends.cuda.matmul.allow_tf32)


def flat(params):
    """All trainable parameters of every slice as one (K, P) matrix, fixed order w1 w2 w3 b1 b2 b3."""
    return torch.cat([p.detach().flatten(1) for p in params], 1)


@torch.no_grad()
def write_back(params, new):
    """Overwrite slices 1.. with rows of new; slice 0 (reference) is never touched."""
    offset = 0
    for p in params:
        size = p[0].numel()
        p[1:] = new[1:, offset:offset + size].reshape(p[1:].shape)
        offset += size


@torch.no_grad()
def evaluate(model, splits, k, chunk=5000):
    """Full-split accuracy and loss per slice. Loss = sum of the 10 squared errors, mean over samples (10 x training MSE)."""
    out = torch.zeros(k, 2 * len(splits), device=splits[0][0].device)
    for i, (x, y) in enumerate(splits):
        for s in range(0, len(y), chunk):
            p = model(x[s:s + chunk].unsqueeze(0).expand(k, -1, -1))
            labels = y[s:s + chunk]
            out[:, i] += p.argmax(-1).eq(labels).sum(1)
            out[:, len(splits) + i] += (p - F.one_hot(labels, 10).to(p.dtype)).square().sum((1, 2))
        out[:, i] /= len(y)
        out[:, len(splits) + i] /= len(y)
    return out


def first_reach(steps, values, threshold):
    hit = np.flatnonzero(np.nan_to_num(values, nan=-1.0) >= threshold)
    return int(steps[hit[0]]) if len(hit) else None


def run(args):
    root, device = args.root, torch.device(args.device)
    if (root / 'summary.json').exists():
        raise RuntimeError('%s already holds a finished run; use a new --root.' % root)
    if 1.0 not in args.cs:
        raise RuntimeError('--cs must contain 1 (the replay control).')
    n, alpha, wd, seed, steps = args.n, args.alpha, args.weight_decay, args.seed, args.steps
    labels = ['ref'] + ['c=%g' % c for c in args.cs]
    k = len(labels)
    _, _, (train_x, train_y), (test_x, test_y) = rd.get_data(args.data_root, device)
    splits = [(train_x[:n], train_y[:n]), (train_x[VAL_START:], train_y[VAL_START:]), (test_x, test_y)]
    batch_seed = seed * 1000000 + 10 * n
    config = dict(seed=seed, n=n, alpha=alpha, lr=LR, weight_decay=wd, batch_size=rd.BATCH, groups=labels, cs=args.cs,
                  steps_after_fork=steps, max_prefork_steps=args.max_prefork_steps,
                  fork_rule='first scheduled evaluation with full train accuracy >= %g%s; test and validation data are not consulted'
                            % (args.fit_accuracy, '' if args.fit_train_mse is None else ' and train MSE <= %g' % args.fit_train_mse),
                  intervention='theta* = theta_t + radial(Delta) + c * tangential(Delta), Delta = actual AdamW update; '
                               'theta_{t+1} = R_{t+1} * theta* / ||theta*||, R_{t+1} = reference norm after the same step',
                  parameter_vector='all weights and biases, order w1 w2 w3 b1 b2 b3',
                  adamw_state='copied from the fork into every slice, then updated normally from each slice\'s own gradients',
                  minibatches='one shared stream for all slices; EpochSampler seed %d' % batch_seed,
                  splits=dict(train='first %d MNIST training images' % n, val='MNIST training images %d..59999' % VAL_START,
                              test='10,000 MNIST test images, recorded only'),
                  loss='nn.MSELoss mean over batch and 10 outputs, one-hot targets', dtype='float64', tf32=False,
                  evaluation_schedule='steps after fork: 0-99 every step, 100-990 every 10, then every 100',
                  metric_names=METRICS, per_step_names=PER_STEP, thresholds=THRESHOLDS,
                  tolerances=dict(replay_window_steps=REPLAY_WINDOW, replay_relative=REPLAY_TOL, norm_relative=NORM_TOL, resume_relative=RESUME_TOL),
                  code_sha256={'run_direction.py': sha256(__file__), 'run_dynamics.py': sha256(rd.__file__)},
                  official_commit=rd.COMMIT)
    rd.atomic_json(root / 'config.json', config)
    rd.atomic_json(root / 'environment.json', environment(device))

    # ---- Phase A: official training up to the fork ----
    model1 = rd.StackedMLP([alpha], seed, device)
    opt1 = torch.optim.AdamW(model1.parameters(), lr=LR, weight_decay=wd)
    sampler = rd.EpochSampler(n, batch_seed)
    pre_schedule = set(rd.eval_schedule(args.max_prefork_steps))
    pre_steps, pre_metrics, pre_norm, fork_step = [], [], [], None
    for step in range(args.max_prefork_steps + 1):
        if step:
            idx, mask = rd.padded([sampler.next()], device)
            rd.update(model1, opt1, train_x, train_y, idx, mask)
        if step in pre_schedule:
            m = evaluate(model1, splits, 1)[0].cpu().numpy()
            pre_steps.append(step); pre_metrics.append(m)
            pre_norm.append(float(model1.layer_norms_squared().detach().sum().sqrt()))
            if m[0] >= args.fit_accuracy and (args.fit_train_mse is None or m[3] / 10 <= args.fit_train_mse):
                fork_step = step
                break
    if fork_step is None:
        raise RuntimeError('Training set not fitted within %d steps; no fork.' % args.max_prefork_steps)
    fork_metrics = dict(zip(METRICS, map(float, pre_metrics[-1])))
    params1 = list(model1.parameters())
    torch.save(dict(seed=seed, n=n, alpha=alpha, step=fork_step, metrics=fork_metrics,
                    parameters=[p[0].detach().cpu() for p in params1], parameter_names=['w1', 'w2', 'w3', 'b1', 'b2', 'b3'],
                    adamw=dict(step=float(opt1.state[params1[0]]['step']), lr=LR, weight_decay=wd,
                               exp_avg=[opt1.state[p]['exp_avg'][0].cpu() for p in params1],
                               exp_avg_sq=[opt1.state[p]['exp_avg_sq'][0].cpu() for p in params1]),
                    sampler=sampler.state()), root / 'fork_checkpoint.pt')
    print(json.dumps(dict(event='fork', step=fork_step, **fork_metrics)), flush=True)

    # ---- Phase B: K slices from the same fork ----
    model = rd.StackedMLP([alpha] * k, seed, device)
    params = list(model.parameters())
    with torch.no_grad():
        for p, q in zip(params, params1):
            p.copy_(q.expand_as(p))
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=wd)
    for p, q in zip(params, params1):  # copy the whole AdamW state, whatever keys this torch version keeps
        optimizer.state[p] = {key: (v.repeat(k, *[1] * (v.dim() - 1)) if torch.is_tensor(v) and v.shape == q.shape
                                    else copy.deepcopy(v)) for key, v in opt1.state[q].items()}
    cvec = torch.tensor([1.0] + list(args.cs), device=device)
    schedule = [s for s in rd.eval_schedule(steps) if s <= steps]
    metrics = torch.zeros(len(schedule), k, len(METRICS), device=device)
    per_step = torch.full((steps + 1, k, len(PER_STEP)), float('nan'), device=device)
    layer_norms_sq = torch.zeros(steps + 1, k, 3, device=device)
    layer_norms_sq[0] = model.layer_norms_squared().detach()
    per_step[0, :, 0] = layer_norms_sq[0].sum(1).sqrt()
    metrics[0] = evaluate(model, splits, k)
    eval_pos, resume_error, diverged, start = 1, 0.0, [None] * k, time.perf_counter()

    for step in range(1, steps + 1):
        batch = sampler.next()
        theta = flat(params)                                   # theta_t
        idx, mask = rd.padded([batch] * k, device)
        losses = rd.update(model, optimizer, train_x, train_y, idx, mask)
        candidate = flat(params)                               # plain AdamW result for every slice
        radius = theta.norm(dim=1)
        u = theta / radius[:, None]
        delta = candidate - theta
        radial = (u * delta).sum(1)                            # signed length of the radial part
        tangential = delta - radial[:, None] * u
        star = theta + radial[:, None] * u + cvec[:, None] * tangential
        star_norm = star.norm(dim=1)
        target = candidate[0].norm()                           # R_{t+1}
        new = star * (target / star_norm)[:, None]
        write_back(params, new)
        final = torch.cat([candidate[:1], new[1:]])            # what is now stored in the parameters
        lsq = model.layer_norms_squared().detach()             # norms re-read from the stored parameters
        layer_norms_sq[step] = lsq
        u_next = final / final.norm(dim=1, keepdim=True)
        angle = 2 * torch.atan2((u_next - u).norm(dim=1), (u_next + u).norm(dim=1))  # stable form of arccos(u.u')
        per_step[step] = torch.stack([lsq.sum(1).sqrt(), radial, tangential.norm(dim=1), star_norm, angle,
                                      (final - final[:1]).norm(dim=1), losses], 1)
        if step <= args.resume_check_steps:  # uninterrupted Phase-A model on the same batches vs the reference slice
            idx1, mask1 = rd.padded([batch], device)
            rd.update(model1, opt1, train_x, train_y, idx1, mask1)
            resume_error = max(resume_error, max(float((p[0] - q[0]).detach().abs().max()) for p, q in zip(params, params1)))
        if step == schedule[eval_pos]:
            metrics[eval_pos] = evaluate(model, splits, k)
            eval_pos += 1
        if step == 500 or step % 5000 == 0:
            finite = torch.isfinite(losses).cpu().numpy()
            for j in range(k):
                if diverged[j] is None and not finite[j]:
                    diverged[j] = step
            elapsed = time.perf_counter() - start
            m = metrics[eval_pos - 1].cpu().numpy()
            status = dict(state='running', step=step, steps=steps, elapsed_seconds=round(elapsed, 1),
                          steps_per_second=round(step / elapsed, 1), projected_total_minutes=round(elapsed / step * steps / 60, 1),
                          test_acc={labels[j]: round(float(m[j, 2]), 4) for j in range(k)}, diverged=diverged,
                          updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            rd.atomic_json(root / 'status.json', status)
            print(json.dumps(status), flush=True)
    elapsed = time.perf_counter() - start

    # ---- Save raw records ----
    metrics, per_step, layer_norms_sq = metrics.cpu().numpy(), per_step.cpu().numpy(), layer_norms_sq.cpu().numpy()
    eval_steps = np.array(schedule)
    np.savez_compressed(root / 'history.npz', groups=np.array(labels), cs=np.array([1.0] + list(args.cs)), fork_step=fork_step,
                        eval_steps=eval_steps, metrics=metrics, metric_names=np.array(METRICS),
                        per_step=per_step, per_step_names=np.array(PER_STEP), layer_norms_squared=layer_norms_sq,
                        prefork_steps=np.array(pre_steps), prefork_metrics=np.array(pre_metrics), prefork_norm=np.array(pre_norm))
    torch.save(dict(groups=labels, step=fork_step + steps, parameter_names=['w1', 'w2', 'w3', 'b1', 'b2', 'b3'],
                    parameters=[p.detach().cpu() for p in params]), root / 'final_checkpoint.pt')

    norm = per_step[:, :, 0]
    with np.errstate(invalid='ignore', divide='ignore'):
        norm_error = np.abs(norm - norm[:, :1]) / norm[:, :1]
        rel_dist = per_step[:, :, 5] / norm[:, :1]
    cum_angle = np.nancumsum(per_step[:, :, 4], 0)
    with open(root / 'curves.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'c', 'steps_since_fork', 'step', *METRICS, 'train_mse', 'global_norm', 'reference_norm',
                         'norm_relative_error', 'cumulative_angle_rad', 'norm_layer1', 'norm_layer2', 'norm_layer3'])
        for j, label in enumerate(labels):
            for e, s in enumerate(schedule):
                writer.writerow([label, 1.0 if j == 0 else args.cs[j - 1], s, fork_step + s, *metrics[e, j], metrics[e, j, 3] / 10,
                                 norm[s, j], norm[s, 0], norm_error[s, j], cum_angle[s, j], *np.sqrt(layer_norms_sq[s, j])])

    # ---- Summary ----
    groups = []
    for j, label in enumerate(labels):
        bad = np.flatnonzero(~np.isfinite(per_step[1:, j, 6]))
        groups.append(dict(
            group=label, c=None if j == 0 else args.cs[j - 1],
            steps_to_test={'%d' % (100 * t): first_reach(eval_steps, metrics[:, j, 2], t) for t in THRESHOLDS},
            steps_to_val={'%d' % (100 * t): first_reach(eval_steps, metrics[:, j, 1], t) for t in THRESHOLDS},
            final=dict(zip(METRICS, map(float, metrics[-1, j]))),
            train_mse_at={'%d' % s: float(metrics[schedule.index(s), j, 3] / 10) for s in MATCHED_STEPS if s <= steps},
            test_acc_at={'%d' % s: float(metrics[schedule.index(s), j, 2]) for s in MATCHED_STEPS if s <= steps},
            max_norm_relative_error=float(np.nanmax(norm_error[:, j])),
            total_angle_rad=float(cum_angle[-1, j]), total_angle_vs_reference=float(cum_angle[-1, j] / cum_angle[-1, 0]),
            mean_tangential_norm=float(np.nanmean(per_step[1:, j, 2])),
            final_relative_distance_to_reference=float(rel_dist[-1, j]),
            first_non_finite_step=int(bad[0]) + 1 if len(bad) else None))
    j1 = 1 + args.cs.index(1.0)
    window = min(REPLAY_WINDOW, steps)
    replay = dict(group=labels[j1], window_steps=window,
                  max_relative_distance_in_window=float(np.nanmax(rel_dist[:window + 1, j1])),
                  max_relative_distance_full_run=float(np.nanmax(rel_dist[:, j1])),
                  max_norm_relative_error=float(np.nanmax(norm_error[:, j1])),
                  max_abs_metric_difference_full_run=dict(zip(METRICS, map(float, np.abs(metrics[:, j1] - metrics[:, 0]).max(0)))))
    replay['passed'] = bool(replay['max_relative_distance_in_window'] <= REPLAY_TOL and replay['max_norm_relative_error'] <= NORM_TOL)
    fork_norm = float(norm[0, 0])
    resume = dict(steps=min(args.resume_check_steps, steps), max_parameter_abs_error=resume_error,
                  passed=bool(resume_error <= RESUME_TOL * fork_norm))
    norms_matched = bool(all(g['max_norm_relative_error'] <= NORM_TOL for g in groups if g['first_non_finite_step'] is None))
    summary = dict(fork=dict(step=fork_step, global_norm=fork_norm, **fork_metrics), groups=groups,
                   replay_control=replay, resume_check=resume, norms_matched=norms_matched,
                   reference_norm=dict(at_fork=fork_norm, minimum=float(np.nanmin(norm[:, 0])), final=float(norm[-1, 0])),
                   timing=dict(seconds_after_fork=elapsed, steps_per_second=steps / elapsed),
                   interpretable=bool(replay['passed'] and resume['passed'] and norms_matched))
    rd.atomic_json(root / 'summary.json', summary)
    rd.atomic_json(root / 'status.json', dict(state='complete', steps=steps, elapsed_seconds=elapsed,
                                             completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    print(json.dumps(dict(event='summary', fork_step=fork_step, replay_control=replay, resume_check=resume, norms_matched=norms_matched,
                          steps_to_test_60={g['group']: g['steps_to_test']['60'] for g in groups},
                          minutes=round(elapsed / 60, 2))), flush=True)
    if not summary['interpretable']:
        print('VERIFICATION_FAILED: do not interpret the c != 1 groups.', flush=True)
        sys.exit(3)
    print('RUN_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='output directory (one per run)')
    parser.add_argument('--data-root', type=Path, required=True, help='directory that holds or will receive MNIST/raw')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n', type=int, default=1000, help='training subset: first N MNIST training images')
    parser.add_argument('--alpha', type=float, default=8.0, help='initialisation scale')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='official notebook default')
    parser.add_argument('--cs', type=float, nargs='+', default=[1.0, 2.0, 0.5], help='tangential multipliers; must include 1')
    parser.add_argument('--steps', type=int, default=100000, help='updates after the fork, same budget for every group')
    parser.add_argument('--max-prefork-steps', type=int, default=20000)
    parser.add_argument('--fit-accuracy', type=float, default=rd.FIT_ACCURACY)
    parser.add_argument('--fit-train-mse', type=float, default=None, help='optional extra fork condition on full-train MSE')
    parser.add_argument('--resume-check-steps', type=int, default=200)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='cpu is for smoke tests')
    args = parser.parse_args()
    if args.n > VAL_START:
        raise RuntimeError('N must leave the validation split untouched (N <= %d).' % VAL_START)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available; pass --device cpu only for smoke tests.')
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    run(args)


if __name__ == '__main__':
    main()
