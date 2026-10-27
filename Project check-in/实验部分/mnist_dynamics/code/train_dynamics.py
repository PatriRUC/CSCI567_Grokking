"""Local FP64 unconstrained MNIST dynamics with resumable AdamW and sampler.

Every step refers to parameters AFTER that many optimizer updates. Evaluation
uses complete splits and cannot consume the independent training RNG stream.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = '3a0a57346230a7255e5390fcd565812d00f1a281'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def setup(seed, device='cuda'):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this config; do not silently change device.')


def environment():
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                platform=platform.platform(), torch=torch.__version__, torchvision=torchvision.__version__,
                numpy=np.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                code_sha256=file_hash(__file__))


def make_model(seed, alpha, device):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 200), nn.ReLU(),
                          nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, 10))
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(alpha)
    return model.to(device)


def make_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],
                             betas=tuple(config['betas']), eps=config['eps'], foreach=False, fused=False)


def load_data(data_root, device, download=False):
    raw_train = torchvision.datasets.MNIST(str(data_root), train=True, download=download)
    raw_test = torchvision.datasets.MNIST(str(data_root), train=False, download=download)
    def convert(ds):
        # Match ToTensor's CPU float64 division before the device transfer.
        return ds.data.to(dtype=torch.float64).flatten(1).div_(255).to(device), ds.targets.to(device)
    train, test = convert(raw_train), convert(raw_test)
    return train, test, raw_train


class Sampler:
    """Shuffle every epoch, retain the final short batch, serialize exact position."""
    def __init__(self, n, batch, seed, device):
        self.n, self.batch, self.device = n, batch, device
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.epoch, self.cursor = -1, n
        self.permutation = torch.empty(0, dtype=torch.long)
        self.device_permutation = self.permutation.to(device)

    def next(self):
        if self.cursor >= self.n:
            self.permutation = torch.randperm(self.n, generator=self.generator)
            self.device_permutation = self.permutation.to(self.device)
            self.cursor = 0
            self.epoch += 1
        start = self.cursor
        self.cursor = min(start + self.batch, self.n)
        return self.device_permutation[start:self.cursor]

    def state_dict(self):
        return dict(n=self.n, batch=self.batch, epoch=self.epoch, cursor=self.cursor,
                    permutation=self.permutation.clone(), generator=self.generator.get_state())

    def load_state_dict(self, state):
        assert (state['n'], state['batch']) == (self.n, self.batch)
        self.epoch, self.cursor = state['epoch'], state['cursor']
        self.permutation = state['permutation'].cpu().clone()
        self.device_permutation = self.permutation.to(self.device)
        self.generator.set_state(state['generator'].cpu())


@torch.no_grad()
def norms(model):
    squares = torch.stack([sum(p.square().sum() for p in model[i].parameters()) for i in (1, 3, 5)])
    return torch.cat((squares.sum().sqrt().reshape(1), squares.sqrt()))


def update(model, optimizer, x, labels):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    target = F.one_hot(labels, 10).to(logits.dtype)
    loss = F.mse_loss(logits, target)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def evaluate(model, x, labels, chunk=2000):
    was_training = model.training
    model.eval()
    correct = torch.zeros((), dtype=torch.int64, device=x.device)
    squared = torch.zeros((), dtype=torch.float64, device=x.device)
    for start in range(0, len(labels), chunk):
        y = labels[start:start+chunk]
        output = model(x[start:start+chunk])
        correct += output.argmax(1).eq(y).sum()
        squared += (output - F.one_hot(y, 10).to(output.dtype)).square().sum()
    values = torch.stack((correct.to(torch.float64), squared)).cpu().tolist()
    model.train(was_training)
    return dict(correct=int(values[0]), count=len(labels), accuracy=values[0]/len(labels),
                mse_element_mean=values[1]/(10*len(labels)), mse_per_sample_sum_outputs=values[1]/len(labels))


def evaluation_steps(config):
    total = config['steps']
    if config['phase'] in ('fig3d', 'reference', 'smoke', 'validation'):
        freq = math.ceil(config.get('schedule_budget', 100000)/150)
        return sorted({t for t in range(total+1) if t < 30 or (t < 150 and t % 10 == 0) or t % freq == 0} | {total})
    return sorted({t for t in (0, 10, 50, 100, 200) if t <= total} | set(range(500, total+1, 500)) | {total})


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def save_checkpoint(path, model, optimizer, sampler, config, step, rows, fit_step, elapsed):
    payload = dict(config=config, config_hash=config_hash(config), step_completed=step,
                   model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state_dict(),
                   rng=capture_rng(), metrics=rows, fit_step=fit_step, elapsed_seconds=elapsed,
                   code_sha256=file_hash(__file__), official_commit=OFFICIAL_COMMIT,
                   parameter_order=[name for name, _ in model.named_parameters()])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def write_metrics(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    tmp = Path(path).with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def event_summary(rows, budget, completed):
    evaluations = [r for r in rows if r.get('test_count')]
    previous = None
    for row in evaluations:
        # Integer comparison avoids rounding ambiguity at exactly 60 percent.
        if row['test_correct'] * 5 > row['test_count'] * 3:
            return dict(event_observed=True, t60=row['step'], previous_eval_step=previous,
                        threshold_status='reached', last_observed_step=evaluations[-1]['step'], budget_steps=budget)
        previous = row['step']
    return dict(event_observed=False, t60=None, previous_eval_step=None,
                threshold_status='threshold_not_reached' if completed else 'incomplete',
                last_observed_step=evaluations[-1]['step'] if evaluations else None, budget_steps=budget)


def run(config, output, data_root, resume=None, stop_after=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ckdir = output/'checkpoints'
    ckdir.mkdir(exist_ok=True)
    setup(config['seed'], config['device'])
    if resume is None and (output/'resolved_config.json').exists():
        raise RuntimeError('Output already exists; use --resume or a fresh output directory.')
    atomic_json(output/'resolved_config.json', config)
    atomic_json(output/'environment.json', environment())
    model = make_model(config['seed'], config['alpha'], config['device'])
    optimizer = make_optimizer(model, config)
    sampler = Sampler(config['n'], config['batch_size'], 100000 + config['seed']*1000000 + config['n'], config['device'])
    train, test, _ = load_data(data_root, config['device'])
    x, y = train[0][:config['n']], train[1][:config['n']]
    indices_path = output/'train_indices.npy'
    if not indices_path.exists():
        np.save(indices_path, np.arange(config['n'], dtype=np.int64))
    total = config['steps']
    target = total if stop_after is None else min(stop_after, total)
    rows, fit_step, start_step, prior_elapsed = [], None, 0, 0.
    norm_path = output/'norms.npy'
    if resume:
        saved = torch.load(resume, map_location=config['device'], weights_only=False)
        if saved['config_hash'] != config_hash(config):
            raise RuntimeError('Resume configuration mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler'])
        restore_rng(saved['rng'])
        start_step, rows, fit_step = saved['step_completed'], saved['metrics'], saved['fit_step']
        prior_elapsed = saved['elapsed_seconds']
        norm_disk = np.lib.format.open_memmap(norm_path, mode='r+')
        assert norm_disk.shape == (total+1, 4)
        norm_disk[start_step+1:] = np.nan
    else:
        norm_disk = np.lib.format.open_memmap(norm_path, mode='w+', dtype=np.float64, shape=(total+1, 4))
        norm_disk[:] = np.nan
        norm_disk[0] = norms(model).cpu().numpy()
    norm_buffer = torch.empty((1000, 4), dtype=torch.float64, device=config['device'])
    pending_start, pending_count = start_step+1, 0
    def flush_norms():
        nonlocal pending_start, pending_count
        if pending_count:
            norm_disk[pending_start:pending_start+pending_count] = norm_buffer[:pending_count].cpu().numpy()
            pending_start += pending_count
            pending_count = 0
        norm_disk.flush()
    planned_evals = set(evaluation_steps(config))
    start_time = time.perf_counter()
    def elapsed():
        return prior_elapsed + time.perf_counter() - start_time
    def record(step, include_test=True):
        nonlocal fit_step
        tr = evaluate(model, x, y)
        row = dict(step=step, elapsed_seconds=elapsed(), **{'train_'+k:v for k,v in tr.items()})
        if include_test:
            te = evaluate(model, *test)
            row.update({'test_'+k:v for k,v in te.items()})
        vals = norms(model).cpu().tolist()
        row.update(dict(global_norm=vals[0], layer1_norm=vals[1], layer2_norm=vals[2], layer3_norm=vals[3]))
        if not all(math.isfinite(float(v)) for v in row.values()):
            raise RuntimeError('Non-finite evaluation')
        rows.append(row)
        if fit_step is None and tr['accuracy'] >= config['fit_accuracy'] and tr['mse_element_mean'] <= config['fit_mse']:
            fit_step = step
            flush_norms()
            save_checkpoint(ckdir/'fit.pt', model, optimizer, sampler, config, step, rows, fit_step, elapsed())
        write_metrics(output/'metrics.csv', rows)
    if not resume:
        record(0)
        save_checkpoint(ckdir/'init.pt', model, optimizer, sampler, config, 0, rows, fit_step, elapsed())
    atomic_json(output/'status.json', dict(state='running', step=start_step, total_steps=total, config_hash=config_hash(config)))
    step = start_step
    try:
        for step in range(start_step+1, target+1):
            batch_indices = sampler.next()
            loss = update(model, optimizer, x[batch_indices], y[batch_indices])
            norm_buffer[pending_count] = norms(model)
            pending_count += 1
            if pending_count == 1000:
                flush_norms()
            is_fit_check = (fit_step is None and config['fit_check_every'] > 0 and step % config['fit_check_every'] == 0)
            if step in planned_evals or is_fit_check:
                record(step, step in planned_evals)
            if step % 1000 == 0 or step == target:
                flush_norms()
                if not torch.isfinite(loss).item() or not np.isfinite(norm_disk[step]).all():
                    raise RuntimeError('Non-finite training loss or norm')
                status = dict(state='running', step=step, total_steps=total, elapsed_seconds=elapsed(), fit_step=fit_step,
                              train_batch_mse=float(loss.cpu()), global_norm=float(norm_disk[step, 0]))
                atomic_json(output/'status.json', status)
                print(json.dumps(status), flush=True)
            if step % config['checkpoint_every'] == 0 or step == target:
                flush_norms()
                save_checkpoint(ckdir/'latest.pt', model, optimizer, sampler, config, step, rows, fit_step, elapsed())
        completed = step == total
        events = event_summary(rows, total, completed)
        atomic_json(output/'events.json', {**events, 'fit_step':fit_step, 'fit_checkpoint_exists':(ckdir/'fit.pt').exists()})
        if completed:
            save_checkpoint(ckdir/'final.pt', model, optimizer, sampler, config, step, rows, fit_step, elapsed())
        atomic_json(output/'status.json', dict(state='complete' if completed else 'paused_for_validation', step=step,
                                              total_steps=total, elapsed_seconds=elapsed(), config_hash=config_hash(config)))
        print('RUN_COMPLETE' if completed else 'RUN_CHECKPOINT_SAVED', flush=True)
    except BaseException as exc:
        flush_norms()
        atomic_json(output/'status.json', dict(state='failed', step=step, total_steps=total,
                                              elapsed_seconds=elapsed(), error=repr(exc)))
        raise
    finally:
        norm_disk.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--download-only', action='store_true')
    args = p.parse_args()
    if args.download_only:
        setup(0, 'cpu')
        load_data(args.data_root, 'cpu', download=True)
        atomic_json(ROOT/'metadata/data_hashes.json', {f.name:file_hash(f) for f in (args.data_root/'MNIST/raw').glob('*') if f.is_file()})
        return
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        config = saved['config']
        output = args.output or args.resume.resolve().parents[1]
    else:
        if not args.config or not args.output:
            p.error('--config and --output required for a new run')
        config = json.loads(args.config.read_text(encoding='utf-8'))
        output = args.output
    run(config, output, args.data_root, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
