"""Single-seed Omnigrok MNIST grokking dynamics (unconstrained AdamW, norm free to change).
Independent (N, alpha) models are stacked; each has its own data subset, batch stream and AdamW state slice.
Experiments: fig3d / fig3e follow the paper figure labels; notebook follows the official notebook defaults.
"""
import argparse
import datetime as dt
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

COMMIT = '3a0a57346230a7255e5390fcd565812d00f1a281'
OFFICIAL_NOTEBOOK = 'mnist/grokking/mnist-grokking.ipynb'
STEPS, BATCH, LR = 100000, 200, 1e-3
FIT_ACCURACY = 0.99
# alphas lists each experiment's alpha values in batch-seed order (seed rule uses the index).
EXPERIMENTS = {
    'fig3d': dict(models=[(1000, float(a)) for a in range(1, 11)], weight_decay=1e-4, alphas=[float(a) for a in range(1, 11)],
                  source='paper Fig.3d title: "MNIST Grokking With Varying alpha, 1000 Train Points, gamma=1e-4"; colour bar alpha 1-10'),
    'fig3e': dict(models=[(128 * 2**k, 8.0) for k in range(9)], weight_decay=5e-3, alphas=[8.0],
                  source='paper Fig.3e: "weight decay ~5e-3", "Steps to Test Accuracy > 60%", N=128*2^k read from the figure; alpha=8 is our assumption'),
    # Official notebook defaults (alpha=8, wd=0.01) over our own N grid, plus an alpha=1 control.
    'notebook': dict(models=[(n, 8.0) for n in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000]] + [(1000, 1.0)],
                     weight_decay=0.01, alphas=[8.0, 1.0],
                     source='official mnist-grokking.ipynb defaults; N grid and alpha=1 control chosen by us'),
}
VALIDATION_MODELS = {'notebook': [(100, 8.0), (500, 8.0), (1000, 8.0), (1000, 1.0)],
                     'fig3d': [(1000, 1.0), (1000, 5.0), (1000, 10.0)],
                     'fig3e': [(128, 8.0), (1024, 8.0), (32768, 8.0)]}


def eval_schedule(steps):
    return sorted(set(range(0, 100)) | set(range(100, 1000, 10)) | set(range(1000, steps + 1, 100)) | {steps})


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def environment():
    return dict(timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), python=sys.version,
                platform=platform.platform(), torch=torch.__version__, torchvision=torchvision.__version__,
                numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                dtype=str(torch.get_default_dtype()), tf32=torch.backends.cuda.matmul.allow_tf32)


def base_model(seed):
    # Same RNG use as the official notebook: seed, then build in float64 on CPU (Kaiming uniform defaults).
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 200), nn.ReLU(), nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, 10))


class StackedMLP(nn.Module):
    def __init__(self, alphas, seed, device):
        super().__init__()
        base = base_model(seed)
        scales = torch.as_tensor(alphas, device=device)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer in base:
            if isinstance(layer, nn.Linear):
                self.weights.append(nn.Parameter(layer.weight.detach().to(device).unsqueeze(0).repeat(len(alphas), 1, 1) * scales[:, None, None]))
                self.biases.append(nn.Parameter(layer.bias.detach().to(device).unsqueeze(0).repeat(len(alphas), 1) * scales[:, None]))

    def forward(self, h):  # h: (models, batch, 784)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b[:, None, :], h, w.transpose(1, 2))
            if i < 2:
                h = F.relu(h)
        return h

    def forward_one(self, j, h):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.addmm(b[j], h, w[j].t())
            if i < 2:
                h = F.relu(h)
        return h

    def layer_norms_squared(self):  # (models, layers), weight and bias together
        return torch.stack([w.square().sum((1, 2)) + b.square().sum(1) for w, b in zip(self.weights, self.biases)], 1)


class EpochSampler:
    """DataLoader(shuffle=True, drop_last=False) semantics: new permutation per epoch, short final batch."""
    def __init__(self, n, seed):
        self.n, self.generator, self.perm, self.cursor = n, torch.Generator().manual_seed(seed), None, n

    def next(self):
        if self.cursor >= self.n:
            self.perm, self.cursor = torch.randperm(self.n, generator=self.generator), 0
        idx = self.perm[self.cursor:self.cursor + BATCH]
        self.cursor += len(idx)
        return idx

    def state(self):
        return dict(generator=self.generator.get_state(), perm=self.perm, cursor=self.cursor)


def padded(batches, device):
    idx = torch.zeros(len(batches), BATCH, dtype=torch.long)
    mask = torch.zeros(len(batches), BATCH)
    for j, b in enumerate(batches):
        idx[j, :len(b)] = b
        mask[j, :len(b)] = 1
    return idx.to(device, non_blocking=True), mask.to(device, non_blocking=True)


def update(model, optimizer, x, y, idx, mask):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x[idx])
    squared = (prediction - F.one_hot(y[idx], 10).to(prediction.dtype)).square().sum(-1) * mask
    # Per model: nn.MSELoss() mean over its real batch rows and 10 outputs; summed so gradients stay per-model.
    losses = squared.sum(1) / (mask.sum(1) * 10)
    losses.sum().backward()
    optimizer.step()
    return losses.detach()


def get_data(data_root, device):
    train = torchvision.datasets.MNIST(str(data_root), train=True, download=True)
    test = torchvision.datasets.MNIST(str(data_root), train=False, download=True)
    def convert(dataset):
        return dataset.data.to(torch.float64).flatten(1).div_(255).to(device), dataset.targets.to(device)
    return train, test, convert(train), convert(test)


@torch.no_grad()
def full_metrics(model, train_x, train_y, test_x, test_y, ns, chunk=5000):
    """Full-split accuracy and loss; loss sums the 10 squared errors per sample (official compute_loss)."""
    k = len(ns)
    out = torch.zeros(k, 4, device=train_x.device)  # train_acc, test_acc, train_loss, test_loss
    for j, n in enumerate(ns):
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            p = model.forward_one(j, train_x[s:e])
            out[j, 0] += p.argmax(-1).eq(train_y[s:e]).sum()
            out[j, 2] += (p - F.one_hot(train_y[s:e], 10).to(p.dtype)).square().sum()
    for s in range(0, len(test_y), chunk):
        p = model(test_x[s:s + chunk].unsqueeze(0).expand(k, -1, -1))
        labels = test_y[s:s + chunk]
        out[:, 1] += p.argmax(-1).eq(labels).sum(1)
        out[:, 3] += (p - F.one_hot(labels, 10).to(p.dtype)).square().sum((1, 2))
    sizes = torch.as_tensor(ns, device=out.device, dtype=out.dtype)
    out[:, 0] /= sizes; out[:, 2] /= sizes
    out[:, 1] /= len(test_y); out[:, 3] /= len(test_y)
    return out


def validate(root, data_root, experiment, steps=50):
    """Official notebook loop vs stacked implementation on identical init and identical batches."""
    wd = EXPERIMENTS[experiment]['weight_decay']
    device = torch.device('cuda')
    raw_train, raw_test, (x, y), (xt, yt) = get_data(data_root, device)
    to_tensor = torchvision.transforms.ToTensor()
    image_match = torch.equal(x[:10].cpu(), torch.stack([to_tensor(raw_train[i][0]).flatten() for i in range(10)]))

    class Indexed(torch.utils.data.Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return self.n
        def __getitem__(self, i): return to_tensor(raw_train[i][0]), raw_train.targets[i], i

    configs = VALIDATION_MODELS[experiment]
    checkpoints = (1, 10, steps)
    serial, batches = [], []
    for n, alpha in configs:
        # Official cell order: seeds, dataset/DataLoader, model, scale on device, AdamW, nn.MSELoss mean.
        torch.manual_seed(0); torch.cuda.manual_seed_all(0); random.seed(0); np.random.seed(0)
        loader = torch.utils.data.DataLoader(Indexed(n), batch_size=BATCH, shuffle=True)
        mlp = base_model(0).to(device)
        with torch.no_grad():
            for p in mlp.parameters(): p.data = alpha * p.data
        opt = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=wd)
        one_hots = torch.eye(10, 10).to(device)
        stream, snaps, it = [], {}, iter(())
        for step in range(1, steps + 1):
            try: xb, lb, ib = next(it)
            except StopIteration:
                it = iter(loader); xb, lb, ib = next(it)
            stream.append(ib)
            opt.zero_grad()
            nn.MSELoss()(mlp(xb.to(device)), one_hots[lb]).backward()
            opt.step()
            if step in checkpoints: snaps[step] = [p.detach().clone() for p in mlp.parameters()]
        serial.append((mlp, snaps)); batches.append(stream)
    model = StackedMLP([a for _, a in configs], 0, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
    comparisons = []
    for step in range(1, steps + 1):
        idx, mask = padded([stream[step - 1] for stream in batches], device)
        update(model, optimizer, x, y, idx, mask)
        if step in checkpoints:
            errors = []
            for j, (_, snaps) in enumerate(serial):
                for l in range(3):
                    errors.append((model.weights[l][j] - snaps[step][2 * l]).abs().max().item())
                    errors.append((model.biases[l][j] - snaps[step][2 * l + 1]).abs().max().item())
            comparisons.append(dict(step=step, max_parameter_abs_error=max(errors)))
    # Official full-split metrics vs ours after the last step.
    stacked = full_metrics(model, x, y, xt, yt, [n for n, _ in configs]).cpu().numpy()
    metric_errors = []
    with torch.no_grad():
        for j, ((n, _), (mlp, _)) in enumerate(zip(configs, serial)):
            p, pt = mlp(x[:n]), mlp(xt)
            ref = [p.argmax(1).eq(y[:n]).double().mean().item(), pt.argmax(1).eq(yt).double().mean().item(),
                   (p - F.one_hot(y[:n], 10)).square().sum().item() / n, (pt - F.one_hot(yt, 10)).square().sum().item() / len(yt)]
            metric_errors.append(float(np.abs(np.array(ref) - stacked[j]).max()))
    max_error = max(c['max_parameter_abs_error'] for c in comparisons)
    passed = bool(image_match and max_error < 1e-9 and max(metric_errors) < 1e-9)
    result = dict(passed=passed, experiment=experiment, weight_decay=wd, configs=configs, steps=steps, image_conversion_exact_match=image_match,
                  serial_comparisons=comparisons, max_metric_abs_error=max(metric_errors),
                  tolerance=1e-9, environment=environment())
    atomic_json(root / 'validation.json', result)
    print(json.dumps(result), flush=True)
    if not passed: raise RuntimeError('Official-loop/stacked validation failed; do not launch training.')


def run(root, data_root, seed, experiment, steps=STEPS):
    validation = json.loads((root / 'validation.json').read_text())
    if not validation['passed']: raise RuntimeError('Validation has not passed.')
    if validation.get('experiment', 'notebook') != experiment: raise RuntimeError('Validation was run for another experiment.')
    exp = EXPERIMENTS[experiment]
    MODELS, WD = exp['models'], exp['weight_decay']
    device = torch.device('cuda')
    _, _, (train_x, train_y), (test_x, test_y) = get_data(data_root, device)
    ns, alphas = [n for n, _ in MODELS], [a for _, a in MODELS]
    k = len(MODELS)
    schedule = eval_schedule(steps)
    config = dict(seed=seed, experiment=experiment, figure_source=exp['source'], official_commit=COMMIT,
                  official_notebook=OFFICIAL_NOTEBOOK, models=MODELS,
                  steps=steps, batch_size=BATCH, lr=LR, weight_decay=WD, optimizer='torch.optim.AdamW (defaults otherwise)',
                  loss='nn.MSELoss mean over batch and 10 outputs, one-hot targets', dtype='float64', tf32=False,
                  training_subset='first N MNIST training samples (official Subset(range(N)); paper text says random subset)',
                  initialization='official Kaiming-uniform base model from torch.manual_seed(seed), all parameters x alpha; shared across models',
                  train_seed_rule='seed*1000000 + 10*N + index of alpha in %s' % exp['alphas'],
                  evaluation='full train subset and full 10,000 test set at every scheduled step (after that many updates)',
                  evaluation_schedule='steps 0-99 every step, 100-990 every 10, 1000-%d every 100' % steps,
                  norms='global and per-layer L2 norms (weight+bias) recorded after every update',
                  fit_checkpoint='first scheduled step with full train accuracy >= %.2f; model slice, AdamW slice and sampler state saved' % FIT_ACCURACY,
                  protocol_changes=['batched independent models', 'GPU tensor cache', 'explicit per-model batch RNG',
                                    'dense full-split evaluation instead of 150 log points'])
    atomic_json(root / 'config.json', config)
    atomic_json(root / 'environment.json', environment())
    model = StackedMLP(alphas, seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    samplers = [EpochSampler(n, seed * 1000000 + 10 * n + exp['alphas'].index(a)) for n, a in MODELS]
    metrics = torch.zeros(len(schedule), k, 4, device=device)
    layer_norms_sq = torch.zeros(steps + 1, k, 3, device=device)
    layer_norms_sq[0] = model.layer_norms_squared().detach()
    fit_step = [None] * k
    eval_pos, start = 0, time.perf_counter()

    def evaluate(step):
        nonlocal eval_pos
        m = full_metrics(model, train_x, train_y, test_x, test_y, ns)
        metrics[eval_pos] = m
        eval_pos += 1
        cpu = m.cpu().numpy()
        if not np.isfinite(cpu).all():
            atomic_json(root / 'failure.json', dict(step=step, reason='non-finite metrics'))
            raise RuntimeError('Non-finite metrics at step %d' % step)
        for j in range(k):
            if fit_step[j] is None and cpu[j, 0] >= FIT_ACCURACY:
                fit_step[j] = step
                save_slice(root / 'checkpoints' / ('fit_N%05d_alpha%g.pt' % MODELS[j]), j, step, cpu[j])

    def save_slice(path, j, step, row):
        state = optimizer.state
        params = list(model.parameters())
        payload = dict(seed=seed, n=MODELS[j][0], alpha=MODELS[j][1], step=step, metrics=row.tolist(),
                       metric_names=['train_acc', 'test_acc', 'train_loss', 'test_loss'],
                       parameters=[p[j].detach().cpu() for p in params],
                       parameter_names=['w1', 'w2', 'w3', 'b1', 'b2', 'b3'],
                       adamw=dict(step=float(state[params[0]]['step']) if params[0] in state else 0.0,
                                  exp_avg=[state[p]['exp_avg'][j].cpu() for p in params] if params[0] in state else None,
                                  exp_avg_sq=[state[p]['exp_avg_sq'][j].cpu() for p in params] if params[0] in state else None,
                                  lr=LR, weight_decay=WD),
                       sampler=samplers[j].state())
        tmp = path.with_suffix('.tmp')
        torch.save(payload, tmp)
        tmp.replace(path)

    (root / 'checkpoints').mkdir(exist_ok=True)
    evaluate(0)
    for step in range(1, steps + 1):
        idx, mask = padded([s.next() for s in samplers], device)
        update(model, optimizer, train_x, train_y, idx, mask)
        layer_norms_sq[step] = model.layer_norms_squared().detach()
        if step == schedule[eval_pos]:
            evaluate(step)
        if step % 5000 == 0:
            m = metrics[eval_pos - 1].cpu().numpy()
            status = dict(state='running', step=step, steps=steps, elapsed_seconds=time.perf_counter() - start,
                          test_acc={'N%d_a%g' % c: round(float(m[j, 1]), 4) for j, c in enumerate(MODELS)},
                          updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            atomic_json(root / 'status.json', status)
            print(json.dumps(status), flush=True)
    for j in range(k):
        save_slice(root / 'checkpoints' / ('final_N%05d_alpha%g.pt' % MODELS[j]), j, steps, metrics[-1, j].cpu().numpy())
    np.savez_compressed(root / 'history.npz', models=np.array(MODELS), eval_steps=np.array(schedule),
                        metrics=metrics.cpu().numpy(), metric_names=np.array(['train_acc', 'test_acc', 'train_loss', 'test_loss']),
                        layer_norms_squared=layer_norms_sq.cpu().numpy(), fit_step=np.array([-1 if s is None else s for s in fit_step]))
    atomic_json(root / 'status.json', dict(state='complete', steps=steps, elapsed_seconds=time.perf_counter() - start,
                                         fit_step={'N%d_a%g' % c: fit_step[j] for j, c in enumerate(MODELS)},
                                         completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    print('TRAINING_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--experiment', choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--steps', type=int, default=STEPS, help='shorter budgets are for smoke checks only')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for this run.')
    if args.validate: validate(args.root, args.data_root, args.experiment)
    else: run(args.root, args.data_root, args.seed, args.experiment, args.steps)


if __name__ == '__main__': main()
