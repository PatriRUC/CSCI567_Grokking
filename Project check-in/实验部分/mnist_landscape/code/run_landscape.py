"""Single-seed Omnigrok Fig.3a-c landscape reproduction.
Independent alpha conditions are stacked; no gradients are shared between models.
"""
import argparse
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

COMMIT = '3a0a57346230a7255e5390fcd565812d00f1a281'
ORIGINAL_NS = [int(v) for v in 10 ** np.linspace(1, 4, 22)] + [int(v) for v in list(10 ** np.linspace(4, 5, 8)[1:6]) + [60000]]
ALPHAS = 10 ** np.linspace(-1, 1, 21)
ALL_NS = sorted(set(ORIGINAL_NS + [200, 8000]))


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
                threads=torch.get_num_threads(), dtype=str(torch.get_default_dtype()),
                tf32=torch.backends.cuda.matmul.allow_tf32)


def base_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 200), nn.ReLU(), nn.Linear(200, 200), nn.ReLU(), nn.Linear(200, 10))


class StackedMLP(nn.Module):
    def __init__(self, alphas, seed, device):
        super().__init__()
        base = base_model(seed)
        scales = torch.as_tensor(alphas, dtype=torch.float32, device=device)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for layer in base:
            if isinstance(layer, nn.Linear):
                self.weights.append(nn.Parameter(layer.weight.detach().to(device).unsqueeze(0).repeat(len(alphas), 1, 1) * scales[:, None, None]))
                self.biases.append(nn.Parameter(layer.bias.detach().to(device).unsqueeze(0).repeat(len(alphas), 1) * scales[:, None]))

    def forward(self, x):
        h = x.flatten(1).unsqueeze(0).expand(self.weights[0].shape[0], -1, -1)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.bmm(h, w.transpose(1, 2)) + b[:, None, :]
            if i < 2:
                h = F.relu(h)
        return h

    def norms_squared(self):
        return torch.stack([p.square().flatten(1).sum(1) for p in self.parameters()]).sum(0)

    @torch.no_grad()
    def project(self, target_sq):
        scale = (target_sq / self.norms_squared()).sqrt()
        for p in self.parameters():
            p.mul_(scale.reshape([-1] + [1] * (p.ndim - 1)))


def update(model, optimizer, target_sq, x, y):
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x)
    target = F.one_hot(y, 10).to(dtype=prediction.dtype).unsqueeze(0).expand_as(prediction)
    # Sum model means, rather than averaging over models: gradients match independent MSE training.
    losses = F.mse_loss(prediction, target, reduction='none').mean((1, 2))
    losses.sum().backward()
    optimizer.step()
    model.project(target_sq)
    return losses.detach()


def get_data(data_root, device):
    train = torchvision.datasets.MNIST(str(data_root), train=True, download=True)
    test = torchvision.datasets.MNIST(str(data_root), train=False, download=True)
    def convert(dataset):
        return dataset.data.to(dtype=torch.float32).unsqueeze(1).div_(255).to(device), dataset.targets.to(device)
    return train, test, convert(train), convert(test)


@torch.no_grad()
def sampled_metrics(model, x, y, generator):
    # Four independent draws reproduce the official train/test accuracy/loss sampling convention.
    # All alphas share each draw as a paired experimental design.
    out = {}
    for metric in ('accuracy', 'loss'):
        idx = torch.randperm(len(y), generator=generator)[:min(2000, len(y))].to(x.device)
        pred = model(x[idx])
        labels = y[idx]
        if metric == 'accuracy':
            values = pred.argmax(-1).eq(labels.unsqueeze(0)).float().mean(1)
        else:
            target = F.one_hot(labels, 10).to(pred.dtype).unsqueeze(0)
            # Official loss_f sums over ten outputs and divides by samples (10x nn.MSELoss).
            values = (pred - target).square().sum((1, 2)) / len(labels)
        out[metric] = values.cpu().numpy()
    return out


@torch.no_grad()
def full_metrics(model, x, y, chunk=2000):
    correct = torch.zeros(len(ALPHAS), device=x.device)
    loss = torch.zeros_like(correct)
    for start in range(0, len(y), chunk):
        pred = model(x[start:start+chunk])
        labels = y[start:start+chunk]
        correct += pred.argmax(-1).eq(labels.unsqueeze(0)).sum(1)
        loss += (pred-F.one_hot(labels, 10).to(pred.dtype).unsqueeze(0)).square().sum((1, 2))
    return (correct/len(y)).cpu().numpy(), (loss/len(y)).cpu().numpy()


def validate(root, data_root):
    device = torch.device('cuda')
    raw_train, _, (x, y), _ = get_data(data_root, device)
    image_match = torch.equal(x[:10].cpu(), torch.stack([torchvision.transforms.ToTensor()(raw_train[i][0]) for i in range(10)]))
    alphas = [.1, 1., 8.]
    batched = StackedMLP(alphas, 0, device)
    target_sq = batched.norms_squared().detach().clone()
    opt = torch.optim.AdamW(batched.parameters(), lr=.001, weight_decay=0., foreach=False)
    serial = []
    for alpha in alphas:
        model = base_model(0).to(device)
        with torch.no_grad():
            for p in model.parameters(): p.mul_(alpha)
        sq = sum(p.square().sum() for p in model.parameters()).detach().clone()
        serial.append((model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False), sq))
    comparisons = []
    for step in range(20):
        xx, yy = x[step*100:(step+1)*100], y[step*100:(step+1)*100]
        update(batched, opt, target_sq, xx, yy)
        for model, optimizer, sq in serial:
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(model(xx), F.one_hot(yy, 10).float()).backward()
            optimizer.step()
            with torch.no_grad():
                scale = (sq/sum(p.square().sum() for p in model.parameters())).sqrt()
                for p in model.parameters(): p.mul_(scale)
        if step in (0, 4, 19):
            errors = []
            for k, (model, _, _) in enumerate(serial):
                for j, idx in enumerate((1, 3, 5)):
                    errors.append((batched.weights[j][k]-model[idx].weight).abs().max().item())
                    errors.append((batched.biases[j][k]-model[idx].bias).abs().max().item())
            comparisons.append(dict(step=step+1, max_parameter_abs_error=max(errors)))
    norm_error = (batched.norms_squared()/target_sq).sqrt().sub(1).abs().max().item()
    max_error = max(v['max_parameter_abs_error'] for v in comparisons)
    passed = bool(image_match and max_error < 2e-4 and norm_error < 2e-6)
    model = StackedMLP(ALPHAS, 0, device)
    sq = model.norms_squared().detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False)
    for i in range(30): update(model, opt, sq, x[:100], y[:100])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(300): update(model, opt, sq, x[(i%10)*100:(i%10+1)*100], y[(i%10)*100:(i%10+1)*100])
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-t0
    result = dict(passed=passed, image_conversion_exact_match=image_match, serial_comparisons=comparisons,
                  maximum_norm_relative_error=norm_error, grouped_21_models_300_steps_seconds=elapsed,
                  grouped_training_10000_steps_projection_seconds=elapsed/300*10000,
                  projected_30_groups_training_hours=elapsed/300*10000*30/3600,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, environment=environment())
    atomic_json(root/'validation.json', result)
    print(json.dumps(result), flush=True)
    if not passed: raise RuntimeError('Independent serial/batched validation failed; do not launch full grid.')


def run(root, data_root, seed):
    validation = json.loads((root/'validation.json').read_text())
    if not validation['passed']: raise RuntimeError('Validation has not passed.')
    device = torch.device('cuda')
    _, _, (train_x, train_y), (test_x, test_y) = get_data(data_root, device)
    config = dict(seed=seed, official_commit=COMMIT, original_data_sizes=ORIGINAL_NS, added_data_sizes=[200, 8000],
                  all_data_sizes=ALL_NS, alphas=ALPHAS.tolist(), original_grid_points=len(ORIGINAL_NS)*len(ALPHAS),
                  total_grid_points=len(ALL_NS)*len(ALPHAS), steps=10000, batch_size=100, lr=.001,
                  weight_decay=0., dtype='float32', optimizer='torch.optim.AdamW', foreach=False,
                  tf32=False, evaluation_every_steps=50, official_evaluation_steps=list(range(1, 10001, 50)),
                  final_extra_evaluation_step=10000, sampled_evaluation_size_max=2000,
                  source_grid_seed_loop=False, method='21 independent alpha models stacked per N',
                  train_seed_rule='100000+seed*1000000+N', evaluation_seed_rule='200000+seed*1000000+N',
                  checkpoint_selection='minimum sampled training loss; never selected by test score',
                  paper_style_metrics='independent best accuracies/minimum losses over scheduled evaluations; test extrema descriptive only',
                  protocol_changes=['explicit seed and paired initialization/batches across alphas', 'GPU tensor cache',
                                    'batched independent models', 'large evaluation batches with independent sampled sets',
                                    'correct GPU labels', 'raw metrics and checkpoints retained', 'extra final/full-data evaluations'])
    atomic_json(root/'config.json', config)
    atomic_json(root/'environment.json', environment())
    rows = []
    start_all = time.perf_counter()
    for group_idx, n in enumerate(ALL_NS):
        group_file = root/'groups'/('N%06d.pt'%n)
        if group_file.exists():
            saved = torch.load(group_file, map_location='cpu', weights_only=False)
            rows.extend(saved['rows'])
            print(json.dumps(dict(event='resume_skipped_completed_group', n=n)), flush=True)
            continue
        start = time.perf_counter()
        model = StackedMLP(ALPHAS, seed, device)
        target_sq = model.norms_squared().detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False)
        train_gen = torch.Generator().manual_seed(100000+seed*1000000+n)
        eval_gen = torch.Generator().manual_seed(200000+seed*1000000+n)
        cursor, perm = n, None
        k = len(ALPHAS)
        best_train_acc = np.zeros(k)
        best_test_acc = np.zeros(k)
        min_train_loss = np.full(k, np.inf)
        min_test_loss = np.full(k, np.inf)
        selected_train_loss = np.full(k, np.inf)
        selected_step = np.zeros(k, dtype=np.int64)
        selected_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
        selected_metrics = np.full((k, 4), np.nan)
        history = []
        max_norm_error = torch.zeros(k, device=device)
        torch.cuda.reset_peak_memory_stats()
        for step in range(1, 10001):
            if cursor >= n:
                perm = torch.randperm(n, generator=train_gen).to(device)
                cursor = 0
            idx = perm[cursor:cursor+100]
            cursor += 100
            losses = update(model, optimizer, target_sq, train_x[idx], train_y[idx])
            if (step-1)%50 == 0 or step == 10000:
                model.eval()
                # Accuracy and loss use separate draws, with fixed seeded streams.
                tr = sampled_metrics(model, train_x[:n], train_y[:n], eval_gen)
                te = sampled_metrics(model, test_x, test_y, eval_gen)
                metrics = np.stack([tr['accuracy'], te['accuracy'], tr['loss'], te['loss']], axis=1)
                relative = (model.norms_squared()/target_sq).sqrt().sub(1).abs().detach()
                max_norm_error = torch.maximum(max_norm_error, relative)
                if not np.isfinite(metrics).all() or not torch.isfinite(losses).all():
                    atomic_json(root/'failure.json', dict(n=n, step=step, reason='non-finite loss or metrics'))
                    raise RuntimeError('Non-finite values at N=%d, step=%d'%(n,step))
                history.append(dict(step=step, official_schedule=((step-1)%50 == 0), metrics=metrics.tolist(), norm_relative_error=relative.cpu().tolist()))
                if (step-1)%50 == 0:
                    best_train_acc = np.maximum(best_train_acc, tr['accuracy'])
                    best_test_acc = np.maximum(best_test_acc, te['accuracy'])
                    min_train_loss = np.minimum(min_train_loss, tr['loss'])
                    min_test_loss = np.minimum(min_test_loss, te['loss'])
                improved = tr['loss'] < selected_train_loss
                if improved.any():
                    selected_train_loss[improved] = tr['loss'][improved]
                    selected_step[improved] = step
                    selected_metrics[improved] = metrics[improved]
                    mask = torch.as_tensor(improved, device=device)
                    with torch.no_grad():
                        for name, p in model.named_parameters(): selected_weights[name][mask] = p[mask]
                model.train()
            if step%1000 == 0:
                status = dict(state='running', group=group_idx+1, groups_total=len(ALL_NS), n=n, step=step,
                              completed_grid_points=len(rows), total_grid_points=config['total_grid_points'],
                              elapsed_current_group_seconds=time.perf_counter()-start,
                              elapsed_process_seconds=time.perf_counter()-start_all,
                              updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                atomic_json(root/'status.json', status)
                print(json.dumps(status), flush=True)
        model.eval()
        final_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        final_train_acc, final_train_loss = full_metrics(model, train_x[:n], train_y[:n])
        final_test_acc, final_test_loss = full_metrics(model, test_x, test_y)
        model.load_state_dict(selected_weights)
        selected_full_train_acc, selected_full_train_loss = full_metrics(model, train_x[:n], train_y[:n])
        selected_full_test_acc, selected_full_test_loss = full_metrics(model, test_x, test_y)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        norm_errors = max_norm_error.cpu().numpy()
        if norm_errors.max() > 2e-6: raise RuntimeError('Norm constraint tolerance exceeded.')
        group_rows = []
        for a_idx, alpha in enumerate(ALPHAS):
            group_rows.append(dict(seed=seed, n=n, alpha_index=a_idx, alpha=float(alpha), original_grid=(n in ORIGINAL_NS),
                target_norm=float(target_sq[a_idx].sqrt().item()), max_norm_relative_error=float(norm_errors[a_idx]),
                paper_best_train_accuracy=float(best_train_acc[a_idx]), paper_best_test_accuracy=float(best_test_acc[a_idx]),
                paper_min_train_loss=float(min_train_loss[a_idx]), paper_min_test_loss=float(min_test_loss[a_idx]),
                train_selected_step=int(selected_step[a_idx]), selected_sample_train_accuracy=float(selected_metrics[a_idx,0]),
                selected_sample_test_accuracy=float(selected_metrics[a_idx,1]), selected_sample_train_loss=float(selected_metrics[a_idx,2]),
                selected_sample_test_loss=float(selected_metrics[a_idx,3]),
                selected_full_train_accuracy=float(selected_full_train_acc[a_idx]), selected_full_test_accuracy=float(selected_full_test_acc[a_idx]),
                selected_full_train_loss=float(selected_full_train_loss[a_idx]), selected_full_test_loss=float(selected_full_test_loss[a_idx]),
                final_full_train_accuracy=float(final_train_acc[a_idx]), final_full_test_accuracy=float(final_test_acc[a_idx]),
                final_full_train_loss=float(final_train_loss[a_idx]), final_full_test_loss=float(final_test_loss[a_idx]),
                group_seconds=elapsed, group_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        payload = dict(n=n, seed=seed, alphas=ALPHAS.tolist(), rows=group_rows, history=history,
                       final_state_dict=final_state, train_selected_state_dict={key:value.cpu() for key,value in selected_weights.items()},
                       optimizer_state_dict=optimizer.state_dict(), train_generator_state=train_gen.get_state(),
                       evaluation_generator_state=eval_gen.get_state(), target_norm_squared=target_sq.cpu())
        temp = group_file.with_suffix('.tmp')
        torch.save(payload, temp)
        temp.replace(group_file)
        rows.extend(group_rows)
        with (root/'results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps(dict(event='group_complete', n=n, grid_points=len(rows), group_seconds=elapsed,
                              max_norm_relative_error=float(norm_errors.max()))), flush=True)
        del model, optimizer, payload, selected_weights
        torch.cuda.empty_cache()
    atomic_json(root/'status.json', dict(state='complete', completed_grid_points=len(rows), total_grid_points=config['total_grid_points'],
                elapsed_process_seconds=time.perf_counter()-start_all, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    print('ALL_GRID_POINTS_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/'groups').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for this run.')
    if args.validate: validate(args.root, args.data_root)
    else: run(args.root, args.data_root, args.seed)

if __name__ == '__main__': main()