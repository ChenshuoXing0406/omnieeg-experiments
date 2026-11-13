"""Paired public-development experiments. Never evaluates legacy test subjects.

Core: single-task dense, joint dense, fixed experts, content-routed experts.
All run identities include source, plan, data, and pretrained weight hashes.
"""
import argparse
import copy
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from search_models import CBraMod, EEGNet, metrics, seed_all, write

OUT = ROOT / 'outputs/preliminary_v1'
TASKS = ['mi', 'ssvep_2s', 'emotion']
FILES = {'mi': 'BNCI2014_001_T_pilot', 'ssvep_2s': 'ssvep_2s', 'emotion': 'emotion_10s'}
PLAN_PATH = ROOT / 'plans/preliminary_20261112.json'
PLAN = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
SOURCE = Path(__file__).read_bytes()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def log(**items):
    print(json.dumps(items, allow_nan=False), flush=True)


def load_data(task, fold):
    path = ROOT / 'data/processed' / (FILES[task] + '.npz')
    with np.load(path, allow_pickle=False) as f:
        subject = f['subject']
        excluded = PLAN['legacy_test_subjects_excluded'][task]
        keep = ~np.isin(subject, excluded)
        # Drop old test arrays immediately; never expose them to feature fitting.
        data = {k: f[k][keep].copy() for k in ['X', 'y', 'subject', 'trial_id']}
        data['channel_names'] = f['channel_names'].tolist()
    val_subjects = PLAN['validation_subjects_by_fold'][task][fold]
    data['val'] = np.flatnonzero(np.isin(data['subject'], val_subjects))
    data['train'] = np.flatnonzero(~np.isin(data['subject'], val_subjects))
    data['classes'] = len(np.unique(data['y']))
    assert not set(data['subject']) & set(excluded)
    assert not set(data['subject'][data['train']]) & set(data['subject'][data['val']])
    assert set(data['y'][data['train']]) == set(data['y'][data['val']]) == set(range(data['classes']))
    assert np.isfinite(data['X']).all()
    data['source_sha256'] = sha(path)
    data['split'] = {p: sorted(np.unique(data['subject'][data[p]]).tolist()) for p in ['train', 'val']}
    return data


class RoutedLayer(nn.Module):
    def __init__(self, original, learned):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(copy.deepcopy(original.linear1), nn.GELU(),
                          copy.deepcopy(original.dropout), copy.deepcopy(original.linear2))
            for _ in range(4)])
        self.base = original
        del self.base.linear1, self.base.linear2
        self.router = nn.Linear(200, 4)  # also present (frozen) in fixed comparator
        self.learned = learned
        self.router.requires_grad_(learned)
        self.task_index = 0
        self.aux = None
        self.register_buffer('counts', torch.zeros(4), persistent=False)
        self.entropy_sum = 0.
        self.token_count = 0

    def reset_stats(self):
        self.counts = torch.zeros(4, device=self.router.weight.device)
        self.entropy_sum = 0.
        self.token_count = 0

    def forward(self, src, src_mask=None):
        b = self.base
        x = src + b._sa_block(b.norm1(src), src_mask, None)
        h = b.norm2(x)
        flat = h.reshape(-1, h.shape[-1])
        if self.learned:
            p = self.router(flat).softmax(-1)
            selected = p.argmax(-1)
            # Forward top-1 weights equal one (dense-equivalent initialization).
            # Explicit straight-through surrogate supplies classification gradients.
            chosen_p = p.gather(1, selected[:, None])
            gain = 1 + chosen_p - chosen_p.detach()
            fraction = torch.bincount(selected, minlength=4).to(p.dtype) / len(selected)
            self.aux = 4 * (fraction.detach() * p.mean(0)).sum()
            entropy = -(p * p.clamp_min(1e-8).log()).sum(-1).detach().sum()
        else:
            selected = torch.full((len(flat),), self.task_index, device=flat.device, dtype=torch.long)
            gain = torch.ones((len(flat), 1), device=flat.device)
            self.aux = flat.sum() * 0
            entropy = torch.zeros((), device=flat.device)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            index = torch.nonzero(selected == expert_id, as_tuple=True)[0]
            if index.numel():
                out = out.index_copy(0, index, expert(flat[index]) * gain[index])
        if not self.training:
            self.counts += torch.bincount(selected, minlength=4).detach()
            self.entropy_sum += float(entropy)
            self.token_count += len(selected)
        return x + b.dropout2(out.reshape_as(h))


class CoreModel(nn.Module):
    def __init__(self, data, group, seed):
        super().__init__()
        self.backbone = CBraMod()
        self.backbone.load_state_dict(torch.load(ROOT / 'checkpoints/cbramod.pth', map_location='cpu', weights_only=True), strict=True)
        self.backbone.proj_out = nn.Identity()
        if group in ['F', 'M']:
            for i in [-2, -1]:
                self.backbone.encoder.layers[i] = RoutedLayer(self.backbone.encoder.layers[i], group == 'M')
        heads = {}
        for task, d in data.items():
            # Independent deterministic head initialization, unaffected by experts.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + TASKS.index(task) * 1000)
                dim = d['X'].shape[1] * (d['X'].shape[-1] // 200) * 200
                heads[task] = nn.Sequential(nn.Flatten(), nn.LayerNorm(dim), nn.Dropout(.5), nn.Linear(dim, d['classes']))
        self.heads = nn.ModuleDict(heads)

    def routed_layers(self):
        return [m for m in self.backbone.encoder.layers if isinstance(m, RoutedLayer)]

    def forward(self, x, task):
        for layer in self.routed_layers():
            layer.task_index = TASKS.index(task)
        x = x.reshape(x.shape[0], x.shape[1], -1, 200)
        return self.heads[task](self.backbone(x))

    def balance(self):
        layers = self.routed_layers()
        return sum(layer.aux for layer in layers) / len(layers) if layers else 0.


def schedules(data, seed, updates, batch=16):
    result = {}
    for task, d in data.items():
        rng = np.random.default_rng(seed + TASKS.index(task) * 10000)
        count = updates * batch
        chunks = [rng.permutation(d['train']) for _ in range((count + len(d['train']) - 1) // len(d['train']))]
        result[task] = np.concatenate(chunks)[:count].reshape(updates, batch)
    return result


def provenance(data, args, extra):
    return {
        'version': 'preliminary-v1', 'group': args.group, 'fold': args.fold,
        'seed': PLAN['paired_training_seeds_by_fold'][args.fold],
        'updates_per_dataset': args.updates, 'source_sha256': hashlib.sha256(SOURCE).hexdigest(),
        'imported_search_source_sha256': sha(ROOT / 'scripts/search_models.py'),
        'plan_sha256': sha(PLAN_PATH), 'pretrained_sha256': sha(ROOT / 'checkpoints/cbramod.pth'),
        'data': {t: {'sha256': d['source_sha256'], 'split': d['split'], 'channels': d['channel_names']} for t, d in data.items()},
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'test_evaluated': False, 'scope': 'three-fold-public-development', **extra}


@torch.no_grad()
def evaluate(model, data, xs):
    model.eval()
    result, predictions, routing = {}, {}, {}
    for task, d in data.items():
        for layer in model.routed_layers() if isinstance(model, CoreModel) else []:
            layer.reset_stats()
        probs = []
        for start in range(0, len(d['val']), 32):
            index = d['val'][start:start + 32]
            x = torch.from_numpy(xs[task][index]).cuda()
            output = model(x, task) if isinstance(model, CoreModel) else model(x)
            probs.append(output.softmax(-1).cpu().numpy())
        p = np.concatenate(probs)
        result[task] = metrics(d['y'][d['val']], p.argmax(-1), d['subject'][d['val']], d['classes'])
        predictions[task] = p
        if isinstance(model, CoreModel):
            routing[task] = [{'counts': l.counts.cpu().tolist(), 'mean_entropy': l.entropy_sum / max(1, l.token_count)} for l in model.routed_layers()]
    return result, predictions, routing


def save_predictions(folder, data, predictions):
    for task, p in predictions.items():
        d = data[task]; index = d['val']
        np.savez_compressed(folder / f'{task}_validation.npz', y=d['y'][index], scores=p,
                            subject=d['subject'][index], trial_id=d['trial_id'][index])


def run_core(args):
    tasks = [args.task] if args.group == 'S' else TASKS
    data = {t: load_data(t, args.fold) for t in tasks}
    seed = PLAN['paired_training_seeds_by_fold'][args.fold]
    seed_all(seed)
    model = CoreModel(data, args.group, seed).cuda()
    xs = {t: (d['X'] / 100).astype('float32') for t, d in data.items()}
    schedule = schedules(data, seed, args.updates)
    config = provenance(data, args, {
        'task': args.task if args.group == 'S' else 'joint',
        'input_normalization': 'microvolt / 100; no target statistics',
        'routing_gradient': 'unit forward gain; selected-softmax straight-through gradient',
        'balance_coefficient': .01 if args.group == 'M' else 0.,
        'sampling_sha256': {t: hashlib.sha256(a.tobytes()).hexdigest() for t, a in schedule.items()},
        'parameters_total': sum(p.numel() for p in model.parameters()),
        'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    folder = OUT / 'core' / f'fold{args.fold + 1}_{args.group}_{args.task if args.group == "S" else "joint"}'
    if (folder / 'result.json').exists():
        assert json.loads((folder / 'config.json').read_text()) == config, 'Cache identity mismatch'
        log(cached=str(folder)); return
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'config.json', config)
    opt = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': model.heads.parameters(), 'lr': 1e-3}], weight_decay=.01)
    # Same per-dataset learning rate progress in all groups.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.updates * len(tasks))
    weights = {}
    for task, d in data.items():
        counts = np.bincount(d['y'][d['train']], minlength=d['classes'])
        weights[task] = torch.tensor(len(d['train']) / (d['classes'] * counts), dtype=torch.float32, device='cuda')
    seed_all(seed + 999)
    begin = time.perf_counter(); best = -1.; history = []; losses = {t: [] for t in tasks}
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.updates):
        model.train()
        for task in tasks:
            d = data[task]; index = schedule[task][step]
            x = torch.from_numpy(xs[task][index]).cuda()
            y = torch.from_numpy(d['y'][index]).long().cuda()
            opt.zero_grad(set_to_none=True)
            logits = model(x, task)
            loss = nn.functional.cross_entropy(logits, y, weight=weights[task])
            loss = loss + (.01 * model.balance() if args.group == 'M' else 0.)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); sched.step()
            losses[task].append(float(loss.detach()))
        if (step + 1) % 100 == 0 or step + 1 == args.updates:
            val, predictions, routing = evaluate(model, data, xs)
            mean = float(np.mean([m['subject_mean_BA'] for m in val.values()]))
            row = {'dataset_updates': step + 1, 'val_BA': {t: m['subject_mean_BA'] for t, m in val.items()},
                   'mean_val_BA': mean, 'training_loss': {t: float(np.mean(v)) for t, v in losses.items()},
                   'elapsed_seconds': time.perf_counter() - begin}
            history.append(row); write(folder / 'history.json', history)
            if mean > best:
                best = mean
                torch.save(model.state_dict(), folder / 'model.pt')
                save_predictions(folder, data, predictions)
                write(folder / 'best_validation.json', {'dataset_updates': step + 1, 'metrics': val, 'routing': routing})
            losses = {t: [] for t in tasks}
            log(run=folder.name, **row)
    best_record = json.loads((folder / 'best_validation.json').read_text())
    result = {'group': args.group, 'fold': args.fold, 'tasks': tasks, 'seed': seed,
              'mean_val_BA': best, 'best': best_record, 'dataset_updates': args.updates,
              'optimizer_updates': args.updates * len(tasks), 'seconds': time.perf_counter() - begin,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
              'test_evaluated': False, 'config_sha256': sha(folder / 'config.json')}
    write(folder / 'result.json', result); log(finished=folder.name, **result)


def smoke(args):
    seed_all(42)
    data = {t: load_data(t, 0) for t in TASKS}
    dense = CoreModel(data, 'J', 42).cuda().eval()
    variants = {g: CoreModel(data, g, 42).cuda().eval() for g in ['F', 'M']}
    checks = []
    for task, d in data.items():
        x = torch.from_numpy(d['X'][d['train'][:2]] / 100).cuda()
        with torch.no_grad():
            ref = dense(x, task)
            for group, model in variants.items():
                error = float((model(x, task) - ref).abs().max())
                assert error < 2e-5, (group, task, error)
                checks.append({'task': task, 'group': group, 'initial_dense_max_error': error})
    learned = variants['M'].train()
    d = data['mi']; x = torch.from_numpy(d['X'][d['train'][:2]] / 100).cuda()
    loss = learned(x, 'mi').square().mean()  # no balance term: classifier must reach router
    loss.backward()
    norms = [float(l.router.weight.grad.norm()) for l in learned.routed_layers()]
    assert all(np.isfinite(norms)) and min(norms) > 0
    del dense, variants, learned, loss, x
    torch.cuda.empty_cache()
    # Real-input largest-window forward/backward timing, not accuracy measurement.
    m = CoreModel(data, 'J', 42).cuda().train()
    d = data['emotion']; x = torch.from_numpy(d['X'][d['train'][:16]] / 100).cuda()
    y = torch.from_numpy(d['y'][d['train'][:16]]).long().cuda()
    torch.cuda.reset_peak_memory_stats()
    start = None
    for i in range(250):
        if i == 50:
            torch.cuda.synchronize(); start = time.perf_counter()
        m.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(m(x, 'emotion'), y); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
    torch.cuda.synchronize()
    report = {'checks': checks, 'router_classification_gradient_norm': norms,
              'timing_scope': 'real-input emotion B16 forward backward clipping; excludes optimizer and validation',
              'seconds_per_batch': (time.perf_counter() - start) / 200,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
              'test_evaluated': False}
    write(OUT / 'smoke.json', report); log(smoke=report)


class FilterBankNet(nn.Module):
    """Local FBCNet-style variant; 9 bands, spatial filters, segmented log variance."""
    def __init__(self, channels, classes):
        super().__init__()
        self.spatial = nn.Conv2d(9, 9 * 16, (channels, 1), groups=9, bias=False)
        self.norm = nn.BatchNorm2d(9 * 16)
        self.head = nn.Linear(9 * 16 * 4, classes)

    def forward(self, x):
        z = nn.functional.silu(self.norm(self.spatial(x))).squeeze(2)
        z = z.reshape(z.shape[0], z.shape[1], 4, -1)
        z = z.var(-1, unbiased=False).clamp(1e-6, 1e6).log().flatten(1)
        return self.head(z)


class SequenceTCN(nn.Module):
    def __init__(self, features, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(features, 64, 3, padding=1), nn.GELU(), nn.Dropout(.5),
                                 nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.GELU(), nn.Dropout(.5),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, classes))

    def forward(self, x): return self.net(x.transpose(1, 2))


def supplement_features(data, kind):
    cache = OUT / 'feature_cache' / f'{data["source_sha256"][:16]}_{kind}.npy'
    if cache.exists(): return np.load(cache, mmap_mode='r')
    cache.parent.mkdir(parents=True, exist_ok=True)
    x = data['X']
    bands = [(i, i + 4) for i in range(4, 40, 4)] if kind == 'fbcnet' else [(.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    features = np.lib.format.open_memmap(cache.with_suffix('.building.npy'), mode='w+', dtype='float32',
        shape=(len(x), 9, x.shape[1], x.shape[-1]) if kind == 'fbcnet' else (len(x), 10, x.shape[1] * 5))
    for band_index, (lo, hi) in enumerate(bands):
        z = sosfiltfilt(butter(4, [lo, hi], fs=200, btype='bandpass', output='sos'), x, axis=-1)
        if kind == 'fbcnet': features[:, band_index] = z / 100
        else:
            # Filter 10 s continuously within each trial, then compute 1 s variance.
            power = np.log(np.var(z.reshape(len(z), z.shape[1], 10, 200), axis=-1).clip(1e-12))
            features[:, :, band_index::5] = power.transpose(0, 2, 1)
        del z
    features.flush(); del features
    cache.with_suffix('.building.npy').replace(cache)
    return np.load(cache, mmap_mode='r')


def run_supplement(args):
    task = 'mi' if args.group in ['fbcnet', 'eegnet'] else 'emotion'
    data = load_data(task, args.fold); seed = PLAN['paired_training_seeds_by_fold'][args.fold]
    seed_all(seed)
    folder = OUT / 'supplement' / f'fold{args.fold + 1}_{args.group}'
    config = provenance({task: data}, args, {'implementation': 'local variants, not exact published reproduction',
                        'lr': .001 if args.group != 'eegnet' else .003, 'weight_decay': .01})
    if (folder / 'result.json').exists():
        assert json.loads((folder / 'config.json').read_text()) == config, 'Cache identity mismatch'
        log(cached=str(folder)); return
    folder.mkdir(parents=True, exist_ok=True); write(folder / 'config.json', config)
    tr = data['train']; va = data['val']
    if args.group == 'eegnet':
        x = data['X']; x = ((x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True).clip(1e-6)).astype('float32')
        model = EEGNet(x.shape[1], data['classes'], filters=16, kernel=64, dropout=.5)
    elif args.group == 'fbcnet':
        x = supplement_features(data, 'fbcnet'); model = FilterBankNet(data['X'].shape[1], data['classes'])
    else:
        x = supplement_features(data, 'sequence')
        scaler = StandardScaler().fit(x[tr].reshape(-1, x.shape[-1]))
        x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype('float32')
        joblib.dump(scaler, folder / 'scaler.joblib')
        if args.group == 'sequence_logistic':
            begin = time.perf_counter()
            clf = LogisticRegression(C=.1, class_weight='balanced', max_iter=2000)
            with threadpool_limits(limits=4): clf.fit(x[tr].mean(1), data['y'][tr])
            p = clf.predict_proba(x[va].mean(1)); joblib.dump(clf, folder / 'model.joblib')
            save_predictions(folder, {task: data}, {task: p})
            result = {'group': args.group, 'fold': args.fold, 'metrics': metrics(data['y'][va], p.argmax(-1), data['subject'][va], data['classes']),
                      'seconds': time.perf_counter() - begin, 'test_evaluated': False, 'deterministic': True}
            write(folder / 'result.json', result); log(finished=folder.name, **result); return
        model = SequenceTCN(x.shape[-1], data['classes'])
    model = model.cuda(); opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.updates)
    schedule = schedules({task: data}, seed, args.updates)[task]
    np.save(folder / 'training_indices.npy', schedule)
    counts = np.bincount(data['y'][tr], minlength=data['classes'])
    weights = torch.tensor(len(tr) / (data['classes'] * counts), dtype=torch.float32, device='cuda')
    best = -1.; history = []; begin = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for i, index in enumerate(schedule):
        model.train(); opt.zero_grad(set_to_none=True)
        logits = model(torch.from_numpy(x[index]).cuda())
        loss = nn.functional.cross_entropy(logits, torch.from_numpy(data['y'][index]).long().cuda(), weight=weights)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); sched.step()
        if (i + 1) % 100 == 0 or i + 1 == args.updates:
            val, predictions, _ = evaluate(model, {task: data}, {task: x})
            score = val[task]['subject_mean_BA']
            history.append({'updates': i + 1, 'val_BA': score, 'train_loss_last_batch': float(loss.detach())})
            write(folder / 'history.json', history)
            if score > best:
                best = score; torch.save(model.state_dict(), folder / 'model.pt')
                save_predictions(folder, {task: data}, predictions)
                write(folder / 'best_validation.json', {'updates': i + 1, 'metrics': val[task]})
            log(run=folder.name, **history[-1])
    result = {'group': args.group, 'fold': args.fold, 'best': json.loads((folder / 'best_validation.json').read_text()),
              'seconds': time.perf_counter() - begin, 'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
              'test_evaluated': False}
    write(folder / 'result.json', result); log(finished=folder.name, **result)


def queue(args):
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / 'source_snapshot.py'
    if snapshot.exists(): assert snapshot.read_bytes() == SOURCE, 'Do not mutate code during the queue'
    else: snapshot.write_bytes(SOURCE)
    jobs = []
    for fold in range(3):
        jobs += [('core', fold, 'S', t) for t in TASKS]
        jobs += [('core', fold, g, 'joint') for g in ['J', 'F', 'M']]
        jobs += [('supplement', fold, g, 'unused') for g in ['fbcnet', 'eegnet', 'sequence_tcn', 'sequence_logistic']]
    for number, (command, fold, group, task) in enumerate(jobs):
        elapsed_training = sum(json.loads(p.read_text()).get('seconds', 0.) for p in OUT.glob('*/*/result.json'))
        if elapsed_training >= 24 * 3600:
            write(OUT / 'queue_status.json', {'status': 'budget_cap_reached', 'training_seconds': elapsed_training})
            export(); return
        name = f'{number + 1:02d}_fold{fold + 1}_{group}_{task}'
        log(starting=name, total_jobs=len(jobs))
        write(OUT / 'queue_status.json', {'status': 'running', 'current': name, 'job': number + 1, 'total': len(jobs)})
        path = OUT / 'logs' / (name + '.log'); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as handle:
            proc = subprocess.run([sys.executable, '-u', str(Path(__file__)), command, '--fold', str(fold),
                                   '--group', group, '--task', task, '--updates', str(args.updates)],
                                  stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
        if proc.returncode:
            write(OUT / 'queue_status.json', {'status': 'failed', 'current': name, 'returncode': proc.returncode, 'log': str(path)})
            raise RuntimeError(path.read_text(encoding='utf-8')[-6000:])
        log(completed=name)
    export()
    write(OUT / 'queue_status.json', {'status': 'finished', 'jobs': len(jobs), 'test_evaluated': False})


def export():
    rows = []; audit = []
    for path in sorted(OUT.glob('*/*/result.json')):
        folder = path.parent; r = json.loads(path.read_text()); config = json.loads((folder / 'config.json').read_text())
        for pred_path in folder.glob('*_validation.npz'):
            task = pred_path.name.removesuffix('_validation.npz')
            with np.load(pred_path) as p:
                m = metrics(p['y'], p['scores'].argmax(-1), p['subject'], p['scores'].shape[1])
                assert not set(p['subject']) & set(PLAN['legacy_test_subjects_excluded'][task])
                assert np.isfinite(p['scores']).all() and np.allclose(p['scores'].sum(-1), 1., atol=1e-5)
            saved = r['best']['metrics'][task] if folder.parent.name == 'core' else r.get('best', r)['metrics']
            assert abs(saved['subject_mean_BA'] - m['subject_mean_BA']) < 1e-10
            rows.append({'family': folder.parent.name, 'group': r['group'], 'fold': r['fold'] + 1, 'task': task,
                         'val_BA': m['subject_mean_BA'], 'macro_f1': m['macro_f1'], 'seconds': r['seconds'],
                         'peak_allocated_MiB': r.get('peak_allocated_MiB', 0), 'path': str(folder.relative_to(ROOT))})
            audit.append({'file': str(pred_path.relative_to(ROOT)), 'passed': True, 'test_subjects_absent': True})
    write(OUT / 'results.json', rows); write(OUT / 'audit.json', audit)
    if rows:
        with (OUT / 'results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for family, group, task in sorted({(r['family'], r['group'], r['task']) for r in rows}):
        values = [r['val_BA'] for r in rows if (r['family'], r['group'], r['task']) == (family, group, task)]
        summary.append({'family': family, 'group': group, 'task': task, 'folds': len(values), 'mean_BA': float(np.mean(values)),
                        'sample_std': float(np.std(values, ddof=1)) if len(values) > 1 else None})
    write(OUT / 'summary.json', summary); log(summary=summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['smoke', 'core', 'supplement', 'queue', 'export'])
    parser.add_argument('--group', default='J'); parser.add_argument('--task', default='joint')
    parser.add_argument('--fold', type=int, default=0, choices=range(3))
    parser.add_argument('--updates', type=int, default=1500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'smoke': smoke(args)
    elif args.command == 'core': run_core(args)
    elif args.command == 'supplement': run_supplement(args)
    elif args.command == 'queue': queue(args)
    else: export()
