"""Matched freezing and initialization controls for preliminary_v1 S models.

Pretrained full fine-tuning is reused after exact identity checks. Frozen
backbones retain train-mode dropout so the intervention is parameter updating,
not simultaneously a change in dropout. Evaluation always uses eval mode.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_preliminary import (ROOT, OUT as BASE, PLAN, TASKS, CoreModel, CBraMod,
                             load_data, schedules, evaluate, save_predictions, sha, seed_all, write, log)

OUT = ROOT / 'outputs/initialization_controls_v1'
SOURCE = Path(__file__).read_bytes()


def tensor_sha(state):
    digest = hashlib.sha256()
    for k, v in sorted(state.items()):
        digest.update(k.encode()); digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ControlModel(CoreModel):
    def __init__(self, data, seed, mode):
        nn.Module.__init__(self)
        self.backbone = CBraMod()
        if mode != 'random_full':
            self.backbone.load_state_dict(torch.load(ROOT / 'checkpoints/cbramod.pth', map_location='cpu', weights_only=True), strict=True)
        self.backbone.proj_out = nn.Identity()
        heads = {}
        for task, d in data.items():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + TASKS.index(task) * 1000)
                dim = d['X'].shape[1] * d['X'].shape[-1]
                heads[task] = nn.Sequential(nn.Flatten(), nn.LayerNorm(dim), nn.Dropout(.5), nn.Linear(dim, d['classes']))
        self.heads = nn.ModuleDict(heads)
        if mode == 'pretrained_frozen': self.backbone.requires_grad_(False)


def assert_baseline(data, task, fold, updates, schedule):
    folder = BASE / 'core' / f'fold{fold + 1}_S_{task}'
    config = json.loads((folder / 'config.json').read_text())
    result = json.loads((folder / 'result.json').read_text())
    assert config['source_sha256'] == sha(ROOT / 'scripts/run_preliminary.py')
    assert config['imported_search_source_sha256'] == sha(ROOT / 'scripts/search_models.py')
    assert config['plan_sha256'] == sha(ROOT / 'plans/preliminary_20261112.json')
    assert config['pretrained_sha256'] == sha(ROOT / 'checkpoints/cbramod.pth')
    assert config['updates_per_dataset'] == updates == result['dataset_updates']
    assert config['seed'] == PLAN['paired_training_seeds_by_fold'][fold]
    assert config['data'][task]['sha256'] == data['source_sha256']
    assert config['data'][task]['split'] == data['split']
    assert config['sampling_sha256'][task] == hashlib.sha256(schedule.tobytes()).hexdigest()
    return folder, result


def preflight():
    seed_all(42)
    d = load_data('mi', 0); data = {'mi': d}
    ref = CoreModel(data, 'S', 42)
    frozen = ControlModel(data, 42, 'pretrained_frozen')
    random = ControlModel(data, 42, 'random_full')
    assert set(ref.state_dict()) == set(frozen.state_dict()) == set(random.state_dict())
    assert tensor_sha(ref.state_dict()) == tensor_sha(frozen.state_dict())
    assert tensor_sha(ref.heads.state_dict()) == tensor_sha(random.heads.state_dict())
    assert tensor_sha(ref.backbone.state_dict()) != tensor_sha(random.backbone.state_dict())
    before_backbone = tensor_sha(frozen.backbone.state_dict())
    before_head = tensor_sha(frozen.heads.state_dict())
    frozen.train()
    optimizer = torch.optim.AdamW(frozen.heads.parameters(), lr=.001)
    x = torch.from_numpy(d['X'][d['train'][:2]] / 100)
    y = torch.from_numpy(d['y'][d['train'][:2]]).long()
    loss = nn.functional.cross_entropy(frozen(x, 'mi'), y)
    loss.backward(); optimizer.step()
    assert tensor_sha(frozen.backbone.state_dict()) == before_backbone
    assert tensor_sha(frozen.heads.state_dict()) != before_head
    assert all(p.grad is None for p in frozen.backbone.parameters())
    schedule = schedules(data, 42, 1500)['mi']
    assert_baseline(d, 'mi', 0, 1500, schedule)
    report = {'matched_state_keys_and_shapes': True, 'pretrained_initial_state_equal': True,
              'head_initialization_equal': True, 'random_backbone_differs': True,
              'frozen_backbone_unchanged_after_optimizer': True, 'frozen_head_updated': True,
              'baseline_schedule_and_provenance_match': True, 'test_evaluated': False,
              'frozen_training_mode': 'train with dropout, identical to full fine-tuning'}
    write(OUT / 'preflight.json', report); log(preflight=report)


def run(args):
    d = load_data(args.task, args.fold); data = {args.task: d}
    seed = PLAN['paired_training_seeds_by_fold'][args.fold]
    schedule = schedules(data, seed, args.updates)[args.task]
    baseline, _ = assert_baseline(d, args.task, args.fold, args.updates, schedule)
    seed_all(seed)
    model = ControlModel(data, seed, args.mode).cuda()
    initial_backbone = tensor_sha(model.backbone.state_dict())
    initial_head = tensor_sha(model.heads.state_dict())
    config = {'version': 'initialization-controls-v1', 'mode': args.mode, 'task': args.task,
              'fold': args.fold, 'seed': seed, 'updates': args.updates, 'batch': 16,
              'source_sha256': hashlib.sha256(SOURCE).hexdigest(),
              'base_source_sha256': sha(ROOT / 'scripts/run_preliminary.py'),
              'search_source_sha256': sha(ROOT / 'scripts/search_models.py'),
              'plan_sha256': sha(ROOT / 'plans/preliminary_20261112.json'),
              'pretrained_source_sha256': sha(ROOT / 'checkpoints/cbramod.pth'),
              'pretrained_loaded': args.mode != 'random_full', 'data_sha256': d['source_sha256'], 'split': d['split'],
              'baseline_directory': str(baseline.relative_to(ROOT)), 'baseline_config_sha256': sha(baseline / 'config.json'),
              'sampling_sha256': hashlib.sha256(schedule.tobytes()).hexdigest(),
              'initial_backbone_sha256': initial_backbone, 'initial_head_sha256': initial_head,
              'backbone_lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': .01,
              'frozen_backbone_training_mode': 'train (dropout retained); eval for validation',
              'selection': 'validation equal-subject mean balanced accuracy', 'test_evaluated': False}
    folder = OUT / 'runs' / f'fold{args.fold + 1}_{args.mode}_{args.task}'
    if (folder / 'result.json').exists():
        assert config == json.loads((folder / 'config.json').read_text()), 'Cache mismatch'
        log(cached=folder.name); return
    folder.mkdir(parents=True, exist_ok=True); write(folder / 'config.json', config)
    x = (d['X'] / 100).astype('float32')
    opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 1e-4},
                            {'params': model.heads.parameters(), 'lr': 1e-3}], weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.updates)
    counts = np.bincount(d['y'][d['train']], minlength=d['classes'])
    weights = torch.tensor(len(d['train']) / (d['classes'] * counts), dtype=torch.float32, device='cuda')
    np.save(folder / 'training_indices.npy', schedule)
    seed_all(seed + 999)
    begin = time.perf_counter(); best = -1; history = []; losses = []
    torch.cuda.reset_peak_memory_stats()
    for step, index in enumerate(schedule):
        model.train()  # freeze only gradients, keep all dropout settings matched
        opt.zero_grad(set_to_none=True)
        logits = model(torch.from_numpy(x[index]).cuda(), args.task)
        loss = nn.functional.cross_entropy(logits, torch.from_numpy(d['y'][index]).long().cuda(), weight=weights)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step(); sched.step(); losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0 or step + 1 == args.updates:
            val, pred, _ = evaluate(model, data, {args.task: x})
            score = val[args.task]['subject_mean_BA']
            row = {'updates': step + 1, 'val_BA': score, 'train_loss': float(np.mean(losses)), 'seconds': time.perf_counter() - begin}
            losses = []; history.append(row); write(folder / 'history.json', history)
            if score > best:
                best = score; torch.save(model.state_dict(), folder / 'model.pt')
                save_predictions(folder, data, pred)
                write(folder / 'best_validation.json', {'updates': step + 1, 'metrics': val[args.task]})
            log(run=folder.name, **row)
    final_backbone = tensor_sha(model.backbone.state_dict())
    if args.mode == 'pretrained_frozen': assert final_backbone == initial_backbone
    else: assert final_backbone != initial_backbone
    assert tensor_sha(model.heads.state_dict()) != initial_head
    result = {'mode': args.mode, 'task': args.task, 'fold': args.fold, 'seed': seed, 'updates': args.updates,
              'best': json.loads((folder / 'best_validation.json').read_text()), 'seconds': time.perf_counter() - begin,
              'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
              'final_backbone_sha256': final_backbone, 'backbone_changed': final_backbone != initial_backbone,
              'test_evaluated': False}
    write(folder / 'result.json', result); log(finished=folder.name, **result)


def export():
    from search_models import metrics
    rows = []; audit = []
    for fold in range(3):
        for task in TASKS:
            folders = [('pretrained_full', BASE / 'core' / f'fold{fold + 1}_S_{task}')]
            folders += [(mode, OUT / 'runs' / f'fold{fold + 1}_{mode}_{task}') for mode in ['pretrained_frozen', 'random_full']]
            for mode, folder in folders:
                if not (folder / 'result.json').exists(): continue
                r = json.loads((folder / 'result.json').read_text())
                with np.load(folder / f'{task}_validation.npz') as p:
                    m = metrics(p['y'], p['scores'].argmax(-1), p['subject'], p['scores'].shape[1])
                    assert not set(p['subject']) & set(PLAN['legacy_test_subjects_excluded'][task])
                    assert np.allclose(p['scores'].sum(-1), 1., atol=1e-5)
                saved = r['best']['metrics'][task] if mode == 'pretrained_full' else r['best']['metrics']
                assert abs(saved['subject_mean_BA'] - m['subject_mean_BA']) < 1e-10
                rows.append({'mode': mode, 'task': task, 'fold': fold + 1, 'BA': m['subject_mean_BA'], 'path': str(folder.relative_to(ROOT))})
                audit.append({'run': folder.name, 'predictions_recomputed': True, 'legacy_test_subjects_absent': True})
    write(OUT / 'results.json', rows); write(OUT / 'audit.json', audit)
    summary = []
    lines = ['# Matched freezing and initialization controls', '', 'Development validation only; three folds with paired training seeds, not official scores.', '',
             '| Task | Mode | Folds | Mean BA % | Fold sample SD, pp |', '|---|---|---:|---:|---:|']
    for task in TASKS:
        for mode in ['pretrained_full', 'pretrained_frozen', 'random_full']:
            values = [r['BA'] for r in rows if r['task'] == task and r['mode'] == mode]
            if not values: continue
            item = {'task': task, 'mode': mode, 'folds': len(values), 'mean_BA': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else None}
            summary.append(item); sd = '-' if item['std'] is None else f"{item['std']*100:.2f}"
            lines.append(f"| {task} | {mode} | {len(values)} | {item['mean_BA']*100:.2f} | {sd} |")
    lines += ['', 'Same flatten, LayerNorm/dropout/Linear head, head initialization, microvolt/100 input, sampling order, updates, optimizer settings and validation cadence.',
              'Frozen backbone retains training-mode dropout to isolate parameter updates; no frozen-eval feature cache.',
              'Pretrained full fine-tuning is reused only after source, split, sample-order, budget and checkpoint hash checks.',
              'Random initialization is trained under the same downstream budget; this does not test large-scale from-scratch pretraining.']
    write(OUT / 'summary.json', summary)
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    log(summary=summary)


def queue(args):
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUT / 'source_snapshot.py'
    if snapshot.exists(): assert snapshot.read_bytes() == SOURCE
    else: snapshot.write_bytes(SOURCE)
    while True:
        status = json.loads((BASE / 'queue_status.json').read_text())
        if status['status'] == 'finished': break
        if status['status'] != 'running': raise RuntimeError(f'Previous queue requires repair: {status}')
        write(OUT / 'queue_status.json', {'status': 'waiting_for_previous_queue', 'previous': status})
        time.sleep(5)
    jobs = [(fold, mode, task) for fold in range(3) for task in TASKS for mode in ['pretrained_frozen', 'random_full']]
    for i, (fold, mode, task) in enumerate(jobs):
        name = f'{i + 1:02d}_fold{fold + 1}_{mode}_{task}'
        write(OUT / 'queue_status.json', {'status': 'running', 'current': name, 'job': i + 1, 'total': len(jobs)})
        log(starting=name)
        path = OUT / 'logs' / (name + '.log'); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            p = subprocess.run([sys.executable, '-u', str(Path(__file__)), 'run', '--mode', mode, '--task', task,
                                '--fold', str(fold), '--updates', str(args.updates)], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        if p.returncode:
            write(OUT / 'queue_status.json', {'status': 'failed', 'current': name, 'log': str(path)})
            raise RuntimeError(path.read_text(encoding='utf-8')[-6000:])
        log(completed=name); export()
    write(OUT / 'queue_status.json', {'status': 'finished', 'jobs': len(jobs), 'test_evaluated': False})
    # Recompute old scores and generate final paired tables after all jobs finish.
    subprocess.run([sys.executable, str(ROOT / 'scripts/report_preliminary.py'), '--checkpoints'], cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['preflight', 'run', 'queue', 'export'])
    parser.add_argument('--mode', choices=['pretrained_frozen', 'random_full'], default='pretrained_frozen')
    parser.add_argument('--task', choices=TASKS, default='mi'); parser.add_argument('--fold', type=int, choices=range(3), default=0)
    parser.add_argument('--updates', type=int, default=1500)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'preflight': preflight()
    elif args.command == 'run': run(args)
    elif args.command == 'queue': queue(args)
    else: export()
