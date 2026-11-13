"""Teammate diagram: spatial/temporal/whole-window spectral residual adapters.

All branches are dense: this is NOT a sparse-compute MoE. Compare uniform
fusion, learned sample gates, and learned fusion without the spectrum branch.
Uses paired preliminary development folds; never evaluates legacy test data.
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

from run_preliminary import (ROOT, OUT as BASE, PLAN, TASKS, CoreModel, load_data,
                            schedules, evaluate, save_predictions, seed_all, write, sha, log)
from run_initialization_controls import assert_baseline, tensor_sha

OUT = ROOT / 'outputs/residual_adapters_v1'
SOURCE = Path(__file__).read_bytes()
MODES = ['uniform', 'learned', 'no_spectrum']


class SpatialAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(200)
        self.down = nn.Linear(200, 32)
        self.attention = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        self.up = nn.Linear(32, 200)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, h):
        b, c, p, _ = h.shape
        z = nn.functional.gelu(self.down(self.norm(h)))
        z = z.transpose(1, 2).reshape(b * p, c, 32)
        z = self.attention(z, z, z, need_weights=False)[0]
        return self.up(z.reshape(b, p, c, 32).transpose(1, 2))


class TemporalAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(200); self.down = nn.Linear(200, 32)
        self.conv = nn.Conv1d(32, 32, 3, padding=1, groups=32)
        self.up = nn.Linear(32, 200)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, h):
        b, c, p, _ = h.shape
        z = nn.functional.gelu(self.down(self.norm(h))).reshape(b * c, p, 32).transpose(1, 2)
        z = nn.functional.gelu(self.conv(z)).transpose(1, 2).reshape(b, c, p, 32)
        return self.up(z)


class SpectrumAdapter(nn.Module):
    def __init__(self, samples):
        super().__init__()
        freq = torch.fft.rfftfreq(samples, 1 / 200)
        mask = (freq >= .5) & (freq <= 90)
        self.register_buffer('frequency_mask', mask)
        features = int(mask.sum()) * 3
        self.spec_norm = nn.LayerNorm(features)
        self.spec_down = nn.Linear(features, 32)
        self.h_norm = nn.LayerNorm(200); self.h_down = nn.Linear(200, 32)
        self.up = nn.Linear(32, 200)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.calls = 0

    def forward(self, raw, h):
        self.calls += 1
        z = torch.fft.rfft(raw - raw.mean(-1, keepdim=True), dim=-1, norm='forward')[..., self.frequency_mask]
        desc = torch.cat([z.real, z.imag, z.abs().clamp_min(1e-8).log()], dim=-1)
        signal = self.spec_down(self.spec_norm(desc))[:, :, None, :]
        z = nn.functional.gelu(signal + self.h_down(self.h_norm(h)))
        return self.up(z)


class ResidualModel(CoreModel):
    def __init__(self, data, seed, mode):
        super().__init__(data, 'S', seed)
        assert len(data) == 1
        samples = next(iter(data.values()))['X'].shape[-1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 50000)
            self.spatial = SpatialAdapter(); self.temporal = TemporalAdapter()
            self.spectral = SpectrumAdapter(samples)
            self.gate = nn.Sequential(nn.LayerNorm(200), nn.Linear(200, 32), nn.GELU(), nn.Linear(32, 3))
            nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
        self.mode = mode
        if mode == 'uniform': self.gate.requires_grad_(False)
        if mode == 'no_spectrum': self.spectral.requires_grad_(False)
        self.register_buffer('gate_sum', torch.zeros(3), persistent=False)
        self.gate_samples = 0; self.gate_entropy = 0.

    def reset_gate_stats(self):
        self.gate_sum.zero_(); self.gate_samples = 0; self.gate_entropy = 0.

    def forward(self, x, task):
        h = self.backbone(x.reshape(x.shape[0], x.shape[1], -1, 200))
        if self.mode == 'uniform':
            weights = h.new_full((len(h), 3), 1 / 3)
        else:
            logits = self.gate(h.mean((1, 2)))
            if self.mode == 'no_spectrum': logits = logits.masked_fill(torch.tensor([False, False, True], device=h.device), -torch.inf)
            weights = logits.softmax(-1)
        delta = weights[:, 0, None, None, None] * self.spatial(h)
        delta = delta + weights[:, 1, None, None, None] * self.temporal(h)
        if self.mode != 'no_spectrum':
            delta = delta + weights[:, 2, None, None, None] * self.spectral(x, h)
        if not self.training:
            self.gate_sum += weights.detach().sum(0)
            self.gate_samples += len(h)
            self.gate_entropy += float(-(weights * weights.clamp_min(1e-8).log()).sum().detach())
        return self.heads[task](h + delta)


def preflight():
    checks = []
    for task in TASKS:
        seed_all(42); d = load_data(task, 0); data = {task: d}
        baseline = CoreModel(data, 'S', 42).cuda().eval()
        x = torch.from_numpy(d['X'][d['train'][:2]] / 100).cuda()
        with torch.no_grad(): ref = baseline(x, task)
        for mode in MODES:
            model = ResidualModel(data, 42, mode).cuda().eval()
            assert tensor_sha(model.heads.state_dict()) == tensor_sha(baseline.heads.state_dict())
            assert tensor_sha(model.backbone.state_dict()) == tensor_sha(baseline.backbone.state_dict())
            with torch.no_grad(): out = model(x, task)
            error = float((out - ref).abs().max()); assert error < 2e-5
            w = model.gate_sum / model.gate_samples
            assert torch.isfinite(w).all() and torch.allclose(w.sum(), torch.ones((), device='cuda'))
            if mode == 'no_spectrum': assert model.spectral.calls == 0 and float(w[2]) == 0
            checks.append({'task': task, 'mode': mode, 'initial_output_max_difference': error, 'gate_weights': w.cpu().tolist()})
            del model
        del baseline, x, data, d
        torch.cuda.empty_cache()
    seed_all(42); d = load_data('mi', 0); model = ResidualModel({'mi': d}, 42, 'learned').cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        x = torch.from_numpy(d['X'][d['train'][:2]] / 100).cuda()
        y = torch.from_numpy(d['y'][d['train'][:2]]).long().cuda()
        loss = nn.functional.cross_entropy(model(x, 'mi'), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    gate_grad = float(model.gate[-1].weight.grad.norm()); assert gate_grad > 0
    expert_grads = {n:float(sum(p.grad.square().sum() for p in m.parameters() if p.grad is not None).sqrt())
                    for n,m in [('spatial',model.spatial),('temporal',model.temporal),('spectral',model.spectral)]}
    assert min(expert_grads.values()) > 0
    write(OUT/'preflight.json', {'checks':checks,'gate_classification_gradient_norm_after_warmup':gate_grad,
                               'expert_gradient_norms':expert_grads,'all_passed':True,'test_evaluated':False})
    log(preflight='passed', gate_gradient=gate_grad, expert_gradients=expert_grads)


def train(args):
    seed = PLAN['paired_training_seeds_by_fold'][args.fold]
    d = load_data(args.task, args.fold); data = {args.task: d}
    schedule = schedules(data, seed, args.updates)[args.task]
    baseline, _ = assert_baseline(d, args.task, args.fold, args.updates, schedule)
    seed_all(seed); model = ResidualModel(data, seed, args.mode).cuda()
    config = {'version':'residual-adapters-v1','mode':args.mode,'task':args.task,'fold':args.fold,'seed':seed,
              'source_sha256':hashlib.sha256(SOURCE).hexdigest(),'base_source_sha256':sha(ROOT/'scripts/run_preliminary.py'),
              'controls_source_sha256':sha(ROOT/'scripts/run_initialization_controls.py'),
              'search_source_sha256':sha(ROOT/'scripts/search_models.py'),'data_sha256':d['source_sha256'],'split':d['split'],
              'pretrained_sha256':sha(ROOT/'checkpoints/cbramod.pth'),'plan_sha256':sha(ROOT/'plans/preliminary_20261112.json'),
              'baseline_directory':baseline.relative_to(ROOT).as_posix(),'sampling_sha256':hashlib.sha256(schedule.tobytes()).hexdigest(),
              'initial_head_sha256':tensor_sha(model.heads.state_dict()),'initial_backbone_sha256':tensor_sha(model.backbone.state_dict()),
              'updates':args.updates,'batch':16,'backbone_lr':1e-4,'head_lr':.001,'adapter_lr':.001,'weight_decay':.01,
              'readout':'flatten','head':'LayerNorm+Dropout(0.5)+Linear','normalization':'microvolt/100',
              'adapter_width':32,'spatial_attention_heads':4,'temporal_kernel':3,'spectral_band':[.5,90],
              'spectral_description':'whole-window real+imag+log-amplitude; natural FFT bins; no time stretching',
              'spectral_bins':int(model.spectral.frequency_mask.sum()),'gate':'sample-level softmax of mean-pooled H',
              'fusion':'H + sum(weight * zero-initialized residual branch)',
              'compute':'dense branches; no sparse-compute claim','loss':'train-fold class-weighted CE; no gate balance loss',
              'parameters_total':sum(p.numel() for p in model.parameters()),
              'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'test_evaluated':False}
    folder = OUT/'runs'/f'fold{args.fold+1}_{args.mode}_{args.task}'
    if (folder/'result.json').exists():
        assert read(folder/'config.json') == config, 'Cache identity mismatch'
        log(cached=folder.name); return
    folder.mkdir(parents=True,exist_ok=True);write(folder/'config.json',config)
    x = (d['X']/100).astype('float32')
    adapter_params=[p for n,p in model.named_parameters() if not n.startswith(('backbone.','heads.'))]
    opt = torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-4},
                            {'params':model.heads.parameters(),'lr':.001},{'params':adapter_params,'lr':.001}],weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.updates)
    counts=np.bincount(d['y'][d['train']],minlength=d['classes'])
    cw=torch.tensor(len(d['train'])/(d['classes']*counts),device='cuda',dtype=torch.float32)
    np.save(folder/'training_indices.npy',schedule)
    seed_all(seed+999);begin=time.perf_counter();best=-1;history=[];losses=[]
    torch.cuda.reset_peak_memory_stats()
    for step,index in enumerate(schedule):
        model.train();opt.zero_grad(set_to_none=True)
        logits=model(torch.from_numpy(x[index]).cuda(),args.task)
        loss=nn.functional.cross_entropy(logits,torch.from_numpy(d['y'][index]).long().cuda(),weight=cw)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();sched.step();losses.append(float(loss.detach()))
        if (step+1)%100==0 or step+1==args.updates:
            model.reset_gate_stats();metrics,pred,_=evaluate(model,data,{args.task:x})
            score=metrics[args.task]['subject_mean_BA']
            gate={'mean_weights':(model.gate_sum/model.gate_samples).cpu().tolist(),
                  'mean_entropy':model.gate_entropy/model.gate_samples}
            row={'updates':step+1,'val_BA':score,'train_loss':float(np.mean(losses)),
                 'seconds':time.perf_counter()-begin,'gate':gate}
            history.append(row);losses=[];write(folder/'history.json',history)
            if score>best:
                best=score;torch.save(model.state_dict(),folder/'model.pt');save_predictions(folder,data,pred)
                write(folder/'best_validation.json',{'updates':step+1,'metrics':metrics[args.task],'gate':gate})
            log(run=folder.name,**row)
    result={'mode':args.mode,'task':args.task,'fold':args.fold,'seed':seed,'updates':args.updates,
            'best':read(folder/'best_validation.json'),'seconds':time.perf_counter()-begin,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'test_evaluated':False}
    write(folder/'result.json',result);log(finished=folder.name,val_BA=best,seconds=result['seconds'])


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def export(checkpoints=False):
    from search_models import metrics
    rows=[];audit=[];reload=[]
    for fold in range(3):
        for task in TASKS:
            base=BASE/'core'/f'fold{fold+1}_S_{task}'
            if (base/'result.json').exists():
                b=read(base/'result.json')['best']['metrics'][task]
                rows.append({'mode':'baseline','task':task,'fold':fold+1,'BA':b['subject_mean_BA'],'path':base.relative_to(ROOT).as_posix()})
            for mode in MODES:
                folder=OUT/'runs'/f'fold{fold+1}_{mode}_{task}'
                if not (folder/'result.json').exists():continue
                r=read(folder/'result.json');c=read(folder/'config.json')
                with np.load(folder/f'{task}_validation.npz') as p:
                    m=metrics(p['y'],p['scores'].argmax(-1),p['subject'],p['scores'].shape[1])
                    assert np.isfinite(p['scores']).all() and np.allclose(p['scores'].sum(-1),1,atol=1e-5)
                    assert not set(p['subject'])&set(PLAN['legacy_test_subjects_excluded'][task])
                    assert abs(m['subject_mean_BA']-r['best']['metrics']['subject_mean_BA'])<1e-10
                    if checkpoints:
                        d=load_data(task,fold);seed_all(c['seed']);model=ResidualModel({task:d},c['seed'],mode)
                        assert tensor_sha(model.heads.state_dict())==c['initial_head_sha256']
                        reference=CoreModel({task:d},'S',c['seed'])
                        assert tensor_sha(reference.heads.state_dict())==c['initial_head_sha256'];del reference
                        model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cpu'),strict=True)
                        model=model.cuda().eval();idx=d['val'][:16]
                        with torch.no_grad():scores=model(torch.from_numpy((d['X'][idx]/100).astype('float32')).cuda(),task).softmax(-1).cpu().numpy()
                        assert np.allclose(scores,p['scores'][:16],rtol=5e-4,atol=1e-4)
                        reload.append({'run':folder.name,'max_abs_difference':float(abs(scores-p['scores'][:16]).max()),'passed':True})
                        del model,d;torch.cuda.empty_cache()
                rows.append({'mode':mode,'task':task,'fold':fold+1,'BA':m['subject_mean_BA'],'path':folder.relative_to(ROOT).as_posix()})
                audit.append({'run':folder.name,'predictions_recomputed':True,'test_subjects_excluded':True})
    write(OUT/'results.json',rows);write(OUT/'audit.json',audit)
    if checkpoints:write(OUT/'checkpoint_audit.json',reload)
    summary=[]
    lines=['# 异构残差适配实验','', '三折开发验证，旧测试被试排除；相同读出、头初值、样本序列和每数据集1500步预算。',
           '空间、时间、整窗频谱分支为密集计算；门控是融合权重，不是第四个专家。','',
           '| 任务 | 条件 | 折数 | 平均验证BA | 折间样本标准差 |','|---|---|---:|---:|---:|']
    for task in TASKS:
        for mode in ['baseline']+MODES:
            values=[r['BA'] for r in rows if r['task']==task and r['mode']==mode]
            if not values:continue
            s={'task':task,'mode':mode,'folds':len(values),'mean_BA':float(np.mean(values)),
               'std':float(np.std(values,ddof=1)) if len(values)>1 else None};summary.append(s)
            std='—' if s['std'] is None else f"{s['std']*100:.2f}个百分点"
            lines.append(f"| {task} | {mode} | {len(values)} | {s['mean_BA']*100:.2f}% | {std} |")
    differences=[]
    for task in TASKS:
        for first,second in [('uniform','baseline'),('learned','baseline'),('learned','uniform'),('learned','no_spectrum')]:
            lookup={(r['mode'],r['fold']):r['BA'] for r in rows if r['task']==task}
            d=[100*(lookup[first,f]-lookup[second,f]) for f in range(1,4) if (first,f) in lookup and (second,f) in lookup]
            if d:differences.append({'task':task,'contrast':first+' minus '+second,'folds':len(d),'fold_deltas_pp':d,'mean_delta_pp':float(np.mean(d))})
    write(OUT/'summary.json',summary);write(OUT/'paired_differences.json',differences)
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');log(summary=summary)


def queue(args):
    assert read(BASE/'queue_status.json')['status']=='finished'
    snapshot=OUT/'source_snapshot.py'
    if snapshot.exists():assert snapshot.read_bytes()==SOURCE
    else:snapshot.write_bytes(SOURCE)
    for i,(fold,task,mode) in enumerate((f,t,m) for f in range(3) for t in TASKS for m in MODES):
        name=f'{i+1:02d}_fold{fold+1}_{mode}_{task}'
        write(OUT/'queue_status.json',{'status':'running','current':name,'job':i+1,'total':27});log(starting=name)
        path=OUT/'logs'/(name+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8') as f:
            p=subprocess.run([sys.executable,'-u',str(Path(__file__)),'train','--fold',str(fold),'--task',task,'--mode',mode,
                              '--updates',str(args.updates)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:
            write(OUT/'queue_status.json',{'status':'failed','current':name,'log':str(path)})
            raise RuntimeError(path.read_text(encoding='utf-8')[-5000:])
        log(completed=name);export()
    export(checkpoints=True)
    write(OUT/'queue_status.json',{'status':'finished','jobs':27,'test_evaluated':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','train','queue','export','audit'])
    p.add_argument('--mode',choices=MODES,default='learned');p.add_argument('--task',choices=TASKS,default='mi')
    p.add_argument('--fold',type=int,choices=range(3),default=0);p.add_argument('--updates',type=int,default=1500)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.command=='preflight':preflight()
    elif args.command=='train':train(args)
    elif args.command=='queue':queue(args)
    else:export(checkpoints=args.command=='audit')
