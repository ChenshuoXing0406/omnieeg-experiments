"""Paired sharing-boundary experiments; preserves original CBraMod S/J protocol."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from run_preliminary import CoreModel,load_data,schedules,save_predictions,sha,PLAN,TASKS
from search_models import seed_all,metrics,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/partial_sharing_v1'
PLAN_PATH=ROOT/'plans/partial_sharing_20261113.json'

def log(**k):print(json.dumps(k,allow_nan=False),flush=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def tensor_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class PartialModel(CoreModel):
    def __init__(self,data,seed,private_layers,mode='task_private'):
        super().__init__(data,'J',seed)
        assert len(data)==3 and private_layers in [2,4] and self.backbone.encoder.norm is None
        original=list(self.backbone.encoder.layers)
        self.private=nn.ModuleDict({t:nn.ModuleList([copy.deepcopy(layer) for layer in original[-private_layers:]]) for t in TASKS})
        self.backbone.encoder.layers=nn.ModuleList(original[:-private_layers])
        self.backbone.encoder.num_layers=len(original)-private_layers
        self.depth=private_layers;self.mode=mode;self.outer_step=0
    def chosen_branch(self,task):
        i=TASKS.index(task)
        return TASKS[(i+self.outer_step)%len(TASKS)] if self.mode=='rotate_assignment' and self.training else task
    def forward(self,x,task):
        h=self.backbone(x.reshape(x.shape[0],x.shape[1],-1,200))
        for layer in self.private[self.chosen_branch(task)]:h=layer(h)
        return self.heads[task](h)
    def counts(self):
        shared=sum(p.numel() for p in self.backbone.parameters())
        return {'total':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
            'shared_backbone':shared,'private_per_task':{t:sum(p.numel() for p in self.private[t].parameters()) for t in TASKS},
            'active_per_task':{t:shared+sum(p.numel() for p in self.private[t].parameters())+sum(p.numel() for p in self.heads[t].parameters()) for t in TASKS}}

@torch.no_grad()
def evaluate(model,data,xs,limit=None):
    model.eval();results={};predictions={}
    for task,d in data.items():
        index=d['val'] if limit is None else d['val'][:limit];p=[]
        for start in range(0,len(index),32):p.append(model(torch.from_numpy(xs[task][index[start:start+32]]).cuda(),task).softmax(-1).cpu().numpy())
        p=np.concatenate(p);predictions[task]=p;results[task]=metrics(d['y'][index],p.argmax(-1),d['subject'][index],d['classes'])
    return results,predictions

def smoke():
    data={t:load_data(t,0) for t in TASKS};seed_all(42);base=CoreModel(data,'J',42).cuda().eval()
    heads=tensor_sha(base.heads.state_dict());samples={t:torch.from_numpy(d['X'][d['train'][:2]]/100).cuda() for t,d in data.items()}
    with torch.no_grad():reference={t:base(samples[t],t) for t in TASKS}
    reference_counts={t:sum(p.numel() for p in base.backbone.parameters())+sum(p.numel() for p in base.heads[t].parameters()) for t in TASKS}
    checks=[]
    for depth in [2,4]:
        seed_all(42);m=PartialModel(data,42,depth).cuda().eval();assert tensor_sha(m.heads.state_dict())==heads
        with torch.no_grad():
            errors={t:float((m(samples[t],t)-reference[t]).abs().max()) for t in TASKS}
        assert max(errors.values())<1e-5 and m.counts()['active_per_task']==reference_counts
        assert len({p.data_ptr() for task in TASKS for p in m.private[task].parameters()})==sum(len(list(m.private[t].parameters())) for t in TASKS)
        m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01)
        before={t:tensor_sha(m.private[t].state_dict()) for t in TASKS}
        opt.zero_grad(set_to_none=True);nn.functional.cross_entropy(m(samples['mi'],'mi'),torch.from_numpy(data['mi']['y'][data['mi']['train'][:2]]).long().cuda()).backward()
        assert any(p.grad is not None and p.grad.norm()>0 for p in m.backbone.parameters())
        assert any(p.grad is not None and p.grad.norm()>0 for p in m.private['mi'].parameters())
        assert all(p.grad is None for t in ['ssvep_2s','emotion'] for p in m.private[t].parameters())
        opt.step();after={t:tensor_sha(m.private[t].state_dict()) for t in TASKS}
        assert before['mi']!=after['mi'] and all(before[t]==after[t] for t in ['ssvep_2s','emotion'])
        checks.append({'depth':depth,'initial_logit_max_abs':errors,'head_initialization_matched':True,
            'active_parameter_count_matches_J':True,'selected_branch_gradient':True,'unselected_branches_unchanged_after_AdamW':True,
            'counts':m.counts()});del m,opt;torch.cuda.empty_cache()
    del base,samples;torch.cuda.empty_cache()
    # The rotation control visits every task/branch pairing equally over1500outer updates.
    coverage=np.zeros((3,3),dtype=int)
    for step in range(1500):
        for i in range(3):coverage[i,(i+step)%3]+=1
    assert np.array_equal(coverage,np.full((3,3),500))
    write(OUT/'smoke.json',{'passed':checks,'rotation_exposures':coverage.tolist(),'test_evaluated':False});log(smoke='passed')

def run(fold,depth,mode):
    data={t:load_data(t,fold) for t in TASKS};seed=PLAN['paired_training_seeds_by_fold'][fold]
    seed_all(seed);model=PartialModel(data,seed,depth,mode).cuda()
    xs={t:(d['X']/100).astype('float32') for t,d in data.items()};schedule=schedules(data,seed,1500)
    base=ROOT/f'outputs/preliminary_v1/core/fold{fold+1}_J_joint';old=read(base/'config.json')
    sampling={t:hashlib.sha256(a.tobytes()).hexdigest() for t,a in schedule.items()}
    assert sampling==old['sampling_sha256']
    assert sha(ROOT/'scripts/run_preliminary.py')==old['source_sha256'] and sha(ROOT/'scripts/search_models.py')==old['imported_search_source_sha256']
    assert all(data[t]['split']==old['data'][t]['split'] and data[t]['source_sha256']==old['data'][t]['sha256'] for t in TASKS)
    cfg={'version':'partial-sharing-v1','fold':fold,'depth':depth,'mode':mode,'seed':seed,'updates_per_task':1500,
        'source_sha256':sha(__file__),'plan_sha256':sha(PLAN_PATH),'pretrained_sha256':sha(ROOT/'checkpoints/cbramod.pth'),
        'baseline_config_sha256':sha(base/'config.json'),'sampling_sha256':sampling,'counts':model.counts(),
        'initial_heads_sha256':tensor_sha(model.heads.state_dict()),'data':{t:{'sha256':d['source_sha256'],'split':d['split']} for t,d in data.items()},
        'bone_lr':1e-4,'head_lr':.001,'wd':.01,'batch':16,'input':'microvolt /100; no target collection statistics',
        'selection':'unweighted mean of all3 validation subject-mean BAs','optimizer':'AdamW','scheduler':'cosine over4500updates',
        'update_pattern':'mi -> ssvep_2s -> emotion for each outer step','torch':torch.__version__,'test_evaluated':False}
    folder=OUT/f'fold{fold+1}_{mode}_P{depth}'
    if (folder/'result.json').exists():assert read(folder/'config.json')==cfg;log(cached=folder.name);return
    folder.mkdir(parents=True,exist_ok=True);write(folder/'config.json',cfg)
    np.savez_compressed(folder/'training_indices.npz',**schedule)
    body=list(model.backbone.parameters())+list(model.private.parameters())
    opt=torch.optim.AdamW([{'params':body,'lr':1e-4},{'params':model.heads.parameters(),'lr':.001}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4500)
    weights={t:torch.tensor(len(d['train'])/(d['classes']*np.bincount(d['y'][d['train']],minlength=d['classes'])),dtype=torch.float32,device='cuda') for t,d in data.items()}
    seed_all(seed+999);start=time.perf_counter();best=-1.;history=[];losses={t:[] for t in TASKS}
    torch.cuda.reset_peak_memory_stats()
    for step in range(1500):
        model.train();model.outer_step=step
        for task in TASKS:
            d=data[task];index=schedule[task][step];opt.zero_grad(set_to_none=True)
            logits=model(torch.from_numpy(xs[task][index]).cuda(),task)
            loss=nn.functional.cross_entropy(logits,torch.from_numpy(d['y'][index]).long().cuda(),weight=weights[task])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step()
            losses[task].append(float(loss.detach()))
        if (step+1)%100==0:
            val,pred=evaluate(model,data,xs);mean=float(np.mean([v['subject_mean_BA'] for v in val.values()]))
            row={'dataset_updates':step+1,'val_BA':{t:v['subject_mean_BA'] for t,v in val.items()},'mean_val_BA':mean,
                 'training_loss':{t:float(np.mean(v)) for t,v in losses.items()},'elapsed_seconds':time.perf_counter()-start}
            history.append(row);write(folder/'history.json',history)
            if mean>best:
                best=mean;torch.save(model.state_dict(),folder/'model.pt');save_predictions(folder,data,pred)
                write(folder/'best_validation.json',{'dataset_updates':step+1,'metrics':val})
            losses={t:[] for t in TASKS};log(run=folder.name,**row)
    peak=torch.cuda.max_memory_allocated()/2**20
    model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cuda'),strict=True)
    # Reload all validation scores with the same batch shape, verifying the complete output.
    _,pred=evaluate(model,data,xs);errors={}
    for task in TASKS:
        saved=np.load(folder/f'{task}_validation.npz')['scores'];errors[task]=float(np.max(np.abs(saved-pred[task])))
        assert errors[task]<1e-5
    result={'fold':fold,'depth':depth,'mode':mode,'seed':seed,'best':read(folder/'best_validation.json'),
        'mean_val_BA':best,'optimizer_updates':4500,'updates_per_task':1500,'seconds':time.perf_counter()-start,
        'peak_allocated_MiB':peak,'reload_max_abs':errors,'test_evaluated':False}
    write(folder/'result.json',result);log(finished=folder.name,mean_val_BA=best,seconds=result['seconds'])

def export():
    rows=[];audits=[]
    for fold in range(3):
        for group in ['S','J']:
            for task in TASKS:
                folder=ROOT/f'outputs/preliminary_v1/core/fold{fold+1}_{group}_{task if group=="S" else "joint"}'
                r=read(folder/'result.json');m=r['best']['metrics'][task]
                rows.append({'group':group,'depth':0,'fold':fold+1,'task':task,'BA':m['subject_mean_BA'],
                    'path':str(folder.relative_to(ROOT)),'reused':True})
        for depth in [2,4]:
            for mode in ['task_private','rotate_assignment']:
                folder=OUT/f'fold{fold+1}_{mode}_P{depth}'
                if not (folder/'result.json').exists():continue
                r=read(folder/'result.json')
                for task in TASKS:
                    with np.load(folder/f'{task}_validation.npz') as p:
                        assert not set(p['subject'])&set(PLAN['legacy_test_subjects_excluded'][task])
                        assert np.isfinite(p['scores']).all() and np.allclose(p['scores'].sum(-1),1,atol=1e-5)
                        m=metrics(p['y'],p['scores'].argmax(-1),p['subject'],p['scores'].shape[-1])
                        assert abs(m['subject_mean_BA']-r['best']['metrics'][task]['subject_mean_BA'])<1e-10
                    rows.append({'group':mode,'depth':depth,'fold':fold+1,'task':task,'BA':m['subject_mean_BA'],
                        'path':str(folder.relative_to(ROOT)),'reused':False})
                    audits.append({'path':str(folder.relative_to(ROOT)),'task':task,'prediction_recomputed':True,'reload_error':r['reload_max_abs'][task]})
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for group,depth,task in sorted({(r['group'],r['depth'],r['task']) for r in rows}):
        a=[r['BA'] for r in rows if (r['group'],r['depth'],r['task'])==(group,depth,task)]
        summary.append({'group':group,'depth':depth,'task':task,'folds':len(a),'mean_BA':float(np.mean(a))})
    write(OUT/'summary.json',summary);write(OUT/'prediction_reload_audit.json',audits)
    return rows,summary

def queue():
    OUT.mkdir(parents=True,exist_ok=True);snapshot=OUT/'source_snapshot.py'
    if snapshot.exists():assert snapshot.read_bytes()==Path(__file__).read_bytes()
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    smoke()
    jobs=[(fold,depth,'task_private') for depth in [2,4] for fold in range(3)]
    def execute(fold,depth,mode):
        write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'depth':depth,'mode':mode})
        logs=OUT/'logs';logs.mkdir(exist_ok=True);path=logs/f'fold{fold+1}_{mode}_P{depth}.log'
        with path.open('w',encoding='utf-8') as f:
            process=subprocess.run([sys.executable,__file__,'run','--fold',str(fold),'--depth',str(depth),'--mode',mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        if process.returncode:
            write(OUT/'queue_status.json',{'status':'failed','fold':fold+1,'depth':depth,'mode':mode,'log':str(path)})
            raise RuntimeError(path.read_text(encoding='utf-8')[-4000:])
        export();log(completed=path.stem)
    for fold,depth,mode in jobs:execute(fold,depth,mode)
    rows,_=export();gate=[]
    for depth in [2,4]:
        delta=[]
        for fold in range(1,4):
            delta.append([next(r['BA'] for r in rows if (r['group'],r['depth'],r['fold'],r['task'])==('task_private',depth,fold,t))-
                next(r['BA'] for r in rows if (r['group'],r['fold'],r['task'])==('J',fold,t)) for t in TASKS])
        a=np.asarray(delta);eligible=bool(a.mean()>=.01 and (a.mean(1)>0).sum()>=2 and a.mean(0).min()>=-.02)
        gate.append({'depth':depth,'eligible':eligible,'mean_gain':float(a.mean()),'task_mean_gains':a.mean(0).tolist(),
                     'positive_folds':int((a.mean(1)>0).sum())})
    write(OUT/'capacity_control_decision.json',gate)
    eligible=[r for r in gate if r['eligible']]
    extra=0
    if eligible:
        chosen=max(eligible,key=lambda r:(r['mean_gain'],-r['depth']))['depth']
        for fold in range(3):execute(fold,chosen,'rotate_assignment')
        extra=3
    export();write(OUT/'queue_status.json',{'status':'finished','new_fits':6+extra,'reused_S_J_fits':12,
        'capacity_control_executed':bool(extra),'prediction_audits':3*(6+extra),'checkpoint_task_audits':3*(6+extra),'test_evaluated':False})
    log(finished=True,new_fits=6+extra)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['queue','run','export','smoke']);p.add_argument('--fold',type=int,default=0)
    p.add_argument('--depth',type=int,default=2);p.add_argument('--mode',default='task_private');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.command=='queue':queue()
    elif args.command=='run':run(args.fold,args.depth,args.mode)
    elif args.command=='smoke':smoke()
    else:export()
