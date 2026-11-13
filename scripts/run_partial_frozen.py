"""Fixed pretrained shared prefix, task-private last4layers; same P4 experiment recipe."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from run_partial_sharing import PartialModel,evaluate,tensor_sha,ROOT,OUT as MAIN_OUT
from run_preliminary import load_data,schedules,save_predictions,sha,PLAN,TASKS
from search_models import seed_all,write as atomic_write

OUT=ROOT/'outputs/partial_sharing_frozen_v1'

def write(path,value):
    # Windows readers can briefly hold the destination open during replace.
    for attempt in range(20):
        try:
            atomic_write(path,value)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.1)

def read(p):return json.loads(Path(p).read_text())

def run(fold):
    data={t:load_data(t,fold) for t in TASKS};seed=42+fold;seed_all(seed)
    model=PartialModel(data,seed,4).cuda();model.backbone.requires_grad_(False)
    xs={t:(d['X']/100).astype('float32') for t,d in data.items()};schedule=schedules(data,seed,1500)
    baseline=read(MAIN_OUT/f'fold{fold+1}_task_private_P4/config.json')
    assert tensor_sha(model.heads.state_dict())==baseline['initial_heads_sha256']
    assert {t:hashlib.sha256(a.tobytes()).hexdigest() for t,a in schedule.items()}==baseline['sampling_sha256']
    folder=OUT/f'fold{fold+1}_frozen_P4';folder.mkdir(parents=True,exist_ok=True)
    cfg={'version':'partial-sharing-frozen-v1.1','fold':fold,'seed':seed,'depth':4,'frozen_prefix':True,
        'source_sha256':sha(__file__),'model_source_sha256':sha(ROOT/'scripts/run_partial_sharing.py'),
        'plan_sha256':sha(ROOT/'plans/partial_sharing_frozen_20261113.json'),'counts':model.counts(),
        'pretrained_sha256':sha(ROOT/'checkpoints/cbramod.pth'),'initial_shared_sha256':tensor_sha(model.backbone.state_dict()),
        'initial_heads_sha256':tensor_sha(model.heads.state_dict()),'paired_P4_config_sha256':sha(MAIN_OUT/f'fold{fold+1}_task_private_P4/config.json'),
        'data':{t:{'sha256':d['source_sha256'],'split':d['split']} for t,d in data.items()},
        'bone_lr':1e-4,'head_lr':.001,'wd':.01,'batch':16,'updates_per_task':1500,'optimizer':'AdamW','scheduler':'cosine4500',
        'dropout':'train mode retained in frozen shared prefix','selection':'primary mean3task checkpoint; secondary independent task-tail/head selection with unchanged shared prefix',
        'input':'microvolt /100; no target statistics','test_evaluated':False}
    if (folder/'result.json').exists():assert read(folder/'config.json')==cfg;return
    write(folder/'config.json',cfg);np.savez_compressed(folder/'training_indices.npz',**schedule)
    shared_initial=tensor_sha(model.backbone.state_dict())
    opt=torch.optim.AdamW([{'params':list(model.backbone.parameters())+list(model.private.parameters()),'lr':1e-4},
        {'params':model.heads.parameters(),'lr':.001}],weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4500)
    weights={t:torch.tensor(len(d['train'])/(d['classes']*np.bincount(d['y'][d['train']],minlength=d['classes'])),dtype=torch.float32,device='cuda') for t,d in data.items()}
    seed_all(seed+999);best=-1.;best_task={t:-1. for t in TASKS};best_task_records={}
    taskwise=folder/'taskwise';taskwise.mkdir(exist_ok=True)
    history=[];losses={t:[] for t in TASKS};start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1500):
        model.train()
        for task in TASKS:
            d=data[task];index=schedule[task][step];opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(torch.from_numpy(xs[task][index]).cuda(),task),torch.from_numpy(d['y'][index]).long().cuda(),weight=weights[task])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if step==0:assert all(p.grad is None for p in model.backbone.parameters())
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step();losses[task].append(float(loss.detach()))
        if (step+1)%100==0:
            val,pred=evaluate(model,data,xs);mean=float(np.mean([m['subject_mean_BA'] for m in val.values()]))
            row={'dataset_updates':step+1,'val_BA':{t:m['subject_mean_BA'] for t,m in val.items()},'mean_val_BA':mean,
                'training_loss':{t:float(np.mean(v)) for t,v in losses.items()},'elapsed_seconds':time.perf_counter()-start}
            history.append(row);write(folder/'history.json',history)
            if mean>best:
                best=mean;torch.save(model.state_dict(),folder/'model.pt');save_predictions(folder,data,pred)
                write(folder/'best_validation.json',{'dataset_updates':step+1,'metrics':val})
            for task in TASKS:
                value=val[task]['subject_mean_BA']
                if value>best_task[task]:
                    best_task[task]=value
                    torch.save({'private':model.private[task].state_dict(),'head':model.heads[task].state_dict()},folder/f'{task}_taskwise.pt')
                    save_predictions(taskwise,{task:data[task]},{task:pred[task]})
                    best_task_records[task]={'dataset_updates':step+1,'metrics':val[task]}
            write(folder/'taskwise_selection.json',best_task_records)
            losses={t:[] for t in TASKS};print(json.dumps({'run':folder.name,**row}),flush=True)
    peak=torch.cuda.max_memory_allocated()/2**20
    assert tensor_sha(model.backbone.state_dict())==shared_initial
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda',weights_only=True),strict=True)
    assert tensor_sha(model.backbone.state_dict())==shared_initial
    _,pred=evaluate(model,data,xs);errors={t:float(np.max(np.abs(pred[t]-np.load(folder/f'{t}_validation.npz')['scores']))) for t in TASKS}
    assert max(errors.values())<1e-5
    # Each task branch is independent once the shared prefix is frozen; composition is exact.
    for task in TASKS:
        chosen=torch.load(folder/f'{task}_taskwise.pt',map_location='cuda',weights_only=True)
        model.private[task].load_state_dict(chosen['private'],strict=True);model.heads[task].load_state_dict(chosen['head'],strict=True)
    assert tensor_sha(model.backbone.state_dict())==shared_initial
    task_metrics,task_pred=evaluate(model,data,xs)
    task_errors={t:float(np.max(np.abs(task_pred[t]-np.load(taskwise/f'{t}_validation.npz')['scores']))) for t in TASKS}
    assert max(task_errors.values())<1e-5
    torch.save(model.state_dict(),folder/'model_taskwise.pt')
    model.load_state_dict(torch.load(folder/'model_taskwise.pt',map_location='cuda',weights_only=True),strict=True)
    _,replay=evaluate(model,data,xs)
    assert max(float(np.max(np.abs(replay[t]-task_pred[t]))) for t in TASKS)<1e-5
    write(folder/'result.json',{'fold':fold,'best':read(folder/'best_validation.json'),'mean_val_BA':best,
        'seconds':time.perf_counter()-start,'peak_allocated_MiB':peak,'reload_max_abs':errors,
        'taskwise_best':best_task_records,'taskwise_composed_metrics':task_metrics,'taskwise_reload_max_abs':task_errors,
        'frozen_shared_unchanged':True,'test_evaluated':False})
    del model,opt,xs,data;torch.cuda.empty_cache()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    write(OUT/'queue_status.json',{'status':'waiting_for_sharing_boundary_gpu_stage'})
    while read(MAIN_OUT/'queue_status.json')['status']!='finished':
        if read(MAIN_OUT/'queue_status.json')['status']=='failed':raise RuntimeError('Partial sharing predecessor failed')
        if time.monotonic()-begin>7200:raise RuntimeError('Predecessor timed out')
        time.sleep(5)
    for fold in range(3):
        write(OUT/'queue_status.json',{'status':'running','fold':fold+1});run(fold)
    write(OUT/'queue_status.json',{'status':'finished','new_fits':3,'prediction_task_records':9,
        'checkpoint_task_audits':9,'frozen_shared_hash_audits':3,'test_evaluated':False})
    print(json.dumps({'finished':True,'new_fits':3}),flush=True)

if __name__=='__main__':main()
