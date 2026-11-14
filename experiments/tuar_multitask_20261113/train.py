"""TUAR pilot, teammate-aligned sequential task updates and classification heads."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix
from model import ExperimentModel,TASKS,MODES

ROOT=Path(__file__).resolve().parent
REFERENCE='ca188b799dd3281b3db8e783a361d9518353fe8b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(p)

class Data:
    def __init__(self,task,part):
        d=ROOT/'processed'/task
        self.x=np.load(d/f'{part}_x.npy',mmap_mode='r');self.y=np.load(d/f'{part}_y.npy')
        self.subject=np.load(d/f'{part}_subject.npy');self.ids=np.load(d/f'{part}_id.npy')
        self.artifact=np.load(d/f'{part}_artifact_mask.npy') if task=='tuar' else None
        assert self.x.shape[1:]==TASKS[task][:2]+(200,) and len(self.x)==len(self.y)==len(self.subject)==len(self.ids)
        assert set(np.unique(self.y))==set(range(TASKS[task][2])) and len(set(self.ids))==len(self.ids)
    def batch(self,index):
        return torch.from_numpy(np.array(self.x[index],copy=True)).cuda(),torch.from_numpy(self.y[index]).long().cuda()

def metrics(y,p,subject,classes):
    pred=p.argmax(-1);cm=confusion_matrix(y,pred,labels=np.arange(classes));recall=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(classes),where=cm.sum(1)>0)
    per={}
    for sub in np.unique(subject):
        m=subject==sub
        present=np.unique(y[m]);per[str(sub)]=float(np.mean([(pred[m][y[m]==k]==k).mean() for k in present]))
    result={'subject_mean_BA':float(np.mean(list(per.values()))),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'per_subject_BA':per,'class_recall':recall.tolist(),
            'confusion_matrix':cm.tolist(),'subjects_missing_a_class':int(sum(len(np.unique(y[subject==s]))<classes for s in np.unique(subject)))}
    if classes==2:result['auroc']=float(roc_auc_score(y,p[:,1]))
    return result

@torch.no_grad()
def evaluate(model,data):
    model.eval();all_metrics={};predictions={}
    for task,d in data.items():
        ps=[];gs=[]
        for a in range(0,len(d.y),32):
            x,y=d.batch(slice(a,a+32));out,g=model(x,task)
            ps.append(out.float().softmax(-1).cpu().numpy())
            if g is not None:gs.append(g.cpu().numpy())
        p=np.concatenate(ps);r=metrics(d.y,p,d.subject,TASKS[task][2]);predictions[task]=p
        if gs:
            g=np.concatenate(gs);r['gate_mean']=g.mean(0).tolist();r['gate_entropy']=float(-(g*np.log(g+1e-9)).sum(-1).mean())
        if d.artifact is not None:
            r['positive_recall_by_artifact_bit']={str(bit):float((p[(d.y==1)&((d.artifact&(1<<bit))>0)].argmax(-1)==1).mean()) for bit in range(8) if np.any((d.y==1)&((d.artifact&(1<<bit))>0))}
        all_metrics[task]=r
    model.train();return all_metrics,predictions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--mode',choices=MODES,default='plain')
    ap.add_argument('--tasks',nargs='+',choices=list(TASKS),required=True);ap.add_argument('--steps',type=int,default=1500)
    ap.add_argument('--batch',type=int,default=16);ap.add_argument('--eval-every',type=int,default=100);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args();dest=ROOT/'results'/a.name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():print('ALREADY_COMPLETE',a.name,flush=True);return
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.30)
    train={t:Data(t,'train') for t in a.tasks};val={t:Data(t,'val') for t in a.tasks}
    for t in a.tasks:assert not set(train[t].subject)&set(val[t].subject)
    model=ExperimentModel(ROOT/'weights/pretrained_weights.pth',a.mode,a.seed).cuda()
    for t,head in model.heads.items():head.requires_grad_(t in a.tasks)
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':1e-4},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad],'lr':1e-3}],weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps*len(a.tasks))
    weights={};schedules={}
    for t,d in train.items():
        counts=np.bincount(d.y,minlength=TASKS[t][2]);weights[t]=torch.tensor(len(d.y)/(TASKS[t][2]*counts),dtype=torch.float32,device='cuda')
        rng=np.random.default_rng(a.seed+list(TASKS).index(t)*10000);total=a.steps*a.batch
        schedules[t]=np.concatenate([rng.permutation(len(d.y)) for _ in range((total+len(d.y)-1)//len(d.y))])[:total].reshape(a.steps,a.batch)
    config=vars(a)|{'reference_commit':REFERENCE,'reference_training':'run_preliminary.py: sequential per-task updates; AdamW1e-4/1e-3 wd.01; cosine; class-weighted CE; head Dropout.5; FP32',
       'implementation':'our common-feature residual expert comparisons; not teammate last-2-FFN sparse MoE',
       'parameters':model.parameter_report(a.tasks),'sampling_sha256':{t:hashlib.sha256(v.tobytes()).hexdigest() for t,v in schedules.items()},
       'data':{t:json.loads((ROOT/'processed'/t/'manifest.json').read_text()) for t in a.tasks},
       'weights_sha256':sha(ROOT/'weights/pretrained_weights.pth'),'sources':{p.name:sha(p) for p in [ROOT/'model.py',ROOT/'train.py']},
       'gpu':torch.cuda.get_device_name(),'gpu_uuid':os.getenv('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'shared_gpu':True,
       'test_evaluated':False,'selection':'mean of per-task subject_mean_BA; one common checkpoint; single-class subject BA uses present-class recall'}
    start=0;best=-1
    torch.manual_seed(a.seed+999);torch.cuda.manual_seed_all(a.seed+999)
    if (dest/'latest.pt').exists():
        assert json.loads((dest/'config.json').read_text())==config,'Resume config mismatch'
        ck=torch.load(dest/'latest.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        start=ck['step'];best=ck['best'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
    write(dest/'config.json',config);print('START',a.name,config['parameters'],flush=True)
    elapsed=time.monotonic();model.train()
    for step in range(start,a.steps):
        losses={}
        for t,d in train.items():
            x,y=d.batch(schedules[t][step]);opt.zero_grad(set_to_none=True)
            logits,g=model(x,t);loss=F.cross_entropy(logits,y,weight=weights[t])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step();losses[t]=float(loss.detach())
        n=step+1
        if n==1 or n%10==0:
            status={'step':n,'optimizer_updates':n*len(a.tasks),'loss':losses,'elapsed_s':time.monotonic()-elapsed,'peak_reserved_MB':torch.cuda.max_memory_reserved()/2**20}
            write(dest/'status.json',status);print('PROGRESS',json.dumps(status),flush=True)
        if n%a.eval_every==0 or n==a.steps:
            results,preds=evaluate(model,val);score=float(np.mean([r['subject_mean_BA'] for r in results.values()]))
            record={'step':n,'validation':results,'selection_score':score,'test_evaluated':False}
            with (dest/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if score>best:
                best=score;torch.save({'model':model.state_dict(),'step':n},dest/'best.tmp');(dest/'best.tmp').replace(dest/'best.pt');write(dest/'best_metrics.json',record)
                for t,p in preds.items():np.savez_compressed(dest/f'{t}_validation.npz',y=val[t].y,scores=p,subject=val[t].subject,window_id=val[t].ids)
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'step':n,'best':best,
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},dest/'latest.tmp');(dest/'latest.tmp').replace(dest/'latest.pt')
            print('VALIDATION',json.dumps({'step':n,'subject_BA':{t:r['subject_mean_BA'] for t,r in results.items()},'score':score}),flush=True)
    model.load_state_dict(torch.load(dest/'best.pt',map_location='cuda',weights_only=True)['model'])
    check,predictions=evaluate(model,val);errors={}
    for t,p in predictions.items():
        with np.load(dest/f'{t}_validation.npz') as saved:
            errors[t]=float(np.max(np.abs(p-saved['scores'])));assert np.allclose(p,saved['scores'],rtol=1e-5,atol=1e-6)
    write(dest/'complete.json',{'steps':a.steps,'best_validation_subject_BA':best,'checkpoint_reload_max_probability_error':errors,'test_evaluated':False})
    print('COMPLETE',a.name,flush=True)

if __name__=='__main__':main()
