"""Normalization-only CBraMod extension; archived scale controls remain immutable."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def tensor_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def metrics(y, p, subjects):
    pred = p.argmax(-1)
    per = {str(s):float(balanced_accuracy_score(y[subjects==s],pred[subjects==s])) for s in np.unique(subjects)}
    return {'subject_mean_BA':float(np.mean(list(per.values()))), 'pooled_BA':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro')), 'per_subject_BA':per,
        'confusion_matrix':confusion_matrix(y,pred,labels=range(4)).tolist()}


def main(args):
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    sys.path.insert(0,str(args.lab_source)); sys.path.insert(0,str(args.cbra_source))
    from hustbciml.algorithms.models.EEGConformer import EEGConformer
    from models.cbramod import CBraMod
    assert sha(args.weights)=='0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178'
    meta=json.loads((args.data/'metadata.json').read_text())
    source_hashes={str(p):sha(p) for root in [args.cbra_source/'models',args.lab_source/'hustbciml/core',args.lab_source/'hustbciml/utils'] for p in root.glob('*.py')}
    lab_module=args.lab_source/'hustbciml/algorithms/models/EEGConformer.py'
    source_hashes[str(lab_module)]=sha(lab_module)

    class ConformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=EEGConformer(n_chans=64,n_times=800,n_classes=4,sfreq=200)
            self.head=nn.Linear(self.backbone.out_features,4)
        def forward(self,x):
            return self.head(self.backbone(x[:,None]))

    class CBraClassifier(nn.Module):
        def __init__(self,pretrained,seed):
            super().__init__(); self.backbone=CBraMod()
            if pretrained:
                self.backbone.load_state_dict(torch.load(args.weights,map_location='cpu',weights_only=True),strict=True)
            self.backbone.proj_out=nn.Identity()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+10000)
                self.head=nn.Sequential(nn.Flatten(),nn.LayerNorm(64*800),nn.Dropout(.5),nn.Linear(64*800,4))
        def forward(self,x):
            return self.head(self.backbone(x.reshape(len(x),64,4,200)))

    def load(ids):
        xs,ys,ss,ts=[],[],[],[]
        for sid in ids:
            p=args.data/f'S{sid:03d}.npz'; assert sha(p)==meta['processed_sha256'][p.name]
            with np.load(p) as d:
                xs.append(d['X']); ys.append(d['y']); ss.extend([sid]*len(d['X'])); ts.extend(d['trial_id'])
        return np.concatenate(xs),np.concatenate(ys),np.array(ss),np.array(ts)

    def normalize(x,policy):
        if policy=='scale':return x/100.
        return (x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)

    @torch.no_grad()
    def predict(model,x,policy):
        model.eval()
        return np.concatenate([model(torch.from_numpy(normalize(x[i:i+16],policy)).to(args.device)).softmax(-1).cpu().numpy() for i in range(0,len(x),16)])

    output=args.output; output.mkdir(parents=True,exist_ok=True)
    write(output/'environment.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
        'script_sha256':sha(__file__),'source_hashes':source_hashes,'weights_sha256':sha(args.weights)})
    results=[]
    try:
        for fold in range(args.folds):
            split=meta['splits'][fold];seed=split['seed']
            assert not set(split['train'])&set(split['val'])
            assert not set(split['train']+split['val'])&set(split['test'])
            x,y,s,t=load(split['train']); vx,vy,vs,vt=load(split['val'])
            schedule=np.load(args.baseline/f'fold{fold+1}_eegnet/training_indices.npy')[:args.updates]
            assert schedule.shape==(args.updates,16)
            assert schedule.min()>=0 and schedule.max()<len(y)
            baseline_cfg=json.loads((args.baseline/f'fold{fold+1}_eegnet/config.json').read_text())
            assert baseline_cfg['split']==split
            heads={};initial_backbones={}
            for name,policy in [('cbra_pretrained','trial'),('cbra_random','trial')]:
                folder=output/f'fold{fold+1}_{name}'
                assert not (folder/'config.json').exists(), 'Existing run: select a fresh output directory'
                seed_all(seed)
                model=ConformerClassifier() if name=='lab_conformer' else CBraClassifier(name=='cbra_pretrained',seed)
                initial_head=tensor_sha(model.head.state_dict());initial_backbone=tensor_sha(model.backbone.state_dict())
                heads[name]=initial_head;initial_backbones[name]=initial_backbone
                if name=='cbra_random':
                    assert heads['cbra_pretrained']==initial_head
                    assert initial_backbones['cbra_pretrained']!=initial_backbone
                cfg={'name':name,'split':split,'seed':seed,'input':[64,800],'normalization':policy,'batch':16,
                    'updates':args.updates,'wd':.01,'optimizer':'AdamW','scheduler':'cosine','validation_interval':100,
                    'backbone_lr':.001 if name=='lab_conformer' else .0001,'head_lr':.001,
                    'sampling_sha256':hashlib.sha256(schedule.tobytes()).hexdigest(),'script_sha256':sha(__file__),
                    'data_metadata_sha256':sha(args.data/'metadata.json'),'initial_head_sha256':initial_head,
                    'initial_backbone_sha256':initial_backbone,'pretrained_loaded':name=='cbra_pretrained',
                    'weights_sha256':sha(args.weights) if name=='cbra_pretrained' else None,
                    'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'pretraining_scope':'Author reports TUEG pretraining; no independent sample-level pretraining audit',
                    'test_evaluated':False}
                control_folder=args.controls/f'fold{fold+1}_{name}'
                control=json.loads((control_folder/'config.json').read_text())
                assert control['normalization']=='scale'
                for k in ['split','seed','input','batch','wd','optimizer','scheduler','validation_interval',
                          'backbone_lr','head_lr','data_metadata_sha256',
                          'initial_head_sha256','initial_backbone_sha256','pretrained_loaded',
                          'weights_sha256','trainable_parameters']:
                    assert cfg[k]==control[k],(k,cfg[k],control[k])
                assert args.updates<=control['updates']
                assert np.array_equal(schedule,np.load(control_folder/'training_indices.npy')[:args.updates])
                if args.updates==control['updates']:
                    assert np.array_equal(schedule,np.load(control_folder/'training_indices.npy'))
                cfg['control_directory']=str(control_folder)
                cfg['control_config_sha256']=sha(control_folder/'config.json')
                cfg['normalization_only_control_verified']=True
                write(folder/'config.json',cfg);np.save(folder/'training_indices.npy',schedule)
                model.to(args.device)
                opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':cfg['backbone_lr']},
                                      {'params':model.head.parameters(),'lr':.001}],weight_decay=.01)
                sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.updates)
                seed_all(seed+999)
                if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
                best=-1;history=[];start=time.perf_counter()
                for step,idx in enumerate(schedule,1):
                    model.train();opt.zero_grad(set_to_none=True)
                    logits=model(torch.from_numpy(normalize(x[idx],policy)).to(args.device))
                    loss=nn.functional.cross_entropy(logits,torch.from_numpy(y[idx]).long().to(args.device))
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();sched.step()
                    if step%100==0 or step==args.updates:
                        p=predict(model,vx,policy);m=metrics(vy,p,vs)
                        row={'step':step,'loss':float(loss.detach()),'val_BA':m['subject_mean_BA'],'seconds':time.perf_counter()-start}
                        history.append(row);write(folder/'history.json',history)
                        if m['subject_mean_BA']>best:
                            best=m['subject_mean_BA'];torch.save(model.state_dict(),folder/'model.pt')
                            np.savez(folder/'validation.npz',scores=p,y=vy,subject=vs,trial_id=vt)
                            write(folder/'best.json',{'step':step,'metrics':m})
                        write(output/'status.json',{'status':'running','fold':fold+1,'model':name,**row})
                        print(json.dumps({'fold':fold+1,'model':name,**row}),flush=True)
                model.load_state_dict(torch.load(folder/'model.pt',map_location=args.device,weights_only=True),strict=True)
                p=predict(model,vx,policy);error=float(np.max(np.abs(p-np.load(folder/'validation.npz')['scores'])));assert error<1e-5
                result={'fold':fold+1,'model':name,'metrics':metrics(vy,p,vs),'seconds':time.perf_counter()-start,
                    'reload_error':error,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else 0.,
                    'trainable_parameters':cfg['trainable_parameters'],'test_evaluated':False}
                results.append(result);write(folder/'result.json',result);write(output/'results.json',results)
                del model,opt;torch.cuda.empty_cache()
        write(output/'status.json',{'status':'finished','fits':len(results),'test_evaluated':False})
    except Exception as exc:
        write(output/'status.json',{'status':'failed','error':repr(exc),'fits_completed':len(results),'test_evaluated':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','baseline','controls','output','lab-source','cbra-source','weights']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--updates',type=int,default=1500)
    p.add_argument('--folds',type=int,choices=[1,2,3],default=3)
    main(p.parse_args())
