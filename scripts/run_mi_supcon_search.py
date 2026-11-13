"""Source-only supervised contrastive objective, with paired CPU lambda0 control."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from run_preliminary import load_data,save_predictions,sha
from run_mi_domain import balanced_schedule
from run_mi_ea_search import predict
from search_models import EEGNet,seed_all,metrics,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mi_supcon_v1'

def supcon(h,y):
    h=nn.functional.normalize(h,dim=-1)
    sim=(h@h.T)/.1;diag=torch.eye(len(h),device=h.device,dtype=torch.bool)
    sim=sim.masked_fill(diag,-1e9)
    logp=sim-torch.logsumexp(sim,dim=-1,keepdim=True)
    positive=(y[:,None]==y[None,:])&~diag
    return -(logp*positive).sum(-1).div(positive.sum(-1).clamp_min(1)).mean()

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];audit=[]
    for fold in range(3):
        d=load_data('mi',fold);x=d['X'];x=((x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)).astype('float32')
        heads=[]
        for lam in [0.,.1,1.]:
            folder=OUT/f'fold{fold+1}_lambda{lam:g}';folder.mkdir(parents=True,exist_ok=True)
            write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'lambda':lam})
            cfg={'fold':fold,'lambda':lam,'seed':42+fold,'source_sha256':sha(__file__),'data_sha256':d['source_sha256'],
                'split':d['split'],'lr':.003,'wd':.01,'dropout':.5,'batch':16,'updates':1500,'temperature':.1,
                'source_only':True,'test_evaluated':False,'device':'cpu','target_statistics':False,
                'implementation':'local EEGNet+supervised contrastive, not complete SCLDGN','positive_pairs':'same class, different source subjects'}
            write(folder/'config.json',cfg);seed_all(42+fold);torch.set_num_threads(2)
            model=EEGNet(22,4,filters=16,kernel=64,dropout=.5)
            initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest();heads.append(initial)
            schedule,_=balanced_schedule(d,42+fold);np.save(folder/'training_indices.npy',schedule)
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1500)
            best=-1.;history=[];start=time.perf_counter()
            for step,index in enumerate(schedule,1):
                model.train();opt.zero_grad(set_to_none=True)
                h=model.features(torch.from_numpy(x[index])[:,None]).flatten(1);y=torch.from_numpy(d['y'][index]).long()
                ce=nn.functional.cross_entropy(model.head(h),y);contrast=supcon(h,y);loss=ce+lam*contrast
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
                if step%100==0:
                    p=predict(model,x[d['val']]);m=metrics(d['y'][d['val']],p.argmax(-1),d['subject'][d['val']],4)
                    history.append({'step':step,'ce_last_batch':float(ce.detach()),'supcon_last_batch':float(contrast.detach()),'val_BA':m['subject_mean_BA']})
                    write(folder/'history.json',history)
                    if m['subject_mean_BA']>best:
                        best=m['subject_mean_BA'];torch.save(model.state_dict(),folder/'model.pt');save_predictions(folder,{'mi':d},{'mi':p})
                        write(folder/'best_validation.json',{'step':step,'metrics':m})
                    print(json.dumps({'run':folder.name,**history[-1]}),flush=True)
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
            p=predict(model,x[d['val']]);saved=np.load(folder/'mi_validation.npz')['scores'];err=float(np.max(np.abs(saved-p)));assert err<1e-5
            m=metrics(d['y'][d['val']],p.argmax(-1),d['subject'][d['val']],4)
            write(folder/'result.json',{'metrics':m,'seconds':time.perf_counter()-start,'initial_model_sha256':initial,
                'reload_error':err,'test_evaluated':False})
            rows.append({'fold':fold+1,'lambda':lam,'BA':m['subject_mean_BA'],'test_evaluated':False})
            audit.append({'run':folder.name,'reload_error':err,'passed':True})
        assert len(set(heads))==1
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'lambda':lam,'mean_BA':float(np.mean([r['BA'] for r in rows if r['lambda']==lam]))} for lam in [0.,.1,1.]]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audit)
    write(OUT/'queue_status.json',{'status':'finished','fits':9,'prediction_audits':9,'checkpoint_audits':9,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
