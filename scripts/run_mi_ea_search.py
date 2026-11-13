"""EEGNet with offline subject EA or cumulative prefix EA; CPU paired controls."""
import csv
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from torch import nn
from run_preliminary import load_data,save_predictions,sha
from run_mi_domain import balanced_schedule
from search_models import EEGNet,seed_all,metrics,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mi_ea_search_v1'
MODES=['offline_subject_EA','causal_prior10','causal_prior50']

def inverse_sqrt(c):
    c=.99*c+.01*np.trace(c)/len(c)*np.eye(len(c));v,u=np.linalg.eigh(c)
    return (u*v.clip(1e-12)**-.5)@u.T

def transform(d,mode):
    raw=d['X'].astype('float64');raw-=raw.mean(-1,keepdims=True)
    cov=raw@raw.transpose(0,2,1)/(raw.shape[-1]-1);prior=cov[d['train']].mean(0)
    x=np.empty_like(raw,dtype='float32');stats={}
    for subject in np.unique(d['subject']):
        idx=np.flatnonzero(d['subject']==subject)
        if mode=='offline_subject_EA':
            w=inverse_sqrt(cov[idx].mean(0));x[idx]=w@raw[idx];stats[str(subject)]=w
        else:
            strength=int(mode.removeprefix('causal_prior'));total=strength*prior.copy()
            for n,i in enumerate(idx,1):
                total+=cov[i];w=inverse_sqrt(total/(strength+n));x[i]=w@raw[i]
            first=inverse_sqrt((strength*prior+cov[idx[0]])/(strength+1))@raw[idx[0]]
            assert np.allclose(first,x[idx[0]],rtol=1e-5,atol=1e-5)
    x=(x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)
    assert np.isfinite(x).all()
    return x,{'prior_from_training':prior,'offline_whitening':stats,'mode':mode}

@torch.no_grad()
def predict(model,x):
    model.eval();p=[]
    for start in range(0,len(x),64):p.append(model(torch.from_numpy(x[start:start+64])).softmax(-1).numpy())
    return np.concatenate(p)

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];audits=[]
    for fold in range(3):
        d=load_data('mi',fold)
        for mode in MODES:
            folder=OUT/f'fold{fold+1}_{mode}';folder.mkdir(parents=True,exist_ok=True)
            write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'mode':mode})
            cfg={'fold':fold,'mode':mode,'seed':42+fold,'source_sha256':sha(__file__),'data_sha256':d['source_sha256'],
                 'split':d['split'],'filters':16,'kernel':64,'dropout':.5,'updates':1500,'batch':16,'lr':.003,'wd':.01,
                 'optimizer':'AdamW','scheduler':'cosine','device':'cpu','strict_DG':False,
                 'normalizer_uses_future_target':mode=='offline_subject_EA','test_evaluated':False,
                 'scope':'normalizer causality only; upstream bandpass is offline; trial order follows source cache'}
            write(folder/'config.json',cfg);x,state=transform(d,mode);joblib.dump(state,folder/'normalizer.joblib')
            seed_all(42+fold);torch.set_num_threads(2);model=EEGNet(22,4,filters=16,kernel=64,dropout=.5)
            schedule,_=balanced_schedule(d,42+fold);np.save(folder/'training_indices.npy',schedule)
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1500)
            best=-1.;history=[];start=time.perf_counter()
            for step,index in enumerate(schedule,1):
                model.train();opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(torch.from_numpy(x[index])),torch.from_numpy(d['y'][index]).long())
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step()
                if step%100==0:
                    p=predict(model,x[d['val']]);m=metrics(d['y'][d['val']],p.argmax(-1),d['subject'][d['val']],4)
                    history.append({'step':step,'loss_last_batch':float(loss.detach()),'val_BA':m['subject_mean_BA']})
                    write(folder/'history.json',history)
                    if m['subject_mean_BA']>best:
                        best=m['subject_mean_BA'];torch.save(model.state_dict(),folder/'model.pt');save_predictions(folder,{'mi':d},{'mi':p})
                        write(folder/'best_validation.json',{'step':step,'metrics':m})
                    print(json.dumps({'run':folder.name,**history[-1]}),flush=True)
            model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cpu'),strict=True)
            p=predict(model,x[d['val']]);saved=np.load(folder/'mi_validation.npz')['scores'];error=float(np.max(np.abs(p-saved)));assert error<1e-5
            m=metrics(d['y'][d['val']],saved.argmax(-1),d['subject'][d['val']],4)
            write(folder/'result.json',{'metrics':m,'seconds':time.perf_counter()-start,'reload_error':error,'test_evaluated':False})
            rows.append({'mode':mode,'fold':fold+1,'BA':m['subject_mean_BA'],'strict_DG':False,'test_evaluated':False})
            audits.append({'run':folder.name,'reload_error':error,'old_test_absent':not bool(set(d['subject'][d['val']])&{8,9})})
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'mode':mode,'mean_BA':float(np.mean([r['BA'] for r in rows if r['mode']==mode])),'strict_DG':False} for mode in MODES]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audits)
    write(OUT/'queue_status.json',{'status':'finished','fits':9,'prediction_audits':9,'checkpoint_audits':9,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
