"""FACED eight-class feature normalization transfer, strict/offline/prefix conditions."""
import csv
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from run_preliminary import load_data,supplement_features,SequenceTCN,sha,save_predictions,schedules
from search_models import seed_all,metrics,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/faced_normalization_v1'

def transform(d,mode):
    raw=np.asarray(supplement_features(d,'sequence')).astype('float64');means=raw.mean(1)
    prior_mean=means[d['train']].mean(0);prior_var=means[d['train']].var(0).clip(1e-8);x=raw.copy();groups=[]
    if mode!='strict':
        for subject in np.unique(d['subject']):
            idx=np.flatnonzero(d['subject']==subject)
            # Existing cache was written in source metadata order; no label-based sorting.
            if mode=='offline':mu=means[idx].mean(0)[None];sd=means[idx].std(0).clip(1e-6)[None]
            else:
                n=(5+np.arange(1,len(idx)+1))[:,None];mu=(5*prior_mean+np.cumsum(means[idx],0))/n
                v=(5*(prior_var+prior_mean**2)+np.cumsum(means[idx]**2,0))/n-mu**2;sd=np.sqrt(v.clip(1e-8))
            x[idx]=(raw[idx]-mu[:,None,:])/sd[:,None,:];groups.append({'subject':int(subject),'mu':mu,'sd':sd})
    return x,{'mode':mode,'prior_mean':prior_mean,'prior_var':prior_var,'groups':groups}

@torch.no_grad()
def predict(model,x):
    model.eval();p=[]
    for start in range(0,len(x),64):p.append(model(torch.from_numpy(x[start:start+64])).softmax(-1).numpy())
    return np.concatenate(p)

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];audits=[]
    for fold in range(3):
        d=load_data('emotion',fold)
        for mode in ['strict','causal','offline']:
            x,state=transform(d,mode)
            for method in ['logistic','tcn']:
                folder=OUT/f'fold{fold+1}_{mode}_{method}';folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
                write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'mode':mode,'method':method})
                cfg={'fold':fold,'mode':mode,'method':method,'data_sha256':d['source_sha256'],'source_sha256':sha(__file__),
                    'split':d['split'],'test_evaluated':False,'strict_DG':mode=='strict','seed':42+fold,
                    'scope':'old ten development subjects only; same-known-video generalization, no new blind test',
                    'feature':'same one-second band log variance as preliminary_v1; ten-frame window',
                    'target_future_normalizer':mode=='offline','target_history_normalizer':mode=='causal',
                    'causal_order':'original cache metadata order; not labels; real chronological mapping needs source audit before online deployment'}
                write(folder/'config.json',cfg);joblib.dump(state,folder/'normalizer.joblib')
                tr=d['train'];va=d['val'];seed_all(42+fold);torch.set_num_threads(2)
                if method=='logistic':
                    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000))
                    with threadpool_limits(limits=2):clf.fit(x[tr].mean(1),d['y'][tr])
                    p=clf.predict_proba(x[va].mean(1));joblib.dump(clf,folder/'model.joblib')
                    err=float(np.max(np.abs(joblib.load(folder/'model.joblib').predict_proba(x[va].mean(1))-p)))
                else:
                    scaler=StandardScaler().fit(x[tr].reshape(-1,x.shape[-1]));shape=x.shape
                    xx=scaler.transform(x.reshape(-1,shape[-1])).reshape(shape).astype('float32');joblib.dump(scaler,folder/'scaler.joblib')
                    model=SequenceTCN(shape[-1],8);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1500);schedule=schedules({'emotion':d},42+fold,1500)['emotion']
                    np.save(folder/'training_indices.npy',schedule)
                    weights=torch.tensor(len(tr)/(8*np.bincount(d['y'][tr],minlength=8)),dtype=torch.float32)
                    best=-1;history=[]
                    for step,index in enumerate(schedule,1):
                        model.train();opt.zero_grad(set_to_none=True)
                        loss=nn.functional.cross_entropy(model(torch.from_numpy(xx[index])),torch.from_numpy(d['y'][index]).long(),weight=weights)
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
                        if step%100==0:
                            p=predict(model,xx[va]);m=metrics(d['y'][va],p.argmax(-1),d['subject'][va],8)
                            history.append({'step':step,'loss_last_batch':float(loss.detach()),'val_BA':m['subject_mean_BA']});write(folder/'history.json',history)
                            if m['subject_mean_BA']>best:
                                best=m['subject_mean_BA'];torch.save(model.state_dict(),folder/'model.pt');save_predictions(folder,{'emotion':d},{'emotion':p})
                                write(folder/'best_validation.json',{'step':step,'metrics':m})
                            print(json.dumps({'run':folder.name,**history[-1]}),flush=True)
                    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
                    p=predict(model,xx[va]);err=float(np.max(np.abs(np.load(folder/'emotion_validation.npz')['scores']-p)))
                assert err<1e-5;save_predictions(folder,{'emotion':d},{'emotion':p});m=metrics(d['y'][va],p.argmax(-1),d['subject'][va],8)
                write(folder/'result.json',{'metrics':m,'seconds':time.perf_counter()-start,'reload_error':err,'test_evaluated':False})
                rows.append({'fold':fold+1,'mode':mode,'method':method,'BA':m['subject_mean_BA'],'strict_DG':mode=='strict','test_evaluated':False})
                audits.append({'run':folder.name,'reload_error':err,'old_test_absent':not bool(set(d['subject'][va])&{3,11})})
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'mode':mode,'method':method,'mean_BA':float(np.mean([r['BA'] for r in rows if (r['mode'],r['method'])==(mode,method)]))}
             for mode in ['strict','causal','offline'] for method in ['logistic','tcn']]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audits)
    write(OUT/'queue_status.json',{'status':'finished','fits':18,'prediction_audits':18,'checkpoint_audits':18,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
