"""Source-template FBTRCA / FBCCA fusion; three-fold subject-disjoint development."""
import csv
import json
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter,sosfiltfilt
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from search_models import metrics,write
from run_preliminary import sha,PLAN

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/ssvep_transfer_v1'
CHANNELS=['PZ','PO5','PO3','POZ','PO4','PO6','O1','OZ','O2']

def standardize(signal):
    signal=signal-signal.mean(-1,keepdims=True)
    return signal/signal.std(-1,keepdims=True).clip(1e-8)

def cca_scores(x,qref):
    result=[]
    for start in range(0,len(x),64):
        batch=x[start:start+64].transpose(0,2,1);batch=batch-batch.mean(1,keepdims=True)
        q=np.linalg.qr(batch,mode='reduced')[0]
        cross=np.einsum('ntc,ktr->nkcr',q,qref,optimize=True)
        result.append(np.linalg.svd(cross,compute_uv=False)[...,0]**2)
    return np.concatenate(result)

def trca_fit(x,y):
    templates=[];filters=[]
    for target in range(40):
        z=x[y==target];total=z.sum(0)
        q=np.einsum('nct,ndt->cd',z,z,optimize=True)
        s=total@total.T-q
        ridge=1e-3*np.trace(q)/len(q)
        values,vectors=eigh(s,q+ridge*np.eye(len(q)))
        w=vectors[:,-1];w/=np.linalg.norm(w)
        templates.append(z.mean(0));filters.append(w)
    return np.stack(filters),np.stack(templates)

def trca_scores(x,filters,templates):
    reference=np.einsum('kc,kct->kt',filters,templates,optimize=True)
    reference-=reference.mean(-1,keepdims=True);reference/=np.linalg.norm(reference,axis=-1,keepdims=True).clip(1e-10)
    result=[]
    for start in range(0,len(x),64):
        z=np.einsum('kc,nct->nkt',filters,x[start:start+64],optimize=True)
        z-=z.mean(-1,keepdims=True);z/=np.linalg.norm(z,axis=-1,keepdims=True).clip(1e-10)
        rho=np.einsum('nkt,kt->nk',z,reference,optimize=True).clip(-1,1)
        result.append(np.sign(rho)*rho**2)
    return np.concatenate(result)

def zscore_classes(x):return (x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-8)

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];audits=[]
    for seconds in [1,2]:
        path=ROOT/f'data/processed/ssvep_{seconds}s.npz';meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path,allow_pickle=False) as raw:
            keep=~np.isin(raw['subject'],[7,8,33,34,35])
            names=[n.upper() for n in raw['channel_names']];pick=[names.index(n) for n in CHANNELS]
            x=raw['X'][keep][:,pick].copy();y=raw['y'][keep];subject=raw['subject'][keep];ids=raw['trial_id'][keep]
        assert len(np.unique(subject))==30 and not set(subject)&{7,8,33,34,35}
        t=np.arange(x.shape[-1])/200
        references=np.stack([np.stack([fun(2*np.pi*f*h*t) for h in range(1,6) for fun in [np.sin,np.cos]],axis=-1) for f in meta['frequencies']])
        references-=references.mean(1,keepdims=True);qref=np.linalg.qr(references,mode='reduced')[0]
        weights=np.arange(1,6)**(-1. if seconds==1 else -.5)+.25
        # Filter once per task, independent of labels/folds. Each 10-class frame uses only its own trial.
        bank=[];analytic=[]
        for lo in [6,14,22,30,38]:
            signal=standardize(sosfiltfilt(butter(4,[lo,90],fs=200,btype='bandpass',output='sos'),x,axis=-1))
            bank.append(signal.astype('float32'))
            with threadpool_limits(limits=4):analytic.append(cca_scores(signal,qref))
        cca=np.einsum('b,bnk->nk',weights,np.stack(analytic))
        for fold in range(3):
            folder=OUT/f'{seconds}s_fold{fold+1}';folder.mkdir(parents=True,exist_ok=True)
            write(OUT/'queue_status.json',{'status':'running','seconds':seconds,'fold':fold+1})
            va=np.flatnonzero(np.isin(subject,PLAN['validation_subjects_by_fold']['ssvep_2s'][fold]))
            tr=np.flatnonzero(~np.isin(subject,PLAN['validation_subjects_by_fold']['ssvep_2s'][fold]))
            assert not set(subject[tr])&set(subject[va]);start=time.perf_counter();models=[];all_scores=[]
            with threadpool_limits(limits=4):
                for signal in bank:
                    filters,templates=trca_fit(signal[tr].astype('float64'),y[tr]);models.append((filters,templates))
                    all_scores.append(trca_scores(signal[va].astype('float64'),filters,templates))
            trca=np.einsum('b,bnk->nk',weights,np.stack(all_scores));joblib.dump(models,folder/'model.joblib')
            cfg={'seconds':seconds,'fold':fold,'channels':CHANNELS,'harmonics':5,'lower':[6,14,22,30,38],'upper':90,
                'weights':weights.tolist(),'ridge_fraction':.001,'source_sha256':sha(__file__),'data_sha256':sha(path),
                'train_subjects':np.unique(subject[tr]).tolist(),'validation_subjects':np.unique(subject[va]).tolist(),
                'strict_subject_generalization':True,'test_evaluated':False,'template':'mean training trials per class',
                'trca':'leading generalized eig of cross-trial S and within-trial Q; shared source-subject filter',
                'scope':'source-template transfer variant, not same-subject calibration FBTRCA; filter context is offline'}
            write(folder/'config.json',cfg)
            for name,alpha in [('fbcca',0.),('fbtrca',1.),('fusion25',.25),('fusion50',.5),('fusion75',.75)]:
                value=(1-alpha)*zscore_classes(cca[va])+alpha*zscore_classes(trca)
                p=softmax(value,axis=-1);m=metrics(y[va],p.argmax(-1),subject[va],40)
                np.savez_compressed(folder/f'{name}_validation.npz',y=y[va],scores=p,subject=subject[va],trial_id=ids[va])
                write(folder/f'{name}_metrics.json',m)
                rows.append({'seconds':seconds,'fold':fold+1,'method':name,'val_BA':m['subject_mean_BA'],
                             'macro_F1':m['macro_f1'],'test_evaluated':False})
                print(json.dumps(rows[-1]),flush=True)
            # Reconstruct five filter-bank score components from persisted model on first32 val trials.
            loaded=joblib.load(folder/'model.joblib')
            with threadpool_limits(limits=4):reload=np.einsum('b,bnk->nk',weights,np.stack([trca_scores(b[va[:32]].astype('float64'),*m) for b,m in zip(bank,loaded)]))
            error=float(np.max(np.abs(reload-trca[:32])));assert error<1e-8
            audits.append({'seconds':seconds,'fold':fold+1,'checkpoint_max_abs_error':error,
                'old_test_subjects_absent':True,'fit_seconds':time.perf_counter()-start})
            write(folder/'fit_result.json',audits[-1])
        del x,bank,analytic
    for row in rows:
        folder=OUT/f'{row["seconds"]}s_fold{row["fold"]}'
        with np.load(folder/f'{row["method"]}_validation.npz') as p:
            score=metrics(p['y'],p['scores'].argmax(-1),p['subject'],40)['subject_mean_BA']
            assert abs(score-row['val_BA'])<1e-12
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'seconds':s,'method':method,'mean_BA':float(np.mean([r['val_BA'] for r in rows if r['seconds']==s and r['method']==method]))}
             for s in [1,2] for method in ['fbcca','fbtrca','fusion25','fusion50','fusion75']]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audits)
    write(OUT/'queue_status.json',{'status':'finished','fitted_FBTRCA_models':6,'score_config_evaluations':30,
                                  'prediction_audits':30,'checkpoint_subset_audits':6,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
