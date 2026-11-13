"""Per-input invariances and causal-normalizer diagnostics; no raw-EEG online claim."""
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
from run_seed_features import load_fold,score,persist,CACHE
from run_preliminary import sha
from search_models import write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/seed_normalization_search_v1'
MODES=['band_center','spatial_center','double_center','causal_prior5','causal_prior20','causal_prior100']

def features(d,mode):
    a=d['X'].mean(1).astype('float64')
    state={}
    if mode in ['band_center','double_center']:a=a-a.mean(-1,keepdims=True)
    if mode in ['spatial_center','double_center']:a=a-a.mean(-2,keepdims=True)
    a=a.reshape(len(a),310)
    if mode.startswith('causal'):
        strength=int(mode.removeprefix('causal_prior'));raw=a.copy()
        mean=raw[d['train']].mean(0);var=raw[d['train']].var(0).clip(1e-8)
        state={'training_prior_mean':mean,'training_prior_variance':var,'prior_strength':strength}
        for s in np.unique(d['subject']):
            for session in np.unique(d['session'][d['subject']==s]):
                idx=np.flatnonzero((d['subject']==s)&(d['session']==session))
                # Cache preserves videos1..15 and increasing non-overlap frame windows.
                sums=strength*mean+np.cumsum(raw[idx],axis=0)
                squares=strength*(var+mean**2)+np.cumsum(raw[idx]**2,axis=0)
                n=(strength+np.arange(1,len(idx)+1))[:,None]
                mu=sums/n;sd=np.sqrt((squares/n-mu**2).clip(1e-8))
                a[idx]=(raw[idx]-mu)/sd
        # Prefix consistency proves normalizer cannot depend on later target windows.
        s=d['subject'][d['val'][0]];session=d['session'][d['val'][0]]
        idx=np.flatnonzero((d['subject']==s)&(d['session']==session))[:8]
        sums=strength*mean+np.cumsum(raw[idx],axis=0);n=(strength+np.arange(1,len(idx)+1))[:,None]
        sq=strength*(var+mean**2)+np.cumsum(raw[idx]**2,axis=0)
        prefix=(raw[idx]-sums/n)/np.sqrt((sq/n-(sums/n)**2).clip(1e-8))
        assert np.allclose(prefix,a[idx],atol=1e-10)
    assert np.isfinite(a).all()
    return a,state

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];audits=[]
    for fold in range(3):
        d=load_fold(fold)
        for mode in MODES:
            folder=OUT/f'fold{fold+1}_{mode}';folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
            x,state=features(d,mode);clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000))
            with threadpool_limits(limits=4):clf.fit(x[d['train']],d['y'][d['train']])
            p=clf.predict_proba(x[d['val']]);m=score(d,p);persist(folder,d,p)
            joblib.dump({'classifier':clf,'normalizer_state':state},folder/'model.joblib')
            cfg={'mode':mode,'fold':fold,'C':.1,'data_sha256':sha(CACHE),'source_sha256':sha(__file__),
                 'strict_DG':not mode.startswith('causal'),'normalizer_uses_target_history':mode.startswith('causal'),
                 'normalizer_uses_future_target':False,'upstream_LDS_offline':True,'test_evaluated':False,
                 'split':{k:np.unique(d['subject'][d[k]]).tolist() for k in ['train','val']}}
            write(folder/'config.json',cfg);write(folder/'result.json',{'metrics':m,'seconds':time.perf_counter()-start,'test_evaluated':False})
            q=joblib.load(folder/'model.joblib')['classifier'].predict_proba(x[d['val']]);error=float(np.max(np.abs(p-q)));assert error<1e-10
            rows.append({'mode':mode,'fold':fold+1,'val_BA':m['window']['subject_mean_BA'],
                'video_BA':m['video_trial']['subject_mean_BA'],'strict_DG':cfg['strict_DG'],'test_evaluated':False})
            audits.append({'run':folder.name,'reload_error':error,'causal_prefix_check':mode.startswith('causal')})
            print(json.dumps(rows[-1]),flush=True)
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'mode':mode,'mean_BA':float(np.mean([r['val_BA'] for r in rows if r['mode']==mode])),
        'mean_video_BA':float(np.mean([r['video_BA'] for r in rows if r['mode']==mode])),'strict_DG':not mode.startswith('causal')}
        for mode in MODES]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audits)
    write(OUT/'queue_status.json',{'status':'finished','fits':18,'reload_audits':18,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
