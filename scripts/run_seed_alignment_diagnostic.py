"""Offline unlabeled-target-session normalization diagnostic, separate from strict DG."""
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

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/seed_alignment_diagnostic_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];audit=[]
    for fold in range(3):
        d=load_fold(fold)
        # Use one temporal-mean vector per 10-second window, identical to strict Logistic.
        original=d['X'].mean(1).reshape(len(d['X']),310)
        for mode in ['session_center','session_zscore']:
            start=time.perf_counter();folder=OUT/f'fold{fold+1}_{mode}';folder.mkdir(parents=True,exist_ok=True)
            x=original.copy();stats=[]
            # No label is used to calculate target statistics; full sessions include future windows.
            for subject in np.unique(d['subject']):
                for session in np.unique(d['session'][d['subject']==subject]):
                    idx=np.flatnonzero((d['subject']==subject)&(d['session']==session))
                    mean=original[idx].mean(0);std=original[idx].std(0).clip(1e-6)
                    x[idx]=(original[idx]-mean)/(std if mode=='session_zscore' else 1.)
                    stats.append({'subject':int(subject),'session':int(session),'mean':mean,'std':std})
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000))
            with threadpool_limits(limits=4):clf.fit(x[d['train']],d['y'][d['train']])
            p=clf.predict_proba(x[d['val']]);m=score(d,p);persist(folder,d,p)
            joblib.dump({'session_statistics':stats,'classifier':clf},folder/'model.joblib')
            cfg={'method':mode,'fold':fold,'C':.1,'data_sha256':sha(CACHE),'source_sha256':sha(__file__),
                 'strict_domain_generalization':False,'uses_unlabeled_target_session':True,
                 'uses_target_labels_for_fitting_or_alignment':False,'uses_future_target_windows':True,
                 'scope':'offline transductive diagnostic only; not online adaptation or competition score',
                 'train_subjects':np.unique(d['subject'][d['train']]).tolist(),'val_subjects':np.unique(d['subject'][d['val']]).tolist()}
            write(folder/'config.json',cfg);write(folder/'result.json',{'metrics':m,'seconds':time.perf_counter()-start,'test_evaluated':False})
            obj=joblib.load(folder/'model.joblib');reload_x=original[d['val']].copy()
            for stat in obj['session_statistics']:
                mask=(d['subject'][d['val']]==stat['subject'])&(d['session'][d['val']]==stat['session'])
                reload_x[mask]=(reload_x[mask]-stat['mean'])/(stat['std'] if mode=='session_zscore' else 1.)
            error=float(np.max(np.abs(obj['classifier'].predict_proba(reload_x)-p)));assert error<1e-9
            npz=np.load(folder/'validation.npz');computed=score(d,npz['scores'])
            assert abs(computed['window']['subject_mean_BA']-m['window']['subject_mean_BA'])<1e-12
            audit.append({'run':folder.name,'checkpoint_max_abs_error':error,'prediction_audit_passed':True})
            rows.append({'fold':fold+1,'method':mode,'window_BA':m['window']['subject_mean_BA'],
                         'video_BA':m['video_trial']['subject_mean_BA'],'uses_unlabeled_target_session':True,'test_evaluated':False})
            print(json.dumps(rows[-1]),flush=True)
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'method':mode,'mean_window_BA':float(np.mean([r['window_BA'] for r in rows if r['method']==mode])),
              'mean_video_BA':float(np.mean([r['video_BA'] for r in rows if r['method']==mode])),
              'strict_domain_generalization':False,'uses_unlabeled_target_session':True}
             for mode in ['session_center','session_zscore']]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audit)
    write(OUT/'queue_status.json',{'status':'finished','fits':6,'audited':6})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
