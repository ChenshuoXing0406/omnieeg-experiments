"""Source-only MI domain controls. Old test subjects never enter these runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.covariance import oas
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from run_preliminary import load_data, PLAN, sha, evaluate, save_predictions
from search_models import EEGNet, seed_all, metrics, write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/mi_domain_v1'
UPDATES=1500
LAMBDAS=[0.,1.,10.]

def log(**kw):print(json.dumps(kw,allow_nan=False),flush=True)

def balanced_schedule(d,seed):
    """Four source subjects, one trial per class each; paired across objectives."""
    rng=np.random.default_rng(seed)
    subjects=np.unique(d['subject'][d['train']])
    pools={(s,c):d['train'][(d['subject'][d['train']]==s)&(d['y'][d['train']]==c)]
           for s in subjects for c in range(d['classes'])}
    schedule=[];domains=[]
    for _ in range(UPDATES):
        chosen=rng.choice(subjects,4,replace=False)
        schedule.append([rng.choice(pools[s,c]) for s in chosen for c in range(d['classes'])])
        domains.append(np.repeat(chosen,d['classes']))
    a=np.asarray(schedule);b=np.asarray(domains)
    assert set(a.ravel()).issubset(set(d['train']))
    assert np.array_equal(d['subject'][a],b)
    return a,b

def coral_loss(features):
    """Deep CORAL pair loss, four source groups of four class-balanced trials."""
    h=features.reshape(4,4,-1)
    h=h-h.mean(1,keepdim=True)
    cov=h.transpose(1,2)@h/3
    return sum((cov[i]-cov[j]).square().sum() for i in range(4) for j in range(i+1,4))/(6*4*features.shape[-1]**2)

def matrix_function(a,kind):
    values,vectors=np.linalg.eigh(a)
    if kind=='exp':values=np.exp(values)
    else:
        values=np.maximum(values,1e-12)
        values=np.log(values) if kind=='log' else values**(-.5)
    return (vectors*values[...,None,:])@np.swapaxes(vectors,-1,-2)

def covariance_features(d):
    """OAS covariance; train-only log-Euclidean reference; tangent projection."""
    # Per-trial trace normalization uses only that input trial, no target pool.
    cov=np.stack([oas(x.T,assume_centered=False)[0] for x in d['X']])
    cov/=np.trace(cov,axis1=1,axis2=2)[:,None,None]
    tr=d['train'];ref=matrix_function(matrix_function(cov[tr],'log').mean(0),'exp')
    white=matrix_function(ref,'invsqrt')
    tangent=matrix_function(white@cov@white,'log')
    i,j=np.triu_indices(cov.shape[-1])
    feature=tangent[:,i,j]*np.where(i==j,1,np.sqrt(2))
    assert np.isfinite(feature).all() and np.linalg.eigvalsh(ref).min()>0
    return feature,ref

def run_tangent(d,fold):
    start=time.perf_counter()
    with threadpool_limits(limits=4):x,ref=covariance_features(d)
    for c in [.1,1.,10.]:
        folder=OUT/f'fold{fold+1}_tangent_C{c:g}'
        if (folder/'result.json').exists():continue
        folder.mkdir(parents=True,exist_ok=True)
        config=identity(d,fold,{'method':'oas_logeuclidean_tangent_logistic','C':c,'reference_fit':'training trials only',
            'normalization':'per-trial covariance / trace; train-only StandardScaler','class_weight':'balanced'})
        write(folder/'config.json',config)
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=2000))
        with threadpool_limits(limits=4):clf.fit(x[d['train']],d['y'][d['train']])
        p=clf.predict_proba(x[d['val']]);save_predictions(folder,{'mi':d},{'mi':p})
        joblib.dump({'reference':ref,'classifier':clf},folder/'model.joblib')
        m=metrics(d['y'][d['val']],p.argmax(-1),d['subject'][d['val']],4)
        write(folder/'result.json',{'metrics':m,'seconds_since_fold_feature_start':time.perf_counter()-start,'test_evaluated':False})
        log(finished=folder.name,BA=m['subject_mean_BA'])

def identity(d,fold,extra):
    return {'version':'mi-domain-v1','source_sha256':sha(__file__),
        'imports':{p:sha(ROOT/'scripts'/p) for p in ['run_preliminary.py','search_models.py']},
        'data_sha256':d['source_sha256'],'split':d['split'],'seed':PLAN['paired_training_seeds_by_fold'][fold],
        'fold':fold,'test_evaluated':False,'target_collection_statistics':False,**extra}

def run_neural(d,fold,lam):
    seed=PLAN['paired_training_seeds_by_fold'][fold];seed_all(seed)
    folder=OUT/f'fold{fold+1}_balanced_coral{lam:g}'
    schedule,domains=balanced_schedule(d,seed)
    config=identity(d,fold,{'method':'eegnet_source_balanced_coral','lambda_coral':lam,'updates':UPDATES,
        'batch_size':16,'subjects_per_batch':4,'trials_per_subject':4,'lr':.003,'weight_decay':.01,
        'filters':16,'kernel':64,'dropout':.5,'optimizer':'AdamW','scheduler':'cosine',
        'val_every':100,'selection':'validation subject mean BA','input':'per-trial channel temporal z-score',
        'schedule_sha256':hashlib.sha256(schedule.tobytes()).hexdigest(),'implementation':'local EEGNet variant; not full SCLDGN'})
    if (folder/'result.json').exists():
        assert json.loads((folder/'config.json').read_text())==config,'Cache identity mismatch'
        return
    folder.mkdir(parents=True,exist_ok=True);write(folder/'config.json',config)
    np.save(folder/'training_indices.npy',schedule)
    x=d['X'];x=((x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)).astype('float32')
    model=EEGNet(x.shape[1],4,filters=16,kernel=64,dropout=.5).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,UPDATES)
    counts=np.bincount(d['y'][d['train']],minlength=4)
    weights=torch.tensor(len(d['train'])/(4*counts),device='cuda',dtype=torch.float32)
    history=[];best=-1.;start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step,index in enumerate(schedule,1):
        model.train();opt.zero_grad(set_to_none=True)
        h=model.features(torch.from_numpy(x[index]).cuda()[:,None]).flatten(1)
        logits=model.head(h)
        ce=nn.functional.cross_entropy(logits,torch.from_numpy(d['y'][index]).long().cuda(),weight=weights)
        align=coral_loss(h);loss=ce+lam*align
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();scheduler.step()
        if step%100==0:
            val,pred,_=evaluate(model,{'mi':d},{'mi':x});score=val['mi']['subject_mean_BA']
            row={'step':step,'train_ce_last_batch':float(ce.detach()),'train_coral_last_batch':float(align.detach()),
                 'val_BA':score};history.append(row);write(folder/'history.json',history)
            if score>best:
                best=score;torch.save(model.state_dict(),folder/'model.pt')
                save_predictions(folder,{'mi':d},pred)
                write(folder/'best_validation.json',{'step':step,'metrics':val['mi']})
            log(run=folder.name,**row)
    write(folder/'result.json',{'metrics':json.loads((folder/'best_validation.json').read_text())['metrics'],
        'seconds':time.perf_counter()-start,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'test_evaluated':False})

def export():
    rows=[]
    for folder in sorted(OUT.glob('fold*')):
        if not (folder/'result.json').exists():continue
        r=json.loads((folder/'result.json').read_text());cfg=json.loads((folder/'config.json').read_text())
        with np.load(folder/'mi_validation.npz',allow_pickle=False) as p:
            assert not set(p['subject'])&{8,9}
            assert np.isfinite(p['scores']).all() and np.allclose(p['scores'].sum(-1),1,atol=1e-5)
            m=metrics(p['y'],p['scores'].argmax(-1),p['subject'],4)
            assert abs(m['subject_mean_BA']-r['metrics']['subject_mean_BA'])<1e-10
        rows.append({'run':folder.name,'fold':cfg['fold']+1,'method':cfg['method'],
            'parameter':cfg.get('lambda_coral',cfg.get('C')),'val_BA':m['subject_mean_BA'],
            'macro_F1':m['macro_f1'],'test_evaluated':False})
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for method,param in sorted({(r['method'],r['parameter']) for r in rows}):
        values=[r['val_BA'] for r in rows if (r['method'],r['parameter'])==(method,param)]
        summary.append({'method':method,'parameter':param,'folds':len(values),'mean_BA':float(np.mean(values)),
            'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None})
    write(OUT/'summary.json',summary);write(OUT/'prediction_audit.json',{'passed':len(rows),'test_evaluated':False})
    log(summary=summary)

def queue():
    OUT.mkdir(parents=True,exist_ok=True)
    snapshot=OUT/'source_snapshot.py'
    if snapshot.exists():assert snapshot.read_bytes()==Path(__file__).read_bytes()
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    # A meaningful gradient check for the source-domain penalty.
    h=torch.randn(16,128,requires_grad=True);z=coral_loss(h);z.backward()
    assert torch.isfinite(h.grad).all() and h.grad.norm()>0
    with threadpool_limits(limits=4):
        for fold in range(3):
            d=load_data('mi',fold)
            write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'test_evaluated':False})
            run_tangent(d,fold)
            for lam in LAMBDAS:run_neural(d,fold,lam)
            export()
    # Reconstruct saved classifiers on validation inputs: verifies preprocessing + checkpoints.
    audits=[]
    for fold in range(3):
        d=load_data('mi',fold);x=d['X'];x=((x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)).astype('float32')
        for lam in LAMBDAS:
            folder=OUT/f'fold{fold+1}_balanced_coral{lam:g}'
            model=EEGNet(x.shape[1],4,filters=16,kernel=64,dropout=.5).cuda()
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda',weights_only=True),strict=True)
            _,p,_=evaluate(model,{'mi':d},{'mi':x})
            saved=np.load(folder/'mi_validation.npz')['scores'];err=float(np.max(np.abs(saved-p['mi'])))
            assert err<1e-5
            audits.append({'run':folder.name,'max_abs_error':err})
        with threadpool_limits(limits=4):features,_=covariance_features(d)
        for c in [.1,1.,10.]:
            folder=OUT/f'fold{fold+1}_tangent_C{c:g}'
            obj=joblib.load(folder/'model.joblib');p=obj['classifier'].predict_proba(features[d['val']])
            err=float(np.max(np.abs(np.load(folder/'mi_validation.npz')['scores']-p)))
            assert err<1e-8;audits.append({'run':folder.name,'max_abs_error':err})
    write(OUT/'checkpoint_audit.json',audits)
    write(OUT/'queue_status.json',{'status':'finished','fits':18,'prediction_audits':18,'checkpoint_audits':len(audits),'test_evaluated':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['queue','export']);args=parser.parse_args()
    if args.command=='queue':queue()
    else:export()
