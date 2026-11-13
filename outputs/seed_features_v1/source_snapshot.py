"""SEED three-class source-only cross-subject controls on author DE-LDS features."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import joblib
import numpy as np
from scipy.io import loadmat
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from search_models import seed_all, metrics, write
from run_preliminary import SequenceTCN, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/seed_features_v1'
CACHE=ROOT/'data/processed/SEED_DE_LDS_10s_v1.npz'
ARCHIVE=Path('D:/seed/SEED/SEED.zip')
UPDATES=1500
VAL_SUBJECTS=[list(range(1,6)),list(range(6,11)),list(range(11,16))]
sys.path.insert(0,str(ROOT/'external/mi_adapter_deps'))

def log(**items):print(json.dumps(items,allow_nan=False),flush=True)

def prepare():
    if CACHE.exists():
        audit=json.loads(CACHE.with_suffix('.json').read_text())
        assert audit['processed_sha256']==sha(CACHE)
        return
    chunks=[];ys=[];subjects=[];sessions=[];trials=[];members=[];dropped=0
    with zipfile.ZipFile(ARCHIVE) as z:
        label_path='SEED/ExtractedFeatures/label.mat';raw=z.read(label_path)
        labels=loadmat(io.BytesIO(raw))['label'].ravel().astype(int)
        assert labels.tolist()==[1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
        paths=[p for p in z.namelist() if re.fullmatch(r'SEED/ExtractedFeatures/\d+_\d{8}\.mat',p)]
        assert len(paths)==45
        dates={s:sorted(p for p in paths if int(Path(p).name.split('_')[0])==s) for s in range(1,16)}
        assert all(len(v)==3 for v in dates.values())
        for subject in range(1,16):
            for session,path in enumerate(dates[subject],1):
                raw=z.read(path)  # zipfile checks member CRC during complete read
                info=z.getinfo(path)
                members.append({'path':path,'bytes':len(raw),'zip_crc32':info.CRC,'sha256':hashlib.sha256(raw).hexdigest()})
                obj=loadmat(io.BytesIO(raw),variable_names=[f'de_LDS{i}' for i in range(1,16)])
                for trial in range(1,16):
                    a=obj[f'de_LDS{trial}'];assert a.shape[0]==62 and a.shape[-1]==5 and np.isfinite(a).all()
                    n=a.shape[1]//10;dropped+=a.shape[1]%10
                    chunks.append(a[:,:n*10].transpose(1,0,2).reshape(n,10,62,5).astype('float32'))
                    ys.extend([int(labels[trial-1]+1)]*n);subjects.extend([subject]*n);sessions.extend([session]*n)
                    trials.extend([f's{subject:02d}-date{Path(path).stem.split("_")[1]}-video{trial:02d}']*n)
                log(prepared_subject=subject,session=session)
    x=np.concatenate(chunks)
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(CACHE,X=x,y=np.asarray(ys),subject=np.asarray(subjects),session=np.asarray(sessions),trial_id=np.asarray(trials))
    write(CACHE.with_suffix('.json'),{'source':str(ARCHIVE),'source_archive_bytes':ARCHIVE.stat().st_size,
        'members':members,'processed_sha256':sha(CACHE),'shape':list(x.shape),'subjects':15,'sessions_per_subject':3,
        'label_names':['negative','neutral','positive'],'source_labels':labels.tolist(),'model_labels':'source + 1',
        'feature':'author de_LDS, 62 channels x 5 bands, one-second feature frames','window_seconds':10,
        'windows_overlap':False,'dropped_trailing_feature_frames':dropped,'trial_count':len(set(trials)),
        'scope':'offline feature task; LDS may use whole-video context; not raw EEG, not strict online latency',
        'validation_subjects_by_fold':VAL_SUBJECTS,'test_set_defined':False})

def load_fold(fold):
    with np.load(CACHE,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    d['train']=np.flatnonzero(~np.isin(d['subject'],VAL_SUBJECTS[fold]))
    d['val']=np.flatnonzero(np.isin(d['subject'],VAL_SUBJECTS[fold]))
    assert not set(d['subject'][d['train']])&set(d['subject'][d['val']])
    assert not set(d['trial_id'][d['train']])&set(d['trial_id'][d['val']])
    return d

def score(d,p):
    va=d['val'];window=metrics(d['y'][va],p.argmax(-1),d['subject'][va],3)
    tids=d['trial_id'][va];pp=[];yy=[];ss=[]
    for trial in np.unique(tids):
        ids=np.flatnonzero(tids==trial);assert len(set(d['y'][va][ids]))==1
        pp.append(p[ids].mean(0));yy.append(d['y'][va][ids[0]]);ss.append(d['subject'][va][ids[0]])
    trial_metrics=metrics(np.asarray(yy),np.asarray(pp).argmax(-1),np.asarray(ss),3)
    return {'window':window,'video_trial':trial_metrics}

def persist(folder,d,p):
    va=d['val'];np.savez_compressed(folder/'validation.npz',scores=p,y=d['y'][va],subject=d['subject'][va],
        session=d['session'][va],trial_id=d['trial_id'][va])

def config(d,fold,method,param):
    return {'version':'seed-features-v1','fold':fold,'seed':42+fold,'method':method,'parameter':param,
        'source_sha256':sha(__file__),'import_sources':{s:sha(ROOT/'scripts'/s) for s in ['run_preliminary.py','search_models.py']},
        'data_sha256':sha(CACHE),'split':{k:np.unique(d['subject'][d[k]]).tolist() for k in ['train','val']},
        'normalization':'StandardScaler fitted on training subjects only','test_evaluated':False,
        'feature':'author DE-LDS 10 consecutive 1-second frames per example; 62x5',
        'scope':'three-fold development; all configurations evaluated on validation; not an independent final test',
        'emt_source_sha256':sha(ROOT/'external/EmT/EmT_Cls.py')}

def build_model(method):
    if method=='tcn':return SequenceTCN(310,3)
    spec=importlib.util.spec_from_file_location('official_emt_cls',ROOT/'external/EmT/EmT_Cls.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.EmT(layers_graph=[1,2],layers_transformer=1,num_adj=2,num_chan=62,num_feature=5,
        hidden_graph=16,K=3,num_head=4,dim_head=16,dropout=.5,num_class=3,alpha=.25,
        graph2token='Linear',encoder_type='Cheby')

def view(x,method):return x.reshape(len(x),10,310) if method=='tcn' else x

@torch.no_grad()
def evaluate(model,x,method):
    model.eval();p=[]
    for start in range(0,len(x),64):p.append(model(torch.from_numpy(view(x[start:start+64],method)).cuda()).softmax(-1).cpu().numpy())
    return np.concatenate(p)

def run_classic(d,fold,method,c):
    folder=OUT/f'fold{fold+1}_{method}_C{c:g}';cfg=config(d,fold,method,c)
    if (folder/'result.json').exists():
        assert json.loads((folder/'config.json').read_text())==cfg;return
    folder.mkdir(parents=True,exist_ok=True);write(folder/'config.json',cfg);start=time.perf_counter()
    x=d['X'].mean(1).reshape(len(d['X']),-1)
    estimator=LogisticRegression(C=c,class_weight='balanced',max_iter=2000) if method=='logistic' else LinearSVC(C=c,class_weight='balanced',dual=False,max_iter=5000)
    clf=make_pipeline(StandardScaler(),estimator)
    with threadpool_limits(limits=4):clf.fit(x[d['train']],d['y'][d['train']])
    p=clf.predict_proba(x[d['val']]) if method=='logistic' else softmax(clf.decision_function(x[d['val']]),axis=-1)
    joblib.dump(clf,folder/'model.joblib');persist(folder,d,p)
    result={'metrics':score(d,p),'seconds':time.perf_counter()-start,'test_evaluated':False,
        'score_note':'SVM softmax margins are uncalibrated; used only for argmax and window-to-video aggregation' if method=='linear_svm' else 'predict_proba'}
    write(folder/'result.json',result);log(finished=folder.name,BA=result['metrics']['window']['subject_mean_BA'])

def run_neural(d,fold,method):
    folder=OUT/f'fold{fold+1}_{method}';cfg=config(d,fold,method,None)
    cfg.update({'updates':UPDATES,'batch_size':24,'lr':.001,'weight_decay':.01,'optimizer':'AdamW','scheduler':'cosine',
        'dropout':.5,'selection':'validation subject mean window BA every 100 updates','sampling':'8 source subjects, one window per class each',
        'implementation':'official EmT classifier with documented small configuration; local TCN; not exact paper training reproduction',
        'emt_config':{'layers_graph':[1,2],'layers_transformer':1,'num_adj':2,'hidden_graph':16,'K':3,'num_head':4,'dim_head':16,'encoder_type':'Cheby'}})
    if (folder/'result.json').exists():assert json.loads((folder/'config.json').read_text())==cfg;return
    folder.mkdir(parents=True,exist_ok=True);write(folder/'config.json',cfg)
    seed_all(42+fold);tr=d['train'];va=d['val'];shape=d['X'].shape
    scaler=StandardScaler().fit(d['X'][tr].reshape(-1,310))
    x=scaler.transform(d['X'].reshape(-1,310)).reshape(shape).astype('float32');joblib.dump(scaler,folder/'scaler.joblib')
    rng=np.random.default_rng(42+fold);subjects=np.unique(d['subject'][tr])
    pools={(s,c):tr[(d['subject'][tr]==s)&(d['y'][tr]==c)] for s in subjects for c in range(3)}
    schedule=np.asarray([[rng.choice(pools[s,c]) for s in rng.choice(subjects,8,replace=False) for c in range(3)] for _ in range(UPDATES)])
    assert set(schedule.ravel()).issubset(set(tr));np.save(folder/'training_indices.npy',schedule)
    model=build_model(method).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,UPDATES)
    history=[];best=-1;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step,index in enumerate(schedule,1):
        model.train();opt.zero_grad(set_to_none=True)
        logits=model(torch.from_numpy(view(x[index],method)).cuda())
        loss=nn.functional.cross_entropy(logits,torch.from_numpy(d['y'][index]).long().cuda())
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step()
        if step%100==0:
            p=evaluate(model,x[va],method);m=score(d,p);value=m['window']['subject_mean_BA']
            row={'step':step,'train_loss_last_batch':float(loss.detach()),'val_BA':value,'val_video_BA':m['video_trial']['subject_mean_BA']}
            history.append(row);write(folder/'history.json',history)
            if value>best:
                best=value;torch.save(model.state_dict(),folder/'model.pt');persist(folder,d,p)
                write(folder/'best_validation.json',{'step':step,'metrics':m})
            log(run=folder.name,**row)
    write(folder/'result.json',{'metrics':json.loads((folder/'best_validation.json').read_text())['metrics'],
        'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'test_evaluated':False})

def export():
    rows=[];audits=[]
    for folder in sorted(OUT.glob('fold*')):
        if not (folder/'result.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text());result=json.loads((folder/'result.json').read_text())
        d=load_fold(cfg['fold'])
        with np.load(folder/'validation.npz',allow_pickle=False) as p:
            assert np.array_equal(p['y'],d['y'][d['val']]) and np.array_equal(p['trial_id'],d['trial_id'][d['val']])
            assert set(p['subject'])==set(cfg['split']['val'])
            m=score(d,p['scores']);assert abs(m['window']['subject_mean_BA']-result['metrics']['window']['subject_mean_BA'])<1e-10
        rows.append({'fold':cfg['fold']+1,'method':cfg['method'],'parameter':str(cfg['parameter']),
            'val_window_BA':m['window']['subject_mean_BA'],'val_video_BA':m['video_trial']['subject_mean_BA'],
            'macro_F1':m['window']['macro_f1'],'seconds':result['seconds'],'test_evaluated':False})
        audits.append({'run':folder.name,'passed':True})
    if not rows:return
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for method,param in sorted({(r['method'],r['parameter']) for r in rows}):
        selected=[r for r in rows if (r['method'],r['parameter'])==(method,param)]
        summary.append({'method':method,'parameter':param,'folds':len(selected),
            'mean_window_BA':float(np.mean([r['val_window_BA'] for r in selected])),
            'mean_video_BA':float(np.mean([r['val_video_BA'] for r in selected]))})
    write(OUT/'summary.json',summary);write(OUT/'prediction_audit.json',audits);log(summary=summary)

def audit_checkpoints():
    audits=[]
    for folder in sorted(OUT.glob('fold*')):
        cfg=json.loads((folder/'config.json').read_text());d=load_fold(cfg['fold']);va=d['val'];method=cfg['method']
        if method in ['tcn','emt']:
            scaler=joblib.load(folder/'scaler.joblib');shape=d['X'][va].shape
            x=scaler.transform(d['X'][va].reshape(-1,310)).reshape(shape).astype('float32')
            model=build_model(method).cuda();model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda',weights_only=True),strict=True)
            p=evaluate(model,x,method)
        else:
            clf=joblib.load(folder/'model.joblib');x=d['X'][va].mean(1).reshape(len(va),-1)
            p=clf.predict_proba(x) if method=='logistic' else softmax(clf.decision_function(x),axis=-1)
        error=float(np.max(np.abs(np.load(folder/'validation.npz')['scores']-p)))
        assert error<1e-5;audits.append({'run':folder.name,'max_abs_error':error})
    write(OUT/'checkpoint_audit.json',audits)

def queue():
    OUT.mkdir(parents=True,exist_ok=True);prepare()
    snapshot=OUT/'source_snapshot.py'
    if snapshot.exists():assert snapshot.read_bytes()==Path(__file__).read_bytes()
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    for fold in range(3):
        d=load_fold(fold);write(OUT/'queue_status.json',{'status':'running','fold':fold+1})
        for method,c in [('logistic',.1),('logistic',1.),('linear_svm',.01),('linear_svm',.1)]:run_classic(d,fold,method,c)
        for method in ['tcn','emt']:run_neural(d,fold,method)
        export()
    export();audit_checkpoints()
    write(OUT/'queue_status.json',{'status':'finished','fits':18,'prediction_audits':18,'checkpoint_audits':18,'test_evaluated':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','queue','export']);a=parser.parse_args()
    if a.command=='queue':queue()
    elif a.command=='prepare':prepare()
    else:export()
