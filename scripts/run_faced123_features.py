"""Separate author-provided FACED123 nine-class feature benchmark, not raw eight-class pilot."""
import csv
import json
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from run_preliminary import SequenceTCN,sha
from search_models import seed_all,metrics,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/faced123_features_v1'
SOURCE=ROOT/'data/raw/FACED_GNN4EEG/FACED_dataset_9_labels.mat'

@torch.no_grad()
def predict(model,x):
    model.eval();out=[]
    for i in range(0,len(x),128):out.append(model(torch.from_numpy(x[i:i+128])).softmax(-1).numpy())
    return np.concatenate(out)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=loadmat(SOURCE)['de_lds'];assert a.shape==(123,840,150) and np.isfinite(a).all()
    # Exact label construction is copied as data, not inferred from file names.
    # Verified in official ge/load_data.py::load_srt_de(label_type='cls9').
    video_labels=np.repeat(np.arange(9),[3,3,3,3,4,3,3,3,3]);assert len(video_labels)==28
    x=a.reshape(123,28,3,10,150).reshape(-1,10,150).astype('float32')
    y=np.tile(np.repeat(video_labels,3),123);subjects=np.repeat(np.arange(123),84)
    trial=np.asarray([f's{s:03d}-video{v:02d}-block{b}' for s in range(123) for v in range(28) for b in range(3)])
    write(OUT/'data_audit.json',{'source':str(SOURCE),'sha256':sha(SOURCE),'source_shape':list(a.shape),'processed_shape':list(x.shape),
        'label_source':'https://raw.githubusercontent.com/Miracle-2001/GNN4EEG/main/ge/load_data.py',
        'loader_source_sha256':sha(ROOT/'tmp/research_20261113/gnn_load_data.py'),'video_labels':video_labels.tolist(),
        'feature_order':'official loader reshapes last dimension to band x30 channels; TCN vector preserves original order',
        'subject_identity':'author array row index; mapping to HF subject IDs not independently established',
        'upstream_normalization':'precomputed DE-LDS provenance retained; complete upstream fitting/causality not verified',
        'scope':'separate nine-class feature benchmark, no claim of raw EEG or end-to-end strict inductive preprocessing'})
    del a;rows=[];audits=[]
    for fold in range(3):
        va=np.flatnonzero((subjects>=fold*41)&(subjects<(fold+1)*41));tr=np.flatnonzero(~np.isin(np.arange(len(x)),va))
        assert not set(subjects[tr])&set(subjects[va])
        for mode in ['train_only','offline_subject_zscore']:
            xx=x.copy();stats={}
            if mode!='train_only':
                for s in np.unique(subjects):
                    idx=np.flatnonzero(subjects==s);mu=xx[idx].mean((0,1));sd=xx[idx].std((0,1)).clip(1e-6)
                    xx[idx]=(xx[idx]-mu)/sd;stats[str(s)]=(mu,sd)
            for method in ['logistic','tcn']:
                folder=OUT/f'fold{fold+1}_{mode}_{method}';folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
                write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'mode':mode,'method':method})
                cfg={'fold':fold,'mode':mode,'method':method,'source_sha256':sha(__file__),'data_sha256':sha(SOURCE),
                    'seed':42+fold,'classes':9,'C':.1,'lr':.001,'wd':.01,'updates':1500,'batch_size':32,'dropout':.5,
                    'train_subjects':np.unique(subjects[tr]).tolist(),'val_subjects':np.unique(subjects[va]).tolist(),
                    'target_subject_statistics':mode!='train_only','upstream_precomputed_feature_provenance_unverified':True,'test_evaluated':False}
                write(folder/'config.json',cfg);joblib.dump(stats,folder/'subject_stats.joblib');seed_all(42+fold);torch.set_num_threads(2)
                if method=='logistic':
                    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000))
                    with threadpool_limits(limits=2):clf.fit(xx[tr].mean(1),y[tr])
                    p=clf.predict_proba(xx[va].mean(1));joblib.dump(clf,folder/'model.joblib')
                    err=float(np.max(np.abs(joblib.load(folder/'model.joblib').predict_proba(xx[va].mean(1))-p)))
                else:
                    scaler=StandardScaler().fit(xx[tr].reshape(-1,150));features=scaler.transform(xx.reshape(-1,150)).reshape(xx.shape).astype('float32')
                    joblib.dump(scaler,folder/'scaler.joblib');model=SequenceTCN(150,9)
                    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1500)
                    rng=np.random.default_rng(42+fold);schedule=rng.choice(tr,size=(1500,32),replace=True);np.save(folder/'training_indices.npy',schedule)
                    best=-1.;history=[]
                    counts=np.bincount(y[tr],minlength=9);weights=torch.tensor(len(tr)/(9*counts),dtype=torch.float32)
                    for step,index in enumerate(schedule,1):
                        model.train();opt.zero_grad(set_to_none=True)
                        loss=nn.functional.cross_entropy(model(torch.from_numpy(features[index])),torch.from_numpy(y[index]).long(),weight=weights)
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
                        if step%100==0:
                            p=predict(model,features[va]);m=metrics(y[va],p.argmax(-1),subjects[va],9)
                            history.append({'step':step,'loss_last_batch':float(loss.detach()),'val_BA':m['subject_mean_BA']});write(folder/'history.json',history)
                            if m['subject_mean_BA']>best:
                                best=m['subject_mean_BA'];torch.save(model.state_dict(),folder/'model.pt')
                                np.savez_compressed(folder/'best_scores.npz',scores=p)
                            print(json.dumps({'run':folder.name,**history[-1]}),flush=True)
                    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
                    p=predict(model,features[va]);err=float(np.max(np.abs(p-np.load(folder/'best_scores.npz')['scores'])));del features
                assert err<1e-5;np.savez_compressed(folder/'validation.npz',y=y[va],scores=p,subject=subjects[va],trial_id=trial[va])
                m=metrics(y[va],p.argmax(-1),subjects[va],9);write(folder/'result.json',{'metrics':m,'reload_error':err,'seconds':time.perf_counter()-start,'test_evaluated':False})
                rows.append({'fold':fold+1,'mode':mode,'method':method,'BA':m['subject_mean_BA'],'test_evaluated':False});audits.append({'run':folder.name,'reload_error':err})
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'mode':mode,'method':method,'mean_BA':float(np.mean([r['BA'] for r in rows if (r['mode'],r['method'])==(mode,method)]))}
             for mode in ['train_only','offline_subject_zscore'] for method in ['logistic','tcn']]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audits)
    write(OUT/'queue_status.json',{'status':'finished','fits':12,'prediction_audits':12,'checkpoint_audits':12,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
