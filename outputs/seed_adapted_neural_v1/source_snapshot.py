"""TCN/EmT with offline or prefix-causal normalization of existing offline DE features."""
import csv
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from run_seed_features import load_fold,score,persist,build_model,view,CACHE
from run_preliminary import sha
from search_models import seed_all,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/seed_adapted_neural_v1'
DEVICE='cpu'  # keep GPU available to the MIRepNet queue

def transform(d,mode):
    raw=d['X'].reshape(len(d['X']),10,310).astype('float64');means=raw.mean(1)
    prior_mean=means[d['train']].mean(0);prior_var=means[d['train']].var(0).clip(1e-8)
    x=raw.copy();state={'mode':mode,'prior_mean':prior_mean,'prior_var':prior_var,'sessions':[]}
    for s in np.unique(d['subject']):
        for session in np.unique(d['session'][d['subject']==s]):
            idx=np.flatnonzero((d['subject']==s)&(d['session']==session))
            if mode=='offline':
                mu=means[idx].mean(0)[None,:];sd=means[idx].std(0).clip(1e-6)[None,:]
            else:
                n=(5+np.arange(1,len(idx)+1))[:,None]
                mu=(5*prior_mean+np.cumsum(means[idx],0))/n
                second=(5*(prior_var+prior_mean**2)+np.cumsum(means[idx]**2,0))/n
                sd=np.sqrt((second-mu**2).clip(1e-8))
                first=(means[idx[0]]-(5*prior_mean+means[idx[0]])/6)/np.sqrt(((5*(prior_var+prior_mean**2)+means[idx[0]]**2)/6-((5*prior_mean+means[idx[0]])/6)**2).clip(1e-8))
                assert np.allclose(first,(means[idx[0]]-mu[0])/sd[0])
            x[idx]=(raw[idx]-mu[:,None,:])/sd[:,None,:]
            state['sessions'].append({'subject':int(s),'session':int(session),'mu':mu,'sd':sd})
    scaler=StandardScaler().fit(x[d['train']].reshape(-1,310))
    x=scaler.transform(x.reshape(-1,310)).reshape(d['X'].shape).astype('float32')
    assert np.isfinite(x).all();state['training_scaler']=scaler
    return x,state

@torch.no_grad()
def predict(model,x,method):
    model.eval();out=[]
    for start in range(0,len(x),128):out.append(model(torch.from_numpy(view(x[start:start+128],method)).to(DEVICE)).softmax(-1).cpu().numpy())
    return np.concatenate(out)

def run(d,fold,mode,method):
    folder=OUT/f'fold{fold+1}_{mode}_{method}';folder.mkdir(parents=True,exist_ok=True)
    cfg={'fold':fold,'mode':mode,'method':method,'source_sha256':sha(__file__),'data_sha256':sha(CACHE),
         'model_source_sha256':sha(ROOT/('external/EmT/EmT_Cls.py' if method=='emt' else 'scripts/run_preliminary.py')),
         'strict_DG':False,'target_future_statistics':mode=='offline','target_history_statistics':mode=='causal',
         'upstream_LDS_offline':True,'seed':42+fold,'updates':1500,'batch_size':24,'lr':.001,'wd':.01,'dropout':.5,
         'optimizer':'AdamW','scheduler':'cosine','device':DEVICE,'selection':'subject mean window validation BA every100updates',
         'normalization':'per-session raw ten-frame mean moments; causal prior5 from training subjects only; train fitted scaler',
         'test_evaluated':False}
    if (folder/'result.json').exists():assert json.loads((folder/'config.json').read_text())==cfg;return
    write(folder/'config.json',cfg);x,state=transform(d,mode);joblib.dump(state,folder/'normalizer.joblib')
    seed_all(42+fold);torch.set_num_threads(2);model=build_model(method).to(DEVICE)
    schedule=np.load(ROOT/f'outputs/seed_features_v1/fold{fold+1}_{method}/training_indices.npy')
    assert set(schedule.ravel()).issubset(set(d['train']));np.save(folder/'training_indices.npy',schedule)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1500)
    best=-1;history=[];start=time.perf_counter()
    for step,index in enumerate(schedule,1):
        model.train();opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(torch.from_numpy(view(x[index],method)).to(DEVICE)),torch.from_numpy(d['y'][index]).long().to(DEVICE))
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step()
        if step%100==0:
            p=predict(model,x[d['val']],method);m=score(d,p);value=m['window']['subject_mean_BA']
            history.append({'step':step,'train_loss_last_batch':float(loss.detach()),'val_BA':value,
                            'video_BA':m['video_trial']['subject_mean_BA']});write(folder/'history.json',history)
            if value>best:
                best=value;torch.save(model.state_dict(),folder/'model.pt');persist(folder,d,p)
                write(folder/'best_validation.json',{'step':step,'metrics':m})
            print(json.dumps({'run':folder.name,**history[-1]}),flush=True)
    model.load_state_dict(torch.load(folder/'model.pt',map_location=DEVICE,weights_only=True),strict=True)
    p=predict(model,x[d['val']],method);err=float(np.max(np.abs(np.load(folder/'validation.npz')['scores']-p)));assert err<1e-5
    write(folder/'result.json',{'metrics':json.loads((folder/'best_validation.json').read_text())['metrics'],
        'seconds':time.perf_counter()-start,'reload_max_abs_error':err,'test_evaluated':False})

def main():
    OUT.mkdir(parents=True,exist_ok=True);snapshot=OUT/'source_snapshot.py'
    if snapshot.exists():assert snapshot.read_bytes()==Path(__file__).read_bytes()
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    for fold in range(3):
        d=load_fold(fold)
        for mode in ['causal','offline']:
            for method in ['tcn','emt']:
                write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'mode':mode,'method':method});run(d,fold,mode,method)
    rows=[];audit=[]
    for folder in sorted(OUT.glob('fold*')):
        cfg=json.loads((folder/'config.json').read_text());r=json.loads((folder/'result.json').read_text());d=load_fold(cfg['fold'])
        p=np.load(folder/'validation.npz');m=score(d,p['scores']);assert abs(m['window']['subject_mean_BA']-r['metrics']['window']['subject_mean_BA'])<1e-12
        rows.append({'fold':cfg['fold']+1,'mode':cfg['mode'],'method':cfg['method'],'BA':m['window']['subject_mean_BA'],
                     'video_BA':m['video_trial']['subject_mean_BA'],'test_evaluated':False})
        audit.append({'run':folder.name,'reload_error':r['reload_max_abs_error'],'passed':True})
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'mode':mode,'method':method,'mean_BA':float(np.mean([r['BA'] for r in rows if (r['mode'],r['method'])==(mode,method)])),
              'mean_video_BA':float(np.mean([r['video_BA'] for r in rows if (r['mode'],r['method'])==(mode,method)]))}
             for mode in ['causal','offline'] for method in ['tcn','emt']]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audit)
    write(OUT/'queue_status.json',{'status':'finished','fits':12,'prediction_audits':12,'checkpoint_audits':12,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
