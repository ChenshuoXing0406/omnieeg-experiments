"""Real MIRepNet source-only and offline-EA controls on paired MI development folds."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import joblib
import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist
import torch
from torch import nn
from run_preliminary import load_data,sha,PLAN,save_predictions
from run_mi_domain import balanced_schedule
from search_models import seed_all,metrics,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mirepnet_search_v1'
sys.path.insert(0,str(ROOT/'external/mi_adapter_deps'))
SETTINGS=[('pretrained_source_lr1e4',True,'source',1e-4),('pretrained_source_lr5e4',True,'source',5e-4),
          ('random_source_lr5e4',False,'source',5e-4),('pretrained_offlineEA_lr1e4',True,'offline_subject_EA',1e-4)]
WEIGHTS=ROOT/'external/MIRepNet/weight/MIRepNet.pth'

def log(**k):print(json.dumps(k,allow_nan=False),flush=True)

def model_class():
    p=ROOT/'external/MIRepNet/model/mlm.py';tree=ast.parse(p.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not(isinstance(n,ast.Import) and any(a.name=='wandb' for a in n.names))]
    module=types.ModuleType('mirepnet_unmodified_numeric_model');exec(compile(tree,str(p),'exec'),module.__dict__)
    return module.mlm_mask

def build(pretrained):
    model=model_class()(emb_size=256,depth=6,n_classes=4,pretrain=None)
    if pretrained:
        weights=torch.load(WEIGHTS,weights_only=True,map_location='cpu');state=model.state_dict()
        loaded={k:weights[k] for k in state if not k.startswith('clshead.')}
        assert all(v.shape==state[k].shape for k,v in loaded.items())
        missing=model.load_state_dict(loaded,strict=False)
        assert sorted(missing.missing_keys)==['clshead.bias','clshead.weight'] and not missing.unexpected_keys
    return model

def whitening(x):
    x=x.astype('float64');x-=x.mean(-1,keepdims=True)
    cov=np.einsum('nct,ndt->cd',x,x)/(len(x)*(x.shape[-1]-1))
    cov=.99*cov+.01*np.trace(cov)/len(cov)*np.eye(len(cov))
    v,u=np.linalg.eigh(cov);assert v.min()>0
    return (u*v**-.5)@u.T

def transform(d,mode):
    spec=importlib.util.spec_from_file_location('mirepnet_names',ROOT/'external/MIRepNet/utils/channel_list.py')
    ch=importlib.util.module_from_spec(spec);spec.loader.exec_module(ch)
    names=[n.upper() for n in d['channel_names']];target=ch.use_channels_names
    distance=cdist([ch.channel_positions[n] for n in target],[ch.channel_positions[n] for n in names])
    mapping=1/(distance+1e-6);mapping/=mapping.sum(-1,keepdims=True)
    for i,n in enumerate(target):
        if n in names:mapping[i]=np.eye(len(names))[names.index(n)]
    x=np.empty((len(d['X']),45,1000),dtype='float32');references={}
    if mode=='source':
        ref=whitening(d['X'][d['train']]);references['training_only']=ref
        x[:]=resample_poly(mapping@ref@d['X'],5,4,axis=-1)
    else:
        for subject in np.unique(d['subject']):
            idx=np.flatnonzero(d['subject']==subject);ref=whitening(d['X'][idx]);references[str(subject)]=ref
            x[idx]=resample_poly(mapping@ref@d['X'][idx],5,4,axis=-1)
    assert np.isfinite(x).all()
    return x,{'mapping':mapping,'references':references,'mode':mode,'resampling':'200->250 Hz; duration 4 seconds preserved'}

@torch.no_grad()
def predict(model,x):
    model.eval();p=[]
    for start in range(0,len(x),32):p.append(model(torch.from_numpy(x[start:start+32]).cuda())[1].softmax(-1).cpu().numpy())
    return np.concatenate(p)

def run(d,fold,setting):
    name,pretrained,mode,lr=setting;folder=OUT/f'fold{fold+1}_{name}';seed=42+fold
    cfg={'name':name,'fold':fold,'seed':seed,'pretrained':pretrained,'alignment':mode,'bone_lr':lr,'head_lr':.001,
         'updates':1500,'batch_size':16,'dropout':.5,'weight_decay':.01,'optimizer':'AdamW','scheduler':'cosine',
         'data_sha256':d['source_sha256'],'source_sha256':sha(__file__),'model_sha256':sha(ROOT/'external/MIRepNet/model/mlm.py'),
         'weight_sha256':sha(WEIGHTS),'split':d['split'],'test_evaluated':False,'strict_DG':mode=='source',
         'sampling':'same balanced source subjects/classes as mi_domain_v1; 4 subjects x 4 classes',
         'scope':'development only; not exact author calibration protocol; random control has equal short update budget'}
    if (folder/'result.json').exists():assert json.loads((folder/'config.json').read_text())==cfg;return
    folder.mkdir(parents=True,exist_ok=True);write(folder/'config.json',cfg)
    x,adapter=transform(d,mode);joblib.dump(adapter,folder/'adapter.joblib')
    seed_all(seed);model=build(pretrained).cuda()
    # Both initialization controls retain identical randomly initialized task head.
    head_hash=__import__('hashlib').sha256(model.clshead.weight.detach().cpu().numpy().tobytes()).hexdigest()
    opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if not n.startswith('clshead.')],'lr':lr},
                           {'params':model.clshead.parameters(),'lr':.001}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1500)
    schedule,_=balanced_schedule(d,seed);np.save(folder/'training_indices.npy',schedule)
    best=-1.;history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step,index in enumerate(schedule,1):
        model.train();opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(model(torch.from_numpy(x[index]).cuda())[1],torch.from_numpy(d['y'][index]).long().cuda())
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step()
        if step%100==0:
            p=predict(model,x[d['val']]);m=metrics(d['y'][d['val']],p.argmax(-1),d['subject'][d['val']],4)
            row={'step':step,'loss_last_batch':float(loss.detach()),'val_BA':m['subject_mean_BA']}
            history.append(row);write(folder/'history.json',history)
            if row['val_BA']>best:
                best=row['val_BA'];torch.save(model.state_dict(),folder/'model.pt');save_predictions(folder,{'mi':d},{'mi':p})
                write(folder/'best_validation.json',{'step':step,'metrics':m})
            log(run=folder.name,**row)
    model.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cuda'),strict=True)
    p=predict(model,x[d['val']]);saved=np.load(folder/'mi_validation.npz')['scores'];err=float(np.max(np.abs(saved-p)));assert err<1e-5
    write(folder/'result.json',{'metrics':json.loads((folder/'best_validation.json').read_text())['metrics'],
        'seconds':time.perf_counter()-start,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'head_initial_sha256':head_hash,'checkpoint_max_abs_error':err,'test_evaluated':False})
    del model,opt,x;torch.cuda.empty_cache()

def export():
    rows=[];summaries=[]
    for folder in sorted(OUT.glob('fold*')):
        if not (folder/'result.json').exists():continue
        cfg=json.loads((folder/'config.json').read_text());r=json.loads((folder/'result.json').read_text())
        with np.load(folder/'mi_validation.npz') as p:
            assert not set(p['subject'])&{8,9};m=metrics(p['y'],p['scores'].argmax(-1),p['subject'],4)
            assert abs(m['subject_mean_BA']-r['metrics']['subject_mean_BA'])<1e-12
        rows.append({'name':cfg['name'],'fold':cfg['fold']+1,'val_BA':m['subject_mean_BA'],'macro_F1':m['macro_f1'],
                     'strict_DG':cfg['strict_DG'],'seconds':r['seconds'],'test_evaluated':False})
    if not rows:return
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name in sorted({r['name'] for r in rows}):
        a=[r for r in rows if r['name']==name];summaries.append({'name':name,'folds':len(a),'mean_BA':float(np.mean([r['val_BA'] for r in a])),
            'strict_DG':a[0]['strict_DG']})
    write(OUT/'summary.json',summaries);log(summary=summaries)

def main():
    OUT.mkdir(parents=True,exist_ok=True);snapshot=OUT/'source_snapshot.py'
    if snapshot.exists():assert snapshot.read_bytes()==Path(__file__).read_bytes()
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    for fold in range(3):
        d=load_data('mi',fold)
        for setting in SETTINGS:
            write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'setting':setting[0]});run(d,fold,setting)
        export()
    # Confirm matched head initialization and draw sequence for pretrained/random controls.
    for fold in range(3):
        a=OUT/f'fold{fold+1}_pretrained_source_lr5e4';b=OUT/f'fold{fold+1}_random_source_lr5e4'
        assert json.loads((a/'result.json').read_text())['head_initial_sha256']==json.loads((b/'result.json').read_text())['head_initial_sha256']
        assert np.array_equal(np.load(a/'training_indices.npy'),np.load(b/'training_indices.npy'))
    write(OUT/'queue_status.json',{'status':'finished','fits':12,'prediction_audits':12,'checkpoint_audits':12,'test_evaluated':False})

if __name__=='__main__':main()
