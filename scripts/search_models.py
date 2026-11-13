"""Validation-only model search, adaptive refinement and seeded confirmation.

Public development data. Model selection never evaluates test predictions. A
separate finalize command evaluates only the selected configuration/ensemble.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import joblib
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/CBraMod'))
from models.cbramod import CBraMod

VERSION='search-v1'
OUT=ROOT/'outputs/search_v1'
SOURCE_BYTES=Path(__file__).read_bytes()
SOURCE_SHA=hashlib.sha256(SOURCE_BYTES).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def key(config):return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def load_task(task):
    name='BNCI2014_001_T_pilot' if task=='mi' else 'emotion_10s' if task=='emotion' else task
    path=ROOT/f'data/processed/{name}.npz'
    data=dict(np.load(path,allow_pickle=False));meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    data['meta']=meta
    data['sha256']=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    data['classes']=len(np.unique(data['y']))
    data['idx']={part:np.flatnonzero(np.isin(data['subject'],ids)) for part,ids in meta['split'].items()}
    assert len(set(data['trial_id']))==len(data['trial_id']) and np.isfinite(data['X']).all()
    all_idx=np.concatenate(list(data['idx'].values()));assert sorted(all_idx.tolist())==list(range(len(data['y'])))
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not set(data['subject'][data['idx'][a]])&set(data['subject'][data['idx'][b]])
    for idx in data['idx'].values():assert set(data['y'][idx])==set(range(data['classes']))
    return data


def metrics(y,pred,subject,classes):
    per_subject={str(s):float(balanced_accuracy_score(y[subject==s],pred[subject==s])) for s in np.unique(subject)}
    return {'subject_mean_BA':float(np.mean(list(per_subject.values()))),
            'pooled_BA':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'per_subject_BA':per_subject,'confusion_matrix':confusion_matrix(y,pred,labels=range(classes)).tolist()}


class EEGNet(nn.Module):
    """Local PyTorch EEGNet-style implementation; not the author's Keras checkpoint."""
    def __init__(self,channels,classes,filters=8,kernel=64,dropout=.5):
        super().__init__();f=filters
        self.features=nn.Sequential(
            nn.Conv2d(1,f,(1,kernel),padding='same',bias=False),nn.BatchNorm2d(f),
            nn.Conv2d(f,2*f,(channels,1),groups=f,bias=False),nn.BatchNorm2d(2*f),nn.ELU(),
            nn.AvgPool2d((1,4)),nn.Dropout(dropout),
            nn.Conv2d(2*f,2*f,(1,16),padding='same',groups=2*f,bias=False),
            nn.Conv2d(2*f,2*f,1,bias=False),nn.BatchNorm2d(2*f),nn.ELU(),
            nn.AvgPool2d((1,8)),nn.Dropout(dropout),nn.AdaptiveAvgPool2d((1,4)))
        self.head=nn.Linear(2*f*4,classes)
    def forward(self,x):return self.head(self.features(x[:,None]).flatten(1))


class CBraClassifier(nn.Module):
    def __init__(self,patches,classes,dropout=.5,channels=1,readout='mean'):
        super().__init__();self.backbone=CBraMod()
        self.backbone.load_state_dict(torch.load(ROOT/'checkpoints/cbramod.pth',weights_only=True,map_location='cpu'),strict=True)
        self.backbone.proj_out=nn.Identity()
        self.readout=readout
        dim=patches*200*(channels if readout=='flatten' else 1)
        self.head=nn.Sequential(nn.Flatten(),nn.LayerNorm(dim),nn.Dropout(dropout),nn.Linear(dim,classes))
    def forward(self,x):
        x=x.reshape(x.shape[0],x.shape[1],-1,200)
        features=self.backbone(x)
        return self.head(features if self.readout=='flatten' else features.mean(dim=1))


def normalize_signal(x,policy):
    if policy=='trial':return (x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)
    if policy=='scale':return x/100.
    raise ValueError(policy)


@torch.no_grad()
def neural_scores(model,loader):
    model.eval();scores=[]
    for (x,) in loader:scores.append(model(x.cuda()).softmax(-1).cpu().numpy())
    return np.concatenate(scores)


def neural(config,data,seed,folder,epochs):
    seed_all(seed)
    model=(CBraClassifier(data['X'].shape[-1]//200,data['classes'],config['dropout'],data['X'].shape[1],config.get('readout','mean')) if config['model']=='cbramod_ft'
           else EEGNet(data['X'].shape[1],data['classes'],config['filters'],config['kernel'],config['dropout'])).cuda()
    X=normalize_signal(data['X'],config['norm']).astype('float32')
    idx=data['idx'];train=idx['train'];val=idx['val'];batch=config.get('batch',32)
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(X[train]),torch.from_numpy(data['y'][train])),batch_size=batch,
                      shuffle=True,generator=gen,num_workers=0)
    validation=DataLoader(TensorDataset(torch.from_numpy(X[val])),batch_size=batch,num_workers=0)
    if config['model']=='cbramod_ft':
        params=[{'params':model.backbone.parameters(),'lr':config['lr']},{'params':model.head.parameters(),'lr':config['head_lr']}]
    else:params=model.parameters()
    opt=torch.optim.AdamW(params,lr=config['lr'],weight_decay=config['wd'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=config['lr']*.1)
    counts=np.bincount(data['y'][train],minlength=data['classes'])
    class_weight=torch.tensor(len(train)/(data['classes']*counts),device='cuda',dtype=torch.float32)
    best=-1;history=[];best_scores=None;updates=0;stale=0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0
        for x,y in loader:
            opt.zero_grad(set_to_none=True)
            logits=model(x.cuda());loss=nn.functional.cross_entropy(logits,y.cuda(),weight=class_weight,label_smoothing=config.get('smoothing',0.))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            updates+=1;loss_sum+=loss.item()*len(y)
        scheduler.step()
        scores=neural_scores(model,validation)
        m=metrics(data['y'][val],scores.argmax(-1),data['subject'][val],data['classes'])
        score=m['subject_mean_BA']
        history.append({'epoch':epoch,'train_loss':loss_sum/len(train),'val_BA':score,'updates':updates})
        write(folder/'history.json',history)
        if score>best:
            best=score;stale=0;best_scores=scores
            torch.save(model.state_dict(),folder/'model.pt')
        else:stale+=1
        if epoch==1 or epoch%5==0:print(json.dumps({'run':folder.name,'epoch':epoch,'val_BA':score,'best':best}),flush=True)
        if stale>=10:break
    peak=torch.cuda.max_memory_allocated()/2**20
    del model,opt,loader,validation,X;torch.cuda.empty_cache()
    return best_scores,{'epochs_completed':len(history),'best_epoch':max(history,key=lambda x:x['val_BA'])['epoch'],
                        'updates':updates,'peak_allocated_MiB':peak}


def cache_array(data,name,builder):
    path=ROOT/'data/features'/f'{data["sha256"][:16]}_{name}.npy'
    if path.exists():return np.load(path,mmap_mode='r')
    path.parent.mkdir(parents=True,exist_ok=True)
    value=builder().astype('float32');assert np.isfinite(value).all();np.save(path,value)
    return value


def posterior(data,policy):
    names=[str(n).upper() for n in data['channel_names']]
    chosen=['O1','OZ','O2'] if policy=='3' else ['PZ','PO5','PO3','POZ','PO4','PO6','O1','OZ','O2']
    return [names.index(name) for name in chosen]


def spectral(data):
    x=data['X'][:,posterior(data,'9')]
    x=x-x.mean(-1,keepdims=True)
    spec=np.fft.rfft(x,axis=-1,norm='forward');freq=np.fft.rfftfreq(x.shape[-1],1/200)
    spec=spec[:,:,(freq>=6)&(freq<=90)]
    return np.concatenate([spec.real,spec.imag,np.log(np.abs(spec).clip(1e-8))],axis=-1).reshape(len(x),-1)


def bandpower(data,mode='absolute'):
    bands=[(.5,4),(4,8),(8,13),(13,30),(30,45)]
    power=np.stack([np.var(sosfiltfilt(butter(4,[lo,hi],fs=200,btype='bandpass',output='sos'),
                                       data['X'],axis=-1),axis=-1).clip(1e-12) for lo,hi in bands],axis=1)
    if mode in ['relative','relative_centered']:power=power/power.sum(1,keepdims=True).clip(1e-12)
    features=np.log(power.clip(1e-12))
    if mode in ['spatial_centered','relative_centered']:features-=features.mean(-1,keepdims=True)
    if mode not in ['absolute','relative','spatial_centered','relative_centered']:raise ValueError(mode)
    return features.reshape(len(power),-1)


def power_features(data,config):
    mode=config.get('power_norm','absolute')
    name='log_bandpower_v1' if mode=='absolute' else f'log_bandpower_{mode}_v1'
    return cache_array(data,name,lambda:bandpower(data,mode))


@torch.no_grad()
def frozen(data,readout='mean'):
    model=CBraMod().cuda().eval()
    model.load_state_dict(torch.load(ROOT/'checkpoints/cbramod.pth',weights_only=True,map_location='cpu'),strict=True)
    model.proj_out=nn.Identity();result=[]
    for start in range(0,len(data['X']),32):
        x=torch.from_numpy(data['X'][start:start+32]/100).cuda()
        feature=model(x.reshape(x.shape[0],x.shape[1],-1,200))
        feature=(feature if readout=='flatten' else feature.mean(1)).flatten(1)
        result.append(feature.cpu().numpy())
    del model;torch.cuda.empty_cache()
    return np.concatenate(result)


def csp_features(data,config):
    train=data['idx']['train'];labels=data['y'][train];features=[];models=[]
    bands=[(8,30)] if config['bands']=='single' else [(4,8),(8,12),(12,20),(20,30),(30,40)]
    for lo,hi in bands:
        x=sosfiltfilt(butter(4,[lo,hi],fs=200,btype='bandpass',output='sos'),data['X'],axis=-1)
        xt=x[train];cov=xt@xt.transpose(0,2,1);cov/=np.trace(cov,axis1=1,axis2=2)[:,None,None].clip(1e-20)
        shr=config['shrink'];cov=(1-shr)*cov+shr*np.eye(x.shape[1])[None]/x.shape[1]
        total=cov.mean(0);filters=[]
        for cls in range(data['classes']):
            _,vectors=eigh(cov[labels==cls].mean(0),total)
            n=config['components']//2;filters.extend([vectors[:,:n].T,vectors[:,-n:].T])
        filters=np.concatenate(filters);z=np.einsum('fc,nct->nft',filters,x,optimize=True)
        power=np.var(z,axis=-1);power/=power.sum(-1,keepdims=True).clip(1e-20)
        features.append(np.log(power.clip(1e-20)));models.append({'band':[lo,hi],'filters':filters})
    return np.concatenate(features,axis=-1),models


@torch.no_grad()
def cca_bank(data,harmonics,channels):
    x=data['X'][:,posterior(data,channels)];n=x.shape[-1];t=np.arange(n)/200
    refs=[]
    for f in data['meta']['frequencies']:
        ref=np.stack([fun(2*np.pi*f*h*t) for h in range(1,harmonics+1) for fun in [np.sin,np.cos]],axis=-1)
        ref-=ref.mean(0,keepdims=True);refs.append(ref)
    qref=torch.linalg.qr(torch.tensor(np.stack(refs),device='cuda',dtype=torch.float32),mode='reduced').Q
    banks=[]
    for lower in [6,14,22,30,38]:
        signal=sosfiltfilt(butter(4,[lower,90],fs=200,btype='bandpass',output='sos'),x,axis=-1).astype('float32')
        scores=[]
        for start in range(0,len(signal),64):
            batch=torch.from_numpy(signal[start:start+64].copy()).cuda().transpose(1,2)
            batch-=batch.mean(1,keepdim=True)
            qx=torch.linalg.qr(batch,mode='reduced').Q
            cross=torch.einsum('ntc,ktr->nkcr',qx,qref)
            scores.append(torch.linalg.svdvals(cross)[...,0].square().cpu().numpy())
        banks.append(np.concatenate(scores))
    return np.stack(banks,axis=1)


def classical(config,data,seed,folder):
    train=data['idx']['train'];val=data['idx']['val'];model=config['model']
    if model=='fbcca':
        bank=cache_array(data,f'cca_h{config["harmonics"]}_c{config["channels"]}',
                         lambda:cca_bank(data,config['harmonics'],config['channels']))
        weights=np.arange(1,config['banks']+1,dtype=float)**(-config['exponent'])+.25
        scores=np.einsum('nbk,b->nk',bank[:,:config['banks']],weights)
        joblib.dump({'config':config,'weights':weights},folder/'model.joblib')
        return scores[val],{'deterministic':True}
    if model=='fbcsp':
        features,csp=csp_features(data,config);joblib.dump(csp,folder/'csp.joblib')
    elif model.startswith('cbramod_'):
        readout=config.get('readout','mean')
        features=cache_array(data,'cbramod_flatten_v1' if readout=='flatten' else 'cbramod_channelmean_v1',lambda:frozen(data,readout))
    elif model.startswith('spectral_'):features=cache_array(data,'complex_spectrum9_v1',lambda:spectral(data))
    else:features=power_features(data,config)
    if model.endswith('ridge'):
        estimator=RidgeClassifier(alpha=config['alpha'],solver='lsqr',class_weight='balanced')
    elif model.endswith('svm'):
        estimator=SVC(C=config['C'],gamma=config['gamma'],class_weight='balanced')
    elif model.endswith('logistic'):
        estimator=LogisticRegression(C=config['C'],max_iter=1000,class_weight='balanced',random_state=seed)
    else:estimator=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
    pipeline=make_pipeline(StandardScaler(),estimator)
    pipeline.fit(features[train],data['y'][train]);joblib.dump(pipeline,folder/'model.joblib')
    scores=pipeline.decision_function(features[val])
    if scores.ndim==1:scores=np.stack([-scores,scores],axis=-1)
    # lbfgs LogisticRegression, SVC, LDA and LSQR Ridge have no stochastic fitting here.
    return scores,{'deterministic':True}


def execute(task,data,config,seed,epochs,stage):
    identity={'version':VERSION,'task':task,'data_sha256':data['sha256'],'config':config,'seed':seed,'epochs':epochs}
    folder=OUT/task/'runs'/key(identity)
    if (folder/'result.json').exists():
        previous=json.loads((folder/'config.json').read_text())
        assert previous['split']==data['meta']['split'], 'Cached run uses a different subject split'
        assert previous['checkpoint_sha256']==hashlib.file_digest((ROOT/'checkpoints/cbramod.pth').open('rb'),'sha256').hexdigest(), 'Checkpoint changed; create a new study/cache version'
        cached=json.loads((folder/'result.json').read_text())
        if config['model'] not in ['eegnet','cbramod_ft'] and cached.get('deterministic') is not True:
            cached['deterministic']=True
            cached['metadata_correction']='Deterministic estimator (including lbfgs logistic); predictions/model unchanged'
            write(folder/'result.json',cached)
        return cached
    folder.mkdir(parents=True,exist_ok=True)
    snapshot=OUT/'source_versions'/f'search_models_{SOURCE_SHA}.py'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():snapshot.write_bytes(SOURCE_BYTES)
    write(folder/'config.json',{**identity,'stage':stage,'split':data['meta']['split'],
                              'code_sha256':SOURCE_SHA,'source_snapshot':str(snapshot.relative_to(ROOT)),
                              'selection_metric':'validation equal-subject mean balanced accuracy',
                              'test_evaluated':False,'torch':torch.__version__,'numpy':np.__version__,
                              'checkpoint_sha256':hashlib.file_digest((ROOT/'checkpoints/cbramod.pth').open('rb'),'sha256').hexdigest()})
    begin=time.perf_counter();seed_all(seed)
    if config['model'] in ['eegnet','cbramod_ft']:scores,extra=neural(config,data,seed,folder,epochs)
    else:scores,extra=classical(config,data,seed,folder)
    val=data['idx']['val'];m=metrics(data['y'][val],scores.argmax(-1),data['subject'][val],data['classes'])
    np.savez_compressed(folder/'validation_predictions.npz',trial_id=data['trial_id'][val],y=data['y'][val],scores=scores,subject=data['subject'][val])
    result={'id':folder.name,'task':task,'config':config,'seed':seed,'budget_epochs':epochs,'stage':stage,
            'val_BA':m['subject_mean_BA'],'val_metrics':m,'seconds':time.perf_counter()-begin,'test_evaluated':False,**extra}
    write(folder/'result.json',result)
    print(json.dumps({'finished':folder.name,'task':task,'model':config['model'],'stage':stage,'val_BA':result['val_BA']}),flush=True)
    return result


def candidates(task):
    configs=[]
    for lr in [1e-5,3e-5]:
        config={'model':'cbramod_ft','lr':lr,'head_lr':3e-4,'wd':.05,'dropout':.5,'norm':'scale'}
        if task!='mi':config['readout']='flatten'
        configs.append(config)
    for filters,drop,lr in [(8,.25,1e-3),(8,.5,1e-3),(16,.5,1e-3),(16,.5,3e-4)]:
        configs.append({'model':'eegnet','filters':filters,'kernel':64,'dropout':drop,'lr':lr,'wd':.01,'norm':'trial'})
    for alpha in [1.,10.,100.,1000.]:configs.append({'model':'cbramod_ridge','alpha':alpha})
    if task=='mi':
        for bands,n in [('single',2),('single',4),('bank',2),('bank',4)]:
            configs.append({'model':'fbcsp','bands':bands,'components':n,'shrink':.1})
        configs.extend([{'model':'bandpower_lda'},{'model':'bandpower_svm','C':1.,'gamma':'scale'}])
    elif task.startswith('ssvep'):
        for channels,harmonics,banks in [('3',3,1),('9',3,1),('3',5,3),('9',3,3),('9',5,3),('9',5,5)]:
            configs.append({'model':'fbcca','channels':channels,'harmonics':harmonics,'banks':banks,'exponent':1.25})
        for alpha in [1.,100.,10000.]:configs.append({'model':'spectral_ridge','alpha':alpha})
    else:
        configs.append({'model':'bandpower_lda'})
        for c in [.01,.1,1.,10.]:configs.append({'model':'bandpower_svm','C':c,'gamma':'scale'})
        for c in [.01,.1,1.]:configs.append({'model':'bandpower_logistic','C':c})
    return configs


def refine(config):
    variants=[];model=config['model']
    if model in ['cbramod_ft','eegnet']:
        for factor in [.3,3.]:
            variant=copy.deepcopy(config);variant['lr']*=factor;variants.append(variant)
        variant=copy.deepcopy(config);variant['wd']=.1;variant['smoothing']=.05;variants.append(variant)
    elif model.endswith('ridge'):
        for factor in [.1,.3,3.,10.]:variants.append({**config,'alpha':config['alpha']*factor})
    elif model=='fbcca':
        for exponent in [.5,1.,2.]:variants.append({**config,'exponent':exponent})
        variants.append({**config,'banks':5 if config['banks']!=5 else 3})
    elif model=='fbcsp':
        for shrink in [.01,.3,.5]:variants.append({**config,'shrink':shrink})
    elif model.endswith('svm'):
        for factor in [.3,3.]:variants.append({**config,'C':config['C']*factor})
        for gamma in [.001,.01,.1]:variants.append({**config,'gamma':gamma})
    elif model.endswith('logistic'):
        for factor in [.3,3.]:variants.append({**config,'C':config['C']*factor})
    return variants


def search(args):
    data=load_task(args.task);folder=OUT/args.task
    plan={'scope':'PUBLIC_DATA_CANDIDATE_SEARCH','coarse_configs':candidates(args.task),'coarse_epochs':args.coarse_epochs,
          'refine_epochs':args.refine_epochs,'screen_seed':42,'confirmation_seeds':[42,43,44],
          'rule':'coarse -> refine top 2 distinct model families -> confirm top 3 configs; rank mean validation subject-BA',
          'test_policy':'no test metrics during search; finalize selected champion only',
          'data_sha256':data['sha256'],'data_metadata':data['meta']}
    if (folder/'plan.json').exists():assert json.loads((folder/'plan.json').read_text())==plan,'Search protocol changed; use new output/version'
    write(folder/'plan.json',plan)
    rows=[]
    for config in plan['coarse_configs']:
        if getattr(args,'classical_only',False) and (config['model'] in ['eegnet','cbramod_ft','fbcca'] or config['model'].startswith('cbramod_')):
            continue
        rows.append(execute(args.task,data,config,42,args.coarse_epochs,'coarse'))
        write(folder/'leaderboard.json',sorted(rows,key=lambda r:r['val_BA'],reverse=True))
    if getattr(args,'classical_only',False):
        print(json.dumps({'classical_precompute_complete':args.task,'runs':len(rows),'test_used':False}),flush=True)
        return
    supplement=folder/'feature_supplement_plan.json'
    if supplement.exists():
        for config in json.loads(supplement.read_text())['configs']:
            rows.append(execute(args.task,data,config,42,args.coarse_epochs,'feature_supplement'))
        write(folder/'leaderboard.json',sorted(rows,key=lambda r:r['val_BA'],reverse=True))
    ranked=sorted(rows,key=lambda r:r['val_BA'],reverse=True)
    families=[];bases=[]
    for row in ranked:
        if row['config']['model'] not in families:
            families.append(row['config']['model']);bases.append(row['config'])
        if len(bases)==2:break
    adaptive=[variant for base in bases for variant in [base]+refine(base)]
    unique={key(config):config for config in adaptive}
    write(folder/'refinement_plan.json',{'based_on_run_ids':[r['id'] for r in ranked], 'configs':list(unique.values()),'test_used':False})
    for config in unique.values():
        rows.append(execute(args.task,data,config,42,args.refine_epochs,'refine'))
        write(folder/'leaderboard.json',sorted(rows,key=lambda r:r['val_BA'],reverse=True))
    # Each configuration competes once using its largest budget, not multiple epoch-budget entries.
    latest={}
    for row in rows:
        k=key(row['config'])
        if k not in latest or row['budget_epochs']>=latest[k]['budget_epochs']:latest[k]=row
    finalists=sorted(latest.values(),key=lambda r:r['val_BA'],reverse=True)[:3]
    confirmed=[]
    for finalist in finalists:
        runs=[]
        seeds=[42] if finalist['config']['model'] not in ['eegnet','cbramod_ft'] else [42,43,44]
        for seed in seeds:runs.append(execute(args.task,data,finalist['config'],seed,args.refine_epochs,'confirmation'))
        confirmed.append({'config':finalist['config'],'run_ids':[r['id'] for r in runs],
                          'val_BA_mean':float(np.mean([r['val_BA'] for r in runs])),
                          'val_BA_std':float(np.std([r['val_BA'] for r in runs],ddof=1)) if len(runs)>1 else 0.,
                          'seeds':seeds,'deterministic':len(seeds)==1,'individual_val_BA':[r['val_BA'] for r in runs]})
        write(folder/'confirmation.json',sorted(confirmed,key=lambda r:r['val_BA_mean'],reverse=True))
    confirmed.sort(key=lambda r:r['val_BA_mean'],reverse=True)
    write(folder/'champion.json',{'task':args.task,'selection':'highest mean validation subject BA; no test selection',
                                'tested_configurations':len(latest),'test_evaluated':False,**confirmed[0]})
    print(json.dumps({'champion':args.task,**confirmed[0]}),flush=True)


def test_scores(data,config,run):
    idx=data['idx']['test'];folder=OUT/run['task']/'runs'/run['id'];model=config['model']
    if model in ['eegnet','cbramod_ft']:
        m=(CBraClassifier(data['X'].shape[-1]//200,data['classes'],config['dropout'],data['X'].shape[1],config.get('readout','mean')) if model=='cbramod_ft'
           else EEGNet(data['X'].shape[1],data['classes'],config['filters'],config['kernel'],config['dropout'])).cuda()
        m.load_state_dict(torch.load(folder/'model.pt',weights_only=True,map_location='cuda'),strict=True)
        x=normalize_signal(data['X'][idx],config['norm']).astype('float32')
        scores=neural_scores(m,DataLoader(TensorDataset(torch.from_numpy(x)),batch_size=32));del m;torch.cuda.empty_cache();return scores
    if model=='fbcca':
        bank=cache_array(data,f'cca_h{config["harmonics"]}_c{config["channels"]}',lambda:cca_bank(data,config['harmonics'],config['channels']))
        return np.einsum('nbk,b->nk',bank[idx,:config['banks']],np.arange(1,config['banks']+1)**(-config['exponent'])+.25)
    if model=='fbcsp':
        pieces=[]
        for band in joblib.load(folder/'csp.joblib'):
            x=sosfiltfilt(butter(4,band['band'],fs=200,btype='bandpass',output='sos'),data['X'][idx],axis=-1)
            z=np.einsum('fc,nct->nft',band['filters'],x,optimize=True);power=np.var(z,axis=-1)
            pieces.append(np.log((power/power.sum(-1,keepdims=True).clip(1e-20)).clip(1e-20)))
        feat=np.concatenate(pieces,axis=-1)
    elif model.startswith('cbramod_'):
        readout=config.get('readout','mean')
        feat=cache_array(data,'cbramod_flatten_v1' if readout=='flatten' else 'cbramod_channelmean_v1',lambda:frozen(data,readout))[idx]
    elif model.startswith('spectral_'):feat=cache_array(data,'complex_spectrum9_v1',lambda:spectral(data))[idx]
    else:feat=power_features(data,config)[idx]
    scores=joblib.load(folder/'model.joblib').decision_function(feat)
    return np.stack([-scores,scores],axis=-1) if scores.ndim==1 else scores


def finalize(args):
    folder=OUT/args.task
    if (folder/'final_test.json').exists():
        previous=json.loads((folder/'final_test.json').read_text())
        current=json.loads((folder/'champion.json').read_text())
        assert previous['champion']['config']==current['config'] and previous['champion']['run_ids']==current['run_ids'], 'Test already evaluated for a different champion; do not silently retest'
        print(json.dumps({'task':args.task,'model':current['config']['model'],
                          'test_BA':previous['test_metrics']['subject_mean_BA'],'existing_result':True}));return
    champion=json.loads((folder/'champion.json').read_text());data=load_task(args.task);seed_all(42)
    run_results=[];probabilities=[]
    for run_id in champion['run_ids']:
        run=json.loads((folder/'runs'/run_id/'result.json').read_text())
        score=test_scores(data,champion['config'],run)
        # Neural outputs are probabilities. Deterministic/classical single runs retain raw decision scores.
        probabilities.append(score)
        idx=data['idx']['test'];m=metrics(data['y'][idx],score.argmax(-1),data['subject'][idx],data['classes'])
        run_results.append({'run_id':run_id,'seed':run['seed'],'test_metrics':m})
    score=np.mean(probabilities,axis=0);idx=data['idx']['test']
    result={'scope':'PUBLIC_DATA_CANDIDATE_BEST_NOT_GLOBAL_OR_COMPETITION','task':args.task,'champion':champion,
            'test_metrics':metrics(data['y'][idx],score.argmax(-1),data['subject'][idx],data['classes']),
            'individual_runs':run_results,'ensemble':len(probabilities)>1,
            'test_previously_used':args.task=='mi','test_subjects':data['meta']['split']['test'],
            'limitation':'Fixed subject split; selected among tested candidates. Training seeds do not measure subject-split variation.'}
    np.savez_compressed(folder/'final_test_predictions.npz',scores=score,y=data['y'][idx],subject=data['subject'][idx],trial_id=data['trial_id'][idx])
    write(folder/'final_test.json',result)
    print(json.dumps({'task':args.task,'model':champion['config']['model'],
                      'test_BA':result['test_metrics']['subject_mean_BA'],
                      'test_macro_f1':result['test_metrics']['macro_f1'],
                      'saved':str(folder/'final_test.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['search','finalize']);p.add_argument('task',choices=['mi','ssvep_1s','ssvep_2s','emotion'])
    p.add_argument('--coarse-epochs',type=int,default=20);p.add_argument('--refine-epochs',type=int,default=40)
    p.add_argument('--classical-only',action='store_true',help='Precompute CPU-only coarse candidates; do not select a champion')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Use CUDA-enabled .venv/Scripts/python.exe')
    with threadpool_limits(limits=4):search(args) if args.command=='search' else finalize(args)
