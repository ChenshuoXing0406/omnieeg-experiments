"""Align MI preprocessing and fixed subject splits to teammate commit ca188b7."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,resample_poly

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'training_20261113'
COMMIT='ca188b799dd3281b3db8e783a361d9518353fe8b'
MI_CHANNELS=['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']

def save(task,x,y,subjects,ids,channels,splits,details):
    d=ROOT/'processed'/task;d.mkdir(parents=True,exist_ok=True)
    assert np.isfinite(x).all() and len(set(ids))==len(ids)
    y=np.array(y);subjects=np.array(subjects);ids=np.array(ids)
    counts={}
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not set(splits[a])&set(splits[b])
    for split,sub in splits.items():
        mask=np.isin(subjects,sub)
        np.save(d/f'{split}_x.npy',x[mask]);np.save(d/f'{split}_y.npy',y[mask]);np.save(d/f'{split}_subject.npy',subjects[mask]);np.save(d/f'{split}_id.npy',ids[mask])
        counts[split]={'subjects':len(sub),'windows':int(mask.sum()),'class_counts':np.bincount(y[mask]).tolist()}
    m={'task':task,'input_shape':list(x.shape[1:]),'num_classes':int(y.max()+1),'channels':channels,'sample_rate':200,'unit':'microvolt /100',
       'splits':splits,'counts':counts,'reference_commit':COMMIT,'test_evaluated':False,**details}
    (d/'manifest.json').write_text(json.dumps(m,indent=2));print('READY',task,counts,flush=True)

def mi():
    if (ROOT/'processed/mi/manifest.json').exists():return
    xs=[];ys=[];subjects=[];ids=[];source=[]
    sos=butter(4,[.3,40],btype='bandpass',fs=250,output='sos')
    for subject in range(1,10):
        p=OLD/f'raw/mi/A{subject:02d}T.mat';runs=loadmat(p,simplify_cells=True)['data'];count=0
        for run_id,run in enumerate(runs):
            labels=np.atleast_1d(run['y']).astype('int64')
            if not labels.size:continue
            assert int(run['fs'])==250
            raw=np.asarray(run['X'][:,:22],dtype='float64').T
            raw-=raw.mean(0,keepdims=True);raw=sosfiltfilt(sos,raw,axis=-1)
            for trial,(start,label) in enumerate(zip(np.atleast_1d(run['trial']).astype('int64')-1,labels)):
                v=resample_poly(raw[:,start+500:start+1500],4,5,axis=-1).astype('float32')/100
                assert v.shape==(22,800)
                xs.append(v.reshape(22,4,200));ys.append(label-1);subjects.append(subject);ids.append(f'A{subject:02d}T-run{run_id}-trial{trial}');count+=1
        assert count==288
        source.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    save('mi',np.stack(xs),ys,subjects,ids,MI_CHANNELS,{'train':[1,2,3,4,5],'val':[6,7],'test':[8,9]},
         {'source':'BNCI2014-001 T only','sources':source,'preprocessing':'EEG-only CAR; continuous-run 0.3-40Hz order4 zero-phase Butterworth; MATLAB index minus1; [2,6)s crop; polyphase250->200Hz; all artifact flags retained',
          'test_history':'Subjects8/9 were evaluated in teammate older runs; no new blind-test claim'})

def ssvep():
    if (ROOT/'processed/ssvep_2s/manifest.json').exists():return
    src=OLD/'processed/ssvep_2s'
    x=np.concatenate([np.load(src/f'{s}_x.npy') for s in ['train','val','test']])
    y=np.concatenate([np.load(src/f'{s}_y.npy') for s in ['train','val','test']])
    subjects=np.concatenate([np.load(src/f'{s}_subject.npy') for s in ['train','val','test']])
    # Verify cached processing against the actual teammate algorithm on a complete subject.
    a=loadmat(OLD/'raw/ssvep/S1.mat')['data'].transpose(2,3,0,1).reshape(240,64,1500)
    a=sosfiltfilt(butter(4,[6,90],fs=250,btype='bandpass',output='sos'),a,axis=-1)
    reference=resample_poly(a[:,:,160:660],4,5,axis=-1).astype('float32').reshape(240,64,2,200)/100
    error=float(np.max(np.abs(reference-x[subjects==1])))
    assert np.allclose(reference,x[subjects==1],rtol=2e-6,atol=2e-6),error
    m=json.loads((src/'manifest.json').read_text());ids=[f'S{s:02d}-target{target}-block{block}' for s in range(1,36) for target in range(40) for block in range(6)]
    assert np.array_equal(subjects,np.repeat(np.arange(1,36),240))
    save('ssvep_2s',x,y,subjects,ids,m['channels'],{'train':list(range(1,6))+list(range(9,30)),'val':[6,30,31,32],'test':[7,8,33,34,35]},
         {'source':'Tsinghua Benchmark','sources':m['sources'],'preprocessing':m['preprocessing'],
          'reference_S1_max_abs_error_scaled':error,'offline_context':True,
          'test_history':'Legacy teammate test; subjects7/8 entered our previous pilot training. Fresh models do not train on these subjects, but this is not a new team-blind test.'})

if __name__=='__main__':mi();ssvep()
