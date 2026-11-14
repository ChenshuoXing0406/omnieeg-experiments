"""Fixed independent SSVEP development baseline; held-out subjects 11/12 never loaded."""
import argparse,csv,hashlib,json,re
from pathlib import Path
import mne
import numpy as np
from scipy.signal import butter,sosfiltfilt,resample_poly
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits

RUN_COUNTS=[2,2,2,2,2,2,3,2,2,4]
EVENT_LABEL={2:0,4:1,3:2,1:3}
FREQS=[13.,17.,21.]

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def ba(y,p,s):
    per={str(k):float(balanced_accuracy_score(y[s==k],p[s==k])) for k in np.unique(s)}
    return {'subject_mean_BA':float(np.mean(list(per.values()))),'pooled_BA':float(balanced_accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'per_subject_BA':per,
            'confusion_matrix':confusion_matrix(y,p,labels=sorted(set(y))).tolist()}

def basis(x):
    x=x-x.mean(axis=0,keepdims=True)
    u,s,_=np.linalg.svd(x,full_matrices=False)
    return u[:,s>s[0]*1e-10]

def main(a):
    assert not (a.output/'status.json').exists(),'Choose a fresh output directory'
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'status.json',{'status':'extracting','test_evaluated':False})
    banks={1:[],2:[]};ys=[];subjects=[];ids=[];sources=[];reference={}
    for seconds in (1,2):
        t=np.arange(seconds*200)/200
        reference[seconds]=[basis(np.stack([fun(2*np.pi*f*h*t) for h in range(1,5) for fun in [np.sin,np.cos]],axis=1)) for f in FREQS]
    for sid,nruns in enumerate(RUN_COUNTS,1):
        count=0
        for run in range(1,nruns+1):
            p=a.source/f'subject{sid:02d}_run{run}_raw.fif';assert p.exists(),p
            raw=mne.io.read_raw_fif(p,preload=True,verbose='ERROR');assert raw.info['sfreq']==256
            events=mne.find_events(raw,stim_channel='STI 014',shortest_event=1,verbose='ERROR')
            picks=mne.pick_types(raw.info,eeg=True,stim=False,exclude=[]);assert len(picks)==8
            x=raw.get_data(picks=picks);assert np.isfinite(x).all()
            x=sosfiltfilt(butter(4,[.3,90],fs=256,btype='bandpass',output='sos'),x,axis=-1)
            x=resample_poly(x,25,32,axis=-1)
            filtered=[sosfiltfilt(butter(4,[lo,90],fs=200,btype='bandpass',output='sos'),x,axis=-1) for lo in [6,14,22,30,38]]
            sources.append({'path':str(p),'sha256':sha(p),'subject':sid,'run':run,'eeg_channels':[raw.ch_names[i] for i in picks],'events':len(events)})
            for e,(sample,_,code) in enumerate(events):
                if code not in EVENT_LABEL:continue
                start=round(((sample-raw.first_samp)/256+2)*200)
                assert start>=0 and start+400<=x.shape[1],(p,e,'incomplete window')
                for seconds in (1,2):
                    result=[]
                    for signal in filtered:
                        q=basis(signal[:,start:start+seconds*200].T)
                        result.append([float(np.linalg.svd(q.T@ref,compute_uv=False)[0]**2) for ref in reference[seconds]])
                    banks[seconds].append(result)
                ys.append(EVENT_LABEL[int(code)]);subjects.append(sid);ids.append(f'S{sid:02d}-R{run}-event{e}');count+=1
        print(json.dumps({'subject':sid,'trials':count}),flush=True)
    y=np.array(ys);s=np.array(subjects);trial=np.array(ids)
    assert len(set(ids))==len(ids) and set(y)=={0,1,2,3}
    write(a.output/'data_manifest.json',{'sources':sources,'trials':len(y),'source_script_sha256':sha(Path(__file__)),'event_label':EVENT_LABEL,'frequencies':FREQS,
        'locked_test_subjects':[11,12],'validation_subjects':[[1,2],[3,4],[5,6]],'input_sfreq':256,'model_sfreq':200,'window_start_seconds':2,'harmonics':4,
        'task_note':'3-class active decoding and source-calibrated 4-class rest rejection are separate tasks','test_evaluated':False})
    rows=[];audits=[]
    for seconds in (1,2):
        bank=np.asarray(banks[seconds]);weights=np.arange(1,6,dtype=float)**(-1 if seconds==1 else -.5)+.25
        for method,scores in [('CCA',bank[:,0]),('FBCCA',np.einsum('nbc,b->nc',bank,weights))]:
            np.savez(a.output/f'{method}_{seconds}s_all_development_scores.npz',scores=scores,y=y,subject=s,trial_id=trial)
            maximum=scores.max(-1);winner=scores.argmax(-1)
            for fold,val_ids in enumerate([[1,2],[3,4],[5,6]],1):
                train=~np.isin(s,val_ids);val=~train
                candidates=np.unique(np.r_[np.nextafter(maximum[train].min(),-np.inf),np.quantile(maximum[train],np.linspace(0,1,51)),np.nextafter(maximum[train].max(),np.inf)])
                best=-1;threshold=None
                for cut in candidates:
                    pred=np.where(maximum[train]<cut,3,winner[train]);v=ba(y[train],pred,s[train])['subject_mean_BA']
                    if v>best:best=v;threshold=float(cut)
                folder=a.output/f'fold{fold}_{method}_{seconds}s';folder.mkdir()
                write(folder/'calibration.json',{'threshold':threshold,'train_subjects':sorted(set(s[train].tolist())),'val_subjects':val_ids,'source_train_BA':best,'candidate_count':len(candidates),'test_evaluated':False})
                for task,idx,pred in [('active3',val&(y!=3),winner),('with_rest4',val,np.where(maximum<threshold,3,winner))]:
                    m=ba(y[idx],pred[idx],s[idx]);name=folder/f'{task}_validation.npz'
                    np.savez(name,scores=scores[idx],y=y[idx],pred=pred[idx],subject=s[idx],trial_id=trial[idx])
                    rows.append({'fold':fold,'method':method,'seconds':seconds,'task':task,'subject_mean_BA':m['subject_mean_BA'],'pooled_BA':m['pooled_BA'],'macro_f1':m['macro_f1']})
                    with np.load(name) as d:check=ba(d['y'],d['pred'],d['subject'])
                    assert check==m
                    audits.append({'fold':fold,'method':method,'seconds':seconds,'task':task,'saved_predictions_recomputed':True,'test_evaluated':False})
    with (a.output/'fold_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'method':m,'seconds':sec,'task':task,'mean_BA':float(np.mean([r['subject_mean_BA'] for r in rows if r['method']==m and r['seconds']==sec and r['task']==task]))} for m in ['CCA','FBCCA'] for sec in [1,2] for task in ['active3','with_rest4']]
    write(a.output/'summary.json',summary);write(a.output/'audit.json',audits)
    write(a.output/'status.json',{'status':'finished','source_threshold_calibrations':12,'validation_records':24,'prediction_audits':24,'neural_fits':0,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=4):main(p.parse_args())
