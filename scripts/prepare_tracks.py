"""Create public-data task caches with predetermined subject-disjoint splits."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, resample_poly

ROOT=Path(__file__).resolve().parents[1]


def write_task(name,X,y,subject,trial_id,channel_names,metadata):
    assert X.ndim==3 and X.shape[0]==len(y)==len(subject)==len(trial_id)
    assert X.shape[1]==len(channel_names) and X.shape[2]%200==0 and np.isfinite(X).all()
    assert len(set(trial_id))==len(trial_id)
    assert sorted(set(ys_int:=list(map(int,y))))==list(range(len(set(ys_int)))), 'Class IDs must be contiguous and explicit'
    splits=metadata['split']
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not set(splits[a])&set(splits[b])
    for part in splits:
        idx=np.isin(subject,splits[part]);assert set(np.asarray(y)[idx])==set(y)
    folder=ROOT/'data/processed';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.npz'
    np.savez_compressed(path,X=X.astype('float32'),y=np.array(y,dtype='int64'),subject=np.array(subject),
                        trial_id=np.array(trial_id),channel_names=np.array(channel_names),sfreq=np.array(200),unit=np.array('uV'))
    metadata.update({'name':name,'scope':'PUBLIC_DATA_DEVELOPMENT_NOT_COMPETITION','shape':list(X.shape),
                     'sfreq':200,'unit':'uV','channel_names':channel_names,
                     'class_counts':np.bincount(y).tolist(),
                     'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()})
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'prepared':name,'shape':list(X.shape)}),flush=True)


def ssvep():
    import py7zr
    folder=ROOT/'data/raw/SSVEP_Benchmark'
    channels=[line.split()[-1] for line in (folder/'64-channels.loc').read_text().splitlines() if line.strip()]
    assert len(channels)==64
    fp=loadmat(folder/'Freq_Phase.mat'); frequencies=fp['freqs'].reshape(-1).tolist()
    assert len(frequencies)==40
    sos=butter(4,[6,90],fs=250,btype='bandpass',output='sos')
    x1=[];x2=[];ys=[];subjects=[];ids=[];sources=[]
    for subject in range(1,36):
        archive=folder/f'S{subject}.mat.7z';path=folder/f'S{subject}.mat'
        if not path.exists():
            with py7zr.SevenZipFile(archive) as f:
                assert f.getnames()==[path.name]
                f.extractall(folder)
        data=loadmat(path)['data']
        assert data.shape==(64,1500,40,6)
        # Reorder to trial, channel, time; labels follow target axis, never trial position guesses.
        trials=data.transpose(2,3,0,1).reshape(240,64,1500)
        filtered=sosfiltfilt(sos,trials,axis=-1)
        begin=160  # 0.5 s pre-stimulus + fixed 0.14 s visual latency, at 250 Hz.
        x1.append(resample_poly(filtered[:,:,begin:begin+250],4,5,axis=-1).astype('float32'))
        x2.append(resample_poly(filtered[:,:,begin:begin+500],4,5,axis=-1).astype('float32'))
        ys.extend(np.repeat(np.arange(40),6));subjects.extend([subject]*240)
        ids.extend([f'S{subject:02d}-target{target}-block{block}' for target in range(40) for block in range(6)])
        sources.append(json.loads(archive.with_name(archive.name+'.source.json').read_text()))
        print(f'SSVEP subject {subject}: 240 trials',flush=True)
    metadata={'source':'https://bci.med.tsinghua.edu.cn/download.html',
              'split':{'train':list(range(1,6))+list(range(9,30)), 'val':[6,30,31,32], 'test':[7,8,33,34,35]},
              'split_note':'Fixed before scores; each partition includes experienced and naive participants. 26/4/5 subjects.',
              'frequencies':frequencies,'sources':sources,
              'preprocessing':['original 64 channels','4th-order 6-90 Hz zero-phase Butterworth within each original 6 s trial',
                               'fixed stimulus-relative crop at +0.14 s','polyphase 250 to 200 Hz'],
              'labels':'target indices 0..39 aligned to Freq_Phase.mat; stimulus frequencies available to CCA baselines',
              'test_previously_used':False,'citation':'Wang et al. 2017 A benchmark dataset for SSVEP-based brain-computer interfaces. DOI:10.1109/TNSRE.2016.2627556'}
    write_task('ssvep_1s',np.concatenate(x1),ys,subjects,ids,channels,{**metadata,'window_seconds':1})
    write_task('ssvep_2s',np.concatenate(x2),ys,subjects,ids,channels,{**metadata,'window_seconds':2})


def emotion():
    import mne
    folder=ROOT/'data/raw/FACED_v1.1.4'
    class_names=['Anger','Disgust','Fear','Sadness','Neutral','Amusement','Inspiration','Joy','Tenderness']
    all_x=[];ys=[];subjects=[];ids=[];sources=[];canonical=None;rates={}
    for subject in range(12):
        eeg=folder/f'sub-{subject:03d}/eeg'; stem=f'sub-{subject:03d}_task-watchingVideoClips'
        path=eeg/(stem+'_eeg.bdf')
        with (eeg/(stem+'_channels.tsv')).open(encoding='utf-8-sig') as f: channels=list(csv.DictReader(f,delimiter='\t'))
        picks=[row['name'] for row in channels if row['type']=='EEG' and row['name'] not in ['A1','A2']]
        assert len(picks)==30
        if canonical is None:canonical=picks
        assert picks==canonical
        with (eeg/(stem+'_events.tsv')).open(encoding='utf-8-sig') as f:
            events=[row for row in csv.DictReader(f,delimiter='\t') if row['emotion_label'] in class_names]
        assert len(events)==28 and len({e['video_index'] for e in events})==28
        raw=mne.io.read_raw_bdf(path,preload=False,verbose='ERROR')
        fs=float(raw.info['sfreq']); rates[str(subject)]=fs
        assert fs in (250,1000)
        ratio=Fraction(200,int(fs));sos=butter(4,[0.5,45],btype='bandpass',fs=fs,output='sos')
        for event in events:
            start=round(float(event['onset'])*fs);n=round(float(event['duration'])*fs)
            assert n>=30*fs and start+n<=raw.n_times
            # Read EEG only. MNE converts the BDF physical unit to Volts; explicitly return to uV.
            signal=raw.get_data(picks=picks,start=start,stop=start+n)*1e6
            signal-=signal.mean(axis=0,keepdims=True)
            signal=sosfiltfilt(sos,signal,axis=-1)
            signal=resample_poly(signal[:,-int(30*fs):],ratio.numerator,ratio.denominator,axis=-1)
            assert signal.shape==(30,6000) and np.isfinite(signal).all()
            for window in range(3):
                all_x.append(signal[:,window*2000:(window+1)*2000].astype('float32'))
                ys.append(class_names.index(event['emotion_label']));subjects.append(subject)
                ids.append(f'sub{subject:03d}-video{int(event["video_index"]):02d}-last30win{window}')
        raw.close();sources.append(json.loads(path.with_name(path.name+'.source.json').read_text()))
        print(f'FACED subject {subject}: 84 windows, original {fs} Hz',flush=True)
    metadata={'source':'https://data.nemar.org/nm000112/v1.1.4/',
              'source_original':'https://www.synapse.org/Synapse:syn50614194',
              'license':'CC-BY-4.0 (NEMAR BIDS conversion)', 'classes':class_names,
              'cohort':'first 12 numeric subject IDs, chosen before scores; 12 of 123 participants',
              'split':{'train':[0,1,4,5,6,7,8,9],'val':[2,10],'test':[3,11]},
              'split_note':'Fixed before scores; 8/2/2 subjects. Each partition contains early 1000 Hz and later 250 Hz recordings.',
              'source_rates':rates,'sources':sources,
              'preprocessing':['30 EEG channels; A1/A2 excluded; source order checked','MNE physical-unit conversion to uV',
                               'per-video CAR and 4th-order 0.5-45 Hz sosfiltfilt','last 30 s of each video',
                               'polyphase resampling to 200 Hz','three non-overlapping 10 s windows per video'],
              'limitation':'Known stimuli across unseen subjects; not unseen-stimulus generalization. Only a 12-subject pilot.',
              'test_previously_used':False,'citation':'Chen et al. (2023), A Large Finer-grained Affective Computing EEG Dataset, DOI:10.1038/s41597-023-02650-w'}
    write_task('emotion_10s',np.stack(all_x),ys,subjects,ids,canonical,metadata)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('track',choices=['ssvep','emotion']);args=p.parse_args()
    ssvep() if args.track=='ssvep' else emotion()
