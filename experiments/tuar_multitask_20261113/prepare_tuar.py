"""Patient-disjoint TUAR artifact detection. Public pilot, not official contest labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,csv,hashlib,json,math
from pathlib import Path
from fractions import Fraction
import numpy as np
import mne
from scipy.signal import butter,sosfiltfilt,resample_poly

CHANNELS=['FP1-F7','F7-T3','T3-T5','T5-O1','FP2-F8','F8-T4','T4-T6','T6-O2',
          'T3-C3','C3-CZ','CZ-C4','C4-T4','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2']
ARTIFACTS={'eyem','chew','shiv','musc','elpp','elec','elst','artf'}
VERSION='tuar20_binary_v1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False));t.replace(path)

def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out

def overlap(intervals,a,b):
    return sum(max(0,min(b,y)-max(a,x)) for x,y in intervals if x<b and y>a)

def annotation_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(line for line in f if line.strip() and not line.startswith('#')))

def canonical(name):
    name=name.upper().strip()
    if name.startswith('EEG '):name=name[4:]
    for suffix in ['-REF','-LE','-AR']:
        if name.endswith(suffix):name=name[:-len(suffix)]
    return {'T7':'T3','T8':'T4','P7':'T5','P8':'T6'}.get(name,name)

def window_plan(path,root):
    raw=mne.io.read_raw_edf(path,preload=False,verbose='ERROR')
    mapping={}
    for name in raw.ch_names:
        c=canonical(name)
        if c in mapping:raise ValueError('duplicate electrode '+c)
        mapping[c]=name
    needed=sorted({c for pair in CHANNELS for c in pair.split('-')})
    missing=set(needed)-set(mapping)
    if missing:raise ValueError('missing electrodes '+str(sorted(missing)))
    fs=float(raw.info['sfreq']);duration=raw.n_times/fs
    assert fs>=200 and fs==int(fs),(fs,path)
    # Unit conversion is handled by MNE; reject missing physical-unit metadata.
    units=getattr(raw,'_orig_units',{})
    for c in needed:
        if units.get(mapping[c],'').lower() not in ['µv','uv','μv','mv','v']:raise ValueError('unknown physical unit '+mapping[c]+':'+str(units.get(mapping[c])))
    raw.close()
    intervals=[];types={k:[] for k in sorted(ARTIFACTS)};uncertain=[];ignored=0
    for row in annotation_rows(path.with_suffix('.csv')):
        channel=row['channel'].strip().upper()
        if channel not in CHANNELS:ignored+=1;continue
        a=float(row['start_time']);b=float(row['stop_time']);label=row['label'].strip().lower()
        assert 0<=a<b<=duration+.1,(path,a,b,duration)
        parts=set(label.split('_'))
        if not parts<=ARTIFACTS:raise ValueError('unknown artifact label '+label)
        if float(row['confidence'])<1:uncertain.append([a,b]);continue
        intervals.append([a,b])
        for k in parts:types[k].append([a,b])
    intervals=merge(intervals);uncertain=merge(uncertain)
    seizure=path.with_name(path.stem+'_seiz.csv')
    if seizure.exists():
        for row in annotation_rows(seizure):
            if row['label'].strip().lower() not in ['bckg','null']:
                # Exclude seizure-context windows and a 2 s margin, not a quality class.
                uncertain.append([max(0,float(row['start_time'])-2),min(duration,float(row['stop_time'])+2)])
        uncertain=merge(uncertain)
    starts=[];labels=[];masks=[];dropped={'seizure_or_uncertain':0,'boundary_artifact':0}
    for a in range(2,math.floor(duration)-3,2):
        b=a+2
        if overlap(uncertain,a,b)>0:dropped['seizure_or_uncertain']+=1;continue
        coverage=overlap(intervals,a,b)
        if 0<coverage<.5-1e-8:dropped['boundary_artifact']+=1;continue
        starts.append(a);labels.append(int(coverage>=.5-1e-8))
        masks.append(sum((1<<i) for i,k in enumerate(sorted(ARTIFACTS)) if overlap(types[k],a,b)>0))
    assert starts,'no eligible windows'
    return {'record':str(path.relative_to(root)),'subject':path.stem.split('_')[0],
      'sfreq':fs,'duration_s':duration,'electrodes':{k:mapping[k] for k in needed},'starts':starts,'y':labels,'artifact_mask':masks,
      'class_counts':np.bincount(labels,minlength=2).tolist(),'dropped_windows':dropped,'ignored_annotation_channels':ignored,
      'source_bytes':path.stat().st_size,'csv_sha256':sha(path.with_suffix('.csv')),
      'seizure_csv_sha256':sha(seizure) if seizure.exists() else None}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='/data2/hust_bciml_eegdata/TUEG/tuh_eeg_artifact/v3.0.1/edf');ap.add_argument('--output',default=str(Path(__file__).resolve().parent/'processed/tuar'));ap.add_argument('--scan-only',action='store_true');args=ap.parse_args()
    root=Path(args.source);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plan_path=out/'split_plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text());assert plan['version']==VERSION and plan['preprocessor_sha256']==sha(__file__),'Changed preprocessing requires a new output directory'
    else:
        records=[];excluded=[]
        for path in sorted(root.rglob('*.edf')):
            if not path.with_suffix('.csv').exists():excluded.append({'record':str(path.relative_to(root)),'reason':'missing same-stem annotation'});continue
            try:record=window_plan(path,root)
            except Exception as e:excluded.append({'record':str(path.relative_to(root)),'reason':repr(e)});continue
            records.append(record)
            if len(records)%25==0:print('SCANNED',len(records),flush=True)
        assert records,excluded[:5]
        subjects=sorted({r['subject'] for r in records},key=lambda s:hashlib.sha256(('TUAR_patient_split_v1|42|'+s).encode()).hexdigest())
        n=len(subjects);ntrain=int(n*.8);nval=int(n*.1)
        splits={'train':sorted(subjects[:ntrain]),'val':sorted(subjects[ntrain:ntrain+nval]),'test':sorted(subjects[ntrain+nval:])}
        for a,b in [('train','val'),('train','test'),('val','test')]:assert not set(splits[a])&set(splits[b])
        counts={}
        for split,subs in splits.items():
            rows=[r for r in records if r['subject'] in subs]
            counts[split]={'subjects':len(subs),'records':len(rows),'eligible_windows':np.sum([r['class_counts'] for r in rows],axis=0).tolist()}
            assert min(counts[split]['eligible_windows'])>0
        plan={'version':VERSION,'source':str(root),'preprocessor_sha256':sha(__file__),'channels':CHANNELS,
              'split_method':'patient ID = basename prefix; SHA256 ordering of TUAR_patient_split_v1|42|ID; first floor(80%) train, next floor(10%) val, remainder test; no score-driven selection',
              'splits':splits,'counts_before_training_cap':counts,'records':records,'excluded':excluded,
              'labels':{'0':'no artifact annotation in selected 20 bipolar channels during window','1':'union of artifact intervals across selected channels covers >=0.5 s of 2 s window'},
              'ambiguous':'0<artifact union<0.5s excluded; low-confidence and seizure-context windows excluded',
              'artifact_mask_bit_order':sorted(ARTIFACTS),'train_cap':'max 64 windows per class per record, seeded by record SHA256; val/test keep all eligible windows',
              'preprocessing':'20 common TCP bipolar derivations; MNE V->uV; continuous-record 4th order 0.5-80Hz zero-phase Butterworth; polyphase to200Hz; 2s nonoverlapping windows, >=2s margins; uV/100; no per-window normalization',
              'scope':'public real-artifact detection pilot; not official contest labels or split','test_evaluated':False}
        write(plan_path,plan);print('SPLIT_LOCKED',json.dumps(counts),'EXCLUDED',len(excluded),flush=True)
    if args.scan_only:return
    cache=out/'record_cache';cache.mkdir(exist_ok=True)
    for i,r in enumerate(plan['records']):
        dest=cache/(Path(r['record']).stem+'.npz');meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists():continue
        split=next(k for k,v in plan['splits'].items() if r['subject'] in v)
        chosen=np.arange(len(r['starts']))
        if split=='train':
            rng=np.random.default_rng(int(hashlib.sha256(r['record'].encode()).hexdigest()[:8],16))
            chosen=np.sort(np.concatenate([rng.choice(np.flatnonzero(np.array(r['y'])==k),min(64,r['y'].count(k)),replace=False) for k in [0,1]]))
        path=root/r['record'];raw=mne.io.read_raw_edf(path,preload=False,verbose='ERROR')
        names=list(r['electrodes']);signal=raw.get_data(picks=[r['electrodes'][k] for k in names])*1e6;raw.close()
        bipolar=np.stack([signal[names.index(pair.split('-')[0])]-signal[names.index(pair.split('-')[1])] for pair in CHANNELS])
        assert np.isfinite(bipolar).all(),path
        bipolar=sosfiltfilt(butter(4,[.5,80],fs=r['sfreq'],btype='bandpass',output='sos'),bipolar,axis=-1)
        ratio=Fraction(200,int(r['sfreq']));bipolar=resample_poly(bipolar,ratio.numerator,ratio.denominator,axis=-1)
        starts=np.array(r['starts'])[chosen];x=np.stack([bipolar[:,int(s*200):int(s*200)+400].reshape(20,2,200) for s in starts]).astype('float32')/100
        assert np.isfinite(x).all()
        np.savez_compressed(dest,x=x,y=np.array(r['y'],dtype='int64')[chosen],start=starts,artifact_mask=np.array(r['artifact_mask'],dtype='int64')[chosen])
        write(meta,{'record':r['record'],'subject':r['subject'],'split':split,'samples':len(x),'edf_sha256':sha(path),'csv_sha256':r['csv_sha256'],'plan_sha256':sha(plan_path)})
        print('PREPARED',i+1,len(plan['records']),split,len(x),flush=True)
    stats={};source_hashes={}
    for split,subjects in plan['splits'].items():
        rows=[r for r in plan['records'] if r['subject'] in subjects]
        metas=[json.loads((cache/(Path(r['record']).stem+'.json')).read_text()) for r in rows]
        n=sum(m['samples'] for m in metas)
        x=np.lib.format.open_memmap(out/f'{split}_x.npy',mode='w+',dtype='float32',shape=(n,20,2,200))
        y=np.empty(n,dtype='int64');sub=[];ids=[];masks=[];offset=0
        for r,m in zip(rows,metas):
            assert m['plan_sha256']==sha(plan_path)
            with np.load(cache/(Path(r['record']).stem+'.npz')) as a:
                end=offset+len(a['y']);x[offset:end]=a['x'];y[offset:end]=a['y'];sub.extend([r['subject']]*len(a['y']));masks.extend(a['artifact_mask'])
                ids.extend([Path(r['record']).stem+':'+str(int(s)) for s in a['start']]);offset=end
            source_hashes[r['record']]=m['edf_sha256']
        x.flush();del x
        assert offset==n and len(set(ids))==n
        np.save(out/f'{split}_y.npy',y);np.save(out/f'{split}_subject.npy',np.array(sub));np.save(out/f'{split}_id.npy',np.array(ids));np.save(out/f'{split}_artifact_mask.npy',np.array(masks))
        stats[split]={'subjects':len(subjects),'records':len(rows),'windows':n,'class_counts':np.bincount(y,minlength=2).tolist()}
    manifest={k:v for k,v in plan.items() if k!='records'}
    manifest.update(task='tuar',num_classes=2,input_shape=[20,2,200],sample_rate=200,unit='microvolt /100',counts=stats,
                    split_plan_sha256=sha(plan_path),source_edf_sha256=source_hashes)
    write(out/'manifest.json',manifest)
    print('TUAR_READY',json.dumps(stats),flush=True)

if __name__=='__main__':main()
