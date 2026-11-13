"""Use Braindecode's pinned FACED window export when raw BDF transport is unavailable.

Decode the explicitly documented Zarr-v3 bytes+gzip chunks without executing
remote dataset code. Preserve original subject identity and event labels.
"""
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import requests
from scipy.signal import butter,sosfiltfilt

from download_tracks import download
from prepare_tracks import write_task

ROOT=Path(__file__).resolve().parents[1]
REPO='braindecode/faced'
REV='18c73355fc71c055e92828384da85bb49bc6e3c0'


def main():
    folder=ROOT/'data/raw/FACED_braindecode'
    folder.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            r=requests.get(f'https://huggingface.co/api/datasets/{REPO}/revision/{REV}',params={'blobs':'true'},timeout=25)
            r.raise_for_status();manifest=r.json();break
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(2)
    (folder/'hub_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    root='sourcedata/braindecode/dataset.zarr/'
    prefixes=[root+f'recording_{i}/' for i in range(12)]
    selected=[item for item in manifest['siblings'] if item['rfilename'] in ['README.md','format_info.json',root+'zarr.json']
              or any(item['rfilename'].startswith(prefix) for prefix in prefixes)]
    jobs=[]
    for item in selected:
        path=item['rfilename'];lfs=item.get('lfs') or {};checksum=lfs.get('sha256') or lfs.get('oid')
        jobs.append((f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{path}',folder/path,checksum))
    with ThreadPoolExecutor(4) as pool:list(pool.map(download,jobs))
    store=folder/root
    global_meta=json.loads((store/'zarr.json').read_text())
    attrs=global_meta['attributes']
    assert '1000000.0' in attrs['raw_preproc_kwargs'] and '2000' in attrs['window_kwargs']
    mapping={'Anger':0,'Disgust':1,'Fear':2,'Sadness':3,'Neutral':4,'Amusement':5,'Inspiration':6,'Joy':7,'Tenderness':8}
    X=[];ys=[];subjects=[];ids=[];channels=None;sources=[];ranges={}
    for recording in range(12):
        record=store/f'recording_{recording}'
        group=json.loads((record/'zarr.json').read_text())['attributes']
        subject=int(group['description']['subject']);assert subject==recording
        names=[item['ch_name'] for item in group['info']['chs']]
        assert len(names)==26 and float(group['info']['sfreq'])==200.
        if channels is None:channels=names
        assert names==channels
        meta=json.loads((record/'data/zarr.json').read_text())
        assert meta['zarr_format']==3 and meta['data_type']=='float32'
        assert [item['name'] for item in meta['codecs']]==['bytes','gzip']
        assert meta['codecs'][0]['configuration']['endian']=='little'
        shape=meta['shape'];chunk=meta['chunk_grid']['configuration']['chunk_shape'];assert shape[1:]==[26,2000]
        pieces=[]
        for i in range((shape[0]+chunk[0]-1)//chunk[0]):
            path=record/f'data/c/{i}/0/0';blob=gzip.decompress(path.read_bytes())
            array=np.frombuffer(blob,dtype='<f4').reshape(-1,26,2000)
            assert array.shape[0] in [chunk[0],min(chunk[0],shape[0]-i*chunk[0])]
            pieces.append(array)
            sources.append(json.loads(path.with_name(path.name+'.source.json').read_text()))
        x=np.concatenate(pieces)[:shape[0]].copy()
        with (record/'metadata.tsv').open(encoding='utf-8-sig') as f:events=list(csv.DictReader(f,delimiter='\t'))
        assert len(events)==len(x) and np.isfinite(x).all()
        # Unit authority is exporter preprocessing (V * 1e6), not stale MNE FIFF_UNIT_V tags.
        ranges[str(subject)]={'median_channel_std_uV':float(np.median(x.std(-1))),'windows':len(x)}
        assert 0.01<ranges[str(subject)]['median_channel_std_uV']<1e5
        x-=x.mean(1,keepdims=True)
        x=sosfiltfilt(butter(4,[.5,45],fs=200,btype='bandpass',output='sos'),x,axis=-1).astype('float32')
        for event in events:
            assert int(event['target'])==mapping[event['emotion_label']]
            assert int(event['i_stop_in_trial'])-int(event['i_start_in_trial'])==2000
            ys.append(int(event['target']));subjects.append(subject)
            ids.append(f'sub{subject:03d}-video{int(event["video_index"]):02d}-start{int(event["i_start_in_trial"])}')
        X.append(x);print(json.dumps({'FACED_hub_subject':subject,'windows':len(x),'unit_check':ranges[str(subject)]}),flush=True)
    present=sorted(set(ys))
    # This pinned export omits Neutral despite the upstream mapping/card listing it.
    # Define the observed task explicitly instead of claiming nine-class FACED.
    target_map={old:new for new,old in enumerate(present)}
    labels=[name for name,source_id in mapping.items() if source_id in present]
    excluded=[name for name,source_id in mapping.items() if source_id not in present]
    ys=[target_map[label] for label in ys]
    metadata={'source':f'https://huggingface.co/datasets/{REPO}/tree/{REV}', 'source_revision':REV,
              'source_original':'https://www.synapse.org/Synapse:syn50614194',
              'source_conversion':'Braindecode FACED processed Zarr export from NEMAR BIDS; 26-channel subset',
              'classes':labels,'source_label_mapping':mapping,'source_target_to_model_target':target_map,
              'excluded_source_classes':excluded,'class_count':len(labels),
              'label_audit':'Neutral is absent from all 12 exported recordings; this is an eight-emotion pilot, not nine-class FACED.',
              'cohort':'first 12 numeric subject IDs, fixed before outcome scores; 12 of 123',
              'split':{'train':[0,1,4,5,6,7,8,9],'val':[2,10],'test':[3,11]},
              'sources':sources,'unit_checks':ranges,
              'upstream_preprocessing':attrs['raw_preproc_kwargs'],'upstream_windowing':attrs['window_kwargs'],
              'local_preprocessing':['no additional unit conversion: upstream explicitly V*1e6','26-channel CAR',
                                     'per-window 4th-order 0.5-45 Hz zero-phase Butterworth'],
              'excluded_inputs':'Self-report ratings, film identities, subject ID and event timing are not model inputs',
              'limitation':'Public processed 26-channel eight-emotion pilot; Neutral absent. Not complete nine-class FACED, raw 30-channel reproduction or competition data. Known stimuli across unseen subjects.',
              'license_note':'Hugging Face card lists license unknown; preserve original dataset attribution and do not redistribute data.',
              'test_previously_used':False,'citation':'Chen et al. (2023), DOI:10.1038/s41597-023-02650-w'}
    write_task('emotion_10s',np.concatenate(X),ys,subjects,ids,channels,metadata)


if __name__=='__main__':main()
