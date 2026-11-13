"""Read-only public access probes plus one small official PhysioNet EDF sample."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/research_20261113'
SAMPLES=ROOT/'data/research_samples'


def probe(item):
    name,url=item
    try:
        r=requests.get(url,headers={'Range':'bytes=0-1023'},stream=True,timeout=(10,25))
        chunk=next(r.iter_content(1024),b'')
        result={'name':name,'url':url,'status':r.status_code,'content_type':r.headers.get('Content-Type'),
                'content_length_header':r.headers.get('Content-Length'),'final_url':r.url,
                'head_hex':chunk[:24].hex(),'head_text':chunk[:100].decode('ascii','replace')}
        r.close();return result
    except Exception as e:return {'name':name,'url':url,'error':type(e).__name__+': '+str(e)[:150]}


def main():
    OUT.mkdir(parents=True,exist_ok=True);SAMPLES.mkdir(parents=True,exist_ok=True)
    shu=requests.get('https://api.figshare.com/v2/articles/19228725',timeout=30).json()
    mat=next(f for f in shu['files'] if f['name']=='mat_files.zip')
    items=[('PhysioNet_MI_EDF','https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf'),
           ('SHU_MI_sample',mat['download_url']),
           ('FACED_features_9class','https://drive.usercontent.google.com/download?id=1ctfPmjUvMKH4xWOkXnIrX_d_AEnhX1Cg&export=download'),
           ('OpenBMI_directory','https://ftp.giga.or.kr/gi_data_4/BCI/2018/'),
           ('FACED_HF_recording_122','https://huggingface.co/datasets/braindecode/faced/resolve/18c73355fc71c055e92828384da85bb49bc6e3c0/sourcedata/braindecode/dataset.zarr/recording_122/zarr.json')]
    with ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(probe,items))
    url=items[0][1];path=SAMPLES/'PhysioNet_S001R04.edf'
    if not path.exists():
        with requests.get(url,stream=True,timeout=(10,30)) as r:
            r.raise_for_status();size=0
            with path.with_suffix('.part').open('wb') as f:
                for chunk in r.iter_content(262144):
                    size+=len(chunk)
                    if size>20*1024**2:raise RuntimeError('Sample exceeds preset 20 MiB bound')
                    f.write(chunk)
            path.with_suffix('.part').replace(path)
    raw=path.read_bytes();header=raw[:256]
    channels=int(header[252:256].decode().strip());record_seconds=float(header[244:252].decode().strip())
    offset=256+channels*(16+80+8+8+8+8+8+80)
    samples=[int(raw[offset+i*8:offset+(i+1)*8].decode().strip()) for i in range(channels)]
    labels=[raw[256+i*16:256+(i+1)*16].decode().strip() for i in range(channels)]
    summary={'file':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
             'signals_including_annotations':channels,'sampling_rates':sorted(set(v/record_seconds for v,l in zip(samples,labels) if 'Annotations' not in l)),
             'annotation_channel_present':any('Annotations' in l for l in labels),'source':url}
    write={'audit_date':'2026-11-13','probes':results,'SHU_sample_metadata':mat,'downloaded_sample':summary}
    (OUT/'access_audit.json').write_text(json.dumps(write,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(write,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
