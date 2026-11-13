"""Audit public pinned FACED metadata only; never fetch EEG signal chunks."""
import csv
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
REV='18c73355fc71c055e92828384da85bb49bc6e3c0'
OUT=ROOT/'tmp/research_20261113/faced_metadata'

def audit(item):
    name=item['rfilename'];target=OUT/('/'.join(name.split('/')[-2:]))
    try:
        if target.exists():raw=target.read_bytes()
        else:
            local=ROOT/'data/raw/FACED_braindecode'/name
            if local.exists():raw=local.read_bytes()
            else:
                r=requests.get(f'https://huggingface.co/datasets/braindecode/faced/resolve/{REV}/{name}',timeout=(10,25))
                r.raise_for_status();raw=r.content
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==item['blobId'],f'Git blob hash mismatch {name}'
        rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')),delimiter='\t'))
        return {'recording':name.split('/')[-2],'windows':len(rows),'labels':dict(Counter(r['emotion_label'] for r in rows)),
                'targets':dict(Counter(r['target'] for r in rows)),'videos':len(set(r['video_index'] for r in rows)),
                'blob_verified':True,'bytes':len(raw)}
    except Exception as e:return {'recording':name.split('/')[-2],'error':str(e)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/raw/FACED_braindecode/hub_manifest.json').read_text())
    items=[i for i in manifest['siblings'] if i['rfilename'].endswith('/metadata.tsv') and '/recording_' in i['rfilename']]
    with ThreadPoolExecutor(max_workers=6) as pool:records=list(pool.map(audit,items))
    labels=Counter()
    for r in records:labels.update(r.get('labels',{}))
    summary={'revision':REV,'metadata_only':True,'records_expected':len(items),'records_verified':sum(r.get('blob_verified',False) for r in records),
             'total_windows':sum(r.get('windows',0) for r in records),'class_counts':dict(labels),
             'no_signal_downloaded':True,'records':records}
    (OUT.parent/'faced_expansion_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},ensure_ascii=True))

if __name__=='__main__':main()
