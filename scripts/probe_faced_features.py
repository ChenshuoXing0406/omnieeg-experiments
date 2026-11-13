"""Follow the public Drive size-warning form and inspect only 1 KiB."""
import json
import hashlib
import re
from pathlib import Path
import requests

out=Path(__file__).resolve().parents[1]/'tmp/research_20261113'
html=(out/'faced_drive_warning.html').read_text()
params=dict(re.findall(r'name="([^"]+)" value="([^"]*)"',html))
with requests.get('https://drive.usercontent.google.com/download',params=params,
                  headers={'Range':'bytes=0-1023'},stream=True,timeout=(10,25)) as r:
    b=next(r.iter_content(1024),b'')
    result={'status':r.status_code,'type':r.headers.get('Content-Type'),
            'content_length_header':r.headers.get('Content-Length'),'content_range':r.headers.get('Content-Range'),
            'head_hex':b[:24].hex(),'head_text':b[:100].decode('ascii','replace'),'sampled_bytes':len(b)}
(out/'faced_feature_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

# Author-linked feature file, bounded at 130 MiB. Preserve original pilot data.
target=out.parents[1]/'data/raw/FACED_GNN4EEG/FACED_dataset_9_labels.mat'
target.parent.mkdir(parents=True,exist_ok=True)
if not target.exists():
    with requests.get('https://drive.usercontent.google.com/download',params=params,stream=True,timeout=(10,30)) as r:
        r.raise_for_status();size=0
        with target.with_suffix('.part').open('wb') as f:
            for chunk in r.iter_content(262144):
                size+=len(chunk)
                if size>130*1024**2:raise RuntimeError('Feature file exceeds 130 MiB limit')
                f.write(chunk)
    target.with_suffix('.part').replace(target)
from scipy.io import whosmat
audit={'source_repository':'https://github.com/Miracle-2001/GNN4EEG','path':str(target),
       'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
       'variables':[{'name':n,'shape':list(s),'matlab_type':t} for n,s,t in whosmat(target)],
       'scope':'Precomputed features only; signal and label semantics still require audit; no new training.'}
(out/'faced_feature_file_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit))
