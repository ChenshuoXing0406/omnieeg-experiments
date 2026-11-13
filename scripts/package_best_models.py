"""Package selected public-pilot models, source, schema and local inference entry point."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/search_v1'

PREDICT=r'''"""Predict from preprocessed microvolt EEG; no network calls or training."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
import search_models as s

p=argparse.ArgumentParser()
p.add_argument('task',choices=['mi','ssvep_1s','ssvep_2s','emotion'])
p.add_argument('input',help='NPZ: X [N,C,T], channel_names, sfreq=200, unit="uV"; same preprocessing as manifest')
p.add_argument('--output',default='predictions.npz')
a=p.parse_args()
root=Path(__file__).resolve().parent
registry=json.loads((root/'MODEL_MANIFEST.json').read_text(encoding='utf-8'))
entry=registry['tasks'][a.task]
data=np.load(a.input,allow_pickle=False)
assert set(['X','channel_names','sfreq','unit']).issubset(data.files)
assert float(data['sfreq'])==200 and str(data['unit'])=='uV'
names=data['channel_names'].tolist()
assert len(names)==len(set(names)), 'Duplicate channel names'
expected=entry['channel_names']
assert set(names)==set(expected), 'Channel identity mismatch; do not silently interpolate or omit channels'
X=np.asarray(data['X'][:,[names.index(ch) for ch in expected]],dtype='float32')
assert X.ndim==3 and list(X.shape[1:])==entry['input_shape'] and np.isfinite(X).all()
metadata=entry['data_metadata']
working={'X':X,'channel_names':np.array(expected),'classes':len(entry['class_names']),
         'meta':metadata,'idx':{'test':np.arange(len(X))},
         'sha256':hashlib.sha256(X.tobytes()+json.dumps(metadata,sort_keys=True).encode()).hexdigest()}
scores=[]
for run_id in entry['champion']['run_ids']:
 run=json.loads((s.OUT/a.task/'runs'/run_id/'result.json').read_text())
 scores.append(s.test_scores(working,entry['champion']['config'],run))
scores=np.mean(scores,axis=0)
np.savez_compressed(a.output,predicted_class=scores.argmax(-1),scores=scores,class_names=np.array(entry['class_names']))
print(json.dumps({'task':a.task,'predictions':len(X),'output':str(Path(a.output).resolve()),
                  'score_kind':'neural: mean softmax probability; classical: decision score'}))
'''


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def main():
    tasks=['mi','ssvep_1s','ssvep_2s','emotion'];registry={'scope':'PUBLIC_PILOT_BEST_AMONG_TESTED_CANDIDATES','tasks':{}}
    for task in tasks:
        folder=OUT/task
        champion=json.loads((folder/'champion.json').read_text())
        final=json.loads((folder/'final_test.json').read_text())
        assert champion['config']==final['champion']['config'] and champion['run_ids']==final['champion']['run_ids']
        name='BNCI2014_001_T_pilot' if task=='mi' else 'emotion_10s' if task=='emotion' else task
        metadata=json.loads((ROOT/f'data/processed/{name}.json').read_text(encoding='utf-8'))
        import numpy as np
        data=np.load(ROOT/f'data/processed/{name}.npz',allow_pickle=False)
        channels=data['channel_names'].tolist()
        labels=metadata.get('classes') or metadata.get('labels')
        if not isinstance(labels,list):labels=[f'target_{i}' for i in range(40)]
        registry['tasks'][task]={'champion':champion,'final_test_metrics':final['test_metrics'],
                                'class_names':labels,'channel_names':channels,'input_shape':metadata['shape'][1:],
                                'data_metadata':metadata,'test_previously_used':task=='mi'}
    identifier=hashlib.sha256(json.dumps(registry,sort_keys=True).encode()).hexdigest()[:12]
    package=OUT/'packages'/f'best_models_{identifier}'
    package.mkdir(parents=True,exist_ok=True)
    (package/'MODEL_MANIFEST.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    (package/'predict.py').write_text(PREDICT,encoding='utf-8')
    for task,entry in registry['tasks'].items():
        folder=OUT/task
        for name in ['champion.json','final_test.json']:
            copy(folder/name,package/(folder/name).relative_to(ROOT))
        for run_id in entry['champion']['run_ids']:
            run=folder/'runs'/run_id
            for name in ['config.json','result.json','model.pt','model.joblib','csp.joblib']:
                if (run/name).exists():copy(run/name,package/(run/name).relative_to(ROOT))
    for script in (ROOT/'scripts').glob('*.py'):copy(script,package/'scripts'/script.name)
    for source in (ROOT/'external/CBraMod/models').glob('*.py'):copy(source,package/'external/CBraMod/models'/source.name)
    copy(ROOT/'external/CBraMod/LICENSE',package/'external/CBraMod/LICENSE')
    if any(entry['champion']['config']['model'].startswith('cbramod_') for entry in registry['tasks'].values()):
        copy(ROOT/'checkpoints/cbramod.pth',package/'checkpoints/cbramod.pth')
    copy(ROOT/'requirements-experiments.txt',package/'requirements-experiments.txt')
    text='''# Selected public-data models

These models are selected within the tested public-pilot configurations. They are not competition results or a unified three-task foundation model.

Input NPZ must contain preprocessed X [N,C,T] in microvolts, channel_names, sfreq=200, and unit="uV". The exact channel identities, lengths, preprocessing, class mapping, source cohort and selected parameters are in MODEL_MANIFEST.json. Missing channels and different sampling rates are rejected. This script does not prepare arbitrary raw EEG.

Use the experiment's CUDA-enabled Python 3.11 environment, or install requirements-experiments.txt in a new environment. Run:

```powershell
python predict.py mi prepared_input.npz --output predictions.npz
```

Task names: mi, ssvep_1s, ssvep_2s, emotion. Neural ensembles average the saved models' probabilities. Classical outputs are decision scores, not calibrated probabilities. There are no network calls during prediction.

FACED is an eight-emotion, 12-subject, 26-channel processed-data pilot. Neutral is absent in the upstream pinned export and explicitly excluded. It is not complete nine-class FACED. SSVEP windows use offline full-trial filtering before cropping, and do not establish strict online latency. MI test subjects were already inspected in the earlier pilot. Fixed-split results do not establish global optimality.

No EEG recordings or participant-level source metadata are included. Original model/data attribution and upstream versions are preserved in the manifest and CBraMod license.
'''
    (package/'README.md').write_text(text,encoding='utf-8')
    hashes={str(p.relative_to(package)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    (package/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    archive=package.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in package.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(package)))
    print(json.dumps({'package':str(package),'archive':str(archive),'files':len(hashes),'bytes':archive.stat().st_size}),flush=True)


if __name__=='__main__':main()
