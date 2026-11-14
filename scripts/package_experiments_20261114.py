"""Package today's immutable experiment outputs, excluding raw EEG and credentials."""
import argparse,gzip,hashlib,json,re,tarfile
from pathlib import Path
import numpy as np

SUITES=['kalunga_classical_20261114','physionet_mi_20261114','physionet_models_20261114','physionet_normalization_20261114']


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(a):
    a.destination.mkdir(parents=True,exist_ok=True)
    paths={}
    for suite in SUITES:
        for p in (a.workspace/'outputs'/suite).rglob('*'):
            if p.is_file():paths[p.relative_to(a.workspace).as_posix()]=p
    selected=json.loads(a.source_list.read_text(encoding='utf-8'))
    for name in selected:paths[name]=(a.source/name)
    patterns=[r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,})',r'\b(?:\d{1,3}\.){3}\d{1,3}\b']
    records=[]
    for name,p in sorted(paths.items()):
        assert p.is_file() and not p.is_symlink(),name
        assert not any(part in ['.git','.venv','__pycache__'] for part in Path(name).parts),name
        assert '_PRIVATE' not in name and not name.startswith(('data/','checkpoints/')),name
        if p.suffix in ['.json','.md','.csv','.log','.txt','.py']:
            text=p.read_text(encoding='utf-8',errors='strict')
            assert not any(re.search(pattern,text) for pattern in patterns),f'Secret/connection check requires review: {name}'
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as d:
                assert not set(d.files)&{'X','x','eeg','raw','signal','cs','ns'},name
        if p.suffix=='.npy':
            v=np.load(p,allow_pickle=False);assert v.ndim==2 and v.shape[1]==16 and v.dtype.kind in 'iu',name
        records.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
    archive=a.destination/'omnieeg-experiments-20261114.tar.gz'
    assert not archive.exists(),'Use a fresh staging directory'
    internal={'date':'2026-11-14','files':records,'neural_fits':21,'short_flow_fits':5,'ssvep_source_threshold_calibrations':12,
              'raw_eeg_included':False,'original_third_party_pretrained_weights_included':False}
    with archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=6) as compressed,tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as tar:
        for rec in records:
            info=tarfile.TarInfo(rec['path']);info.size=rec['bytes'];info.mode=0o644;info.mtime=0
            with paths[rec['path']].open('rb') as src:tar.addfile(info,src)
        payload=json.dumps(internal,indent=2,ensure_ascii=False).encode('utf-8')
        import io
        info=tarfile.TarInfo('file_manifest.json');info.size=len(payload);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(payload))
    verified=0
    with tarfile.open(archive,'r:gz') as tar:
        expected={r['path']:r for r in records}
        for member in tar:
            assert member.isfile()
            if member.name=='file_manifest.json':continue
            with tar.extractfile(member) as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
            assert h==expected[member.name]['sha256'];verified+=1
    assert verified==len(records)
    parts=[]
    with archive.open('rb') as src:
        index=1
        while payload:=src.read(64*1024**2):
            p=archive.with_name(archive.name+f'.part{index:02d}');p.write_bytes(payload)
            parts.append({'name':p.name,'bytes':len(payload),'sha256':sha(p),'url':'https://github.com/ChenshuoXing0406/omnieeg-experiments/releases/download/artifacts-20261114/'+p.name});index+=1
    manifest={'tag':'artifacts-20261114','release_url':'https://github.com/ChenshuoXing0406/omnieeg-experiments/releases/tag/artifacts-20261114',
        'files':records,'parts':parts,'all_parts_remote_verified':False,'local_archive_readback_verified':True,
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'counts':{'formal_neural_fits':21,'short_flow_fits':5,'source_threshold_calibrations':12},
        'scope':'Original experiment source bytes, trained checkpoints, predictions, indices, logs, reports and audits; no raw EEG or original third-party pretrained weights.'}
    a.manifest.parent.mkdir(parents=True,exist_ok=True);a.manifest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'files':len(records),'archive_bytes':archive.stat().st_size,'parts':len(parts),'readback_verified':verified}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','source','source-list','destination','manifest']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
