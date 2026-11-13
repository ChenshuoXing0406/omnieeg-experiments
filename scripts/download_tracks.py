"""Download predefined public SSVEP/FACED development cohorts with file hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def download(job):
    url, path, expected_hash = job
    path.parent.mkdir(parents=True, exist_ok=True)
    record_path = path.with_name(path.name + '.source.json')
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record['bytes'] == path.stat().st_size and (not expected_hash or record['sha256'] == expected_hash):
            return
    if path.suffix=='.bdf':
        # The published data-plane supports Range. Resume real bytes on unstable long transfers.
        manifest=json.loads((ROOT/'data/raw/FACED_v1.1.4/manifest.json').read_text(encoding='utf-8'))
        total=next(item['size'] for item in manifest if item['url']==url)
        temp=path.with_name(path.name+'.part'); failures=0
        while (temp.stat().st_size if temp.exists() else 0)<total:
            start=temp.stat().st_size if temp.exists() else 0
            end=min(start+16*2**20,total)-1
            try:
                headers={'User-Agent':'Mozilla/5.0','Origin':'https://nemar.org','Range':f'bytes={start}-{end}'}
                with requests.get(url,headers=headers,stream=True,timeout=(15,25)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                        raise ValueError('Server did not honor resume range')
                    with temp.open('ab') as out:
                        for chunk in response.iter_content(2**18):out.write(chunk)
                assert temp.stat().st_size<=total
                failures=0
                print(json.dumps({'progress':path.name,'MiB':round(temp.stat().st_size/2**20,1),'total_MiB':round(total/2**20,1)}),flush=True)
            except Exception as exc:
                failures+=1
                print(json.dumps({'resume_retry':failures,'file':path.name,'error':str(exc)[:100]}),flush=True)
                if failures>=6:raise
                time.sleep(1)
        checksum=hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()
        if checksum!=expected_hash:raise ValueError('Resumed BDF SHA256 mismatch')
        temp.replace(path)
        record_path.write_text(json.dumps({'source_url':url,'bytes':total,'sha256':checksum,
                               'expected_sha256':expected_hash,'time_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
        print(json.dumps({'downloaded':path.name,'sha256_verified':True}),flush=True)
        return
    for attempt in range(3):
        try:
            temp = path.with_name(path.name + '.part')
            digest = hashlib.sha256()
            with requests.get(url, stream=True, timeout=(25, 90)) as response:
                response.raise_for_status()
                expected_size = response.headers.get('Content-Length')
                with temp.open('wb') as out:
                    for chunk in response.iter_content(2**20):
                        out.write(chunk); digest.update(chunk)
                if expected_size and temp.stat().st_size != int(expected_size):
                    raise ValueError('Incomplete file')
                checksum = digest.hexdigest()
                if expected_hash and checksum != expected_hash:
                    raise ValueError(f'SHA256 mismatch for {path.name}')
                temp.replace(path)
                record = {'source_url': url, 'resolved_url': response.url.split('?')[0],
                          'bytes': path.stat().st_size, 'sha256': checksum,
                          'expected_sha256': expected_hash, 'time_utc': datetime.now(timezone.utc).isoformat()}
                record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(json.dumps({'downloaded': str(path.relative_to(ROOT)), 'MiB': round(path.stat().st_size/2**20, 1)}), flush=True)
            return
        except Exception as exc:
            print(json.dumps({'retry': attempt+1,'url':url,'error':str(exc)[:250]}),flush=True)
            if attempt == 2: raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('track', choices=['ssvep','emotion'])
    parser.add_argument('--subjects',type=int,default=None)
    args=parser.parse_args()
    jobs=[]
    if args.track=='ssvep':
        n=args.subjects or 10
        base='https://bci.med.tsinghua.edu.cn/upload/yijun/'
        names=['Readme.txt','Sub_info.txt','64-channels.loc','Freq_Phase.mat']+[f'S{i}.mat.7z' for i in range(1,n+1)]
        jobs=[(base+name,ROOT/'data/raw/SSVEP_Benchmark'/name,None) for name in names]
    else:
        n=args.subjects or 12
        base='https://data.nemar.org/nm000112/v1.1.4/'
        for attempt in range(3):
            try:
                response=requests.get(base+'manifest.json',timeout=30);response.raise_for_status()
                manifest=response.json();break
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(2**attempt)
        if not isinstance(manifest,list):raise TypeError('Unexpected manifest shape')
        folder=ROOT/'data/raw/FACED_v1.1.4';folder.mkdir(parents=True,exist_ok=True)
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        subjects={f'sub-{i:03d}' for i in range(n)}
        selected=[item for item in manifest if item['path'].split('/')[0] in subjects
                  or item['path'] in ['README.md','dataset_description.json','participants.tsv']]
        selected=[item for item in selected if item['path'].endswith(('.bdf','.json','.tsv','.md'))]
        assert sum(item['path'].endswith('.bdf') for item in selected)==n
        print(json.dumps({'cohort':'first numeric IDs; selected before labels/results','subjects':n,
                          'download_GiB':sum(item['size'] for item in selected)/2**30}),flush=True)
        for item in selected:
            jobs.append((item['url'],folder/item['path'],item['checksum'] if item.get('checksum_algorithm')=='sha256' else None))
    # Fetch small metadata first. A failed subject must not cancel other public files.
    jobs.sort(key=lambda job: (job[1].suffix=='.bdf',job[1].name))
    failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(download,job):job[1].name for job in jobs}
        for future in as_completed(pending):
            try:future.result()
            except Exception as exc:
                failures.append(pending[future])
                print(json.dumps({'failed_file':pending[future],'error_type':type(exc).__name__}),flush=True)
    if failures:raise RuntimeError(f'Incomplete files (rerun resumes): {failures}')


if __name__=='__main__':main()
