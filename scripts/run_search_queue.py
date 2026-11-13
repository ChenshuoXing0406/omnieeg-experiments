"""Execute the remaining GPU studies sequentially, keeping persistent live logs."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]


def main():
    logs=ROOT/'outputs/search_v1/logs';logs.mkdir(parents=True,exist_ok=True)
    status={}
    for task in ['ssvep_1s','ssvep_2s','emotion']:
        cache=ROOT/'data/processed'/('emotion_10s.npz' if task=='emotion' else f'{task}.npz')
        if not cache.exists():
            status[task]={'status':'missing_data'};continue
        status[task]={'status':'running','started_utc':datetime.now(timezone.utc).isoformat()}
        (logs/'queue_status.json').write_text(json.dumps(status,indent=2))
        command=[sys.executable,str(ROOT/'scripts/search_models.py'),'search',task]
        with (logs/f'{task}.log').open('a',encoding='utf-8') as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            for line in process.stdout:
                f.write(line);f.flush();print(line,end='',flush=True)
            code=process.wait()
        status[task]['status']='complete' if code==0 else 'failed'
        status[task]['exit_code']=code;status[task]['ended_utc']=datetime.now(timezone.utc).isoformat()
        (logs/'queue_status.json').write_text(json.dumps(status,indent=2))
    (logs/'queue_status.json').write_text(json.dumps(status,indent=2))
    if any(item['status']!='complete' for item in status.values()):raise RuntimeError(status)


if __name__=='__main__':main()
