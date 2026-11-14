"""Wait for a genuinely idle GPU, run the fixed MI experiment, then audit results."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--max-wait',type=int,default=3600)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True);state=a.root/'queue_status.json'
    started=time.monotonic();streak={}
    try:
        while True:
            smoke=a.root/'smoke/status.json'
            if smoke.exists():
                s=json.loads(smoke.read_text())
                if s['status']=='failed':raise RuntimeError('CPU preflight failed: '+str(s))
                if s['status']=='finished':break
            save(state,{'status':'waiting_for_cpu_preflight'});time.sleep(10)
            if time.monotonic()-started>a.max_wait:raise TimeoutError('Preflight wait exceeded limit')
        while True:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
            candidates=[]
            for row in output.splitlines():
                idx,used,util=map(int,row.split(','));streak[idx]=streak.get(idx,0)+1 if used<500 and util<10 else 0
                if streak[idx]>=3:candidates.append(idx)
            if candidates:gpu=candidates[0];break
            save(state,{'status':'waiting_for_idle_gpu','wait_seconds':round(time.monotonic()-started),'required':'memory<500MiB and utilization<10%, three consecutive checks 15s apart'})
            if time.monotonic()-started>a.max_wait:raise TimeoutError('No idle GPU within configured wait; no training started')
            time.sleep(15)
        receipt=json.loads((a.root/'command.json').read_text());cmd=receipt['command']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        save(state,{'status':'training','gpu_physical_index':gpu,'command':cmd})
        with (a.root/'formal.stdout.log').open('xb') as log:subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        report=[sys.executable,str(a.root/'report_physionet_normalization.py'),'--root',str(a.root),'--controls',receipt['controls']]
        with (a.root/'audit.stdout.log').open('xb') as log:subprocess.run(report,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        save(state,{'status':'finished','fits':6,'reused_control_fits':6,'audited_prediction_sets':12,'test_evaluated':False,'gpu_physical_index':gpu})
    except Exception as exc:
        save(state,{'status':'failed','error':repr(exc)});raise
