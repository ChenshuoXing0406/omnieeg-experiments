import json,os,subprocess,sys,traceback
from pathlib import Path
root=Path(sys.argv[1])
def save(value):
 p=root/'immediate_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
try:
 cmd=json.loads((root/'command.json').read_text())['command']
 save({'status':'training','gpu_physical_index':3,'fits_planned':6,'test_evaluated':False})
 with (root/'formal.stdout.log').open('xb') as log:
  subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
 save({'status':'auditing','fits':6,'test_evaluated':False})
 with (root/'audit.stdout.log').open('xb') as log:
  subprocess.run([sys.executable,str(root/'report_physionet_normalization.py'),'--root',str(root),'--controls','/mnt/data2/xcs/physionet_models_20261114/formal'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
 save({'status':'finished','fits':6,'audited_prediction_sets':12,'test_evaluated':False})
except Exception as exc:
 save({'status':'failed','error':repr(exc)});raise
