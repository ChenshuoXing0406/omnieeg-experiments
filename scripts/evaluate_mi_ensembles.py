"""Fixed validation-only MI probability combinations; no new fits or test scoring."""
import csv
import json
from pathlib import Path
import numpy as np
from search_models import metrics,write
from run_preliminary import sha

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mi_ensembles_v1'

def main():
    assert json.loads((ROOT/'outputs/mi_specialists_v1/queue_status.json').read_text())['status']=='finished'
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for fold in range(1,4):
        base_path=ROOT/f'outputs/mi_domain_v1/fold{fold}_balanced_coral0/mi_validation.npz';base=dict(np.load(base_path))
        for name in ['shallow_lr1e3','shallow_lr3e3','eegnet_frequency_shift']:
            other_path=ROOT/f'outputs/mi_specialists_v1/fold{fold}_{name}/mi_validation.npz';other=dict(np.load(other_path))
            for k in ['y','subject','trial_id']:assert np.array_equal(base[k],other[k])
            assert not set(base['subject'])&{8,9}
            for alpha in [.25,.5,.75]:
                p=(1-alpha)*base['scores']+alpha*other['scores'];m=metrics(base['y'],p.argmax(-1),base['subject'],4)
                folder=OUT/f'fold{fold}_{name}_alpha{alpha:g}';folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'mi_validation.npz',scores=p,y=base['y'],subject=base['subject'],trial_id=base['trial_id'])
                write(folder/'config.json',{'fold':fold,'name':name,'alpha_specialist':alpha,'new_fits':0,
                    'source_predictions':[str(p.relative_to(ROOT)) for p in [base_path,other_path]],
                    'source_prediction_sha256':[sha(p) for p in [base_path,other_path]],'test_evaluated':False,'source_only':True})
                write(folder/'result.json',{'metrics':m,'new_fits':0,'test_evaluated':False})
                rows.append({'fold':fold,'name':name,'alpha':alpha,'BA':m['subject_mean_BA'],'new_fits':0,'test_evaluated':False})
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'name':name,'alpha':a,'mean_BA':float(np.mean([r['BA'] for r in rows if r['name']==name and r['alpha']==a])),'new_fits':0}
             for name in ['shallow_lr1e3','shallow_lr3e3','eegnet_frequency_shift'] for a in [.25,.5,.75]]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',{'score_configurations':27,'fits':0,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
