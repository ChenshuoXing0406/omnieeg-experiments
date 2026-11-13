"""Validation-only fixed-grid probability ensembles; zero additional model fits."""
import csv
import json
from pathlib import Path
import numpy as np
from search_models import metrics,write
from run_preliminary import sha

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/emotion_ensembles_v1'
WEIGHTS=[(1,1,1),(0,1,1),(1,1,0),(1,0,1),(1,2,1),(1,1,2)]

def path_seed(fold,condition):
    if condition=='strict':return [ROOT/f'outputs/seed_features_v1/fold{fold}_logistic_C0.1/validation.npz',
        ROOT/f'outputs/seed_features_v1/fold{fold}_tcn/validation.npz',ROOT/f'outputs/seed_features_v1/fold{fold}_emt/validation.npz']
    logistic=ROOT/f'outputs/seed_normalization_search_v1/fold{fold}_causal_prior5/validation.npz' if condition=='causal' else ROOT/f'outputs/seed_alignment_diagnostic_v1/fold{fold}_session_zscore/validation.npz'
    return [logistic,ROOT/f'outputs/seed_adapted_neural_v1/fold{fold}_{condition}_tcn/validation.npz',
        ROOT/f'outputs/seed_adapted_neural_v1/fold{fold}_{condition}_emt/validation.npz']

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for condition in ['strict','causal','offline']:
        for fold in range(1,4):
            paths=path_seed(fold,condition);source=[dict(np.load(p,allow_pickle=False)) for p in paths]
            for obj in source[1:]:
                for k in ['y','subject','trial_id']:assert np.array_equal(source[0][k],obj[k])
            for i,weights in enumerate(WEIGHTS):
                w=np.asarray(weights)/sum(weights);p=np.einsum('m,mnk->nk',w,np.stack([s['scores'] for s in source]))
                base=source[0];m=metrics(base['y'],p.argmax(-1),base['subject'],3)
                folder=OUT/f'{condition}_fold{fold}_ensemble{i}';folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'validation.npz',scores=p,y=base['y'],subject=base['subject'],trial_id=base['trial_id'])
                write(folder/'config.json',{'weights_logistic_tcn_emt':w.tolist(),'source_predictions':[str(x.relative_to(ROOT)) for x in paths],
                    'source_prediction_sha256':[sha(x) for x in paths],'source_sha256':sha(__file__),'new_fits':0,
                    'condition':condition,'test_evaluated':False,'upstream_LDS_offline':True})
                write(folder/'result.json',{'metrics':m,'test_evaluated':False,'new_fits':0})
                rows.append({'condition':condition,'fold':fold,'ensemble':i,'weights':str(weights),'BA':m['subject_mean_BA'],'new_fits':0,'test_evaluated':False})
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'condition':condition,'ensemble':i,'weights':WEIGHTS[i],
              'mean_BA':float(np.mean([r['BA'] for r in rows if r['condition']==condition and r['ensemble']==i])),'new_fits':0}
             for condition in ['strict','causal','offline'] for i in range(len(WEIGHTS))]
    write(OUT/'summary.json',summary)
    write(OUT/'audit.json',{'source_identity_checks':9,'score_config_evaluations':54,'new_fits':0,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
