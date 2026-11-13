"""Adapt the MI search to the observed channel-mean readout weakness, using validation only."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
import search_models as s


def main():
    task='mi';data=s.load_task(task);folder=s.OUT/task
    champion=json.loads((folder/'champion.json').read_text())
    if champion.get('readout_supplement_done'):
        print('Readout supplement already complete');return
    configs=[{'model':'cbramod_ft','lr':lr,'head_lr':3e-4,'wd':.05,'dropout':.5,'norm':'scale','readout':'flatten'} for lr in [1e-5,3e-5]]
    configs.extend({'model':'cbramod_ridge','alpha':alpha,'readout':'flatten'} for alpha in [10.,100.,1000.,10000.])
    s.write(folder/'readout_supplement_plan.json',{'reason':'channel-mean variants weak on validation; test not inspected',
            'configs':configs,'screen_epochs':20,'confirm_epochs':40,'seeds':[42,43,44],
            'previous_champion':champion,'selection':'compare confirmation validation mean with previous champion'})
    rows=[s.execute(task,data,config,42,20,'readout_screen') for config in configs]
    best=max(rows,key=lambda r:r['val_BA'])
    seeds=[42] if best.get('deterministic') else [42,43,44]
    runs=[s.execute(task,data,best['config'],seed,40,'readout_confirmation') for seed in seeds]
    mean=float(np.mean([r['val_BA'] for r in runs]));std=float(np.std([r['val_BA'] for r in runs],ddof=1)) if len(runs)>1 else 0.
    summary={'config':best['config'],'run_ids':[r['id'] for r in runs],'seeds':seeds,'deterministic':len(seeds)==1,
             'val_BA_mean':mean,'val_BA_std':std,'individual_val_BA':[r['val_BA'] for r in runs]}
    s.write(folder/'readout_supplement_result.json',{'screen':rows,'confirmation':summary,'test_used':False})
    s.write(folder/'champion_before_readout.json',champion)
    if mean>champion['val_BA_mean']:
        champion={**champion,**summary}
    champion['tested_configurations']+=len(configs);champion['readout_supplement_done']=True
    s.write(folder/'champion.json',champion)
    print(json.dumps({'updated_champion':champion}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
