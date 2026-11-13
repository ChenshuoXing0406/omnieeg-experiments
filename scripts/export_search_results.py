"""Recompute saved metrics and export an auditable search summary and figure."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from search_models import metrics,OUT,ROOT,write


def main():
    rows=[];champions=[];audit=[]
    tasks=['mi','ssvep_1s','ssvep_2s','emotion']
    for task in tasks:
        folder=OUT/task
        if not folder.exists():continue
        name='BNCI2014_001_T_pilot' if task=='mi' else 'emotion_10s' if task=='emotion' else task
        cache=ROOT/f'data/processed/{name}.npz';data=np.load(cache,allow_pickle=False)
        meta=json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
        val=np.flatnonzero(np.isin(data['subject'],meta['split']['val']));test=np.flatnonzero(np.isin(data['subject'],meta['split']['test']))
        by_id={};count=0
        for path in sorted((folder/'runs').glob('*/result.json'),key=lambda p:p.stat().st_mtime):
            r=json.loads(path.read_text());p=np.load(path.parent/'validation_predictions.npz',allow_pickle=False)
            assert np.array_equal(p['trial_id'],data['trial_id'][val]) and np.array_equal(p['y'],data['y'][val])
            score=metrics(p['y'],p['scores'].argmax(-1),p['subject'],len(np.unique(data['y'])))['subject_mean_BA']
            assert abs(score-r['val_BA'])<1e-12 and r['test_evaluated'] is False
            history=path.parent/'history.json'
            if history.exists():
                h=json.loads(history.read_text());best=max(h,key=lambda row:row['val_BA'])
                assert r['best_epoch']==best['epoch'] and abs(best['val_BA']-score)<1e-12
            row={'task':task,'run_id':r['id'],'stage':r['stage'],'model':r['config']['model'],'seed':r['seed'],
                 'val_BA':r['val_BA'],'budget_epochs':r['budget_epochs'],'epochs_completed':r.get('epochs_completed'),
                 'best_epoch':r.get('best_epoch'),'updates':r.get('updates'),'seconds':r['seconds'],
                 'config':json.dumps(r['config'],sort_keys=True),'test_evaluated':False}
            rows.append(row);by_id[r['id']]=r;count+=1
        champion_path=folder/'champion.json'
        if champion_path.exists():
            c=json.loads(champion_path.read_text());chosen=[by_id[i] for i in c['run_ids']]
            assert all(r['config']==c['config'] for r in chosen)
            assert abs(np.mean([r['val_BA'] for r in chosen])-c['val_BA_mean'])<1e-12
            vals=[np.load(folder/'runs'/i/'validation_predictions.npz',allow_pickle=False)['scores'] for i in c['run_ids']]
            ensemble_val=metrics(data['y'][val],np.mean(vals,axis=0).argmax(-1),data['subject'][val],len(np.unique(data['y'])))
            info={'task':task,'model':c['config']['model'],'tested_configurations':c['tested_configurations'],
                  'val_BA_mean':c['val_BA_mean'],'val_BA_std':c['val_BA_std'],
                  'val_ensemble_BA':ensemble_val['subject_mean_BA'],'seeds':c['seeds'],'config':c['config'],
                  'checkpoint_directories':[str((folder/'runs'/i).relative_to(ROOT)) for i in c['run_ids']]}
            final_path=folder/'final_test.json'
            if final_path.exists():
                result=json.loads(final_path.read_text());p=np.load(folder/'final_test_predictions.npz',allow_pickle=False)
                assert result['champion']['config']==c['config'] and result['champion']['run_ids']==c['run_ids']
                assert np.array_equal(p['trial_id'],data['trial_id'][test]) and np.array_equal(p['y'],data['y'][test])
                m=metrics(p['y'],p['scores'].argmax(-1),p['subject'],len(np.unique(data['y'])))
                assert abs(m['subject_mean_BA']-result['test_metrics']['subject_mean_BA'])<1e-12
                info['test_BA']=m['subject_mean_BA'];info['test_macro_f1']=m['macro_f1']
                info['test_previously_used']=result['test_previously_used']
            champions.append(info)
        audit.append({'task':task,'verified_run_count':count,'subject_overlap':False,'validation_predictions_recomputed':True,
                      'champion_available':champion_path.exists(),'final_test_available':(folder/'final_test.json').exists()})
    OUT.mkdir(parents=True,exist_ok=True)
    if rows:
        with (OUT/'all_runs.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'champions_summary.json',champions)
    write(OUT/'audit.json',{'passed':True,'tasks':audit,'runs_verified':len(rows),
                           'scope':'metric integrity; not proof of global optimality or clinical validity'})
    if champions:
        keys=['task','model','tested_configurations','val_BA_mean','val_BA_std','val_ensemble_BA','test_BA','test_macro_f1','seeds','config']
        with (OUT/'champions_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader()
            for r in champions:writer.writerow({**r,'config':json.dumps(r['config'],sort_keys=True)})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    labels={'mi':'Motor imagery (4 classes)','ssvep_1s':'SSVEP 1 s (40 classes)',
            'ssvep_2s':'SSVEP 2 s (40 classes)','emotion':'Emotion (8 classes, 12 subjects)'}
    for ax,task in zip(axes.flat,tasks):
        chosen=[r for r in rows if r['task']==task]
        if not chosen:ax.set_visible(False);continue
        values=np.array([r['val_BA'] for r in chosen])*100
        ax.scatter(np.arange(1,len(values)+1),values,s=15,color='#8194a5',alpha=.65,label='Run validation BA')
        ax.plot(np.arange(1,len(values)+1),np.maximum.accumulate(values),color='#135b9c',linewidth=2,label='Best single-seed run so far')
        ax.set_title(labels[task]);ax.set_xlabel('Completed run (includes seeded confirmation)');ax.set_ylabel('Validation subject-mean BA (%)')
        ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Public-data model search — validation only; fixed subject splits',fontsize=13)
    fig.savefig(OUT/'validation_search.png',dpi=180);plt.close(fig)
    print(json.dumps({'verified_runs':len(rows),'champions':champions}),flush=True)


if __name__=='__main__':main()
