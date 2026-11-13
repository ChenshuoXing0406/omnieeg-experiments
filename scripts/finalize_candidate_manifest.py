"""Link measured development candidates to their actual trained artifacts."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_preliminary import sha
from search_models import write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/next_round_20261113'

def read(path):return json.loads((ROOT/path).read_text())
def files(paths):
    return [{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in paths]

def main():
    base=read('outputs/mi_domain_v1/summary.json')
    mi_single=next(x['mean_BA'] for x in base if x['method']=='eegnet_source_balanced_coral' and x['parameter']==0)
    mi_combo=max(read('outputs/mi_ensembles_v1/summary.json'),key=lambda r:r['mean_BA'])
    ss=read('outputs/ssvep_transfer_v1/summary.json')
    seed=read('outputs/seed_features_v1/summary.json')
    seeden=read('outputs/emotion_ensembles_v1/summary.json')
    fa=read('outputs/faced_normalization_v1/summary.json')
    fa123=read('outputs/faced123_features_v1/summary.json')
    seed_strict=max(seed,key=lambda r:r['mean_window_BA'])
    seed_causal=max((r for r in seeden if r['condition']=='causal'),key=lambda r:r['mean_BA'])
    seed_offline=max((r for r in seeden if r['condition']=='offline'),key=lambda r:r['mean_BA'])
    candidates=[]
    for seconds in [1,2]:
        chosen=next(r for r in ss if r['seconds']==seconds and r['method']=='fusion25')
        candidates.append({'task':f'SSVEP {seconds}s','classes':40,'model':'FBCCA .75 + source-template FBTRCA .25',
            'protocol':'subject-disjoint development; offline filter context','mean_BA':chosen['mean_BA'],
            'weights':files([ROOT/f'outputs/best_candidates_20261113/ssvep_{seconds}s_fusion25.joblib']),
            'artifact_training':'all30development subjects refit; no additional validation estimate',
            'inference':'scripts/ssvep_transfer_candidate.py::predict_preprocessed'})
    mixture_paths=[]
    for fold in range(1,4):
        mixture_paths.extend([ROOT/f'outputs/mi_domain_v1/fold{fold}_balanced_coral0/model.pt',
                              ROOT/f'outputs/mi_specialists_v1/fold{fold}_{mi_combo["name"]}/model.pt'])
    candidates.append({'task':'MI','classes':4,'model':f'EEGNet {1-mi_combo["alpha"]:.2f} + {mi_combo["name"]} {mi_combo["alpha"]:.2f}',
        'protocol':'strict source-only; old test8/9 excluded','mean_BA':mi_combo['mean_BA'],'weights':files(mixture_paths),
        'artifact_training':'three separate paired folds; score uses only each fold validation predictions',
        'normalization':'22EEG,200Hz,4s, existingCAR/.3-40Hzcache, per-trial temporal zscore'})
    strict_paths=[ROOT/f'outputs/seed_features_v1/fold{fold}_{seed_strict["method"]}/{name}'
        for fold in range(1,4) for name in ['model.pt','scaler.joblib']]
    candidates.append({'task':'SEED','classes':3,'model':seed_strict['method'],'protocol':'train-only downstream normalization; author DE-LDS offline',
        'mean_BA':seed_strict['mean_window_BA'],'weights':files(strict_paths),'input':'10x62x5 author DE-LDS features, not raw EEG'})
    for condition,best in [('causal',seed_causal),('offline',seed_offline)]:
        paths=[ROOT/f'outputs/seed_adapted_neural_v1/fold{fold}_{condition}_{method}/{name}'
               for fold in range(1,4) for method in ['tcn','emt'] for name in ['model.pt','normalizer.joblib']]
        assert best['weights']==[0,1,1]
        candidates.append({'task':'SEED','classes':3,'model':'TCN .5 + EmT .5','protocol':condition+'_normalization; upstream DE-LDS offline',
            'mean_BA':best['mean_BA'],'weights':files(paths),'input':'10x62x5 author features',
            'warning':'reset normalization per new session; do not reuse saved validation-subject moments as a new-user reference; use training prior for causal mode'})
    for dataset,items in [('FACED8',fa),('FACED123features9',fa123)]:
        best=max(items,key=lambda r:r['mean_BA'])
        candidates.append({'task':dataset,'classes':8 if dataset=='FACED8' else 9,'model':best['method'],
            'protocol':best['mode'],'mean_BA':best['mean_BA'],
            'scope':'separate feature benchmark; do not compare directly with other label spaces',
            'upstream_provenance_unverified':dataset=='FACED123features9'})
    write(ROOT/'outputs/best_candidates_20261113/candidate_manifest.json',{'date':'2026-11-13',
        'status':'best among evaluated development candidates, not global optimum or independent test results',
        'candidates':candidates,'test_evaluated_this_round':False,'experimental_fits':105,'additional_deployment_refits':2})
    with (OUT/'当前候选模型.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['task','classes','model','protocol','mean_BA']);writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in candidates)
    report=OUT/'结果报告.md';text=report.read_text(encoding='utf-8')
    headline=['## 当前候选（开发选择结果）','',
        '| 任务 | 类数 | 模型 | 条件 | 平均BA |','|---|---:|---|---|---:|']
    for r in candidates:headline.append(f'| {r["task"]} | {r["classes"]} | {r["model"]} | {r["protocol"]} | {100*r["mean_BA"]:.2f}% |')
    headline+=['',f'MI严格开发基线：均衡采样EEGNet {100*mi_single:.2f}%；最佳组合比其高 {100*(mi_combo["mean_BA"]-mi_single):.2f} 个百分点。',
        '组合权重是在开发结果上选出的候选，尚无独立测试确认。SEED组合对单模型的收益很小（不足0.1个百分点），部署简化时可优先保留单TCN或单EmT。',
        'MI组合额外评分27份，0次新增拟合；SEED组合评分54份，0次新增拟合。SSVEP两套全开发集重拟合模型已保存，重载完全一致，没有新增准确率。','']
    # Insert the selected-candidate overview before the complete configuration table.
    marker='| 任务 | 条件 | 模型／参数 |'
    if '## 当前候选（开发选择结果）' in text:
        begin=text.index('## 当前候选（开发选择结果）');end=text.index(marker,begin)
        text=text[:begin]+text[end:]
    text=text.replace(marker,'\n'.join(headline)+'\n'+marker,1)
    report.write_text(text,encoding='utf-8')
    audit=read('outputs/next_round_20261113/audit.json');audit['mi_ensemble_score_configurations']=27
    audit['candidate_artifact_paths_verified']=sum(len(c.get('weights',[])) for c in candidates)
    write(OUT/'audit.json',audit)
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    comparisons=[('SSVEP 2s: source-only',[next(r['mean_BA'] for r in ss if r['seconds']==2 and r['method']=='fbcca'),next(r['mean_BA'] for r in ss if r['seconds']==2 and r['method']=='fusion25')],['FBCCA','+ source template']),
                 ('MI: source-only',[mi_single,mi_combo['mean_BA']],['EEGNet','+ Shallow']),
                 ('SEED: different adaptation conditions',[seed_strict['mean_window_BA'],seed_causal['mean_BA'],seed_offline['mean_BA']],['Train-only','Past/current*','Whole session*'])]
    for ax,(title,values,labels) in zip(axes,comparisons):
        bars=ax.bar(labels,np.asarray(values)*100,color=['#708090','#237f8b','#bd843f'][:len(values)])
        ax.set_ylim(0,100);ax.set_title(title,fontsize=10);ax.set_ylabel('Development subject-mean BA (%)');ax.tick_params(axis='x',labelsize=8)
        for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.2f}',ha='center',fontsize=9)
        ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Paired development folds; no independent final test',fontsize=12)
    fig.supxlabel('* SEED author DE-LDS is offline upstream; normalization conditions differ.',fontsize=9)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps({'candidates':len(candidates),'artifact_paths_verified':audit['candidate_artifact_paths_verified'],
        'MI_best_BA':mi_combo['mean_BA'],'SEED_causal_best_BA':seed_causal['mean_BA']}))

if __name__=='__main__':main()
