"""Audit and summarize completed multi-task search; preserve protocol boundaries."""
import csv
import json
from pathlib import Path
import numpy as np
from search_models import metrics,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/next_round_20261113'
BATCHES={
    'mirepnet_search_v1':('MI四类','run_mirepnet_search.py',12),
    'seed_normalization_search_v1':('SEED三类','run_seed_normalization_search.py',18),
    'seed_adapted_neural_v1':('SEED三类','run_seed_adapted_neural.py',12),
    'mi_ea_search_v1':('MI四类','run_mi_ea_search.py',9),
    'mi_supcon_v1':('MI四类','run_mi_supcon_search.py',9),
    'faced_normalization_v1':('FACED八类','run_faced_normalization_search.py',18),
    'faced123_features_v1':('FACED123九类作者特征','run_faced123_features.py',12),
    'mi_specialists_v1':('MI四类','run_mi_specialists.py',9)}

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];audits=[]
    for batch,(task,script,count) in BATCHES.items():
        path=ROOT/'outputs'/batch;status=read(path/'queue_status.json');assert status['status']=='finished',batch
        folders=sorted(p.parent for p in path.glob('fold*/result.json'));assert len(folders)==count
        for folder in folders:
            cfg=read(folder/'config.json');result=read(folder/'result.json')
            name='mi_validation.npz' if task=='MI四类' else 'emotion_validation.npz' if batch=='faced_normalization_v1' else 'validation.npz'
            with np.load(folder/name,allow_pickle=False) as p:
                assert np.isfinite(p['scores']).all() and np.allclose(p['scores'].sum(-1),1,atol=1e-5)
                k=p['scores'].shape[-1];m=metrics(p['y'],p['scores'].argmax(-1),p['subject'],k)
                expected=result['metrics'].get('window',result['metrics'])
                assert abs(m['subject_mean_BA']-expected['subject_mean_BA'])<1e-10
                if task=='MI四类':assert not set(p['subject'])&{8,9}
                if batch=='faced_normalization_v1':assert not set(p['subject'])&{3,11}
            if batch=='mirepnet_search_v1':model=cfg['name'];protocol='strict_source_only' if cfg['strict_DG'] else 'offline_target_subject_EA'
            elif batch=='mi_supcon_v1':model=f'EEGNet SupCon lambda={cfg["lambda"]:g}';protocol='strict_source_only'
            elif batch=='mi_specialists_v1':model=cfg['name'];protocol='strict_source_only'
            elif batch=='mi_ea_search_v1':model='EEGNet';protocol=cfg['mode']
            elif batch=='seed_normalization_search_v1':
                model='Logistic C=.1 '+cfg['mode'];protocol='strict_source_only' if cfg['strict_DG'] else cfg['mode']
            elif batch=='faced123_features_v1':model=cfg['method'];protocol='author_features_'+cfg['mode']+'_upstream_unverified'
            else:model=cfg['method'];protocol={'strict':'strict_source_only','causal':'causal_normalizer_offline_upstream','offline':'offline_target_statistics'}[cfg['mode']]
            rows.append({'task':task,'batch':batch,'protocol':protocol,'model':model,'fold':cfg['fold']+1,
                'classes':k,'validation_BA':m['subject_mean_BA'],'macro_F1':m['macro_f1'],'test_evaluated':False,
                'path':str(folder.relative_to(ROOT)),'source_script':script})
            audits.append({'path':str(folder.relative_to(ROOT)),'prediction_recomputed':True,'checkpoint_reload_error':result.get('reload_error',result.get('checkpoint_max_abs_error',result.get('reload_max_abs_error'))),
                'test_evaluated':False})
    ss=ROOT/'outputs/ssvep_transfer_v1';assert read(ss/'queue_status.json')['status']=='finished'
    ssrows=list(csv.DictReader((ss/'results.csv').open(encoding='utf-8-sig')));assert len(ssrows)==30
    for r in ssrows:
        folder=ss/f'{r["seconds"]}s_fold{r["fold"]}'
        with np.load(folder/f'{r["method"]}_validation.npz') as p:
            m=metrics(p['y'],p['scores'].argmax(-1),p['subject'],40);assert abs(m['subject_mean_BA']-float(r['val_BA']))<1e-10
            assert not set(p['subject'])&{7,8,33,34,35}
        rows.append({'task':f'SSVEP{r["seconds"]}秒','batch':'ssvep_transfer_v1','protocol':'strict_source_only_offline_filter_context',
            'model':r['method'],'fold':int(r['fold']),'classes':40,'validation_BA':m['subject_mean_BA'],'macro_F1':m['macro_f1'],
            'test_evaluated':False,'path':str(folder.relative_to(ROOT))+':'+r['method'],'source_script':'run_ssvep_transfer_search.py'})
    summary=[]
    for task,protocol,model in sorted({(r['task'],r['protocol'],r['model']) for r in rows}):
        selected=sorted([r for r in rows if (r['task'],r['protocol'],r['model'])==(task,protocol,model)],key=lambda r:r['fold'])
        assert len(selected)==3
        vals=[r['validation_BA'] for r in selected]
        summary.append({'task':task,'protocol':protocol,'model':model,'fold1':vals[0],'fold2':vals[1],'fold3':vals[2],
            'mean_BA':float(np.mean(vals)),'sample_std':float(np.std(vals,ddof=1))})
    for name,records in [('逐折结果.csv',rows),('配置汇总.csv',summary)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write(OUT/'summary.json',summary);write(OUT/'prediction_audit.json',audits)
    fits=sum(v[2] for v in BATCHES.values())+6;assert fits==105
    write(OUT/'audit.json',{'completed_fits':fits,'pipeline_configuration_rows':len(rows),
        'prediction_recomputations':len(rows),'ssvep_fits':6,'ssvep_score_configurations':30,
        'emotion_ensemble_score_configurations':54,'ensemble_new_fits':0,'ssvep_deployment_refits':2,'test_evaluated':False})
    lines=['# 三任务下一轮实验汇总（2026-11-13）','',
        '本轮105次对照拟合完成，另有两套SSVEP全开发集部署候选重拟合（未新增准确率）。SSVEP六个训练模板模型另评估30个分数组合；情绪54个概率组合未新增拟合。不使用旧测试选方案。以下为三折开发验证，不是正式比赛或独立盲测。','',
        '| 任务 | 条件 | 模型／参数 | 第1折 | 第2折 | 第3折 | 平均BA |','|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+ ' | '.join([r['task'],r['protocol'],r['model'],*[f'{100*r[k]:.2f}%' for k in ['fold1','fold2','fold3','mean_BA']]])+' |')
    lines+=['','## 解释与使用边界','',
        '- SSVEP同一三折比较FBCCA、训练被试模板FBTRCA及固定分数融合。没有目标标签或目标统计拟合，仍有原6秒试次零相位滤波上下文，不声称严格1/2秒在线延迟。',
        '- MI源侧共享参考与目标被试整批EA分开；随机初始化与预训练控制同头同采样。MIRepNet未超过基线时也完整保留，不以模型名称替代实测选择。',
        '- SEED所有会话按被试分组。causal仅指归一化使用历史/当前特征，上游作者DE-LDS仍离线；offline还使用未来目标窗口。不得将两者成绩写成严格零校准泛化。',
        '- FACED八类仍是原12人中排除旧测试后10个开发被试。新的123人九类是作者预计算特征文件；标签顺序由作者load_srt_de核验，完整上游拟合/归一化来源未独立重建，因此单列，不称全链路严格泛化。',
        '- 训练预算与参数数量不同的模型比较是方法对照；CPU与GPU条件也不用于速度排名。折和训练种子共同变化，不作统计显著性声明。',
        '- 所有配置均值用于开发选择，不是最优模型的无偏泛化估计；后续需要新数据或固定未使用外部测试验证。',
        '- 公开作者预训练数据清单将BNCI2014001放在下游，未列入七个预训练集；这仅依据作者声明，不能重建实际权重全部训练样本或跨库身份重叠。','',
        'SEED概率组合另见 outputs/emotion_ensembles_v1/summary.json（六个固定权重配置×三条件×三折，0次新增拟合）。',
        '模型和预处理状态保存在各运行目录：model.pt / model.joblib、normalizer.joblib / adapter.joblib、config.json、history.json以及验证逐样本分数。']
    (OUT/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'fits':fits,'configuration_rows':len(rows),'report':str(OUT/'结果报告.md')}))

if __name__=='__main__':main()
