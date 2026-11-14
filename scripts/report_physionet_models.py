"""Independent prediction, sampling and initialization audit for the MI model study."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def main(args):
    root=args.root;formal=root/'formal'
    status=json.loads((formal/'status.json').read_text())
    assert status['status']=='finished' and status['fits']==9,status
    env=json.loads((formal/'environment.json').read_text())
    for path,digest in env['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    rows=[];audits=[];configs={}
    names=['lab_conformer','cbra_pretrained','cbra_random']
    for fold in range(1,4):
        reference=np.load(args.baseline/f'fold{fold}_eegnet/validation.npz')
        expected_indices=np.load(args.baseline/f'fold{fold}_eegnet/training_indices.npy')
        for name in names:
            folder=formal/f'fold{fold}_{name}'
            cfg=json.loads((folder/'config.json').read_text());configs[fold,name]=cfg
            r=json.loads((folder/'result.json').read_text())
            d=np.load(folder/'validation.npz');pred=d['scores'].argmax(-1)
            assert np.isfinite(d['scores']).all() and np.allclose(d['scores'].sum(-1),1,atol=1e-6)
            for k in ['trial_id','subject','y']:assert np.array_equal(d[k],reference[k]),(fold,name,k)
            assert set(d['subject'])==set(cfg['split']['val'])
            assert not set(d['subject'])&set(cfg['split']['test'])
            assert len(set(d['trial_id']))==len(d['trial_id'])
            assert np.array_equal(np.load(folder/'training_indices.npy'),expected_indices)
            assert hashlib.sha256(expected_indices.tobytes()).hexdigest()==cfg['sampling_sha256']
            m={'subject_mean_BA':float(np.mean([balanced_accuracy_score(d['y'][d['subject']==s],pred[d['subject']==s]) for s in cfg['split']['val']])),
               'pooled_BA':float(balanced_accuracy_score(d['y'],pred)), 'macro_f1':float(f1_score(d['y'],pred,average='macro'))}
            for k,v in m.items():assert abs(r['metrics'][k]-v)<1e-12,(fold,name,k)
            assert r['reload_error']<1e-5
            rows.append({'fold':fold,'model':name,**m,'seconds':r['seconds'],'peak_allocated_MiB':r['peak_allocated_MiB'],'trainable_parameters':r['trainable_parameters']})
            audits.append({'fold':fold,'model':name,'metrics_verified':True,'sampling_matches_previous_baseline':True,
                           'checkpoint_reload_error':r['reload_error'],'test_evaluated':False})
        a,b=configs[fold,'cbra_pretrained'],configs[fold,'cbra_random']
        assert a['initial_head_sha256']==b['initial_head_sha256']
        assert a['initial_backbone_sha256']!=b['initial_backbone_sha256']
        for k in ['split','seed','normalization','batch','updates','wd','backbone_lr','head_lr','sampling_sha256']:
            assert a[k]==b[k],k
    summary=[]
    for name in names:
        rr=[r for r in rows if r['model']==name]
        summary.append({'model':name,'mean_BA':float(np.mean([r['subject_mean_BA'] for r in rr])),
            'development_split_sd':float(np.std([r['subject_mean_BA'] for r in rr],ddof=1)),
            'mean_seconds':float(np.mean([r['seconds'] for r in rr])),
            'max_peak_allocated_MiB':max(r['peak_allocated_MiB'] for r in rr),'trainable_parameters':rr[0]['trainable_parameters']})
    deltas=[next(r['subject_mean_BA'] for r in rows if r['fold']==fold and r['model']=='cbra_pretrained')-
            next(r['subject_mean_BA'] for r in rows if r['fold']==fold and r['model']=='cbra_random') for fold in range(1,4)]
    out=root/'report';out.mkdir(exist_ok=True)
    with (out/'fold_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'models':summary,'pretrained_minus_random_by_split':deltas,'mean_initialization_delta':float(np.mean(deltas))},indent=2),encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    lines=['# 64通道MI：实验室模型与基础模型对照','',
        'PhysioNet MI，109人、9837trial、64通道×4秒；每次87人训练/11人验证，固定11人测试池未评分。三个本地开发重划分，验证被试集合可能重叠，不是官方种子或独立盲测。','',
        '| 条件 | 平均验证逐被试BA | 划分标准差 | 参数量 | 平均训练/验证秒数 | 峰值allocated显存MiB |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['model']} | {s['mean_BA']*100:.2f}% | {s['development_split_sd']*100:.2f}pp | {s['trainable_parameters']} | {s['mean_seconds']:.1f} | {s['max_peak_allocated_MiB']:.1f} |")
    lines+=['',f"CBraMod预训练减随机的逐划分差值：{' / '.join(f'{d*100:+.2f}pp' for d in deltas)}；平均{np.mean(deltas)*100:+.2f}pp。",
        '此配对保持相同架构、头初值、输入uV/100、更新数、优化配置和采样顺序；只支持本轮1500步预算下的初始化比较，不代表随机大模型充分收敛的性能上限。',
        '', '计时覆盖训练、验证及最终检查点重载复核，不含数据载入和模型初始化；显存为PyTorch allocated峰值，不含完整CUDA上下文成本。',
        'EEGConformer沿用实验室实现并外接四类线性头，使用每trial时间zscore；CBraMod使用uV/100。与既有小模型的比较同时涉及结构和配方差异，不能归因于单一架构。',
        '已有同开发协议的EEGNet/Shallow/固定融合平均BA为59.13%/61.29%/62.26%；它们是系统参照，不是本轮预训练初始化控制的一部分。',
        '作者报告CBraMod在TUEG预训练，未做逐样本独立预训练重叠审计。连续run离线滤波使用窗外上下文。',
        '', '正式9次拟合均完成检查点重载；保存预测的三项指标、采样索引、被试身份、初始头匹配及依赖源码哈希核验全部通过。3次两步短流程另列，不混入正式拟合。']
    (out/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'pretrained_minus_random_by_split':deltas}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    main(p.parse_args())
