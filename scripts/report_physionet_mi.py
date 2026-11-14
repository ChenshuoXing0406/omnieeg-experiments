"""Audit saved validation predictions and summarize the independent MI experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    status = json.loads((root/'formal/status.json').read_text())
    assert status['status'] == 'finished', status
    meta = json.loads((root/'data/metadata.json').read_text())
    rows, audits = [], []
    models = [('eegnet','EEGNet'),('shallow','Shallow'),('fixed_fusion','Fixed fusion 0.75/0.25')]
    for fold, split in enumerate(meta['splits'],1):
        saved = {}
        for name,label in models:
            folder = root/'formal'/f'fold{fold}_{name}'
            with np.load(folder/'validation.npz') as src:
                d = {k:src[k] for k in src.files}
            assert np.isfinite(d['scores']).all()
            assert np.allclose(d['scores'].sum(-1),1,atol=1e-6)
            assert set(d['subject']) == set(split['val'])
            assert not set(d['subject']) & set(split['test'])
            assert len(set(d['trial_id'])) == len(d['trial_id'])
            pred = d['scores'].argmax(-1)
            per = [balanced_accuracy_score(d['y'][d['subject']==s],pred[d['subject']==s]) for s in split['val']]
            row = {'fold':fold,'model':label,'subject_mean_BA':float(np.mean(per)),
                   'pooled_BA':float(balanced_accuracy_score(d['y'],pred)),
                   'macro_f1':float(f1_score(d['y'],pred,average='macro'))}
            if name != 'fixed_fusion':
                result = json.loads((folder/'result.json').read_text())
                assert abs(result['metrics']['subject_mean_BA']-row['subject_mean_BA'])<1e-12
                assert result['reload_error']<1e-5
            else:
                for k in ['y','subject','trial_id']:
                    assert np.array_equal(d[k],saved['eegnet'][k]) and np.array_equal(d[k],saved['shallow'][k])
                assert np.allclose(d['scores'],.75*saved['eegnet']['scores']+.25*saved['shallow']['scores'],atol=1e-7)
            saved[name] = d
            rows.append(row); audits.append({'fold':fold,'model':label,'predictions_verified':True,'test_evaluated':False})
    summary = [{'model':label,'mean_BA':float(np.mean([r['subject_mean_BA'] for r in rows if r['model']==label])),
                'development_split_sd':float(np.std([r['subject_mean_BA'] for r in rows if r['model']==label],ddof=1))}
               for _,label in models]
    out = root/'report';out.mkdir(exist_ok=True)
    with (out/'fold_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    lines = ['# 64通道MI独立数据集验证','',
        f"PhysioNet MI：{len(meta['subjects'])}人、{sum(s['trials'] for s in meta['subjects'])}个trial；64通道、200Hz、4秒。",
        '三个本地开发重划分，每次87人训练、11人验证；固定11人测试池未评分。验证集合可能重叠。',
        '模型在新数据集从随机权重重新训练；这不是旧模型的零样本迁移，也不是官方比赛结果。','',
        '| 模型 | 验证逐被试平均BA（三划分均值） | 开发划分标准差 |','|---|---:|---:|']
    for s in summary:lines.append(f"| {s['model']} | {s['mean_BA']*100:.2f}% | {s['development_split_sd']*100:.2f}pp |")
    lines += ['', '标准差混合被试划分与训练种子的变化，不代表纯训练随机性。固定融合权重沿用旧BNCI开发选择，本轮不搜索融合权重。',
        '连续run离线滤波使用窗外上下文；部分原始文件为128Hz，按真实头信息处理，不声称已经修复源数据时钟问题。',
        '六次正式拟合均完成检查点重载预测核验；九份验证预测完成独立评分复算和固定融合核验；短流程结果未混入正式结果。']
    (out/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
