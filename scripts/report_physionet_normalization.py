"""Audit the paired 2x2 initialization/normalization development experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score,f1_score


def main(a):
    new=a.root/'formal';old=a.controls
    status=json.loads((new/'status.json').read_text());assert status['status']=='finished' and status['fits']==6
    oldenv=json.loads((old/'environment.json').read_text());newenv=json.loads((new/'environment.json').read_text())
    assert oldenv['source_hashes']==newenv['source_hashes']
    for path,digest in newenv['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    rows=[];audits=[]
    for fold in range(1,4):
        for model in ['cbra_pretrained','cbra_random']:
            pair={}
            for norm,base in [('scale',old),('trial',new)]:
                f=base/f'fold{fold}_{model}'
                c=json.loads((f/'config.json').read_text());r=json.loads((f/'result.json').read_text())
                d=np.load(f/'validation.npz');pred=d['scores'].argmax(-1)
                assert c['normalization']==norm and c['updates']==1500
                assert not set(d['subject'])&set(c['split']['test'])
                assert set(d['subject'])==set(c['split']['val'])
                assert np.isfinite(d['scores']).all() and np.allclose(d['scores'].sum(-1),1,atol=1e-6)
                m={'subject_mean_BA':float(np.mean([balanced_accuracy_score(d['y'][d['subject']==s],pred[d['subject']==s]) for s in c['split']['val']])),
                   'pooled_BA':float(balanced_accuracy_score(d['y'],pred)),'macro_f1':float(f1_score(d['y'],pred,average='macro'))}
                for k,v in m.items():assert abs(v-r['metrics'][k])<1e-12
                assert r['reload_error']<1e-5
                pair[norm]=(c,{k:d[k] for k in ['y','subject','trial_id']},np.load(f/'training_indices.npy'))
                rows.append({'fold':fold,'model':model,'normalization':norm,**m})
            c0,d0,i0=pair['scale'];c1,d1,i1=pair['trial']
            for k in ['split','seed','input','batch','updates','wd','optimizer','scheduler','validation_interval','backbone_lr','head_lr','sampling_sha256','data_metadata_sha256','initial_head_sha256','initial_backbone_sha256','weights_sha256']:
                assert c0[k]==c1[k],k
            for k in d0:assert np.array_equal(d0[k],d1[k])
            assert np.array_equal(i0,i1)
            audits.append({'fold':fold,'model':model,'normalization_only_pair_verified':True,'predictions_recomputed':True,'test_evaluated':False})
    summary=[{'model':m,'normalization':n,'mean_BA':float(np.mean([r['subject_mean_BA'] for r in rows if r['model']==m and r['normalization']==n])),
              'development_split_sd':float(np.std([r['subject_mean_BA'] for r in rows if r['model']==m and r['normalization']==n],ddof=1))}
             for m in ['cbra_pretrained','cbra_random'] for n in ['scale','trial']]
    deltas={m:[next(r['subject_mean_BA'] for r in rows if r['fold']==f and r['model']==m and r['normalization']=='trial')-
               next(r['subject_mean_BA'] for r in rows if r['fold']==f and r['model']==m and r['normalization']=='scale') for f in range(1,4)] for m in ['cbra_pretrained','cbra_random']}
    out=a.root/'report';out.mkdir(exist_ok=True)
    with (out/'fold_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'conditions':summary,'trial_minus_scale_by_split':deltas},indent=2),encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    lines=['# MI归一化与预训练的2×2配对对照','',
      '新增6次逐trial时间zscore训练，复用6次uV/100训练；109人，三次87人训练/11人验证，固定11人测试池未评分。所有训练均1500步。',
      '除归一化外，跨条件初始主干、头、训练采样、数据划分、学习率和更新数完全匹配。','',
      '| 初始化 | 输入 | 平均验证逐被试BA | 开发划分标准差 |','|---|---|---:|---:|']
    for s in summary:lines.append(f"| {s['model']} | {s['normalization']} | {s['mean_BA']*100:.2f}% | {s['development_split_sd']*100:.2f}pp |")
    for m,ds in deltas.items():lines+=['',f"{m}：zscore相对uV/100逐划分差值为{' / '.join(f'{v*100:+.2f}pp' for v in ds)}，均值{np.mean(ds)*100:+.2f}pp。"]
    lines+=['','这些是重复使用开发验证集的条件对照，不是独立测试；验证集合可能重叠，标准差不是纯随机种子方差。',
       '结论仅限本轮预处理、分类头和1500步预算，不代表随机模型充分训练上限。连续run离线滤波仍使用窗外上下文。',
       '六组跨归一化配对与十二份保存预测复算通过；六个新增checkpoint重载核验通过。']
    (out/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'conditions':summary,'trial_minus_scale_by_split':deltas}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--controls',type=Path,required=True)
    main(p.parse_args())
