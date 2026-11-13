"""Figures and readable tables for the dense residual-adapter ablations."""
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/residual_adapters_v1'


def fmt(value):
    return str(Decimal(str(round(float(value),10))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))


def main():
    rows=json.loads((OUT/'results.json').read_text())
    summaries=json.loads((OUT/'summary.json').read_text())
    diffs=json.loads((OUT/'paired_differences.json').read_text())
    assert len(rows)==36 and all(s['folds']==3 for s in summaries), 'Final figures require all three paired folds'
    with (OUT/'逐折结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'配对差值.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(diffs[0]));w.writeheader();w.writerows(diffs)
    modes=['baseline','uniform','learned','no_spectrum']
    labels=['Baseline','Uniform\n3 branches','Learned\n3 branches','Learned\nno spectrum']
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.8))
    colors=['#355C7D','#3C8DAD','#D88C45','#7C6A9B']
    for ax,task,title,limit in zip(axes,['mi','ssvep_2s','emotion'],['Motor imagery','SSVEP, 2 seconds','Emotion, 8 classes'],[70,100,35]):
        for i,mode in enumerate(modes):
            values=np.array([r['BA'] for r in rows if r['task']==task and r['mode']==mode])*100
            if not len(values):continue
            ax.bar(i,values.mean(),color=colors[i],width=.66,alpha=.85)
            sd=values.std(ddof=1) if len(values)>1 else 0.
            ax.errorbar(i,values.mean(),yerr=sd,color='#222222',capsize=4,linewidth=1)
            ax.scatter(i+np.linspace(-.13,.13,len(values)),values,color='#222222',s=20,zorder=3)
            ax.text(i,values.mean()+sd+limit*.03,fmt(values.mean()),ha='center',fontsize=10)
        ax.set_xticks(range(4),labels,fontsize=9);ax.set_title(title)
        ax.set_ylim(0,limit);ax.set_ylabel('Validation balanced accuracy (%)')
        ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Heterogeneous residual adapters: paired development ablations',fontsize=15)
    fig.text(.5,.02,'Dots: development folds. Error bars: across-fold sample SD. All active branches are computed; dense fusion.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(OUT/'comparison.png',dpi=200);plt.close(fig)
    lines=['# 异构残差适配结果解读','',
           '基线复用同协议的单任务预训练全量微调。新模型保持flatten、头初值、样本顺序和1500步预算。',
           '各数据集独立适配，尚不是联合训练结果；旧测试被试全部排除。','',
           '| 任务 | 条件 | 折数 | 平均验证BA | 折间标准差 |','|---|---|---:|---:|---:|']
    for s in summaries:
        sd='—' if s['std'] is None else f"{s['std']*100:.2f}个百分点"
        lines.append(f"| {s['task']} | {s['mode']} | {s['folds']} | {fmt(s['mean_BA']*100)}% | {sd} |")
    lines+=['','## 配对差值','', '| 任务 | 对照 | 折数 | 平均差值 | 逐折差值（百分点） |','|---|---|---:|---:|---|']
    for d in diffs:
        vals=', '.join(f'{v:+.2f}' for v in d['fold_deltas_pp'])
        lines.append(f"| {d['task']} | {d['contrast']} | {d['folds']} | {d['mean_delta_pp']:+.2f}个百分点 | {vals} |")
    lines+=['','## 当前结论','',
            '本轮27次训练与27项预测重算、27个检查点重载均已完成。',
            'MI三个新条件均未超过基线；SSVEP最高为去谱68.50%，比基线67.63%高0.875个百分点；情绪最高为均匀融合17.22%，比基线16.42%高约0.80个百分点。',
            '按事先约定的平均提升至少1个百分点等工程门槛，本轮未获得保留新结构的充分证据。',
            '学习门控未稳定优于均匀融合，整窗频谱分支也未显示一致额外收益。',
            '同折已有MI EEGNet平均47.74%、情绪时序TCN平均18.21%，仍高于本轮对应新结构；不据局部CBraMod基线差值宣称任务最优。',
            '本轮适合作为负面对照保留，不能据此宣称异构专家或样本门控有效，也不排除其他预算和实现下的可能性。',
            '', '## 判读限制','',
            '均匀融合对基线检验新增结构整体；学习融合对均匀融合检验门控；有谱对无谱检验附加谱分支。',
            '无谱模型总实例数相同但活跃参数/计算量减少，不是严格等激活容量的谱信息对照。',
            '适配器输出零初始化使初始预测与基线一致。门控是融合权重生成器，不是额外专家。',
            '所有活跃分支均计算，不宣称稀疏计算或生理机制解耦。门控权重不等同于分支因果贡献。',
            '开发折与训练种子共同变化，折间标准差不是置信区间；这些结果不是独立盲测或赛事成绩。','']
    (OUT/'结果解读.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'task_records':len(rows),'summary_rows':len(summaries),'paired_comparisons':len(diffs)}))


if __name__=='__main__':main()
