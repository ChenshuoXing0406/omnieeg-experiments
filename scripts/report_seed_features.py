"""Create a source-linked report of completed SEED runs; no training or test scoring."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/seed_features_v1'
summary=json.loads((OUT/'summary.json').read_text());diag=ROOT/'outputs/seed_alignment_diagnostic_v1'
aligned=json.loads((diag/'summary.json').read_text())
for p in [OUT,diag]:assert json.loads((p/'queue_status.json').read_text())['status']=='finished'
assert len(json.loads((OUT/'checkpoint_audit.json').read_text()))==18
assert len(json.loads((diag/'audit.json').read_text()))==6
names={'logistic':'DE均值＋Logistic','linear_svm':'DE均值＋线性SVM','tcn':'DE序列＋TCN','emt':'DE序列＋EmT'}
lines=['# SEED 三分类跨被试实验结果（2026-11-13）','',
       '24 次新拟合完成：18 次严格跨被试特征实验，6 次使用无标签目标会话统计的离线诊断。两类设置不能合并排名。',
       'D:/seed/SEED/SEED.zip 包含原版SEED完整结构；本轮只读取45份作者DE-LDS特征及标签，不是原始EEG实验。15人×3会话×15视频=675视频试次，生成14,895个非重叠十秒特征窗口。末尾不足10个特征帧的片段共丢弃3780帧。',
       '验证三折按被试固定为1–5、6–10、11–15，其他人为训练。同人的所有会话在同一侧；所有模型标准化只训练拟合，诊断条件例外明确如下。没有独立保留测试集，全部均为开发验证成绩。','',
       '## 严格跨被试：不使用目标会话集合统计','',
       '| 方法 | 参数 | 三折平均窗口BA | 三折平均视频BA |','|---|---|---:|---:|']
for r in summary:lines.append(f'| {names[r["method"]]} | {r["parameter"]} | {100*r["mean_window_BA"]:.2f}% | {100*r["mean_video_BA"]:.2f}% |')
lines+=['','视频BA来自同一视频窗口分数平均后分类，不是另一次模型拟合。窗口BA为各被试BA先计算后平均。作者DE-LDS可能含同视频前后文，因此属于离线特征条件，不代表实时十秒推理。',
        'TCN与EmT使用相同源被试/类别均衡采样、1500更新、batch24、AdamW lr.001 wd.01、cosine、dropout.5，每100步按验证被试平均窗口BA选模型。EmT使用作者分类器的小配置，并非论文训练流程完整复现；TCN为本地变体。线性模型使用十帧均值，相比序列模型还改变了读出方式，因此不能把差值只归因于网络结构。','',
        '## 离线诊断：使用无标签目标会话整批统计','',
        '| 相同Logistic C=0.1，变更归一化 | 平均窗口BA | 平均视频BA |','|---|---:|---:|']
strict=next(r for r in summary if r['method']=='logistic' and r['parameter']=='0.1')
lines.append(f'| 仅训练集合StandardScaler | {100*strict["mean_window_BA"]:.2f}% | {100*strict["mean_video_BA"]:.2f}% |')
for r in aligned:lines.append(f'| {r["method"]}，再拟合训练StandardScaler | {100*r["mean_window_BA"]:.2f}% | {100*r["mean_video_BA"]:.2f}% |')
gain=100*(aligned[1]['mean_window_BA']-strict['mean_window_BA'])
lines+=['',f'相同特征读出与分类器，训练和目标各会话分别标准化后窗口BA提高 {gain:.2f} 个百分点。这支持会话/被试特征分布偏移是当前重要因素，但不能单独区分源域归一化与目标域归一化各贡献多少，也不证明所有模型瓶颈都相同。',
        '**诊断条件使用目标会话内未来窗口，不使用目标标签进行拟合或对齐。它是离线传导设置，不能写成严格零校准泛化或在线预测成绩，不能直接当比赛成绩。** 所有会话统计只在本会话内计算，不跨会话或被试拟合目标统计。',
        '下一步应比较因果在线累计统计与固定训练参考；只有比赛推理允许目标流适应时才启用。严格泛化方向优先研究输入自身的相对频谱特征、训练被试分布约束及CLISA训练被试对齐，而不是仅增加网络复杂度。','',
        '## 审计与限制','',
        '- 18份严格配置与6份诊断配置的预测复算和模型重载全部通过。',
        '- 原始ZIP不改写；逐特征成员SHA256、ZIP CRC32及处理后NPZ SHA256在数据元信息中。没有声称完整校验尚未读取的原始EEG或其他两个ZIP。',
        '- 三折训练种子42/43/44共同改变，不作统计显著性声明。所有候选配置都报告，不能将调参最佳验证值称作独立测试。',
        '- 标签来自包内label.mat和readme，原始-1/0/+1映射0/1/2。',
        '- SEED-IV、SEED-VIG暂未训练；SEED原始EEG的EEGNet/CBraMod、CLISA也尚未完成。',
        '- SEED三分类与FACED八分类不作百分比直接比较；没有改变旧FACED、MI、SSVEP结果。','',
        '来源：[SEED官方说明](https://bcmi.sjtu.edu.cn/home/seed/seed.html)，[EmT官方代码](https://github.com/yi-ding-cs/EmT)。',
        '逐折结果：results.csv；训练曲线与逐试次分数保存在各fold目录；离线诊断文件位于 ../seed_alignment_diagnostic_v1。']
(OUT/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
rows=[]
for p,protocol in [(OUT,'strict_cross_subject'),(diag,'offline_target_session_statistics')]:
    for row in csv.DictReader((p/'results.csv').open(encoding='utf-8-sig')):
        rows.append({'protocol':protocol,'fold':row['fold'],'method':row['method'],'parameter':row.get('parameter','.1'),
            'window_BA':row.get('val_window_BA',row.get('window_BA')),'video_BA':row.get('val_video_BA',row.get('video_BA')),
            'test_evaluated':False})
with (OUT/'全部24次结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
selected=[next(r for r in summary if r['method']==m and r['parameter']==p) for m,p in [('logistic','0.1'),('linear_svm','0.01'),('tcn','None'),('emt','None')]]
axes[0].bar(['LogReg C=.1','LinearSVM .01','TCN','EmT'],[100*r['mean_window_BA'] for r in selected],color=['#708090','#708090','#267e8e','#458f73'])
axes[0].set_title('Strict cross-subject development')
vals=[100*strict['mean_window_BA']]+[100*r['mean_window_BA'] for r in aligned]
axes[1].bar(['Train-only scaler','Session center','Session z-score'],vals,color=['#708090','#c88c40','#c88c40'])
axes[1].set_title('Same Logistic: offline target statistics')
for ax in axes:
    ax.axhline(100/3,color='gray',ls='--',lw=1,label='Chance (3 classes)');ax.set_ylim(0,100)
    ax.set_ylabel('Subject-mean window BA (%)');ax.tick_params(axis='x',labelsize=9)
    for bar in ax.patches:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.2f}',ha='center')
    ax.legend(loc='upper left',fontsize=8);ax.spines[['right','top']].set_visible(False)
fig.suptitle('SEED: 15 subjects, grouped 3-fold validation; no independent test',fontsize=11)
fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
print(json.dumps({'fits':len(rows),'report':str(OUT/'结果报告.md'),'offline_diagnostic_gain_pp':gain}))
