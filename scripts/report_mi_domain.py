"""Report paired development results, without scoring legacy test inputs."""
import csv
import json
from pathlib import Path
import numpy as np
from search_models import write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mi_domain_v1'
status=json.loads((OUT/'queue_status.json').read_text())
assert status['status']=='finished'
rows=list(csv.DictReader((OUT/'results.csv').open(encoding='utf-8-sig')))
legacy=[json.loads((ROOT/f'outputs/preliminary_v1/supplement/fold{i}_eegnet/result.json').read_text())['best']['metrics']['subject_mean_BA'] for i in range(1,4)]
groups=[('原 EEGNet，普通采样',legacy)]
for method,param,label in [('eegnet_source_balanced_coral',0,'均衡采样 EEGNet，λ=0'),
        ('eegnet_source_balanced_coral',1,'均衡采样 EEGNet，λ=1'),
        ('eegnet_source_balanced_coral',10,'均衡采样 EEGNet，λ=10'),
        ('oas_logeuclidean_tangent_logistic',.1,'OAS 切空间 Logistic，C=0.1'),
        ('oas_logeuclidean_tangent_logistic',1,'OAS 切空间 Logistic，C=1'),
        ('oas_logeuclidean_tangent_logistic',10,'OAS 切空间 Logistic，C=10')]:
    selected=sorted([r for r in rows if r['method']==method and float(r['parameter'])==param],key=lambda r:int(r['fold']))
    assert len(selected)==3
    groups.append((label,[float(r['val_BA']) for r in selected]))
paired=[]
for fold in range(3):
    paths=[OUT/f'fold{fold+1}_balanced_coral{lam}' for lam in [0,1,10]]
    cfg=[json.loads((p/'config.json').read_text()) for p in paths]
    assert len({c['schedule_sha256'] for c in cfg})==1
    for p in paths:
        index=np.load(p/'training_indices.npy')
        assert index.shape==(1500,16)
    paired.append({'fold':fold+1,'same_sampling_all_lambdas':True})
write(OUT/'paired_sampling_audit.json',paired)
report=['# MI 源域泛化对照结果（2026-11-13）','',
    '18 次新拟合完成；沿用三折开发被试与训练种子，旧测试被试 8、9 排除。以下都是验证被试平均 BA，不是新测试成绩。','',
    '| 方法 | 第1折 | 第2折 | 第3折 | 均值 |','|---|---:|---:|---:|---:|']
for label,values in groups:
    report.append('| '+label+' | '+' | '.join(f'{100*v:.2f}%' for v in [*values,np.mean(values)])+' |')
baseline=np.array(groups[1][1]);old=np.array(legacy)
report+=['',f'均衡采样 λ=0 相对旧普通采样平均变化：{100*(baseline-old).mean():+.2f} 个百分点。两者训练预算、架构和各折初始化种子相同，采样序列不同；旧配置复用，不算新拟合。',
    '同折及同采样的 CORAL 对照：']
for label,values in groups[2:4]:
    delta=np.array(values)-baseline
    report.append(f'- {label}：相对 λ=0 平均 {100*delta.mean():+.2f} 个百分点；逐折 '+', '.join(f'{100*v:+.2f}' for v in delta)+'。')
report+=['','配置固定为 λ=0/1/10 和 C=0.1/1/10，全部配置均报告。每个神经模型每100步验证，按验证BA取checkpoint；不同配置均值选择仍属于开发集选择。三折与随机种子共同变化，不能称为显著性证据或纯种子标准差。',
    'CORAL 是本地 EEGNet 训练目标对照，不是完整 SCLDGN 复现。切空间方法使用 log-Euclidean 参考，不声称复现迭代仿射不变均值算法。',
    '18 份预测独立复算、18 份模型重载以及3组配对抽样核验通过。未重新评分旧测试。','',
    'MIRepNet：108 个主干张量加载成功，真实输入 2×45×1000 前向反向通过；四分类头新建。归一化参考仅训练被试拟合，区别于作者的目标被试 EA。当前未训练、没有MIRepNet准确率。',
    'SEED：等待用户下载，本轮没有新增情绪结果。']
(OUT/'结果报告.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print('\n'.join(report))
