"""Scientific figures from saved development summaries (no EEG or test access)."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/research_update_20261113'


def load(path): return json.loads((ROOT / path).read_text())


def fmt(value):
    return str(Decimal(str(round(float(value), 10))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def panel(ax, series, title, ylim):
    colors = ['#355C7D', '#3C8DAD', '#D88C45', '#7C6A9B']
    for i, (label, values) in enumerate(series):
        values = np.array(values) * 100
        ax.bar(i, values.mean(), color=colors[i], alpha=.82, width=.65)
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), color='#222222', capsize=4, linewidth=1)
        ax.scatter(i + np.linspace(-.13, .13, len(values)), values, color='#222222', s=20, zorder=3)
        ax.text(i, values.mean() + values.std(ddof=1) + ylim[1] * .03, fmt(values.mean()), ha='center', fontsize=10)
    ax.set_xticks(range(len(series)), [s[0] for s in series], fontsize=9)
    ax.set_title(title); ax.set_ylim(*ylim); ax.set_ylabel('Validation balanced accuracy (%)')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load('outputs/preliminary_v1/results.json')
    controls = load('outputs/initialization_controls_v1/results.json')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.dpi': 150})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    for ax, task, title, limit in zip(axes, ['mi', 'ssvep_2s', 'emotion'], ['Motor imagery', 'SSVEP, 2 seconds', 'Emotion, 8 classes'], [(0, 65), (0, 95), (0, 28)]):
        series = [(label, [r['val_BA'] for r in rows if r['family'] == 'core' and r['group'] == group and r['task'] == task])
                  for group, label in [('S', 'Single'), ('J', 'Joint'), ('F', 'Fixed'), ('M', 'Learned')]]
        assert all(len(v) == 3 for _, v in series)
        panel(ax, series, title, limit)
    fig.suptitle('Shared-backbone and expert-routing controls', fontsize=15)
    fig.text(.5, .02, 'Dots: paired development folds. Bars: means. Error bars: sample SD across folds; not confidence intervals.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93)); fig.savefig(OUT / 'joint_routing.png', dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    for ax, task, title, limit in zip(axes, ['mi', 'ssvep_2s', 'emotion'], ['Motor imagery', 'SSVEP, 2 seconds', 'Emotion, 8 classes'], [(0, 65), (0, 95), (0, 35)]):
        series = [(label, [r['BA'] for r in controls if r['mode'] == mode and r['task'] == task])
                  for mode, label in [('pretrained_full', 'Pretrained\nfull FT'), ('pretrained_frozen', 'Pretrained\nfrozen'), ('random_full', 'Random\nfull FT')]]
        assert all(len(v) == 3 for _, v in series)
        panel(ax, series, title, limit)
    fig.suptitle('Matched initialization and backbone-freezing controls', fontsize=15)
    fig.text(.5, .02, 'Identical flatten/head, paired folds, 1,500 updates. Frozen backbone retains training-mode dropout.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93)); fig.savefig(OUT / 'initialization_controls.png', dpi=200); plt.close(fig)
    summary = load('outputs/initialization_controls_v1/summary.json')
    lines = ['# 11月13日实验更新', '', '本轮结果均为三折开发验证BA，不是赛事成绩或独立盲测。折与训练种子共同变化。', '',
             '## 初始化与冻结对照', '', '| 任务 | 模式 | 三折均值BA | 折间样本标准差 |', '|---|---|---:|---:|']
    for r in summary:
        assert r['folds'] == 3
        lines.append(f"| {r['task']} | {r['mode']} | {fmt(r['mean_BA']*100)}% | {fmt(r['std']*100)}个百分点 |")
    lines += ['', 'P-full为原S组，经数据、样本顺序、头初始化与更新预算检查后复用；另补18次训练。冻结组保留训练态dropout，仅冻结骨干参数。',
              '随机初始化组仅在同一下游预算下比较，不代表决赛大规模从零预训练。', '',
              '## 当前可以支持的判断', '',
              '- MI：全量微调比冻结平均高12.50个百分点；预训练比随机平均高2.22个百分点，三个折方向一致。',
              '- SSVEP两秒：全量微调比冻结高43.79个百分点；预训练比随机高38.39个百分点，三个折方向一致。此结论限于当前1500步短训预算。',
              '- 情绪八类：冻结与全量微调均约16.4%，随机初始化平均18.08%；当前没有预训练优势证据。随机结果折间波动较大，不据此断言预训练普遍有害。',
              '- 现有联合稠密及同构FFN专家没有稳定超过同架构单任务对照；内容路由未超过固定路由。不能把当前MoE写成已验证贡献。',
              '- 优先保留已支持的预训练适配路径；将图示异构残差专家与样本门控作为独立待验证扩展。', '',
              '## 方案边界', '', '当前四个同构FFN专家、token级Top-1路由不等同于队友图示的空间/时间/频谱异构专家和样本级门控。',
              '三路全部计算再融合时按密集混合报告；新增谱旁路和门控应分别消融。', '',
              '## 核验文件', '', '../preliminary_v1/audit.json：前一批保存预测重算。',
              '../preliminary_v1/checkpoint_reload_audit.json：核心模型重载检查。',
              '../initialization_controls_v1/audit.json：新对照及复用基线保存预测重算。',
              '../initialization_controls_v1/checkpoint_audit.json：新增模型初值重建、冻结权重与检查点重载检查。', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(OUT)


if __name__ == '__main__': main()
