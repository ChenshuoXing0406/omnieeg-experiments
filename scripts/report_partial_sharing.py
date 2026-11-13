"""Competition-oriented comparison of completed sharing-boundary controls."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_partial_sharing import ROOT,OUT,TASKS,read,export
from search_models import write
from run_preliminary import PLAN

def main():
    status=read(OUT/'queue_status.json');assert status['status']=='finished'
    frozen_out=ROOT/'outputs/partial_sharing_frozen_v1';assert read(frozen_out/'queue_status.json')['status']=='finished'
    rows,summary=export();names={'S':'Single task','J':'Fully shared','task_private':'Task-private','rotate_assignment':'Rotating assignment','frozen_prefix':'Frozen prefix','frozen_taskwise':'Frozen/taskwise selection'}
    from search_models import metrics
    extra_audit=[]
    for fold in range(1,4):
        folder=frozen_out/f'fold{fold}_frozen_P4';r=read(folder/'result.json')
        assert r['frozen_shared_unchanged']
        for group in ['frozen_prefix','frozen_taskwise']:
            for task in TASKS:
                pred_path=folder/f'{task}_validation.npz' if group=='frozen_prefix' else folder/'taskwise'/f'{task}_validation.npz'
                expected=r['best']['metrics'][task] if group=='frozen_prefix' else r['taskwise_composed_metrics'][task]
                error=r['reload_max_abs'][task] if group=='frozen_prefix' else r['taskwise_reload_max_abs'][task]
                with np.load(pred_path) as p:
                    assert not set(p['subject']) & set(PLAN['legacy_test_subjects_excluded'][task])
                    assert np.isfinite(p['scores']).all() and np.allclose(p['scores'].sum(-1),1,atol=1e-5)
                    m=metrics(p['y'],p['scores'].argmax(-1),p['subject'],p['scores'].shape[-1])
                    assert abs(m['subject_mean_BA']-expected['subject_mean_BA'])<1e-10
                rows.append({'group':group,'depth':4,'fold':fold,'task':task,'BA':m['subject_mean_BA'],
                    'path':str(pred_path.parent.relative_to(ROOT)),'reused':False})
                extra_audit.append({'path':str(pred_path.parent.relative_to(ROOT)),'task':task,'reload_error':error,'frozen_shared_unchanged':True})
    for group in ['frozen_prefix','frozen_taskwise']:
        for task in TASKS:summary.append({'group':group,'depth':4,'task':task,'folds':3,
            'mean_BA':float(np.mean([r['BA'] for r in rows if r['group']==group and r['task']==task]))})
    with (OUT/'complete_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(OUT/'complete_summary.json',summary)
    write(OUT/'complete_prediction_reload_audit.json',read(OUT/'prediction_reload_audit.json')+extra_audit)
    def folder_for(group,depth,fold):
        if group=='J':return ROOT/f'outputs/preliminary_v1/core/fold{fold}_J_joint'
        if group in ['frozen_prefix','frozen_taskwise']:return frozen_out/f'fold{fold}_frozen_P4'
        return OUT/f'fold{fold}_{group}_P{depth}'
    groups=sorted({(r['group'],r['depth']) for r in rows},key=lambda x:({'S':0,'J':1,'task_private':2,'rotate_assignment':3,'frozen_prefix':4,'frozen_taskwise':5}[x[0]],x[1]))
    means={(r['group'],r['depth'],r['task']):r['mean_BA'] for r in summary}
    resources=[];paired=[];selection=[];structures=[]
    for group,depth in groups:
        if group=='S':
            configs=[read(ROOT/f'outputs/preliminary_v1/core/fold1_S_{task}/config.json') for task in TASKS]
            total=sum(c['parameters_total'] for c in configs)
            trainable=sum(c['parameters_trainable'] for c in configs)
            count_note='sum of3 separately stored task models'
        elif group=='J':
            cfg=read(ROOT/'outputs/preliminary_v1/core/fold1_J_joint/config.json');total=cfg['parameters_total'];trainable=cfg['parameters_trainable'];count_note='one shared model +3heads'
        else:
            cfg=read(folder_for(group,depth,1)/'config.json');total=cfg['counts']['total'];trainable=cfg['counts']['trainable'];count_note='one shared prefix +3tails +3heads'
        structures.append({'group':group,'depth':depth,'stored_parameters':total,'trainable_parameters':trainable,
            'active_parameters_per_task':read(OUT/'smoke.json')['passed'][0]['counts']['active_per_task'],
            'taskwise_reuses_frozen_training':group=='frozen_taskwise'})
        for fold in range(1,4):
            if group=='S':
                paths=[ROOT/f'outputs/preliminary_v1/core/fold{fold}_S_{task}' for task in TASKS]
                seconds=sum(read(p/'result.json')['seconds'] for p in paths)
                memory=max(read(p/'result.json')['peak_allocated_MiB'] for p in paths)
            else:
                folder=folder_for(group,depth,fold)
                result=read(folder/'result.json');seconds=result['seconds'];memory=result['peak_allocated_MiB']
            resources.append({'group':group,'depth':depth,'fold':fold,'stored_parameters':total,'training_wall_seconds':seconds,
                              'peak_allocated_MiB':memory,'note':count_note})
        for task in TASKS:
            values=[]
            for fold in range(1,4):
                value=next(r['BA'] for r in rows if (r['group'],r['depth'],r['fold'],r['task'])==(group,depth,fold,task))
                joint=next(r['BA'] for r in rows if (r['group'],r['fold'],r['task'])==('J',fold,task))
                single=next(r['BA'] for r in rows if (r['group'],r['fold'],r['task'])==('S',fold,task))
                values.append(value);paired.append({'group':group,'depth':depth,'fold':fold,'task':task,'BA':value,
                    'delta_vs_joint_pp':100*(value-joint),'delta_vs_single_pp':100*(value-single)})
            if group!='S':
                for fold in range(1,4):
                    folder=folder_for(group,depth,fold)
                    history=read(folder/'history.json')
                    selection.append({'group':group,'depth':depth,'fold':fold,'task':task,
                        'taskwise_validation_max':max(r['val_BA'][task] for r in history),
                        'mean_selected_checkpoint_BA':next(r['BA'] for r in rows if (r['group'],r['depth'],r['fold'],r['task'])==(group,depth,fold,task)),
                        'note':('frozen shared prefix permits exact composition of independently selected branches; no extra fit' if group=='frozen_taskwise' else 'diagnostic validation maxima may occur at different checkpoints; not one deployable shared model')})
    for filename,records in [('逐折配对差值.csv',paired),('参数与资源.csv',resources),('选模耦合诊断.csv',selection)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write(OUT/'parameter_structure.json',structures)
    lines=['# 面向比赛方案的共享边界实验','',
        f'新训练{status["new_fits"]+3}次（共享边界{status["new_fits"]}次＋冻结前缀3次），复用已完成S/J共12次；新预测复算和完整验证集checkpoint重载均通过。所有结果为公开开发数据，旧测试被试排除。',
        '## 单一联合检查点与可组合任务分支结果','',
        '| 组别 | MI BA | SSVEP两秒 BA | 情绪八类 BA | 三任务均值 | 总存储参数 |',
        '|---|---:|---:|---:|---:|---:|']
    for group,depth in groups:
        values=[means[group,depth,t] for t in TASKS];params=next(r['stored_parameters'] for r in resources if r['group']==group and r['depth']==depth)
        label=names[group]+(f' P{depth}' if depth else '')
        lines.append('| '+label+' | '+' | '.join(f'{100*v:.2f}%' for v in [*values,np.mean(values)])+f' | {params:,} |')
    lines+=['','## 对方案的直接结论','']
    def changes(group,depth,reference,reference_depth):
        return ', '.join(f'{t} {100*(means[group,depth,t]-means[reference,reference_depth,t]):+.2f}pp' for t in TASKS)
    lines.append('部分共享P4相对完全共享J：'+changes('task_private',4,'J',0)+'；相对单任务S：'+changes('task_private',4,'S',0)+'。')
    lines.append('冻结共享前缀、保持联合选模，相对可训练P4：'+changes('frozen_prefix',4,'task_private',4)+'。这是当前学习率与训练预算下的冻结策略对照，不排除另行优化冻结方案。')
    lines.append('同一次冻结训练，仅改为独立选分支，相对联合选模：'+changes('frozen_taskwise',4,'frozen_prefix',4)+'。该差值包含验证集选优自由度增加，不能作为独立测试泛化收益；但组合后的模型确实可部署。')
    lines.append('当前证据用于决定哪些设计值得保留为候选，不能把“共享预训练骨干＋独立适配”的工程可行性写成已证明跨任务协同。初赛材料应如实呈现有效和无效对照；后续正式数据需重新固定官方协议与模型选择。')
    lines+=['','S各任务单独按自身验证BA选模型；J/P2/P4及轮转对照均按三个验证BA等权均值选一个检查点。Frozen prefix也保留这个规则，Frozen/taskwise则在同一次训练中独立选私有分支和头，再与哈希不变的共享前缀组合成一份model_taskwise.pt；该组没有额外训练，已验证组合输出与各任务选中时一致。S和联合组的选择约束不同，差值不能全归因为梯度冲突。另附选模耦合诊断CSV；共享权重会变化的模型，其逐任务历史上界不等于能组合部署的一份共享权重。','',
        '## 受控条件与有效范围','',
        '- 相同公开CBraMod、头初始化、数据划分、采样索引、每任务1500更新、batch16、主干lr1e-4／头lr1e-3、AdamW、cosine、类别权重和更新次序。',
        '- P2共享前10层、每任务独占最后2层；P4共享前8层、每任务独占最后4层。每次路径仍经过完整12层，激活参数与J一致，总参数不同。',
        '- Frozen prefix P4与P4完全相同的初始化、结构与采样，但共享patch embedding及前8层不更新，训练时仍保留原Dropout；逐折核验共享权重哈希在训练前后完全不变。它隔离共享前缀更新策略，仍不等价于重新预训练的模型。',
        '- smoke核验两种模型初始化logits与J完全一致、私有参数不别名、未选分支不更新；避免把初始化与计算深度变化算成共享边界收益。',
        '- 如触发容量控制，轮转组存储参数、激活深度和每分支更新数与被选P组一致，每个分支对各任务训练500次，验证使用预先固定分支，不选最佳分支。它检验固定任务分工与混合训练的区别，但轮转动态本身也是条件，不能宣称完全隔离所有机制。',
        '- 参数量和训练时间分别报告；本轮部分时段检测到电池供电与GPU降频，墙钟还包含数据/验证等开销，不作为架构之间的严格吞吐排名或云端算力估计。',
        '- 当前只有MI22x800、SSVEP64x400、FACED26x2000真实代理数据；不是官方六数据集。六种官方形状另做合成接口检查，假标签和两类输出不代表官方类别数或真实成绩。',
        '- 三任务BA均值仅为本地联合检查点选择指标，不是比赛主赛道积分；我们选择三个分赛道，每个正式分赛道还需两个官方数据集。',
        '- 本轮只验证下游共享与适配方式，不代表已完成决赛从零基础预训练。任务专用先验、均衡采样、时序头的联合收益不能由旧专用模型成绩替代证明。',
        '- 三折和种子共同变化，不作统计显著性声明，不把开发选优结果当独立测试。','',
        '## 容量控制决策','', '```json',json.dumps(read(OUT/'capacity_control_decision.json'),ensure_ascii=False,indent=2),'```','']
    if status['capacity_control_executed']:
        control=next((g,d) for g,d in groups if g=='rotate_assignment');depth=control[1]
        delta=[means['task_private',depth,t]-means['rotate_assignment',depth,t] for t in TASKS]
        lines.append('固定任务分工相对同容量轮转组的平均变化：'+', '.join(f'{t} {100*d:+.2f}pp' for t,d in zip(TASKS,delta))+f'；三任务均值 {100*np.mean(delta):+.2f}pp。')
    else:lines.append('两种私有深度均未通过预先约定的实用门槛，未追加容量对照；不能据此宣称部分共享有效。')
    lines+=['','报告附表：complete_results.csv、逐折配对差值.csv、参数与资源.csv、选模耦合诊断.csv。原始训练曲线、完整配置和预测见各fold目录。']
    (OUT/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    labels=[names[g]+(f' {d}' if d else '') for g,d in groups]
    for ax,task,title in zip(axes,TASKS,['MI (4 classes)','SSVEP (40 classes)','FACED (8 classes)']):
        values=[100*means[g,d,task] for g,d in groups]
        ax.bar(np.arange(len(values)),values,color=['#83909c','#617586','#277c8e','#459177','#c58c42','#8b6baa','#ab6288'][:len(values)])
        ax.set_xticks(np.arange(len(values)),labels,rotation=23,ha='right',fontsize=8);ax.set_title(title)
        ax.set_ylabel('Subject-mean validation BA (%)');ax.spines[['right','top']].set_visible(False)
        for i,v in enumerate(values):ax.text(i,v+.5,f'{v:.2f}',ha='center',fontsize=8)
        ax.set_ylim(0,max(values)*1.18)
    fig.suptitle('Paired sharing-boundary study: three development folds, no independent test',fontsize=11)
    fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    write(OUT/'report_audit.json',{'new_fits':status['new_fits']+3,'new_prediction_task_records':len([r for r in rows if not r['reused']]),
        'reused_baseline_task_records':18,'taskwise_selection_extra_training':0,'capacity_control_executed':status['capacity_control_executed'],'test_evaluated':False})
    print(json.dumps({'new_fits':status['new_fits']+3,'groups':groups,'report':str(OUT/'结果报告.md')}))

if __name__=='__main__':main()
