"""Consolidate all completed real-data runs; no fitting or new test evaluation."""
import csv
import hashlib
import json
import re
import statistics
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiment_summary_complete_20261113'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pack(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(v):return str(Decimal(str(v*100)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))+'%'

BATCH={
 'mi_domain_v1':'MI均衡采样与域对齐','mi_ea_search_v1':'MI目标侧EA','mi_specialists_v1':'MI专用网络',
 'mi_supcon_v1':'MI监督对比学习','mirepnet_search_v1':'MIRepNet初始化与EA',
 'seed_features_v1':'SEED训练侧特征基线','seed_alignment_diagnostic_v1':'SEED离线对齐诊断',
 'seed_normalization_search_v1':'SEED归一化搜索','seed_adapted_neural_v1':'SEED目标侧归一化神经网络',
 'faced_normalization_v1':'FACED八类归一化','faced123_features_v1':'FACED九类作者特征',
 'partial_sharing_v1':'CBraMod部分共享','partial_sharing_frozen_v1':'CBraMod冻结共享前缀'}

def unwrap(m):return m.get('window',m)
def canonical(c):
    excluded={'fold','seed','split','train_subjects','val_subjects','validation_subjects','data','counts',
              'initial_shared_sha256','initial_heads_sha256','paired_P4_config_sha256'}
    return {k:v for k,v in c.items() if k not in excluded and 'sha256' not in k and k not in ['import_sources','imports','source_predictions']}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=read(ROOT/'outputs/experiment_summary_20261113/ledger.json')
    for m in old['manifest']:
        p=ROOT/m['实验ID'];assert sha(p/'result.json')==m['结果SHA256'];assert sha(p/'config.json')==m['配置SHA256']
    rows=[];evals=[];curves=[];audits=[];covered=set()
    def add(folder,batch,task,model,condition,c,r,m,source,*,fold=None,params=None,fit=True,selection='任务自身验证BA',note='',pred=None,test=None,protocol='公开开发集三折',record_suffix=''):
        m=unwrap(m) if m else {};fid=rel(folder);covered.add(rel(source))
        p=canonical(c) if params is None else params
        key=hashlib.sha256(pack([protocol,task,model,condition,p]).encode()).hexdigest()[:14]
        split=c.get('split',{})
        train=split.get('train',c.get('train_subjects'));val=split.get('val',c.get('val_subjects',c.get('validation_subjects')))
        if task=='SEED三类' and train is None and fold:
            val=list(range((fold-1)*5+1,fold*5+1));train=sorted(set(range(1,16))-set(val))
        metric=m.get('subject_mean_BA')
        row={'任务':task,'模型':model,'条件':condition,'协议':protocol,'批次':batch,'折':fold,'种子':c.get('seed',r.get('seed')),
          '验证BA':metric,'验证宏F1':m.get('macro_f1'),'视频级BA':r.get('metrics',{}).get('video_trial',{}).get('subject_mean_BA'),
          '测试BA':test,'记录类型':'拟合任务结果' if fit else '复用评分','选模方式':selection,
          '学习率':p.get('lr'),'主干学习率':p.get('backbone_lr',p.get('bone_lr')),'头学习率':p.get('head_lr'),
          '权重衰减':p.get('weight_decay',p.get('wd')),'Batch':p.get('batch_size',p.get('batch')),
          '更新预算':p.get('updates',p.get('updates_per_task')),'轮数预算':p.get('epochs'),
          '最佳位置':None,'运行耗时秒':r.get('seconds',r.get('fit_seconds')) if fit else None,
          '峰值显存MiB':r.get('peak_allocated_MiB') if fit else None,'训练被试':pack(train) if train is not None else None,
          '验证被试':pack(val) if val is not None else None,'完整参数JSON':pack(p),'原始配置JSON':pack(c),
          '配置ID':key,'拟合ID':fid,'任务记录ID':fid+':'+task+record_suffix,'结果来源':rel(source),
          '预测来源':rel(pred) if pred is not None else None,'结果SHA256':sha(source),'配置SHA256':sha(folder/'config.json') if (folder/'config.json').exists() else None,
          '说明':note}
        if metric is not None:assert 0<=metric<=1
        if pred is not None:
            with np.load(pred,allow_pickle=False) as a:
                scores=a['scores'];y=a['y'];s=a['subject'];yhat=scores.argmax(-1)
                assert scores.ndim==2 and len(scores)==len(y)==len(s) and np.isfinite(scores).all()
                # Some classical scores are decision values rather than probabilities.
                subj=[]
                for subject in np.unique(s):
                    ii=s==subject;subj.append(np.mean([np.mean(yhat[ii & (y==cl)]==cl) for cl in np.unique(y[ii])]))
                computed=float(np.mean(subj));assert abs(computed-metric)<1e-10,(pred,computed,metric)
                if task.startswith('MI'):assert not set(s)&{8,9}
                if task.startswith('SSVEP'):assert not set(s)&{7,8,33,34,35}
                if task=='FACED八类':assert not set(s)&{3,11}
            audits.append({'prediction':rel(pred),'BA':metric,'recomputed_BA':computed,'passed':True})
        if fit:
            rows.append(row)
            if (folder/'history.json').exists():
                h=read(folder/'history.json')
                for z in h:
                    ba=z.get('val_BA',z.get('val_balanced_accuracy'));loss=z.get('training_loss',z.get('train_loss',z.get('loss_last_batch',z.get('train_loss_last_batch',z.get('train_ce_last_batch')))))
                    internal={'MI四类':'mi','SSVEP两秒':'ssvep_2s','FACED八类':'emotion'}.get(task,task)
                    if isinstance(ba,dict):ba=ba.get(internal)
                    if isinstance(loss,dict):loss=loss.get(internal)
                    curves.append({'任务':task,'模型':model,'条件':condition,'折':fold,'种子':row['种子'],'位置':z.get('step',z.get('dataset_updates',z.get('updates',z.get('epoch')))),
                      '验证BA':ba,'训练损失':loss,'累计秒':z.get('elapsed_seconds',z.get('seconds')),'拟合ID':fid,'日志来源':rel(folder/'history.json')})
        else:evals.append(row)
        return row

    # Preserve all historical rows and their exact protocol/parameter distinctions.
    taskmap={'MI四分类':'MI四类','SSVEP一秒':'SSVEP一秒','SSVEP两秒':'SSVEP两秒','情绪八类':'FACED八类'}
    for x in old['records']:
        folder=ROOT/x['实验ID'];c=read(folder/'config.json');r=read(folder/'result.json')
        m={'subject_mean_BA':x['验证BA'],'macro_f1':x['验证宏F1']}
        row=add(folder,x['批次'],taskmap[x['任务']],x['模型'],'历史记录：'+x['主干状态'] if x['主干状态'] else '原配置',c,r,m,folder/'result.json',
          fold=x['折'],params={**json.loads(x['完整模型参数JSON']),'epochs_budget':x['轮数预算'],'updates_per_task_budget':x['每数据集步数预算']},protocol=x['协议'],test=x['测试BA'],note=x['说明'],
          selection='三任务平均验证BA选同一检查点' if x['模型代码'] in ['J','F','M'] else '任务自身验证BA')
        row.update({'最佳位置':x['最佳步'] or x['最佳轮'],'轮数预算':x['轮数预算'],'更新预算':x['每数据集步数预算'],
                    '训练被试':x['训练被试'],'验证被试':x['验证被试'],'种子':x['训练种子']})
    # Historical curves use their original recorded loss definitions.
    curves=[]
    for h in old['histories']:
        curves.append({'任务':taskmap[h['任务']],'模型':h['模型'],'条件':h['协议'],'折':h['折'],'种子':h['种子'],
          '位置':h['更新位置'] or h['轮'],'验证BA':h['验证BA'],'训练损失':h['训练损失'],'累计秒':h['累计秒'],'拟合ID':h['实验ID'],'日志来源':h['实验ID']+'/history.json'})
    old_fits={r['拟合ID'] for r in rows};assert len(old_fits)==218
    assert len({r['配置ID'] for r in rows})==old['counts']['configs'], 'Historical budgets/configurations must remain distinct'

    for family,batch in BATCH.items():
        for source in sorted((ROOT/'outputs'/family).glob('*/result.json')):
            folder=source.parent;c=read(folder/'config.json');r=read(source);f=c.get('fold',r.get('fold',0))+1
            if family.startswith('partial_sharing'):
                depth=c['depth'];frozen=family=='partial_sharing_frozen_v1';condition=f'P{depth}'+('冻结共享前缀' if frozen else '共享前缀可训练')
                p={'depth':depth,'readout':'flatten','head':'LayerNorm+Dropout.5+Linear','backbone_lr':1e-4,'head_lr':.001,'wd':.01,'batch':16,'updates':1500,'optimizer':'AdamW','scheduler':'cosine4500','frozen_prefix':frozen}
                for t,task in [('mi','MI四类'),('ssvep_2s','SSVEP两秒'),('emotion','FACED八类')]:
                    cc={**c,'split':c['data'][t]['split']}
                    rr=add(folder,batch,task,'CBraMod部分共享',condition,cc,r,r['best']['metrics'][t],source,fold=f,params=p,
                      selection='三任务平均验证BA选同一检查点',pred=folder/f'{t}_validation.npz',note='一套模型展开三任务；耗时不重复累加')
                    rr['最佳位置']=r['best']['dataset_updates']
                    if frozen:
                        add(folder,batch,task,'CBraMod部分共享',condition+'独立选分支',cc,r,r['taskwise_composed_metrics'][t],source,
                          fold=f,params=p,fit=False,selection='各任务独立选最佳私有分支，与哈希不变共享前缀组合',pred=folder/'taskwise'/f'{t}_validation.npz',record_suffix=':taskwise',note='复用同一次训练，0次新增拟合；开发选优自由度增加')
                continue
            seed_family=family.startswith('seed_');task='SEED三类' if seed_family else 'FACED九类作者特征' if family=='faced123_features_v1' else 'FACED八类' if family=='faced_normalization_v1' else 'MI四类'
            method=c.get('method',c.get('name',folder.name.split('_',1)[-1]))
            if family=='mi_domain_v1':method='EEGNet均衡采样' if c['method'].startswith('eegnet') else '切空间Logistic'
            if family=='mi_supcon_v1':method='EEGNet监督对比';condition=f"lambda={c['lambda']}"
            elif family=='mi_domain_v1':condition=f"CORAL={c['lambda_coral']}" if 'lambda_coral' in c else f"C={c.get('C')}"
            else:condition=c.get('mode',c.get('alignment','训练侧归一化' if seed_family else 'source_only'))
            if family=='seed_features_v1':condition='训练侧归一化'
            p=canonical(c)
            if family=='faced_normalization_v1':
                if method=='tcn':p.update(lr=.001,wd=.01,batch=16,updates=1500,dropout=.5,optimizer='AdamW',scheduler='cosine1500')
                else:p.update(C=.1,class_weight='balanced',max_iter=2000)
            if family=='seed_normalization_search_v1':method='Logistic';p.update(C=.1,class_weight='balanced',max_iter=2000)
            if family=='seed_alignment_diagnostic_v1':method='Logistic '+c['method']
            is_classical=('logistic' in method.lower() or 'svm' in method.lower())
            if is_classical:
                for k in ['lr','wd','weight_decay','updates','batch','batch_size','dropout','optimizer','scheduler']:p.pop(k,None)
                p.setdefault('C',c.get('parameter',.1))
            pred=next((x for x in [folder/'validation.npz',folder/'mi_validation.npz',folder/'emotion_validation.npz'] if x.exists()),None)
            assert pred is not None,folder
            note='开发折和种子共同变化；不作显著性声明。'
            if seed_family:note+='上游作者DE-LDS为离线特征；causal仅指下游归一化。'
            if family=='faced123_features_v1':note+='123人九类作者预计算特征，上游来源未独立重建；与八类分表。'
            if family=='faced_normalization_v1' and condition!='strict':note+='使用无标签目标统计，历史顺序需部署前核验。'
            row=add(folder,batch,task,method,condition,c,r,r['metrics'],source,fold=f,params=p,pred=pred,note=note)
            if (folder/'best_validation.json').exists():
                best=read(folder/'best_validation.json');row['最佳位置']=best.get('step',best.get('updates'))

    for family,task in [('mi_ensembles_v1','MI四类'),('emotion_ensembles_v1','SEED三类')]:
        for source in sorted((ROOT/'outputs'/family).glob('*/result.json')):
            folder=source.parent;c=read(folder/'config.json');r=read(source);assert r['new_fits']==0
            f=c.get('fold') if family.startswith('mi_') else int(re.search(r'fold(\d+)',folder.name)[1])
            condition=c.get('condition','source_only');model='EEGNet融合'+c['name'] if family.startswith('mi_') else 'Logistic/TCN/EmT概率融合'
            pred=next(folder.glob('*validation.npz'))
            add(folder,'MI融合评分' if family.startswith('mi_') else 'SEED融合评分',task,model,condition,c,r,r['metrics'],source,fold=f,
              params={k:v for k,v in canonical(c).items() if k!='source_prediction_sha256'},fit=False,pred=pred,note='0次新增训练；权重在开发结果上选择')

    for folder in sorted((ROOT/'outputs/ssvep_transfer_v1').glob('*s_fold*')):
        c=read(folder/'config.json');r=read(folder/'fit_result.json');task='SSVEP一秒' if c['seconds']==1 else 'SSVEP两秒';f=c['fold']+1
        p={k:v for k,v in canonical(c).items() if k not in ['seconds','strict_subject_generalization','test_evaluated']}
        for method in ['fbtrca','fbcca','fusion25','fusion50','fusion75']:
            source=folder/f'{method}_metrics.json';m=read(source);fit=method=='fbtrca'
            add(folder,'SSVEP源模板迁移',task,method,'源被试模板；离线滤波上下文',c,r,m,source,fold=f,
              params={**p,'method':method,'template_fusion_weight':int(method[6:])/100 if method.startswith('fusion') else None},fit=fit,
              pred=folder/f'{method}_validation.npz',record_suffix=':'+method,note='6套模板拟合共30项方法评分；FBCCA与融合评分不计新增拟合')
        covered.add(rel(folder/'fit_result.json'))

    manifest=ROOT/'outputs/best_candidates_20261113/ssvep_manifest.json'
    for a in read(manifest)['artifacts']:
        artifact=ROOT/a['path'];assert artifact.exists() and sha(artifact)==a['sha256']
        task='SSVEP一秒' if a['seconds']==1 else 'SSVEP两秒';p={'fusion_template_weight':.25,'training_subjects':30,'window_seconds':a['seconds']}
        rr=add(artifact,'SSVEP全开发集重拟合',task,'FBCCA+源模板FBTRCA','30开发被试重拟合',{}, {},None,manifest,params=p,protocol='部署候选重拟合',note='没有新增验证或测试分数；权重与预处理已打包')
        rr['拟合ID']=rel(artifact);rr['任务记录ID']=rel(artifact)+':'+task
    fits={r['拟合ID'] for r in rows};assert len(fits)==376,len(fits)
    assert len(rows)==412,len(rows)
    assert len(evals)==114,len(evals)
    assert len({r['任务记录ID'] for r in rows})==len(rows)
    assert len({r['任务记录ID'] for r in rows+evals})==len(rows)+len(evals)
    seen=set()
    for r in rows:r['独立拟合计数']=int(r['拟合ID'] not in seen);seen.add(r['拟合ID'])
    # Every primary result file is accounted for; unpacked historical inference copies are excluded.
    primary=[p for p in (ROOT/'outputs').rglob('result.json') if not {'inference_check','packages'} & set(p.parts)]
    uncovered=[rel(p) for p in primary if rel(p) not in covered];assert not uncovered,uncovered
    groups=defaultdict(list)
    for r in rows+evals:
        if r['验证BA'] is not None:groups[(r['配置ID'],r['记录类型'])].append(r)
    configs=[]
    for (key,kind),rs in groups.items():
        x=rs[0];values=[r['验证BA'] for r in rs]
        configs.append({'任务':x['任务'],'模型':x['模型'],'条件':x['条件'],'协议':x['协议'],'记录类型':kind,'次数':len(rs),
          '验证BA均值':statistics.mean(values),'验证BA标准差':statistics.stdev(values) if len(values)>1 else None,
          '验证BA最小':min(values),'验证BA最大':max(values),'折':pack([r['折'] for r in rs]),'配置ID':key,'完整参数JSON':x['完整参数JSON']})
    configs.sort(key=lambda x:(x['任务'],x['协议'],x['模型'],x['条件']))
    batch=[]
    for name in dict.fromkeys(r['批次'] for r in rows+evals):
        rr=[r for r in rows if r['批次']==name];ee=[r for r in evals if r['批次']==name]
        batch.append({'批次':name,'独立拟合':len({r['拟合ID'] for r in rr}),'拟合任务记录':len(rr),'复用评分':len(ee)})
    def candidate(label,task,predicate,condition):
        selected=[x for x in rows+evals if x['任务']==task and predicate(x)]
        assert len(selected)==3,(label,len(selected))
        return {'任务':task,'候选':label,'条件':condition,'平均BA':statistics.mean(x['验证BA'] for x in selected),'记录ID':[x['任务记录ID'] for x in selected]}
    candidates=[
      candidate('FBCCA .75 + 源模板FBTRCA .25','SSVEP一秒',lambda x:x['模型']=='fusion25','仅源被试模板；离线滤波上下文'),
      candidate('FBCCA .75 + 源模板FBTRCA .25','SSVEP两秒',lambda x:x['模型']=='fusion25','仅源被试模板；离线滤波上下文'),
      candidate('均衡采样EEGNet','MI四类',lambda x:x['拟合ID'].startswith('outputs/mi_domain_v1/') and x['条件']=='CORAL=0.0','训练侧数据'),
      candidate('EEGNet .75 + Shallow .25','MI四类',lambda x:'mi_ensembles_v1/' in x['拟合ID'] and x['模型']=='EEGNet融合shallow_lr1e3' and json.loads(x['原始配置JSON'])['alpha_specialist']==.25,'训练侧数据；开发选择融合权重'),
      candidate('TCN','SEED三类',lambda x:x['批次']==BATCH['seed_features_v1'] and x['模型']=='tcn','仅训练侧下游归一化；上游DE-LDS离线'),
      candidate('TCN','SEED三类',lambda x:x['批次']==BATCH['seed_adapted_neural_v1'] and x['模型']=='tcn' and x['条件']=='causal','目标历史/当前窗口归一化；上游仍离线'),
      candidate('TCN .5 + EmT .5','SEED三类',lambda x:x['批次']=='SEED融合评分' and x['条件']=='causal' and json.loads(x['原始配置JSON'])['weights_logistic_tcn_emt']==[0,.5,.5],'目标历史/当前窗口归一化；上游仍离线'),
      candidate('TCN .5 + EmT .5','SEED三类',lambda x:x['批次']=='SEED融合评分' and x['条件']=='offline' and json.loads(x['原始配置JSON'])['weights_logistic_tcn_emt']==[0,.5,.5],'整会话目标统计；离线条件'),
      candidate('TCN','FACED八类',lambda x:x['批次']==BATCH['faced_normalization_v1'] and x['模型']=='tcn' and x['条件']=='strict','仅训练侧下游归一化'),
      candidate('TCN','FACED八类',lambda x:x['批次']==BATCH['faced_normalization_v1'] and x['模型']=='tcn' and x['条件']=='offline','整被试目标统计；离线条件'),
      candidate('TCN','FACED九类作者特征',lambda x:x['模型']=='tcn' and x['条件']=='train_only','作者预计算特征；仅训练侧下游归一化'),
      candidate('TCN','FACED九类作者特征',lambda x:x['模型']=='tcn' and x['条件']=='offline_subject_zscore','作者预计算特征；目标被试离线标准化')]
    control=[]
    specs=[('单任务S','outputs/preliminary_v1/core/',lambda x:'_S_' in x['拟合ID']),
      ('联合J','outputs/preliminary_v1/core/',lambda x:'_J_' in x['拟合ID']),
      ('固定路由F','outputs/preliminary_v1/core/',lambda x:'_F_' in x['拟合ID']),
      ('学习路由M','outputs/preliminary_v1/core/',lambda x:'_M_' in x['拟合ID']),
      ('预训练全冻结','outputs/initialization_controls_v1/',lambda x:'pretrained_frozen' in x['拟合ID']),
      ('随机初始化全量','outputs/initialization_controls_v1/',lambda x:'random_full' in x['拟合ID']),
      ('残差均匀融合','outputs/residual_adapters_v1/',lambda x:x['模型']=='异构残差均匀融合'),
      ('残差学习门控','outputs/residual_adapters_v1/',lambda x:x['模型']=='异构残差学习门控'),
      ('残差去谱门控','outputs/residual_adapters_v1/',lambda x:x['模型']=='异构残差去谱'),
      ('私有末两层P2','outputs/partial_sharing_v1/',lambda x:x['条件'].startswith('P2')),
      ('私有末四层P4','outputs/partial_sharing_v1/',lambda x:x['条件'].startswith('P4')),
      ('冻结前缀P4联合选模','outputs/partial_sharing_frozen_v1/',lambda x:x['记录类型']=='拟合任务结果'),
      ('冻结前缀P4独立选分支','outputs/partial_sharing_frozen_v1/',lambda x:x['记录类型']=='复用评分')]
    for label,prefix,predicate in specs:
        item={'设置':label,'记录ID':{}}
        for task in ['MI四类','SSVEP两秒','FACED八类']:
            ss=[x for x in rows+evals if x['任务']==task and x['拟合ID'].startswith(prefix) and predicate(x)]
            assert len(ss)==3,(label,task,len(ss))
            item[task]=statistics.mean(x['验证BA'] for x in ss);item['记录ID'][task]=[x['任务记录ID'] for x in ss]
        control.append(item)
    payload={'records':rows,'evaluations':evals,'configs':configs,'histories':curves,'historical_tests':old['tests'],
      'datasets':old['datasets'],'batches':batch,'candidates':candidates,'controls':control,'counts':{'fits':len(fits),'training_task_records':len(rows),'reuse_evaluations':len(evals),
      'historical_test_records':len(old['tests']),'configurations':len(configs),'curve_records':len(curves),'recomputed_predictions':len(audits),
      'deployment_refits_without_new_accuracy':2},'audit':{'uncovered_primary_results':uncovered,'source_hash_verified_old_fits':len(old_fits),
      'excluded_inference_copies':12,'new_test_evaluation':False,'prediction_checks':audits}}
    save(OUT/'ledger.json',payload)
    csvout('全部拟合任务记录.csv',rows);csvout('复用评分.csv',evals);csvout('配置汇总.csv',configs);csvout('批次统计.csv',batch)
    csvout('训练曲线.csv',curves)
    lines=['# OmniEEG 总实验汇总','', '截至2026-11-13。仅汇总本机保存的已完成实验，不新增训练或测试评分。',
      '',f'共 **{len(fits)}次独立拟合、{len(rows)}条拟合任务记录、{len(evals)}条复用评分、{len(configs)}个参数与评分配置**。其中2次是部署候选重拟合，没有新增准确率。另保留10条历史测试/集成记录，不额外计为拟合。',
      '', '一套联合模型展开多个任务行；同一次冻结训练的独立选分支、MI/SEED概率融合、SSVEP固定公式及融合评分均不重复计训练。12份打包/解包副本排除。',
      '', '## 数据与比较规则','',
      '- MI：BNCI2014-001，仅T会话，四类、22通道、4秒。当前三折开发使用被试1—7，旧测试8/9排除。',
      '- SSVEP：Tsinghua Benchmark，40类，1秒/2秒；当前开发30人，旧测试7/8/33/34/35排除。源缓存64通道，模板方法选择9个后枕通道。整试次离线滤波后裁窗，不代表严格在线时延。',
      '- FACED八类：早期公开导出12人、26通道、10秒，缺Neutral；当前开发10人，旧测试3/11排除。',
      '- SEED：15人、每人3会话、三类，62通道×5频带作者DE-LDS，10秒非重叠特征窗。按被试五人一组作三折，上游LDS可能使用整段视频上下文。',
      '- FACED九类作者特征：123人、30通道×5频带、10帧窗口。作者数组行索引按41人分三折；完整上游处理来源未独立重建，不等同原八类原始EEG实验。',
      '- BA为被试平均平衡准确率。SEED视频级BA另列，不混入窗口级BA。不同数据、类别数、目标统计条件、固定划分和三折均值不得直接排名。折和种子共同变化，不作统计显著性声明。',
      '- 本表保留历史测试成绩，但当前候选仍是开发选优结果，没有新增独立测试或官方成绩。半决赛可用公开权重；决赛从零预训练尚未完成。',
      '', '## 当前专用候选','', '|任务|模型|条件|三折平均BA|','|---|---|---|---:|']
    lines += [f"|{x['任务']}|{x['候选']}|{x['条件']}|{pct(x['平均BA'])}|" for x in candidates]
    lines += ['', '以上是不同条件下的候选，不是同一协议的全局最优排名。SEED目标侧归一化使用无标签目标数据，causal仅指归一化器；作者DE-LDS上游仍离线。FACED八类目标历史归一化还需核实真实采集顺序。',
      '', '## 统一模型与消融','', '|设置|MI四类BA|SSVEP两秒BA|FACED八类BA|','|---|---:|---:|---:|']
    lines += [f"|{x['设置']}|{pct(x['MI四类'])}|{pct(x['SSVEP两秒'])}|{pct(x['FACED八类'])}|" for x in control]
    lines += ['', 'CBraMod核心对照使用同一预训练主干、flatten读出和任务分类头；每任务1500次更新、batch16、主干学习率1e-4、头学习率1e-3、AdamW、wd0.01。主干冻结对照保留训练态Dropout。S按各任务BA选模，J/F/M/P按三任务BA均值选一个检查点。参数总量和训练预算有差异，不能把所有差值归因于路由。',
      '', 'P4相对J：MI−0.71pp、SSVEP+0.90pp、情绪+1.43pp，三任务均值仅+0.54pp；未过预设1pp门槛，未追加条件性容量对照。冻结前缀的任务分支可精确组合，独立选模没有新增训练，但增加验证选优自由度。',
      '', '## 关键结论','',
      '1. SSVEP源模板融合是当前成熟的专用候选。早期“98%”来自固定划分验证；当时两秒FBCCA验证98.33%、历史测试84.92%。当前三折融合95.67%属于另一协议，不可混写。',
      '2. MI的均衡采样EEGNet和Shallow融合优于当前CBraMod。CORAL、目标侧EA、监督对比、频移及MIRepNet在已测参数下未稳定超过基线；负结果仍保留，不排除更充分调参。',
      '3. 情绪提升与目标统计处理高度相关。SEED训练侧TCN55.64%与目标历史归一化TCN78.06%的差异提示域偏移重要，但不构成对域偏移唯一原因的证明。',
      '4. 统一联合、固定/学习路由、异构残差与部分共享未得到稳定协同收益。当前应把统一预训练初始化与任务特定适配作为基线，复杂共享结构保留为待验证方向。',
      '5. 公开开发实验不等于比赛正式结果。官方六数据集、官方被试划分与种子、决赛从零预训练仍需独立完成。',
      '', '## 实验批次','', '|批次|独立拟合|拟合任务记录|复用评分|','|---|---:|---:|---:|']
    lines += [f"|{x['批次']}|{x['独立拟合']}|{x['拟合任务记录']}|{x['复用评分']}|" for x in batch]
    lines += ['', '## 文件与核验','',
      f'- Excel含总览、配置汇总、逐次训练、复用评分、早期测试与数据协议；完整训练曲线另存CSV（{len(curves)}行）。',
      '- 逐次训练与评分保留完整参数JSON、原始配置、源记录和SHA256；日志未记录的值不补造。归一化补充实验中可从源码确认的参数已单列，经典模型的更新/学习率字段留空。',
      f'- 旧218次拟合的结果/配置哈希均与原总表一致；新纳入的{len(audits)}份保存预测已重新计算BA并核验。打包/解包副本不计数，所有主结果文件均已纳入。',
      '- 本次仅汇总与复算保存预测，没有重新训练模型、重新使用测试集或重新优化参数。']
    (OUT/'总实验汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'collection_audit.json',payload['audit']);print(pack(payload['counts']))

if __name__=='__main__':main()
