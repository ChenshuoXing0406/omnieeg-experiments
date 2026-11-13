"""Collect completed real-data fits only; never fit models or score new tests."""
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/experiment_summary_20261113'
TASKS = {'mi': ('MI四分类', 'BNCI2014-001 T', 22, 4, 4),
         'ssvep_1s': ('SSVEP一秒', 'Tsinghua Benchmark', 64, 1, 40),
         'ssvep_2s': ('SSVEP两秒', 'Tsinghua Benchmark', 64, 2, 40),
         'emotion': ('情绪八类', 'FACED公开处理导出', 26, 10, 8)}
NAMES = {'cbramod_ft':'CBraMod全量微调','cbramod_ridge':'CBraMod冻结＋Ridge','eegnet':'EEGNet式CNN',
         'fbcsp':'滤波组CSP＋LDA','bandpower_lda':'频带功率＋LDA','bandpower_svm':'频带功率＋SVM',
         'bandpower_logistic':'频带功率＋Logistic','fbcca':'FBCCA','spectral_ridge':'复频谱＋Ridge',
         'S':'CBraMod单任务稠密','J':'CBraMod联合稠密','F':'CBraMod固定路由','M':'CBraMod学习路由',
         'fbcnet':'FBCNet式网络','sequence_tcn':'一秒频带序列＋TCN','sequence_logistic':'一秒特征时间均值＋Logistic',
         'pretrained_frozen':'CBraMod预训练冻结主干','random_full':'CBraMod随机初始化全量训练',
         'linear_probe':'CBraMod五轮冻结探测','full_finetune':'CBraMod五轮全量微调','random':'CBraMod五轮随机初始化'}
NAMES.update(residual_uniform='异构残差均匀融合',residual_learned='异构残差学习门控',residual_no_spectrum='异构残差去谱')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def packed(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(x): return hashlib.sha256(packed(x).encode()).hexdigest()[:12]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records=[]; histories=[]; test_rows=[]; manifest=[]; checks=[]
    plan=read(ROOT/'plans/preliminary_20261112.json')
    metas={t:read(ROOT/'data/processed'/f'{"BNCI2014_001_T_pilot" if t=="mi" else "emotion_10s" if t=="emotion" else t}.json') for t in TASKS}

    def add(folder, batch, task, model, params, result, config, metric, protocol, **extra):
        fit=folder.relative_to(ROOT).as_posix(); dims=TASKS[task]
        split=extra.pop('split'); val=metric.get('subject_mean_BA',extra.pop('val',None))
        key=ident([protocol, task, model, params, extra.get('epochs_budget'),extra.get('updates_budget')])
        row={'实验ID':fit,'任务记录ID':fit+':'+task,'批次':batch,'协议':protocol,'任务':dims[0],
             '数据集':dims[1],'模型':NAMES.get(model,model),'模型代码':model,'配置ID':key,
             '阶段':extra.pop('stage',None),'折':extra.pop('fold',None),'训练种子':result.get('seed',config.get('seed')),
             '验证BA':val,'验证宏F1':metric.get('macro_f1'),'验证合并BA':metric.get('pooled_BA'),
             '测试BA':None,'测试宏F1':None,'测试是否评估':False,
             '预训练权重':extra.pop('pretrained',None),'主干状态':extra.pop('backbone',None),
             '读出':params.get('readout'),'分类头':params.get('head'),
             '骨干学习率':params.get('backbone_lr'),'头学习率':params.get('head_lr'),'学习率':params.get('lr'),
             '权重衰减':params.get('wd'),'Dropout':params.get('dropout'),'Batch':params.get('batch'),
             '归一化':params.get('norm'),'优化器':params.get('optimizer'),'学习率调度':params.get('scheduler'),
             '轮数预算':extra.pop('epochs_budget',None),'实际轮数':result.get('epochs_completed'),
             '最佳轮':result.get('best_epoch'),'每数据集步数预算':extra.pop('updates_budget',None),
             '实际更新步数':result.get('updates',result.get('optimizer_updates')),
             '最佳步':extra.pop('best_step',None),'运行耗时秒':result.get('seconds',result.get('training_seconds')),
             '耗时统计口径':'联合模型总耗时；跨任务行不累加' if model in ['J','F','M'] else '该次拟合日志计时',
             '峰值显存MiB':result.get('peak_allocated_MiB'),'输入通道数':dims[2],'输入秒数':dims[3],'采样率Hz':200,'类别数':dims[4],
             '滤波器数':params.get('filters'),'卷积核':params.get('kernel'),'alpha':params.get('alpha'),
             'C':params.get('C'),'gamma':params.get('gamma'),'功率归一化':params.get('power_norm'),
             '频带':packed(params['bands']) if 'bands' in params else None,'CSP分量':params.get('components'),
             '协方差收缩':params.get('shrink'),'CCA通道子集':params.get('channels'),'谐波数':params.get('harmonics'),
             '滤波器组数':params.get('banks'),'频带权重指数':params.get('exponent'),
             '专家数':params.get('experts'),'TopK':params.get('top_k'),'均衡损失系数':params.get('balance'),
             '训练被试':packed(split['train']),'验证被试':packed(split['val']),
             '测试或排除被试':packed(split.get('test',plan['legacy_test_subjects_excluded'].get(task,[]))),
             '确定性拟合':extra.pop('deterministic',False),'完整模型参数JSON':packed(params),
             '参数来源':'config.json；未显式记录的默认值按训练源码补齐',
             '验证逐被试BA':packed(metric.get('per_subject_BA',{})),
             '结果来源':fit+'/result.json','代码SHA256':config.get('source_sha256',config.get('code_sha256')),
             '数据SHA256':config.get('data_sha256',config.get('data',{}).get(task,{}).get('sha256')),
             '权重来源SHA256':config.get('checkpoint_sha256',config.get('pretrained_sha256',config.get('pretrained_source_sha256'))),
             '说明':extra.pop('note','')}
        assert not extra,extra
        records.append(row)
        checks.append({'record':row['任务记录ID'],'value':val,'passed':isinstance(val,(int,float)) and 0<=val<=1})
        if (folder/'history.json').exists():
            for h in read(folder/'history.json'):
                ba=h.get('val_BA',h.get('val_balanced_accuracy'))
                if isinstance(ba,dict):ba=ba[task]
                loss=h.get('train_loss',h.get('training_loss',h.get('train_loss_last_batch')))
                if isinstance(loss,dict):loss=loss[task]
                histories.append({'实验ID':fit,'任务':dims[0],'协议':protocol,'模型':row['模型'],'折':row['折'],
                                  '种子':row['训练种子'],'轮':h.get('epoch'),'更新位置':h.get('updates',h.get('dataset_updates')),
                                  '验证BA':ba,'训练损失':loss,'训练损失口径':'M含均衡辅助项；补充神经模型为末批损失' if model in ['M','eegnet','fbcnet','sequence_tcn'] and batch=='三折对照' else '日志记录值',
                                  '累计秒':h.get('elapsed_seconds',h.get('seconds'))})
        return row

    backbone_arch={'backbone_d_model':200,'backbone_layers':12,'backbone_heads':8,'ffn_dim':800,'patch':200}
    for folder in sorted((ROOT/'outputs/pilot').glob('initial_5ep_*_seed42')):
        c=read(folder/'config.json'); r=read(folder/'result.json'); mode=r['mode']
        p={**backbone_arch,'readout':'flatten','head':'Flatten+LayerNorm+Linear','batch':c['batch_size'],
           'backbone_lr':c['backbone_lr'],'head_lr':c['head_lr'],'wd':.01,'norm':'uV/100','optimizer':'AdamW','scheduler':'none',
           'backbone_mode':'eval' if mode=='linear_probe' else 'train'}
        hist=read(folder/'history.json');best=next(h for h in hist if h['epoch']==r['best_epoch'])
        row=add(folder,'五轮试跑','mi',mode,p,r,c,{'macro_f1':best['val_macro_f1']},'P0：MI固定划分五轮',
                split=c['split'],val=r['best_val_BA'],epochs_budget=5,pretrained=mode!='random',
                backbone='冻结eval' if mode=='linear_probe' else '更新',note='五轮短训；验证指标为合并BA，MI每被试类别均衡，等于被试平均BA')
        row['实际轮数']=5;row['测试BA']=r['test_BA'];row['测试宏F1']=r['test_macro_f1'];row['测试是否评估']=True
        test_rows.append({'协议':row['协议'],'任务':row['任务'],'模型':row['模型'],'类型':'单模型','运行ID':row['实验ID'],
                          '验证BA':row['验证BA'],'验证口径':'单模型验证BA','测试BA':r['test_BA'],'测试宏F1':r['test_macro_f1'],'测试被试':row['测试或排除被试']})

    neural={'eegnet','cbramod_ft'}
    for task in TASKS:
        for folder in sorted((ROOT/'outputs/search_v1'/task/'runs').iterdir()):
            if not (folder/'result.json').exists():continue
            c=read(folder/'config.json');r=read(folder/'result.json');p=dict(r['config']);model=p.pop('model')
            if model in neural:
                p.update(batch=p.get('batch',32),optimizer='AdamW',scheduler='CosineAnnealingLR',smoothing=p.get('smoothing',0.))
                if model=='cbramod_ft':
                    p.update(backbone_arch);p['backbone_lr']=p.pop('lr');p.setdefault('readout','mean')
                    p['head']='Flatten+LayerNorm+Dropout+Linear'
                else:p.update(readout='AdaptiveAvgPool(1,4)+Flatten',head='Linear',depth_multiplier=2)
            elif model=='cbramod_ridge':p.update(backbone_arch);p.setdefault('readout','mean');p.update(head='StandardScaler+Ridge(lsqr,balanced)',norm='uV/100')
            else:
                p['head']='StandardScaler+classifier' if model!='fbcca' else 'CCA score argmax'
                if model.startswith('bandpower'):p.setdefault('power_norm','absolute')
                if model=='bandpower_logistic':p.update(max_iter=1000,class_weight='balanced')
                if model=='fbcsp':p['head']='StandardScaler+LDA(lsqr,shrinkage=auto)'
            if model.endswith('ridge'):p.update(solver='lsqr',class_weight='balanced')
            if model.endswith('lda') or model=='fbcsp':p.update(solver='lsqr',lda_shrinkage='auto')
            if model.endswith('svm'):p.update(kernel='rbf',class_weight='balanced')
            if model.endswith('logistic'):p['solver']='lbfgs'
            if model.startswith('bandpower'):p['band_edges']=[[.5,4],[4,8],[8,13],[13,30],[30,45]]
            if model=='fbcsp':p['band_edges']=[[8,30]] if p['bands']=='single' else [[4,8],[8,12],[12,20],[20,30],[30,40]]
            if model=='fbcca':p.update(lower_edges=[6,14,22,30,38][:p['banks']],upper_edge=90,filter_order=4,weight_addend=.25)
            row=add(folder,'参数搜索',task,model,p,r,c,r['val_metrics'],'P1：固定被试划分搜索',split=c['split'],
                    stage=r['stage'],epochs_budget=r['budget_epochs'] if model in neural else None,
                    pretrained=model.startswith('cbramod'),backbone='更新' if model=='cbramod_ft' else '冻结eval' if model=='cbramod_ridge' else '不适用',
                    deterministic=r.get('deterministic',model not in neural),
                    note='传统方法日志中的epochs预算不适用，表中留空' if model not in neural else '')
        final=read(ROOT/'outputs/search_v1'/task/'final_test.json');ch=final['champion']
        for ind in final['individual_runs']:
            row=next(x for x in records if x['实验ID']==f'outputs/search_v1/{task}/runs/{ind["run_id"]}')
            tm=ind['test_metrics'];row['测试BA']=tm['subject_mean_BA'];row['测试宏F1']=tm['macro_f1'];row['测试是否评估']=True
            if task=='mi':test_rows.append({'协议':row['协议'],'任务':row['任务'],'模型':row['模型'],'类型':'集成成员单独评估',
                '运行ID':row['实验ID'],'验证BA':row['验证BA'],'验证口径':'单个训练种子','测试BA':row['测试BA'],'测试宏F1':row['测试宏F1'],'测试被试':row['测试或排除被试']})
        summary=next(x for x in read(ROOT/'outputs/search_v1/champions_summary.json') if x['task']==task)
        test_rows.append({'协议':'P1：固定被试划分搜索','任务':TASKS[task][0],'模型':NAMES[ch['config']['model']],
            '类型':'三模型概率均值集成' if final['ensemble'] else '选定单模型','运行ID':packed(ch['run_ids']),
            '验证BA':summary['val_ensemble_BA'],'验证口径':'集成概率预测' if final['ensemble'] else '单模型验证BA',
            '测试BA':final['test_metrics']['subject_mean_BA'],'测试宏F1':final['test_metrics']['macro_f1'],'测试被试':packed(final['test_subjects'])})

    for family in ['core','supplement']:
        for folder in sorted((ROOT/'outputs/preliminary_v1'/family).iterdir()):
            if not (folder/'result.json').exists():continue
            c=read(folder/'config.json');r=read(folder/'result.json');g=r['group'];fold=r['fold']
            p={'batch':16,'wd':.01,'optimizer':'AdamW','scheduler':'CosineAnnealingLR','lr':c.get('lr'), 'dropout':.5}
            if family=='core':
                p.update(backbone_arch);p.update(readout='flatten',head='Flatten+LayerNorm+Dropout+Linear',backbone_lr=1e-4,head_lr=1e-3,norm='uV/100')
                if g in ['F','M']:p.update(experts=4,top_k=1,replaced_layers='last 2 FFN',balance=.01 if g=='M' else 0.,routing='token content' if g=='M' else 'fixed task',gradient='unit-gain straight-through' if g=='M' else None)
                task_list=r['tasks']
            else:
                task_list=list(c['data'])
                if g=='eegnet':p.update(filters=16,kernel=64,depth_multiplier=2,norm='per-trial channel zscore',readout='AdaptiveAvgPool(1,4)+Flatten',head='Linear')
                elif g=='fbcnet':p.update(bands=[[i,i+4] for i in range(4,40,4)],filters=16,readout='4-segment logvariance',head='Linear',norm='uV/100',activation='SiLU',dropout=None)
                else:p.update(bands=[[.5,4],[4,8],[8,13],[13,30],[30,45]],norm='train-fold StandardScaler',window_seconds=1)
                if g=='sequence_tcn':p.update(hidden=64,kernel=3,dilations=[1,2],readout='temporal average',head='Linear')
                if g=='sequence_logistic':p.update(C=.1,max_iter=2000,class_weight='balanced',head='Logistic',readout='time mean',optimizer='lbfgs',scheduler=None,batch=None,wd=None,lr=None,dropout=None)
            for task in task_list:
                m=r['best']['metrics'][task] if family=='core' else r.get('best',r)['metrics']
                row=add(folder,'三折对照',task,g,p,r,c,m,'P2：旧开发被试三折',split=c['data'][task]['split'],fold=fold+1,
                        updates_budget=1500 if g!='sequence_logistic' else None,
                        best_step=r['best'].get('dataset_updates',r['best'].get('updates')) if 'best' in r else None,
                        pretrained=family=='core',backbone='更新' if family=='core' else '不适用',deterministic=g=='sequence_logistic',
                        note='联合模型共享检查点按三任务验证BA均值选择' if g in ['J','F','M'] else '本地结构变体，不是论文完整复现' if g=='fbcnet' else '')
                if family=='supplement' and g!='sequence_logistic':row['实际更新步数']=1500

    for folder in sorted((ROOT/'outputs/initialization_controls_v1/runs').iterdir()):
        c=read(folder/'config.json');r=read(folder/'result.json');mode=r['mode'];task=r['task']
        p={**backbone_arch,'readout':'flatten','head':'Flatten+LayerNorm+Dropout+Linear','batch':16,'wd':.01,'dropout':.5,
           'optimizer':'AdamW','scheduler':'CosineAnnealingLR','backbone_lr':1e-4,'head_lr':1e-3,'norm':'uV/100',
           'initialization':mode,'backbone_training_mode':'train; dropout retained'}
        add(folder,'初始化与冻结',task,mode,p,r,c,r['best']['metrics'],'P2：旧开发被试三折',split=c['split'],fold=r['fold']+1,
            updates_budget=1500,best_step=r['best']['updates'],pretrained=c['pretrained_loaded'],
            backbone='冻结train，保留dropout' if mode=='pretrained_frozen' else '更新',
            note='全量预训练基线复用三折S组，不重复计为新增拟合')

    adapter_fits=0
    adapter_root=ROOT/'outputs/residual_adapters_v1/runs'
    if adapter_root.exists():
        for folder in sorted(adapter_root.iterdir()):
            if not (folder/'result.json').exists():continue
            c=read(folder/'config.json');r=read(folder/'result.json');task=r['task'];mode=r['mode']
            p={**backbone_arch,'mode':mode,'readout':'flatten','head':'Flatten+LayerNorm+Dropout+Linear',
               'batch':c['batch'],'wd':c['weight_decay'],'dropout':.5,'backbone_lr':c['backbone_lr'],
               'head_lr':c['head_lr'],'adapter_lr':c['adapter_lr'],'optimizer':'AdamW','scheduler':'CosineAnnealingLR',
               'norm':'uV/100','adapter_width':32,'spatial_heads':4,'temporal_kernel':3,
               'spectral_band':c['spectral_band'],'spectral_bins':c['spectral_bins'],
               'gate':'fixed 1/3' if mode=='uniform' else 'sample softmax; spectrum masked' if mode=='no_spectrum' else 'sample softmax',
               'fusion':'H + weighted residual branches','branch_output_init':'zeros','compute':'dense'}
            add(folder,'异构残差适配',task,'residual_'+mode,p,r,c,r['best']['metrics'],'P2：旧开发被试三折',
                split=c['split'],fold=r['fold']+1,updates_budget=1500,best_step=r['best']['updates'],pretrained=True,backbone='更新',
                note='全部有效分支密集计算；验证门控均值='+packed(r['best']['gate']['mean_weights']))
            adapter_fits+=1

    assert len(records)==209+adapter_fits
    assert len({r['实验ID'] for r in records})==191+adapter_fits
    assert len({r['任务记录ID'] for r in records})==209+adapter_fits
    assert all(c['passed'] for c in checks)
    for row in records:
        folder=ROOT/row['实验ID']
        if not any(x['实验ID']==row['实验ID'] for x in manifest):
            manifest.append({'实验ID':row['实验ID'],'结果SHA256':digest(folder/'result.json'),'配置SHA256':digest(folder/'config.json')})
    groups=defaultdict(list)
    for r in records:groups[r['配置ID']].append(r)
    configs=[]
    for k,rs in groups.items():
        r=rs[0];vals=[x['验证BA'] for x in rs]
        configs.append({'配置ID':k,'协议':r['协议'],'任务':r['任务'],'模型':r['模型'],'次数':len(rs),'验证BA均值':statistics.mean(vals),
                        '验证BA样本标准差':statistics.stdev(vals) if len(vals)>1 and (r['协议'].startswith('P2') or not all(x['确定性拟合'] for x in rs)) else None,
                        '验证BA最小':min(vals),'验证BA最大':max(vals),'种子':packed(sorted({x['训练种子'] for x in rs})),
                        '折':packed(sorted({x['折'] for x in rs if x['折'] is not None})),
                        '统计口径':'固定划分训练种子差异' if r['协议'].startswith('P1') else '开发折与种子共同变化' if r['协议'].startswith('P2') else '单次五轮试跑',
                        '预训练权重':r['预训练权重'],'轮数预算':r['轮数预算'],'每数据集步数预算':r['每数据集步数预算'],
                        '完整参数JSON':r['完整模型参数JSON']})
    configs.sort(key=lambda x:(x['协议'],x['任务'],x['模型'],-x['验证BA均值']))
    datasets=[]
    for task,meta in metas.items():
        dims=TASKS[task]
        for protocol,split,fold in ([('P0：MI固定划分五轮',meta['split'],None)] if task=='mi' else [])+[('P1：固定被试划分搜索',meta['split'],None)]+[
            ('P2：旧开发被试三折',{'train':sorted(set(sum(plan['validation_subjects_by_fold'][task],[]))-set(v)), 'val':v,'test':plan['legacy_test_subjects_excluded'][task]},i+1)
            for i,v in enumerate(plan['validation_subjects_by_fold'].get(task,[]))]:
            datasets.append({'协议':protocol,'任务':dims[0],'数据集':dims[1],'折':fold,'通道':dims[2],'秒':dims[3],'Hz':200,'类数':dims[4],
                             '训练被试':packed(split['train']),'验证被试':packed(split['val']),'测试或排除被试':packed(split['test']),
                             '来源':meta.get('source',meta.get('source_url','公开数据；见README及数据元信息')),
                             '说明':'同一Benchmark的两个时长；整试次离线滤波后裁窗，不是严格在线时延' if task.startswith('ssvep') else '前12名被试、26通道、缺Neutral；不是完整九类FACED' if task=='emotion' else '仅T会话，保留伪迹试次，排除EOG'})
    payload={'records':records,'configs':configs,'histories':histories,'tests':test_rows,'datasets':datasets,
             'counts':{'fits':191+adapter_fits,'task_records':len(records),'configs':len(configs),'curve_records':len(histories),'test_records':len(test_rows),'adapter_fits':adapter_fits},
             'batches':dict(Counter(r['批次'] for r in records)),'manifest':manifest}
    (OUT/'ledger.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    with (OUT/'全部模型参数与结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (OUT/'collection_audit.json').write_text(json.dumps({'counts':payload['counts'],'checks':checks,'manifest':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload['counts'],ensure_ascii=False))


if __name__=='__main__':main()
