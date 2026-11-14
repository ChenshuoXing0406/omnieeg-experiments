"""Independent recall/threshold verification for saved classical SSVEP predictions."""
import argparse,json
from pathlib import Path
import numpy as np


def main(root):
    checks=[]
    for folder in sorted(p for p in root.glob('fold*') if p.is_dir()):
        c=json.loads((folder/'calibration.json').read_text())
        assert not set(c['train_subjects'])&set(c['val_subjects'])
        assert not set(c['train_subjects']+c['val_subjects'])&{11,12}
        name=folder.name.split('_');method=name[1];seconds=int(name[2][:-1])
        allp=np.load(root/f'{method}_{seconds}s_all_development_scores.npz')
        for task in ['active3','with_rest4']:
            d=np.load(folder/f'{task}_validation.npz')
            mask=np.isin(allp['subject'],c['val_subjects'])
            if task=='active3':mask&=allp['y']!=3
            for k in ['scores','y','subject','trial_id']:assert np.array_equal(d[k],allp[k][mask])
            expected=d['scores'].argmax(-1)
            if task=='with_rest4':expected=np.where(d['scores'].max(-1)<c['threshold'],3,expected)
            assert np.array_equal(expected,d['pred'])
            per=[]
            for s in c['val_subjects']:
                y=d['y'][d['subject']==s];p=d['pred'][d['subject']==s]
                per.append(float(np.mean([np.mean(p[y==cls]==cls) for cls in np.unique(y)])))
            checks.append({'run':folder.name,'task':task,'independent_recall_BA':float(np.mean(per)),'threshold_and_saved_identity_verified':True})
    assert len(checks)==24
    (root/'independent_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    summary=json.loads((root/'summary.json').read_text());manifest=json.loads((root/'data_manifest.json').read_text())
    lines=['# 独立SSVEP数据集经典基线','',f"Kalunga2016：开发侧10人、{len(manifest['sources'])}个官方run、{manifest['trials']}个事件，8个真实EEG电极，重采样200Hz。验证覆盖被试1–6，7–10仅用于源阈值校准。11/12锁定且未下载或评分。",'',
        '| 方法 | 输入时长 | 三类激活BA | 含rest四类BA |','|---|---:|---:|---:|']
    for method in ['CCA','FBCCA']:
        for sec in [1,2]:
            vals={r['task']:r['mean_BA'] for r in summary if r['method']==method and r['seconds']==sec}
            actual=[r['independent_recall_BA'] for r in checks if f'_{method}_{sec}s' in r['run']]
            assert len(actual)==6
            for task in ['active3','with_rest4']:
                actual=[r['independent_recall_BA'] for r in checks if f'_{method}_{sec}s' in r['run'] and r['task']==task]
                assert abs(float(np.mean(actual))-vals[task])<1e-12
            lines.append(f"| {method} | {sec}秒 | {vals['active3']*100:.2f}% | {vals['with_rest4']*100:.2f}% |")
    lines+=['','数字为三个本地开发划分的逐被试BA均值，不是官方比赛或独立测试成绩。三类激活是免训练频率识别；四类包含仅在训练被试校准的rest拒识阈值，不是免校准四类方法。',
        '源阈值校准12次，神经模型训练0次；两种任务共24条验证记录。24份保存预测已独立复核试次/被试身份、阈值应用和逐类召回BA，全部通过。',
        '事件映射遵循MOABB现行修正：13Hz=2、17Hz=4、21Hz=3、rest=1。公共盘额外run未纳入官方列表，未使用其他项目缓存。',
        '输入窗口为事件后[2,3)/[2,4)秒；全run离线滤波使用窗外上下文，不代表严格在线1秒/2秒延迟。四谐波最高84Hz，低于200Hz采样的Nyquist100Hz。',
        '这项实验说明经典频率参考法在第二个代理库上可运行，不能由两库百分比直接宣布跨数据集模型泛化，也不等同比赛的60通道数据。',
        '初步发现：该固定配方下FBCCA在两种窗口/两种任务的开发均值均高于CCA；rest拒识与纯频率辨识仍有明显差距。未按验证结果继续搜索阈值或滤波参数。']
    (root/'结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Independent audits:',len(checks),'all passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
