"""Pre-launch checks for grouping, interval labels, initialized heads and gradients."""
import json
from pathlib import Path
import numpy as np
import torch
from model import ExperimentModel,TASKS,MODES,count
from prepare_tuar import merge,overlap,canonical

root=Path(__file__).resolve().parent
torch.set_num_threads(2)
assert merge([(0,.4),(.2,.8),(.8,1),(2,3)])==[[0,1],[2,3]]
assert abs(overlap(merge([(0,.4),(.2,.8)]),.3,.9)-.5)<1e-8
assert overlap(merge([(1,2)]),2,4)==0
assert canonical('EEG T7-REF')=='T3'
reference=None;reports={}
for mode in MODES:
    model=ExperimentModel(root/'weights/pretrained_weights.pth',mode).eval()
    heads={k:v.clone() for k,v in model.heads.state_dict().items()}
    if reference is None:reference=heads
    else:assert all(torch.equal(v,reference[k]) for k,v in heads.items())
    for task,(c,n,classes) in TASKS.items():
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        x=torch.randn(1,c,n,200)*.2
        logits,g=model(x,task)
        assert logits.shape==(1,classes) and torch.isfinite(logits).all()
        logits.square().sum().backward()
        for expert in model.experts:
            assert sum(float(p.grad.abs().sum()) for p in expert.parameters() if p.grad is not None)>0
        if model.router is not None:assert sum(float(p.grad.abs().sum()) for p in model.router.parameters())>0
    reports[mode]=model.parameter_report(list(TASKS))
    print('MODEL_OK',mode,flush=True)
caps=[reports[m]['experts'] for m in MODES if m!='plain']
assert max(caps)/min(caps)<1.01
for task in TASKS:
    folder=root/'processed'/task
    if not (folder/'manifest.json').exists():continue
    manifest=json.loads((folder/'manifest.json').read_text());subsets=[]
    for part in ['train','val','test']:
        x=np.load(folder/f'{part}_x.npy',mmap_mode='r');y=np.load(folder/f'{part}_y.npy');sub=np.load(folder/f'{part}_subject.npy')
        assert x.shape[1:]==TASKS[task][:2]+(200,) and len(x)==len(y)==len(sub)
        assert set(np.unique(y))==set(range(TASKS[task][2]))
        for offset in range(0,len(x),256):assert np.isfinite(x[offset:offset+256]).all()
        subsets.append(set(sub.tolist()))
    assert all(not subsets[i]&subsets[j] for i in range(3) for j in range(i+1,3))
    print('DATA_OK',task,flush=True)
(root/'setup_checks.json').write_text(json.dumps({'models':reports,'checks':'interval union/boundaries; common head initialization; all shapes forward/backward; expert/router gradients; capacity within1%; available data finite and subject disjoint'},indent=2))
print('ALL_CHECKS_PASSED',flush=True)
