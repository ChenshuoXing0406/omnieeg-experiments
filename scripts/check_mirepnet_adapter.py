"""Real development-input shape/weight/backprop check, no accuracy experiment."""
import ast
import importlib.util
import json
from pathlib import Path
import types
import sys

import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist
import torch
from torch import nn

from run_preliminary import load_data, sha
from search_models import seed_all, write

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/mi_adapter_deps'))
OUT=ROOT/'outputs/mi_domain_v1'
source=ROOT/'external/MIRepNet/model/mlm.py'
# wandb is imported but unused by the model; exclude only that optional import.
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name=='wandb' for a in n.names))]
module=types.ModuleType('mirepnet_adapter_numeric_unchanged')
exec(compile(tree,str(source),'exec'),module.__dict__)
spec=importlib.util.spec_from_file_location('mirepnet_channels',ROOT/'external/MIRepNet/utils/channel_list.py')
channels=importlib.util.module_from_spec(spec);spec.loader.exec_module(channels)

seed_all(42)
d=load_data('mi',0);tr=d['train']
names=[s.upper() for s in d['channel_names']];target=channels.use_channels_names
positions=channels.channel_positions
dist=cdist(np.asarray([positions[s] for s in target]),np.asarray([positions[s] for s in names]))
mapping=1/(dist+1e-6);mapping/=mapping.sum(-1,keepdims=True)
for i,name in enumerate(target):
    if name in names:mapping[i]=np.eye(len(names))[names.index(name)]
assert mapping.shape==(45,22) and np.allclose(mapping.sum(1),1)
# Source-only covariance reference: no validation subject batch statistics.
xx=d['X'][tr].astype('float64');xx-=xx.mean(-1,keepdims=True)
cov=np.einsum('nct,ndt->cd',xx,xx)/(len(xx)*(xx.shape[-1]-1))
ref=.99*cov+.01*np.trace(cov)/len(cov)*np.eye(len(cov))
v,u=np.linalg.eigh(ref);whitening=(u*v**-.5)@u.T
x=resample_poly(mapping@whitening@d['X'][tr[:2]],5,4,axis=-1).astype('float32')
assert x.shape==(2,45,1000) and np.isfinite(x).all()
weights_path=ROOT/'external/MIRepNet/weight/MIRepNet.pth'
weights=torch.load(weights_path,map_location='cpu',weights_only=True)
model=module.mlm_mask(emb_size=256,depth=6,n_classes=4,pretrain=None)
state=model.state_dict();backbone=[k for k in state if not k.startswith('clshead.')]
assert all(k in weights and weights[k].shape==state[k].shape for k in backbone)
loaded={k:weights[k] for k in backbone}
info=model.load_state_dict(loaded,strict=False)
assert sorted(info.missing_keys)==['clshead.bias','clshead.weight'] and not info.unexpected_keys
model.eval();feature,logits=model(torch.from_numpy(x))
assert feature.shape==(2,256) and logits.shape==(2,4) and torch.isfinite(logits).all()
nn.functional.cross_entropy(logits,torch.from_numpy(d['y'][tr[:2]]).long()).backward()
grad=float(model.embedding.conv1.weight.grad.norm());assert np.isfinite(grad) and grad>0
write(OUT/'mirepnet_adapter_smoke.json',{'status':'passed_real_input_forward_backward','training_completed':False,
    'accuracy_measured':False,'shape':list(x.shape),'features_shape':list(feature.shape),'logits_shape':list(logits.shape),
    'loaded_backbone_tensors':len(loaded),'new_four_class_head':True,'backbone_gradient_norm':grad,
    'source_model_sha256':sha(source),'weight_sha256':sha(weights_path),'data_sha256':d['source_sha256'],
    'target_template':target,'source_channel_names':names,'exact_observed_channels':len(set(names)&set(target)),
    'reference_fit_subjects':d['split']['train'],'normalization':'source pooled covariance + 1% shrinkage; not subjectwise target EA',
    'interpolation':'official inverse-distance formula and coordinate table; observed channels preserved exactly',
    'resampling':'200 to 250 Hz using resample_poly(5,4), duration kept at 4 seconds',
    'optional_import_removed':'unused wandb only; upstream numerical model unchanged','test_evaluated':False,
    'caveat':'This source-only adapter differs from author subjectwise EA; pretraining overlap still needs audit before performance claims.'})
print(json.dumps({'status':'passed','loaded_backbone_tensors':len(loaded),'gradient_norm':grad}))
