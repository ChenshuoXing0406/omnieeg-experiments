"""Nontrivial synthetic check: combine branches saved at different update counts."""
import copy
import json
import numpy as np
import torch
from torch import nn
from run_partial_sharing import PartialModel,tensor_sha,ROOT,TASKS
from search_models import seed_all,write

def main():
    seed_all(42);torch.set_num_threads(2)
    fake={t:{'X':np.empty((0,c,n),dtype='float32'),'classes':2} for t,c,n in [('mi',22,800),('ssvep_2s',64,400),('emotion',26,2000)]}
    model=PartialModel(fake,42,4);model.backbone.requires_grad_(False)
    shared=copy.deepcopy(model.backbone.state_dict());shared_hash=tensor_sha(shared)
    x={t:torch.randn(2,d['X'].shape[1],d['X'].shape[2])*.1 for t,d in fake.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    chosen={};expected={};saved_steps={'mi':1,'ssvep_2s':2,'emotion':3}
    for step in range(1,4):
        for task in TASKS:
            model.train();opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(model(x[task],task),torch.tensor([0,1])).backward();opt.step()
            assert tensor_sha(model.backbone.state_dict())==shared_hash
            if step==saved_steps[task]:
                chosen[task]={'tail':copy.deepcopy(model.private[task].state_dict()),'head':copy.deepcopy(model.heads[task].state_dict())}
                model.eval()
                with torch.no_grad():expected[task]=model(x[task],task).softmax(-1).clone()
    assert tensor_sha(chosen['mi']['tail'])!=tensor_sha(model.private['mi'].state_dict())
    composite=PartialModel(fake,42,4);composite.backbone.load_state_dict(shared,strict=True)
    for task in TASKS:
        composite.private[task].load_state_dict(chosen[task]['tail'],strict=True)
        composite.heads[task].load_state_dict(chosen[task]['head'],strict=True)
    composite.eval();errors={}
    with torch.no_grad():
        for task in TASKS:errors[task]=float((composite(x[task],task).softmax(-1)-expected[task]).abs().max())
    assert max(errors.values())<1e-5
    out={'scope':'synthetic correctness check, not a dataset accuracy experiment or new fit',
         'private_branch_save_steps':saved_steps,'frozen_prefix_hash_unchanged':True,'nontrivial_earlier_branch_restore':True,
         'composed_logits_max_abs_error':errors,'passed':True,'device':'cpu'}
    write(ROOT/'outputs/partial_sharing_frozen_v1/composition_smoke.json',out);print(json.dumps(out))

if __name__=='__main__':main()
