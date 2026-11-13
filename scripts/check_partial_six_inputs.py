"""Six handbook shape probes only: synthetic inputs, dummy labels, no accuracy claim."""
import json
import time
import numpy as np
import torch
from torch import nn
from run_partial_sharing import PartialModel,ROOT,OUT
from run_preliminary import TASKS
from search_models import seed_all,write

SHAPES={'mi64':('mi',64,800),'mi22':('mi',22,800),'ssvep1s':('ssvep_2s',60,200),
        'ssvep2s':('ssvep_2s',60,400),'emotion30':('emotion',30,2000),'emotion62':('emotion',62,2000)}

class SixHeadProbe(PartialModel):
    def __init__(self,depth):
        fake={t:{'X':np.empty((0,c,n),dtype='float32'),'classes':2} for t,c,n in [('mi',22,800),('ssvep_2s',60,400),('emotion',30,2000)]}
        super().__init__(fake,42,depth)
        self.heads=nn.ModuleDict({key:nn.Sequential(nn.Flatten(),nn.LayerNorm(c*n),nn.Dropout(.5),nn.Linear(c*n,2))
                                 for key,(_,c,n) in SHAPES.items()})
    def forward(self,x,key):
        task=SHAPES[key][0];h=self.backbone(x.reshape(len(x),x.shape[1],-1,200))
        for layer in self.private[task]:h=layer(h)
        return self.heads[key](h)

def main():
    results=[]
    for depth in [2,4]:
        seed_all(42);model=SixHeadProbe(depth).cuda().train()
        for key,(task,c,n) in SHAPES.items():
            model.zero_grad(set_to_none=True);x=torch.randn(2,c,n,device='cuda')*.1
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            p=model(x,key);assert p.shape==(2,2) and torch.isfinite(p).all()
            nn.functional.cross_entropy(p,torch.tensor([0,1],device='cuda')).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert any(p.grad is not None and p.grad.norm()>0 for p in model.private[task].parameters())
            assert all(p.grad is None for t in TASKS if t!=task for p in model.private[t].parameters())
            torch.cuda.synchronize()
            results.append({'private_depth':depth,'dataset_shape_key':key,'task_branch':task,'shape':[2,c,n],
                'dummy_output_shape':[2,2],'finite_forward_backward':True,'inactive_task_branches_no_gradient':True,
                'seconds_single_probe':time.perf_counter()-start,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
        del model;torch.cuda.empty_cache()
    write(OUT/'six_input_interface_smoke.json',{'scope':'synthetic interface/backprop probe only; no official EEG or labels; no classification accuracy',
        'class_count':2,'class_count_note':'dummy two-class heads only; official class counts not yet disclosed',
        'layout_note':'tensor shape compatibility does not demonstrate electrode-layout semantics or generalization',
        'timing_note':'single tiny-batch probes; not warmed-up throughput benchmark','records':results,'passed':12})
    print(json.dumps({'synthetic_interface_probes_passed':12}))

if __name__=='__main__':main()
