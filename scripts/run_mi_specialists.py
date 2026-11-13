"""MI specialists: learned log-variance ShallowConv and EEGNet frequency-shift augmentation."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from run_preliminary import load_data,save_predictions,sha
from run_mi_domain import balanced_schedule
from search_models import EEGNet,seed_all,metrics,write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/mi_specialists_v1'
SETTINGS=[('shallow_lr1e3',.001),('shallow_lr3e3',.003),('eegnet_frequency_shift',.003)]

class Shallow(nn.Module):
    """Local ShallowConvNet-style: temporal/spatial filters, square-average-log readout."""
    def __init__(self):
        super().__init__()
        self.temporal=nn.Conv2d(1,40,(1,25),bias=True)
        self.spatial=nn.Conv2d(40,40,(22,1),bias=False)
        self.norm=nn.BatchNorm2d(40)
        self.pool=nn.AvgPool2d((1,75),stride=(1,15))
        self.dropout=nn.Dropout(.5);self.head=nn.Linear(40*47,4)
    def forward(self,x):
        h=self.norm(self.spatial(self.temporal(x[:,None])))
        h=self.pool(h.square()).clamp_min(1e-6).log()
        return self.head(self.dropout(h).flatten(1))

def build(name):return Shallow() if name.startswith('shallow') else EEGNet(22,4,filters=16,kernel=64,dropout=.5)

@torch.no_grad()
def predict(model,x):
    model.eval();p=[]
    for start in range(0,len(x),32):p.append(model(torch.from_numpy(x[start:start+32]).cuda()).softmax(-1).cpu().numpy())
    return np.concatenate(p)

def shift(x):
    # Analytic signal via an exact DFT Hilbert multiplier; same shift across channels of a trial.
    n=x.shape[-1];h=torch.zeros(n,device=x.device,dtype=x.dtype);h[0]=h[n//2]=1;h[1:n//2]=2
    analytic=torch.fft.ifft(torch.fft.fft(x,dim=-1)*h,dim=-1)
    hz=(torch.rand((len(x),1,1),device=x.device)*2-1)
    phase=2*torch.pi*hz*torch.arange(n,device=x.device)[None,None,:]/200
    return (analytic*torch.exp(1j*phase)).real

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # GPU stages stay sequential; independent CPU studies may continue.
    start_wait=time.monotonic()
    while json.loads((ROOT/'outputs/mirepnet_search_v1/queue_status.json').read_text())['status']!='finished':
        write(OUT/'queue_status.json',{'status':'waiting_for_mirepnet_gpu_stage'})
        if time.monotonic()-start_wait>3600:raise RuntimeError('GPU predecessor did not complete within one hour')
        time.sleep(5)
    rows=[];audits=[]
    for fold in range(3):
        d=load_data('mi',fold);x=d['X'];x=((x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)).astype('float32')
        for name,lr in SETTINGS:
            folder=OUT/f'fold{fold+1}_{name}';folder.mkdir(parents=True,exist_ok=True)
            cfg={'name':name,'fold':fold,'seed':42+fold,'source_sha256':sha(__file__),'data_sha256':d['source_sha256'],
                'split':d['split'],'lr':lr,'wd':.01,'dropout':.5,'batch':16,'updates':1500,'optimizer':'AdamW','scheduler':'cosine',
                'input':'per-trial temporal zscore','augmentation':'uniform frequency shift +/-1Hz during training only' if name.endswith('shift') else 'none',
                'scope':'strict source-only; local ShallowConv-style variant; not exact published reproduction','test_evaluated':False}
            write(folder/'config.json',cfg);write(OUT/'queue_status.json',{'status':'running','fold':fold+1,'name':name})
            seed_all(42+fold);model=build(name).cuda();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1500);schedule,_=balanced_schedule(d,42+fold)
            np.save(folder/'training_indices.npy',schedule);start=time.perf_counter();best=-1.;history=[]
            for step,index in enumerate(schedule,1):
                model.train();opt.zero_grad(set_to_none=True);batch=torch.from_numpy(x[index]).cuda()
                if name.endswith('shift'):batch=shift(batch)
                loss=nn.functional.cross_entropy(model(batch),torch.from_numpy(d['y'][index]).long().cuda())
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sched.step()
                if step%100==0:
                    p=predict(model,x[d['val']]);m=metrics(d['y'][d['val']],p.argmax(-1),d['subject'][d['val']],4)
                    history.append({'step':step,'loss_last_batch':float(loss.detach()),'val_BA':m['subject_mean_BA']});write(folder/'history.json',history)
                    if m['subject_mean_BA']>best:
                        best=m['subject_mean_BA'];torch.save(model.state_dict(),folder/'model.pt');save_predictions(folder,{'mi':d},{'mi':p})
                        write(folder/'best_validation.json',{'step':step,'metrics':m})
                    print(json.dumps({'run':folder.name,**history[-1]}),flush=True)
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda',weights_only=True),strict=True)
            p=predict(model,x[d['val']]);error=float(np.max(np.abs(p-np.load(folder/'mi_validation.npz')['scores'])));assert error<1e-5
            m=metrics(d['y'][d['val']],p.argmax(-1),d['subject'][d['val']],4)
            write(folder/'result.json',{'metrics':m,'seconds':time.perf_counter()-start,'reload_error':error,'test_evaluated':False})
            rows.append({'name':name,'fold':fold+1,'BA':m['subject_mean_BA'],'test_evaluated':False});audits.append({'run':folder.name,'reload_error':error})
            del model,opt;torch.cuda.empty_cache()
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[{'name':name,'mean_BA':float(np.mean([r['BA'] for r in rows if r['name']==name]))} for name,_ in SETTINGS]
    write(OUT/'summary.json',summary);write(OUT/'audit.json',audits)
    write(OUT/'queue_status.json',{'status':'finished','fits':9,'prediction_audits':9,'checkpoint_audits':9,'test_evaluated':False})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
