"""Matched short public-data pilots and six-shape GPU checks using official CBraMod."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external/CBraMod'))
from models.cbramod import CBraMod


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def provenance():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'torch': torch.__version__, 'numpy': np.__version__,
            'gpu': torch.cuda.get_device_name(0),
            'cbramod_commit': subprocess.check_output(['git','-C',str(ROOT/'external/CBraMod'),'rev-parse','HEAD'], text=True).strip()}


class Classifier(nn.Module):
    def __init__(self, channels, patches, classes, checkpoint=None):
        super().__init__()
        self.backbone = CBraMod()
        if checkpoint:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
        self.backbone.proj_out = nn.Identity()
        # Local simple head. Preserve channel and patch positions rather than mean pooling.
        self.head = nn.Sequential(nn.Flatten(), nn.LayerNorm(channels*patches*200),
                                  nn.Linear(channels*patches*200, classes))

    def forward(self, x):
        return self.head(self.backbone(x.reshape(x.shape[0], x.shape[1], -1, 200)))


def smoke(args):
    seed_all(args.seed)
    rows = []
    for channels, samples in [(60,200),(60,400),(64,800),(22,800),(30,2000),(62,2000)]:
        model = Classifier(channels, samples//200, 4).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
        x = torch.randn(2,channels,samples,device='cuda'); y=torch.tensor([0,1],device='cuda')
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin=time.perf_counter()
        before=model.head[-1].weight.detach().clone()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x),y); loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step()
        torch.cuda.synchronize()
        assert not torch.equal(before,model.head[-1].weight)
        row={'channels':channels,'samples':samples,'batch_size':2,'steps':3,
             'seconds_per_step':(time.perf_counter()-begin)/3,
             'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
             'parameters':sum(p.numel() for p in model.parameters()),'finite_gradients':True,'updated':True}
        rows.append(row); print(json.dumps(row),flush=True)
        del opt, model,x,y; torch.cuda.empty_cache()
    backbone=CBraMod().cuda().eval()
    x=torch.randn(2,22,4,200,device='cuda'); mask=torch.zeros(2,22,4,device='cuda',dtype=torch.bool);mask[:,:,1]=True
    altered=x.clone();altered[mask]=1000*torch.randn_like(altered[mask])
    with torch.no_grad():
        a=backbone.patch_embedding(x,mask); b=backbone.patch_embedding(altered,mask)
    assert torch.equal(a,b), 'Masked raw signal leaked into an embedding branch'
    save_json(ROOT/'outputs/smoke/summary.json',{'scope':'SYNTHETIC_ENGINEERING_ONLY',
              'environment':provenance(),'rows':rows,'masked_signal_invariance':True})


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); probabilities=[]; labels=[]
    for x,y in loader:
        probabilities.append(model(x.cuda()).softmax(-1).cpu().numpy());labels.append(y.numpy())
    prob=np.concatenate(probabilities); y=np.concatenate(labels); pred=prob.argmax(-1)
    return {'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2,3]).tolist()},y,prob


def pilot(args):
    data_path=ROOT/'data/processed/BNCI2014_001_T_pilot.npz'
    data=np.load(data_path,allow_pickle=False)
    X=data['X']/100; y=data['y']; subject=data['subject']; ids=data['trial_id']
    split={'train':[1,2,3,4,5],'val':[6,7],'test':[8,9]}
    assert X.shape==(2592,22,800) and len(np.unique(ids))==len(ids) and np.isfinite(X).all()
    assert all(not set(split[a])&set(split[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    indices={k:np.flatnonzero(np.isin(subject,v)) for k,v in split.items()}
    for idx in indices.values(): assert set(y[idx])=={0,1,2,3}
    checkpoint=ROOT/'checkpoints/cbramod.pth'
    summaries=[]
    for mode in args.modes:
        seed_all(args.seed)
        out=ROOT/'outputs/pilot'/f'{args.tag}_{mode}_seed{args.seed}'
        if out.exists(): raise FileExistsError(f'Refusing to overwrite run: {out}')
        out.mkdir(parents=True)
        gen=torch.Generator().manual_seed(args.seed)
        loaders={k:DataLoader(TensorDataset(torch.from_numpy(X[idx]),torch.from_numpy(y[idx])),
                              batch_size=args.batch_size,shuffle=k=='train',generator=gen if k=='train' else None,
                              num_workers=0) for k,idx in indices.items()}
        model=Classifier(22,4,4,checkpoint if mode!='random' else None).cuda()
        if mode=='linear_probe': model.backbone.requires_grad_(False)
        opt=torch.optim.AdamW([
            {'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':args.backbone_lr},
            {'params':model.head.parameters(),'lr':args.head_lr}],weight_decay=0.01)
        config={**vars(args),'mode':mode,'scope':'PUBLIC_DATA_SHORT_PILOT_NOT_COMPETITION',
                'split':split,'trials':{k:ids[idx].tolist() for k,idx in indices.items()},
                'environment':provenance(),'parameters':sum(p.numel() for p in model.parameters()),
                'data_sha256':hashlib.file_digest(data_path.open('rb'),'sha256').hexdigest(),
                'checkpoint_sha256':hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest() if mode!='random' else None,
                'selection':'maximum validation balanced accuracy; earliest epoch on ties',
                'normalization':'input uV / 100; no fitted cross-subject statistics'}
        save_json(out/'config.json',config)
        best=-1.;best_epoch=0;history=[];updates=0
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize();begin=time.perf_counter()
        for epoch in range(1,args.epochs+1):
            model.train()
            if mode=='linear_probe':model.backbone.eval()
            total=0.; count=0
            for x,labels in loaders['train']:
                opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(x.cuda()),labels.cuda())
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                loss.backward()
                norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();updates+=1;total+=loss.item()*len(labels);count+=len(labels)
            metrics,_,_=evaluate(model,loaders['val'])
            row={'epoch':epoch,'train_loss':total/count,'val_balanced_accuracy':metrics['balanced_accuracy'],
                 'val_macro_f1':metrics['macro_f1'],'updates':updates}
            history.append(row);save_json(out/'history.json',history)
            if metrics['balanced_accuracy']>best:
                best=metrics['balanced_accuracy'];best_epoch=epoch
                torch.save(model.state_dict(),out/'best.pt')
            print(json.dumps({'mode':mode,**row}),flush=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        peak=torch.cuda.max_memory_allocated()/2**20
        model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=True))
        test,truth,prob=evaluate(model,loaders['test'])
        np.savez_compressed(out/'test_predictions.npz',trial_id=ids[indices['test']],subject=subject[indices['test']],y=truth,probability=prob)
        result={'scope':'PUBLIC_DATA_SHORT_PILOT_NOT_COMPETITION','mode':mode,'seed':args.seed,
                'epochs':args.epochs,'updates':updates,'best_epoch':best_epoch,'best_val_BA':best,
                'test_BA':test['balanced_accuracy'],'test_macro_f1':test['macro_f1'],
                'test_confusion_matrix':test['confusion_matrix'],'training_seconds':elapsed,'peak_allocated_MiB':peak,
                'limitation':'T session only; fixed 5/2/2 subjects; one development seed; short budget; no claim of convergence or official performance'}
        save_json(out/'result.json',result);summaries.append(result)
        save_json(ROOT/'outputs/pilot'/f'{args.tag}_summary.json',summaries)
        print(json.dumps({'completed':result}),flush=True)
        del opt,model,loaders;torch.cuda.empty_cache()
    with (ROOT/'outputs/pilot'/f'{args.tag}_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        keys=['mode','epochs','updates','best_epoch','best_val_BA','test_BA','test_macro_f1','training_seconds','peak_allocated_MiB']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(summaries)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','pilot'])
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--backbone-lr',type=float,default=3e-5)
    p.add_argument('--head-lr',type=float,default=5e-4);p.add_argument('--tag',default='initial_5ep')
    p.add_argument('--modes',nargs='+',choices=['linear_probe','full_finetune','random'],
                   default=['linear_probe','full_finetune','random'])
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required: use .venv/Scripts/python.exe')
    if args.epochs<1 or args.batch_size<1:p.error('Positive epochs and batch size required')
    smoke(args) if args.command=='smoke' else pilot(args)


if __name__=='__main__':main()
