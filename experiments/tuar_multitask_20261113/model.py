"""Shared CBraMod and controlled feature/expert ablations. No channel padding."""
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from models.cbramod import CBraMod

TASKS = {'mi': (22, 4, 4), 'ssvep_2s': (64, 2, 40), 'tuar': (20, 2, 2)}
MODES = ['plain', 'adapter', 'homo_fixed', 'homo_gate', 'hetero_fixed', 'hetero_gate']
FEATURE_DIM = 605

class MLPExpert(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.GELU(), nn.Linear(hidden, 200))
    def forward(self, z, x, spectrum):
        return self.net(z)

class OperationExpert(nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.common = nn.Linear(FEATURE_DIM, 64)
        self.out = nn.Linear(64, 200)
        if operation in ('frequency', 'temporal'):
            self.conv = nn.Sequential(nn.Conv1d(1, 8, 9, padding=4), nn.GELU(), nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(64, 64))
        if operation == 'spatial':
            self.attention = nn.MultiheadAttention(64, 4, dropout=0, batch_first=True)
        if operation == 'temporal':
            self.temporal = nn.Conv1d(64, 64, 3, padding=1, groups=64)
    def forward(self, z, x, spectrum):
        b,c,n,_ = z.shape
        h = self.common(z)
        if self.operation == 'frequency':
            h = h + self.conv(spectrum.reshape(b*c, 1, 201)).reshape(b,c,1,64)
        elif self.operation == 'spatial':
            v = h.permute(0,2,1,3).reshape(b*n,c,64)
            v,_ = self.attention(v,v,v,need_weights=False)
            h = h + v.reshape(b,n,c,64).permute(0,2,1,3)
        else:
            v = self.temporal(h.reshape(b*c,n,64).transpose(1,2)).transpose(1,2).reshape(b,c,n,64)
            local = self.conv(x.reshape(b*c*n,1,200)).reshape(b,c,n,64)
            h = h + v + local
        return self.out(F.gelu(h))

def count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class ExperimentModel(nn.Module):
    def __init__(self, weights, mode='plain', seed=42):
        super().__init__()
        assert mode in MODES
        self.mode = mode
        # Always instantiate all heads, in the same order and with the same seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.backbone = CBraMod()
            state = torch.load(weights, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state, strict=True)
            self.backbone.proj_out = nn.Identity()
            self.heads = nn.ModuleDict()
            for i,(k,(c,n,classes)) in enumerate(TASKS.items()):
                torch.manual_seed(seed+i*1000)
                self.heads[k]=nn.Sequential(nn.Flatten(),nn.LayerNorm(c*n*200),nn.Dropout(.5),nn.Linear(c*n*200,classes))
        self.experts = nn.ModuleList()
        self.router = None
        if mode != 'plain':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+1000)
                hetero = nn.ModuleList([OperationExpert(op) for op in ['frequency','spatial','temporal']])
                target = count(hetero)
                if mode == 'adapter':
                    hidden = max(1, round((target-200)/(FEATURE_DIM+201)))
                    self.experts = nn.ModuleList([MLPExpert(hidden)])
                elif mode.startswith('homo'):
                    hidden = max(1, round((target/3-200)/(FEATURE_DIM+201)))
                    self.experts = nn.ModuleList([MLPExpert(hidden) for _ in range(3)])
                else:
                    self.experts = hetero
            if mode.endswith('gate'):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+2000)
                    self.router = nn.Sequential(nn.Linear(400,32),nn.GELU(),nn.Linear(32,3))

    def forward(self, x, task):
        assert tuple(x.shape[1:]) == TASKS[task][:2]+(200,), (task,x.shape)
        h = self.backbone(x)
        gates = None
        if self.mode != 'plain':
            # Full-window FFT uses actual time samples, not an FFT over embedding coordinates.
            with torch.autocast(device_type=x.device.type, enabled=False):
                signal = x.float().flatten(2)
                s = torch.log1p(torch.fft.rfft(signal, norm='forward').abs())
                s = F.interpolate(s.reshape(-1,1,s.shape[-1]),size=201,mode='linear',align_corners=True).reshape(x.shape[0],x.shape[1],201)
                stats = torch.stack([x.float().square().mean(-1).sqrt(), x.float().abs().mean(-1),
                                     x.float().amax(-1)-x.float().amin(-1), x.float().diff(dim=-1).square().mean(-1).sqrt()],dim=-1)
            z = torch.cat([h, s.unsqueeze(2).expand(-1,-1,x.shape[2],-1), x, stats],dim=-1)
            if self.router is None:
                gates = h.new_full((len(x),len(self.experts)),1/len(self.experts))
            else:
                pooled = torch.cat([h.mean((1,2)),h.float().std((1,2),unbiased=False)],dim=-1)
                gates = self.router(pooled).softmax(-1)
            correction = sum(expert(z,x,s)*gates[:,i,None,None,None] for i,expert in enumerate(self.experts))
            h = h + .1*correction
        return self.heads[task](h), gates

    def parameter_report(self, tasks):
        return {'backbone': count(self.backbone), 'active_heads': sum(count(self.heads[t]) for t in tasks),
                'experts': count(self.experts), 'router': 0 if self.router is None else count(self.router),
                'gate_compute': 'all experts evaluated; soft routing, not sparse compute'}
