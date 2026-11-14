"""Independent 64-channel MI development experiment; never scores held-out test subjects."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

RUNS = (4, 6, 8, 10, 12, 14)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prepare(args):
    import mne
    from scipy.signal import butter, sosfiltfilt, resample_poly
    from fractions import Fraction
    out = args.output
    if (out / 'metadata.json').exists():
        raise FileExistsError('Prepared data exists; choose a new output to preserve provenance')
    rows, rejected, sources, subjects, names = [], [], [], [], None
    out.mkdir(parents=True, exist_ok=True)
    for folder in sorted(args.source.glob('S[0-9][0-9][0-9]')):
        sid = int(folder.name[1:])
        if not all((folder / f'{folder.name}R{r:02d}.edf').is_file() for r in RUNS):
            rejected.append({'subject': sid, 'reason': 'missing imagined-movement run'})
            continue
        xs, ys, ids = [], [], []
        for run in RUNS:
            p = folder / f'{folder.name}R{run:02d}.edf'
            raw = mne.io.read_raw_edf(p, preload=True, verbose='ERROR')
            ch = [c.rstrip('.').upper() for c in raw.ch_names]
            assert len(ch) == len(set(ch)) == 64, (sid, run, ch)
            if names is None:
                names = ch
            assert ch == names, (sid, run, 'channel order differs')
            fs = float(raw.info['sfreq'])
            x = raw.get_data() * 1e6  # MNE EDF physical scaling -> volts -> uV.
            assert np.isfinite(x).all()
            x -= x.mean(axis=0, keepdims=True)
            x = sosfiltfilt(butter(4, [.3, 40], fs=fs, btype='bandpass', output='sos'), x)
            ratio = Fraction(200 / fs).limit_denominator(10000)
            for event, (onset, duration, desc) in enumerate(zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)):
                if desc not in ('T1', 'T2'):
                    continue
                start, n = int(round(onset * fs)), int(round(4 * fs))
                if duration < 4 - 1 / fs or start + n > x.shape[1]:
                    rejected.append({'subject': sid, 'run': run, 'event': event, 'reason': 'less than 4 seconds within annotated imagery'})
                    continue
                epoch = resample_poly(x[:, start:start+n], ratio.numerator, ratio.denominator, axis=-1)
                assert epoch.shape == (64, 800) and np.isfinite(epoch).all()
                label = int(desc[1]) - 1 + (2 if run in (6, 10, 14) else 0)
                xs.append(epoch.astype('float32')); ys.append(label); ids.append(f'{folder.name}R{run:02d}-event{event}')
            sources.append({'path': str(p), 'sha256': sha(p), 'sfreq': fs, 'channel_count': len(ch)})
        if set(ys) != {0, 1, 2, 3}:
            rejected.append({'subject': sid, 'reason': 'not all four classes after temporal checks'})
            continue
        subjects.append(sid)
        np.savez(out / f'S{sid:03d}.npz', X=np.stack(xs), y=np.array(ys), trial_id=np.array(ids))
        rows.append({'subject': sid, 'trials': len(ys), 'counts': np.bincount(ys, minlength=4).tolist()})
        print(json.dumps(rows[-1]), flush=True)
        write(out / 'prepare_progress.json', {'subjects_completed': len(subjects), 'last_subject': sid})
    assert len(subjects) >= 20
    # A single locked test pool is excluded from ALL development repeats.
    rng = np.random.default_rng(20261114)
    perm = rng.permutation(subjects)
    ntest = max(1, round(len(perm) * .1)); test = sorted(perm[:ntest].tolist())
    dev = perm[ntest:]
    splits = []
    for seed in (42, 43, 44):
        p = np.random.default_rng(seed).permutation(dev)
        nval = max(1, round(len(subjects) * .1))
        splits.append({'seed': seed, 'train': sorted(p[nval:].tolist()), 'val': sorted(p[:nval].tolist()), 'test': test})
    write(out / 'metadata.json', {'scope': 'PUBLIC_DEVELOPMENT_NOT_OFFICIAL; fixed untouched test pool, three development re-splits',
        'source_root': str(args.source), 'source_url': 'https://physionet.org/content/eegmmidb/1.0.0/',
        'labels': ['left_fist', 'right_fist', 'both_fists', 'both_feet'], 'runs': RUNS,
        'input': [64, 800], 'sfreq': 200, 'unit': 'uV', 'channel_names': names,
        'preprocessing': 'MNE EDF physical scaling; EEG CAR; continuous-run offline 0.3-40Hz Butterworth order4; onset [0,4)s inside imagery annotation; polyphase resampling; no target-set fitted statistics',
        'subjects': rows, 'sources': sources, 'rejected': rejected, 'splits': splits,
        'processed_sha256': {p.name: sha(p) for p in sorted(out.glob('S*.npz'))}, 'script_sha256': sha(__file__)})


def train(args):
    import torch
    from torch import nn
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    meta = json.loads((args.data / 'metadata.json').read_text())

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__(); f = 16
            self.features = nn.Sequential(nn.Conv2d(1,f,(1,64),padding='same',bias=False), nn.BatchNorm2d(f),
                nn.Conv2d(f,2*f,(64,1),groups=f,bias=False),nn.BatchNorm2d(2*f),nn.ELU(),nn.AvgPool2d((1,4)),nn.Dropout(.5),
                nn.Conv2d(2*f,2*f,(1,16),padding='same',groups=2*f,bias=False),nn.Conv2d(2*f,2*f,1,bias=False),
                nn.BatchNorm2d(2*f),nn.ELU(),nn.AvgPool2d((1,8)),nn.Dropout(.5),nn.AdaptiveAvgPool2d((1,4)))
            self.head = nn.Linear(128,4)
        def forward(self,x):
            return self.head(self.features(x[:,None]).flatten(1))

    class Shallow(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(1,40,(1,25)),nn.Conv2d(40,40,(64,1),bias=False),nn.BatchNorm2d(40))
            self.pool = nn.AvgPool2d((1,75),stride=(1,15)); self.drop = nn.Dropout(.5); self.head = nn.Linear(40*47,4)
        def forward(self,x):
            return self.head(self.drop(self.pool(self.net(x[:,None]).square()).clamp_min(1e-6).log()).flatten(1))

    def load(ids):
        xs, ys, ss, ts = [], [], [], []
        for sid in ids:
            p = args.data / f'S{sid:03d}.npz'
            assert sha(p) == meta['processed_sha256'][p.name]
            with np.load(p) as d:
                x = d['X']; x = (x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True).clip(1e-6)
                xs.append(x.astype('float32')); ys.append(d['y']); ss.extend([sid]*len(x)); ts.extend(d['trial_id'])
        return np.concatenate(xs), np.concatenate(ys), np.array(ss), np.array(ts)

    def score(y,p,s):
        pred = p.argmax(-1)
        per = {str(k): float(balanced_accuracy_score(y[s==k],pred[s==k])) for k in np.unique(s)}
        return {'subject_mean_BA': float(np.mean(list(per.values()))), 'pooled_BA': float(balanced_accuracy_score(y,pred)),
            'macro_f1': float(f1_score(y,pred,average='macro')), 'per_subject_BA':per, 'confusion_matrix':confusion_matrix(y,pred,labels=range(4)).tolist()}

    @torch.no_grad()
    def predict(model,x):
        model.eval()
        return np.concatenate([model(torch.from_numpy(x[i:i+32]).to(args.device)).softmax(-1).cpu().numpy() for i in range(0,len(x),32)])

    results = []
    for fold in range(args.folds):
        split = meta['splits'][fold]; seed = split['seed']
        assert not (set(split['train']) & set(split['val']) or set(split['train']+split['val']) & set(split['test']))
        x,y,s,t = load(split['train']); vx,vy,vs,vt = load(split['val'])
        rng = np.random.default_rng(seed)
        pools = {(sid,c): np.flatnonzero((s==sid)&(y==c)) for sid in split['train'] for c in range(4)}
        assert all(len(v) for v in pools.values())
        schedule = np.array([[rng.choice(pools[sid,c]) for sid in rng.choice(split['train'],4,replace=False) for c in range(4)] for _ in range(args.updates)])
        probs = {}
        for name, factory, lr in [('eegnet',EEGNet,.003), ('shallow',Shallow,.001)]:
            folder = args.output / f'fold{fold+1}_{name}'
            if (folder/'config.json').exists(): raise FileExistsError(folder)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = factory().to(args.device)
            opt = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.updates)
            cfg = {'split':split,'model':name,'seed':seed,'lr':lr,'wd':.01,'dropout':.5,'updates':args.updates,'batch':16,
                'normalization':'per-trial channel-wise temporal zscore','selection':'validation subject-mean BA; earliest on ties',
                'script_sha256':sha(__file__),'metadata_sha256':sha(args.data/'metadata.json'),'test_evaluated':False}
            write(folder/'config.json',cfg); np.save(folder/'training_indices.npy',schedule)
            best = -1; history = []; started = time.monotonic()
            for step,idx in enumerate(schedule,1):
                model.train(); opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(torch.from_numpy(x[idx]).to(args.device)),torch.from_numpy(y[idx]).long().to(args.device))
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True); opt.step(); sched.step()
                if step % 100 == 0 or step == args.updates:
                    p = predict(model,vx); m = score(vy,p,vs)
                    row = {'step':step,'loss':float(loss.detach()),'val_BA':m['subject_mean_BA']}; history.append(row)
                    write(folder/'history.json',history)
                    if m['subject_mean_BA'] > best:
                        best = m['subject_mean_BA']; torch.save(model.state_dict(),folder/'model.pt')
                        np.savez(folder/'validation.npz',scores=p,y=vy,subject=vs,trial_id=vt)
                        write(folder/'best.json',{'step':step,'metrics':m})
                    write(args.output/'status.json',{'status':'running','fold':fold+1,'model':name,**row})
                    print(json.dumps({'fold':fold+1,'model':name,**row}),flush=True)
            model.load_state_dict(torch.load(folder/'model.pt',map_location=args.device,weights_only=True))
            p = predict(model,vx); saved = np.load(folder/'validation.npz')['scores']
            error = float(np.max(np.abs(saved-p))); assert error < 1e-5
            probs[name] = p
            result = {'fold':fold+1,'model':name,'metrics':score(vy,p,vs),'reload_error':error,'seconds':time.monotonic()-started,'test_evaluated':False}
            results.append(result); write(folder/'result.json',result)
            del model,opt; torch.cuda.empty_cache()
        # Existing BNCI-selected weight, fixed in advance; no target-data weight search.
        p = .75*probs['eegnet'] + .25*probs['shallow']
        folder = args.output / f'fold{fold+1}_fixed_fusion'; folder.mkdir(parents=True)
        np.savez(folder/'validation.npz',scores=p,y=vy,subject=vs,trial_id=vt)
        results.append({'fold':fold+1,'model':'fixed_fusion_75_25','metrics':score(vy,p,vs),'test_evaluated':False})
        write(args.output/'results.json',results)
    write(args.output/'status.json',{'status':'finished','fits':2*args.folds,'fusion_evaluations':args.folds,'test_evaluated':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('train'); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--updates',type=int,default=1500); p.add_argument('--folds',type=int,choices=[1,2,3],default=3)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else train(args)
