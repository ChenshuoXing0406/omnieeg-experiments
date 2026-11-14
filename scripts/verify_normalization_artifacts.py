"""Verify transferred checkpoints and independently recompute MI class-recall BA."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(a):
    manifest=json.loads((a.root/'completion_manifest.json').read_text())
    for rec in manifest['files']:
        p=a.root/rec['path'];assert p.stat().st_size==rec['bytes'],p
        assert sha(p)==rec['sha256'],p
    env=json.loads((a.root/'formal/environment.json').read_text())
    assert sha(a.runner)==env['script_sha256']
    summary=json.loads((a.root/'report/summary.json').read_text())
    rows=[];checkpoint_count=0
    for fold in range(1,4):
        for model in ['cbra_pretrained','cbra_random']:
            previous=None
            for norm,base in [('scale',a.controls),('trial',a.root/'formal')]:
                folder=base/f'fold{fold}_{model}'
                cfg=json.loads((folder/'config.json').read_text());result=json.loads((folder/'result.json').read_text())
                data=np.load(folder/'validation.npz');pred=data['scores'].argmax(-1)
                assert np.isfinite(data['scores']).all()
                assert np.allclose(data['scores'].sum(-1),1,atol=1e-6)
                assert set(data['y'])=={0,1,2,3}
                assert len(set(data['trial_id']))==len(pred)
                assert set(data['subject'])==set(cfg['split']['val'])
                assert not set(data['subject'])&set(cfg['split']['train']+cfg['split']['test'])
                per=[]
                for sid in sorted(set(data['subject'])):
                    idx=data['subject']==sid;y=data['y'][idx];p=pred[idx]
                    assert set(y)=={0,1,2,3}
                    per.append(float(np.mean([np.count_nonzero(p[y==c]==c)/np.count_nonzero(y==c) for c in range(4)])))
                score=float(np.mean(per));assert abs(score-result['metrics']['subject_mean_BA'])<1e-12
                assert result['reload_error']<1e-5 and not result['test_evaluated']
                indices=np.load(folder/'training_indices.npy')
                if norm=='scale':previous=(cfg,{k:data[k] for k in ['y','subject','trial_id']},indices)
                else:
                    old,identity,schedule=previous
                    for k in ['split','seed','input','batch','updates','wd','backbone_lr','head_lr','initial_head_sha256','initial_backbone_sha256','sampling_sha256']:
                        assert cfg[k]==old[k],(fold,model,k)
                    assert cfg['normalization']=='trial' and old['normalization']=='scale'
                    for k,v in identity.items():assert np.array_equal(data[k],v)
                    assert np.array_equal(indices,schedule)
                    state=torch.load(folder/'model.pt',weights_only=True,map_location='cpu')
                    assert state['head.3.weight'].shape==(4,51200)
                    assert all(torch.isfinite(v).all() for v in state.values())
                    checkpoint_count+=1;del state
                rows.append({'fold':fold,'model':model,'normalization':norm,'independent_recall_BA':score})
    for condition in summary['conditions']:
        actual=float(np.mean([r['independent_recall_BA'] for r in rows if r['model']==condition['model'] and r['normalization']==condition['normalization']]))
        assert abs(actual-condition['mean_BA'])<1e-12
    assert checkpoint_count==6 and len(rows)==12
    status=json.loads((a.root/'formal/status.json').read_text());assert status['status']=='finished' and status['fits']==6
    supervisor=json.loads((a.root/'immediate_status.json').read_text());assert supervisor['status']=='finished'
    receipt={'status':'verified','transfer_sha256_files':len(manifest['files']),'new_checkpoints_readable_and_finite':checkpoint_count,
             'prediction_sets_independently_recomputed':len(rows),'normalization_pairs_verified':6,'training_source_hash_verified':True,'test_evaluated':False,'scores':rows}
    (a.root/'completion_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='scores'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--controls',type=Path,required=True);p.add_argument('--runner',type=Path,required=True)
    torch.set_num_threads(4);main(p.parse_args())
