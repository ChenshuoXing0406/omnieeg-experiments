"""Fit/use the selected source-template fusion candidate; no legacy test scoring."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from run_ssvep_transfer_search import standardize,trca_fit,trca_scores,cca_scores,zscore_classes,CHANNELS
from run_preliminary import sha
from search_models import write

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/best_candidates_20261113'

def predict_preprocessed(x,model):
    """x: [batch,9,200*seconds], posterior order CHANNELS, already initial6-90Hz filtered.

    Original experiment initially filters complete6s trials before latency-aligned crop.
    Matching this offline context is caller responsibility; this API does not promise online latency.
    Outputs uncalibrated softmax class scores, not calibrated posterior probabilities.
    """
    assert x.ndim==3 and x.shape[1:]==(9,200*model['seconds']) and np.isfinite(x).all()
    a=[];b=[]
    for lower,pair in zip([6,14,22,30,38],model['filter_models']):
        signal=standardize(sosfiltfilt(butter(4,[lower,90],fs=200,btype='bandpass',output='sos'),x,axis=-1))
        a.append(cca_scores(signal,model['reference_qr']));b.append(trca_scores(signal,*pair))
    ca=np.einsum('b,bnk->nk',model['bank_weights'],np.stack(a));tr=np.einsum('b,bnk->nk',model['bank_weights'],np.stack(b))
    logits=.75*zscore_classes(ca)+.25*zscore_classes(tr)
    return softmax(logits,axis=-1)

def fit():
    OUT.mkdir(parents=True,exist_ok=True);artifacts=[]
    for seconds in [1,2]:
        path=ROOT/f'data/processed/ssvep_{seconds}s.npz';meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path,allow_pickle=False) as raw:
            keep=~np.isin(raw['subject'],[7,8,33,34,35]);names=[n.upper() for n in raw['channel_names']]
            pick=[names.index(n) for n in CHANNELS];x=raw['X'][keep][:,pick].copy();y=raw['y'][keep];subjects=raw['subject'][keep]
        assert len(np.unique(subjects))==30
        t=np.arange(200*seconds)/200
        references=np.stack([np.stack([fun(2*np.pi*f*h*t) for h in range(1,6) for fun in [np.sin,np.cos]],axis=-1) for f in meta['frequencies']])
        references-=references.mean(1,keepdims=True);models=[]
        with threadpool_limits(limits=2):
            for lower in [6,14,22,30,38]:
                signal=standardize(sosfiltfilt(butter(4,[lower,90],fs=200,btype='bandpass',output='sos'),x,axis=-1))
                models.append(trca_fit(signal,y))
            obj={'seconds':seconds,'reference_qr':np.linalg.qr(references,mode='reduced')[0],
                'filter_models':models,'bank_weights':np.arange(1,6)**(-1. if seconds==1 else -.5)+.25,
                'posterior_channel_order':CHANNELS,'frequencies':meta['frequencies'],'fusion_alpha':.25,
                'training_subjects':np.unique(subjects).tolist(),'data_sha256':sha(path),'source_sha256':sha(__file__),
                'source_algorithm_sha256':sha(ROOT/'scripts/run_ssvep_transfer_search.py'),
                'scope':'all30development subjects refit; zero new validation/test estimate; offline filter context'}
            p=predict_preprocessed(x[:4],obj);target=OUT/f'ssvep_{seconds}s_fusion25.joblib';joblib.dump(obj,target)
            reload=predict_preprocessed(x[:4],joblib.load(target));err=float(np.max(np.abs(p-reload)))
            assert err<1e-10 and np.allclose(p.sum(-1),1) and np.isfinite(p).all()
        artifacts.append({'seconds':seconds,'path':str(target.relative_to(ROOT)),'sha256':sha(target),'reload_max_abs_error':err,
            'new_refit':True,'new_accuracy_evaluated':False})
        del x,signal,obj
    write(OUT/'ssvep_manifest.json',{'deployment_refits':2,'artifacts':artifacts,'test_evaluated':False,
        'inference_entry':'scripts/ssvep_transfer_candidate.py::predict_preprocessed',
        'security_note':'load only trusted locally generated joblib files'})
    print(json.dumps(artifacts),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['fit']);parser.parse_args();fit()
