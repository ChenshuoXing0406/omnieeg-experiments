"""Predeclare and run validation-driven amplitude-normalization variants on CPU."""
import json
from threadpoolctl import threadpool_limits
import search_models as s


def main():
    data=s.load_task('emotion');folder=s.OUT/'emotion'
    assert not (folder/'final_test.json').exists(), 'Do not tune after final test evaluation'
    configs=[]
    for mode in ['relative','spatial_centered','relative_centered']:
        configs.append({'model':'bandpower_lda','power_norm':mode})
        for c in [.01,.1,1.]:configs.append({'model':'bandpower_logistic','C':c,'power_norm':mode})
        for c in [1.,10.]:configs.append({'model':'bandpower_svm','C':c,'gamma':'scale','power_norm':mode})
    plan={'reason':'Absolute bandpower weak on validation; check within-trial scale invariance across subjects.',
          'configs':configs,'test_used':False,'fit_policy':'Normalization within each trial; StandardScaler fitted to training subjects only'}
    path=folder/'feature_supplement_plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    s.write(path,plan)
    results=[s.execute('emotion',data,config,42,20,'feature_supplement') for config in configs]
    s.write(folder/'feature_supplement_result.json',sorted(results,key=lambda r:r['val_BA'],reverse=True))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
