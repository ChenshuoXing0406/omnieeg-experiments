"""Independently reload paired controls and export per-fold effect sizes."""
import json
from pathlib import Path
import numpy as np
import torch
from run_initialization_controls import OUT, BASE, ROOT, PLAN, TASKS, CoreModel, ControlModel, load_data, tensor_sha, write, sha


def main():
    torch.set_num_threads(4)
    rows = json.loads((OUT / 'results.json').read_text())
    audit = []
    for path in sorted((OUT / 'runs').glob('*/result.json')):
        folder = path.parent; r = json.loads(path.read_text()); c = json.loads((folder / 'config.json').read_text())
        task = r['task']; data = {task: load_data(task, r['fold'])}
        torch.manual_seed(c['seed'])
        model = ControlModel(data, c['seed'], r['mode'])
        reference = CoreModel(data, 'S', c['seed'])
        assert set(reference.state_dict()) == set(model.state_dict())
        assert tensor_sha(reference.heads.state_dict()) == tensor_sha(model.heads.state_dict())
        if r['mode'] == 'pretrained_frozen':
            assert tensor_sha(reference.backbone.state_dict()) == tensor_sha(model.backbone.state_dict())
        del reference
        assert tensor_sha(model.heads.state_dict()) == c['initial_head_sha256']
        # Reconstruct the original random/loaded initialization before checkpoint load.
        assert tensor_sha(model.backbone.state_dict()) == c['initial_backbone_sha256']
        model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        if r['mode'] == 'pretrained_frozen':
            assert tensor_sha(model.backbone.state_dict()) == c['initial_backbone_sha256']
        model = model.cuda().eval(); d = data[task]; index = d['val'][:16]
        with torch.no_grad():
            scores = model(torch.from_numpy((d['X'][index] / 100).astype('float32')).cuda(), task).softmax(-1).cpu().numpy()
        with np.load(folder / f'{task}_validation.npz') as p:
            assert np.array_equal(p['trial_id'][:16], d['trial_id'][index])
            error = float(np.max(np.abs(scores - p['scores'][:16])))
            assert np.allclose(scores, p['scores'][:16], rtol=5e-4, atol=1e-4)
        audit.append({'run': folder.name, 'initialization_reconstructed': True,
                      'head_and_architecture_match_original_S': True,
                      'checkpoint_predictions_reproduced': True, 'max_abs_score_error': error,
                      'checkpoint_sha256': sha(folder / 'model.pt'), 'frozen_backbone_verified': r['mode'] == 'pretrained_frozen'})
        del model, data
        torch.cuda.empty_cache()
    write(OUT / 'checkpoint_audit.json', audit)
    pairs = []
    for task in TASKS:
        for alternate in ['pretrained_frozen', 'random_full']:
            differences = []
            for fold in [1, 2, 3]:
                lookup = {(r['task'], r['mode'], r['fold']): r['BA'] for r in rows}
                if (task, alternate, fold) in lookup and (task, 'pretrained_full', fold) in lookup:
                    differences.append(100 * (lookup[(task, 'pretrained_full', fold)] - lookup[(task, alternate, fold)]))
            if differences:
                pairs.append({'task': task, 'contrast': f'pretrained_full minus {alternate}',
                              'folds': len(differences), 'fold_differences_pp': differences,
                              'mean_difference_pp': float(np.mean(differences)),
                              'positive_folds': int(np.sum(np.array(differences) > 0))})
    write(OUT / 'paired_effects.json', pairs)
    print(json.dumps({'audited_checkpoints': len(audit), 'paired_effects': pairs}, indent=2))


if __name__ == '__main__': main()
