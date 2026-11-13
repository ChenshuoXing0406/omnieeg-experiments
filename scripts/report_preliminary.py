"""Summarize paired development folds, never reading legacy test predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_preliminary import OUT, ROOT, TASKS, CoreModel, load_data, sha, write
import torch


def report(checkpoints=False):
    rows = json.loads((OUT / 'results.json').read_text())
    grouped = {(r['family'], r['group'], r['task'], r['fold']): r for r in rows}
    comparisons = []
    for group in ['J', 'F', 'M']:
        for task in TASKS:
            pairs = [(grouped[('core', group, task, f)], grouped[('core', 'S', task, f)])
                     for f in [1, 2, 3] if ('core', group, task, f) in grouped and ('core', 'S', task, f) in grouped]
            if pairs:
                delta = [(a['val_BA'] - b['val_BA']) * 100 for a, b in pairs]
                comparisons.append({'group': group, 'reference': 'S', 'task': task,
                                    'folds': len(pairs), 'mean_delta_pp': float(np.mean(delta)),
                                    'fold_delta_pp': delta, 'positive_folds': int(np.sum(np.array(delta) > 0))})
    for task in TASKS:
        pairs = [(grouped[('core', 'M', task, f)], grouped[('core', 'F', task, f)])
                 for f in [1, 2, 3] if ('core', 'M', task, f) in grouped and ('core', 'F', task, f) in grouped]
        if pairs:
            delta = [(a['val_BA'] - b['val_BA']) * 100 for a, b in pairs]
            comparisons.append({'group': 'M', 'reference': 'F', 'task': task, 'folds': len(pairs),
                                'mean_delta_pp': float(np.mean(delta)), 'fold_delta_pp': delta,
                                'positive_folds': int(np.sum(np.array(delta) > 0))})
    write(OUT / 'paired_comparisons.json', comparisons)
    summary = json.loads((OUT / 'summary.json').read_text())
    lines = ['# Preliminary public-development experiments', '',
             'New grouped development folds; legacy test subjects excluded. Fold and training seed both vary.',
             'These are validation-selected development scores, not official results or independent blind tests.', '',
             '| Family | Group | Task | Folds | Mean BA % | Across-fold sample SD, pp |',
             '|---|---|---|---:|---:|---:|']
    for r in summary:
        sd = '-' if r['sample_std'] is None else f"{r['sample_std'] * 100:.2f}"
        lines.append(f"| {r['family']} | {r['group']} | {r['task']} | {r['folds']} | {r['mean_BA'] * 100:.2f} | {sd} |")
    lines += ['', '## Paired differences', '', '| Group | Reference | Task | Folds | Mean delta, pp | Positive folds |', '|---|---|---|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['group']} | {r['reference']} | {r['task']} | {r['folds']} | {r['mean_delta_pp']:+.2f} | {r['positive_folds']} |")
    lines += ['', 'S: independent dense; J: joint dense; F: fixed-task top-1 experts; M: learned top-1 experts.',
              'Four total experts; fixed task routing uses three. M uses a selected-probability straight-through gradient with unit forward gain.',
              'Differences do not establish statistical significance, and capacity/active-compute matched dense ablations remain future work.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    if checkpoints:
        torch.set_num_threads(4)
        checks = []
        for path in sorted((OUT / 'core').glob('*/result.json')):
            folder = path.parent; result = json.loads(path.read_text()); config = json.loads((folder / 'config.json').read_text())
            data = {t: load_data(t, config['fold']) for t in result['tasks']}
            model = CoreModel(data, config['group'], config['seed']).cuda().eval()
            model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
            for task, d in data.items():
                with np.load(folder / f'{task}_validation.npz') as saved:
                    index = d['val'][:16]
                    assert np.array_equal(d['trial_id'][index], saved['trial_id'][:16])
                    with torch.no_grad():
                        pred = model(torch.from_numpy((d['X'][index] / 100).astype('float32')).cuda(), task).softmax(-1).cpu().numpy()
                    difference = float(np.max(np.abs(pred - saved['scores'][:16])))
                    assert np.allclose(pred, saved['scores'][:16], rtol=5e-4, atol=1e-4), (folder, difference)
                    checks.append({'run': folder.name, 'task': task, 'subset_samples': len(index),
                                   'max_abs_score_difference': difference, 'checkpoint_sha256': sha(folder / 'model.pt'), 'passed': True})
            del model, data
            torch.cuda.empty_cache()
        write(OUT / 'checkpoint_reload_audit.json', checks)
    print(json.dumps({'summary_rows': len(summary), 'comparisons': comparisons}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--checkpoints', action='store_true')
    report(parser.parse_args().checkpoints)
