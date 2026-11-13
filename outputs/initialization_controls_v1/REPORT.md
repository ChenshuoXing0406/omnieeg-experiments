# Matched freezing and initialization controls

Development validation only; three folds with paired training seeds, not official scores.

| Task | Mode | Folds | Mean BA % | Fold sample SD, pp |
|---|---|---:|---:|---:|
| mi | pretrained_full | 3 | 43.89 | 1.18 |
| mi | pretrained_frozen | 3 | 31.39 | 1.15 |
| mi | random_full | 3 | 41.67 | 1.22 |
| ssvep_2s | pretrained_full | 3 | 67.62 | 5.85 |
| ssvep_2s | pretrained_frozen | 3 | 23.83 | 3.61 |
| ssvep_2s | random_full | 3 | 29.24 | 2.82 |
| emotion | pretrained_full | 3 | 16.42 | 0.59 |
| emotion | pretrained_frozen | 3 | 16.35 | 0.86 |
| emotion | random_full | 3 | 18.08 | 3.10 |

Same flatten, LayerNorm/dropout/Linear head, head initialization, microvolt/100 input, sampling order, updates, optimizer settings and validation cadence.
Frozen backbone retains training-mode dropout to isolate parameter updates; no frozen-eval feature cache.
Pretrained full fine-tuning is reused only after source, split, sample-order, budget and checkpoint hash checks.
Random initialization is trained under the same downstream budget; this does not test large-scale from-scratch pretraining.