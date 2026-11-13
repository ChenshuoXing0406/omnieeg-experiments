# Preliminary public-development experiments

New grouped development folds; legacy test subjects excluded. Fold and training seed both vary.
These are validation-selected development scores, not official results or independent blind tests.

| Family | Group | Task | Folds | Mean BA % | Across-fold sample SD, pp |
|---|---|---|---:|---:|---:|
| core | F | emotion | 3 | 16.97 | 0.84 |
| core | F | mi | 3 | 40.49 | 1.44 |
| core | F | ssvep_2s | 3 | 66.54 | 5.95 |
| core | J | emotion | 3 | 15.37 | 1.72 |
| core | J | mi | 3 | 40.53 | 0.26 |
| core | J | ssvep_2s | 3 | 66.22 | 5.08 |
| core | M | emotion | 3 | 16.83 | 1.67 |
| core | M | mi | 3 | 39.89 | 1.54 |
| core | M | ssvep_2s | 3 | 65.06 | 5.66 |
| core | S | emotion | 3 | 16.42 | 0.59 |
| core | S | mi | 3 | 43.89 | 1.18 |
| core | S | ssvep_2s | 3 | 67.62 | 5.85 |
| supplement | eegnet | mi | 3 | 47.74 | 3.25 |
| supplement | fbcnet | mi | 3 | 34.09 | 4.01 |
| supplement | sequence_logistic | emotion | 3 | 18.01 | 2.10 |
| supplement | sequence_tcn | emotion | 3 | 18.21 | 2.11 |

## Paired differences

| Group | Reference | Task | Folds | Mean delta, pp | Positive folds |
|---|---|---|---:|---:|---:|
| J | S | mi | 3 | -3.36 | 0 |
| J | S | ssvep_2s | 3 | -1.40 | 0 |
| J | S | emotion | 3 | -1.05 | 1 |
| F | S | mi | 3 | -3.40 | 0 |
| F | S | ssvep_2s | 3 | -1.08 | 1 |
| F | S | emotion | 3 | +0.56 | 2 |
| M | S | mi | 3 | -3.99 | 0 |
| M | S | ssvep_2s | 3 | -2.57 | 0 |
| M | S | emotion | 3 | +0.41 | 2 |
| M | F | mi | 3 | -0.60 | 1 |
| M | F | ssvep_2s | 3 | -1.49 | 0 |
| M | F | emotion | 3 | -0.14 | 1 |

S: independent dense; J: joint dense; F: fixed-task top-1 experts; M: learned top-1 experts.
Four total experts; fixed task routing uses three. M uses a selected-probability straight-through gradient with unit forward gain.
Differences do not establish statistical significance, and capacity/active-compute matched dense ablations remain future work.
