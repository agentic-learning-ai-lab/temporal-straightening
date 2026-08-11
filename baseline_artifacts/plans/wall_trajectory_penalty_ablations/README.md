# Wall trajectory-penalty ablation

This study compares speed-sensitive latent trajectory penalties with the
existing direction-only Wall baseline. Every condition uses DINOv2 patch
features, the learned channel projector, and the epoch-20 checkpoint.

## Objectives

- `R0 = 1 - cos(theta)`
- `R1 = r + 1/r - 2`
- `R2 = R1 + 2*R0`
- `R3 = R0 + R1`

R2 uses coefficient 0.05 so its effective direction coefficient is 0.1,
matching the R0 baseline. R1 and R3 use coefficient 0.1.

## Planning results

| Condition | Objective | Seed 100 | Seed 200 | Seed 300 | Mean | Population SD | Delta vs R0 |
|---|---|---: | ---: | ---:|---:|---:|---:|
| R0 direction-only baseline | `0.1 × R0` | 100.00% | 92.00% | 88.00% | 93.33% | 4.99 pp | baseline |
| R1 speed-only | `0.1 × R1` | 34.00% | 48.00% | 62.00% | 48.00% | 11.43 pp | -45.33 pp |
| R2 full penalty | `0.05 × R2` | 88.00% | 88.00% | 80.00% | 85.33% | 3.77 pp | -8.00 pp |
| R3 direction + speed | `0.1 × (R0 + R1)` | 88.00% | 92.00% | 82.00% | 87.33% | 4.11 pp | -6.00 pp |

Planning uses seeds 100, 200, and 300. Each seed contains 50
evaluations split into five deterministic 10-evaluation chunks, for
150 evaluations per condition.

## Final training losses

| Condition | Training loss | Validation loss |
|---|---:|---:|
| R0 direction-only baseline | 0.0531 | 0.0508 |
| R1 speed-only | 0.0048 | 0.0043 |
| R2 full penalty | 0.0904 | 0.0898 |
| R3 direction + speed | 0.1042 | 0.1021 |

## Artifact layout

- `r1_speed_only/`, `r2_full_matched/`, and `r3_beta1/`: raw planner chunks
- `seed_*/aggregate.json`: validated 50-evaluation seed summaries
- `comparison.json`: aggregate metrics and paired deltas against R0
- `comparison.stdout`: human-readable aggregation output

## Validation

- All epoch-20 checkpoints are non-empty.
- Every planner chunk contains `final_eval/success_rate`.
- Every seed aggregate has `status: complete` and `n_evals: 50`.
- Paired comparisons use identical planning seeds across conditions.
