# UMaze straightening-penalty results

All conditions: open-loop GD planning, `plan_gd.yaml`, seeds 100/200/300,
50 evaluations per seed, epoch-20 checkpoints.

The penalty is our R3 trajectory-penalty term (`trajectory_penalties` in
`models/visual_world_model.py`): `r3 = r0 + beta * r1`, where `r0` is the
direction (curvature) term and `r1` is the speed term. Applied to the loss
as `scale * (r0 + beta * r1)`, each condition is two coefficients:
**direction** (`scale`) and **speed** (`scale * beta`).
Every row holds the direction coefficient at 0.1; rows are ordered by
speed coefficient.

| condition | token | direction | speed | seeds | success rate | state dist | visual dist |
|---|---|---:|---:|---:|---:|---:|---:|
| r0_direction_only | `aggcos1e-1` | 0.1000 | 0.0000 | 3 | 0.960 ± 0.016 | 2.730 ± 0.164 | 0.395 |
| r2_full_matched | `aggr2_5e-2` | 0.1000 | 0.0500 | 3 | 0.960 ± 0.016 | 2.412 ± 0.112 | 0.404 |
| r3_beta1 | `aggr3b1_1e-1` | 0.1000 | 0.1000 | 3 | 0.947 ± 0.034 | 2.655 ± 0.071 | 0.408 |
| r3_beta2 | `aggr3b2_1e-1` | 0.1000 | 0.2000 | 3 | 0.927 ± 0.034 | 2.518 ± 0.169 | 0.405 |
| r3_beta5 | `aggr3b5_1e-1` | 0.1000 | 0.5000 | 3 | 0.913 ± 0.019 | 2.766 ± 0.145 | 0.442 |
| r3_beta10 | `aggr3b10_1e-1` | 0.1000 | 1.0000 | 3 | 0.753 ± 0.047 | 2.745 ± 0.028 | 0.459 |

## Notes

- Error bars are population std across seeds, not across episodes.
- Every condition is a single training run, so seed spread reflects
  planning-seed variation only; training variation is unmeasured.
- `r2` expands to `scale * (r1 + 2*r0)`, so `aggr2_5e-2` carries a
  direction coefficient of 0.1 -- matched to `r0` by construction.
