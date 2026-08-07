# UMaze straightening-penalty results

All conditions: open-loop GD planning, `plan_gd.yaml`, seeds 100/200/300,
50 evaluations per seed, epoch-20 checkpoints.

The penalty is `scale * (r0 + beta * r1)`, so each condition is two
coefficients: **direction** (`scale`) and **speed** (`scale * beta`).
Rows are ordered by direction coefficient.

| condition | token | direction | speed | seeds | success rate | state dist | visual dist |
|---|---|---:|---:|---:|---:|---:|---:|
| r0_direction_only | `aggcos1e-1` | 0.1000 | 0.0000 | 3 | 0.960 ± 0.016 | 2.730 ± 0.164 | 0.395 |
| r2_full_matched | `aggr2_5e-2` | 0.1000 | 0.0500 | 3 | 0.960 ± 0.016 | 2.412 ± 0.112 | 0.404 |
| r3_beta1 | `aggr3b1_1e-1` | 0.1000 | 0.1000 | 3 | 0.947 ± 0.034 | 2.655 ± 0.071 | 0.408 |
| r3_beta2 | `aggr3b2_1e-1` | 0.1000 | 0.2000 | 3 | 0.927 ± 0.034 | 2.518 ± 0.169 | 0.405 |
| r3_beta5 | `aggr3b5_1e-1` | 0.1000 | 0.5000 | 3 | 0.913 ± 0.019 | 2.766 ± 0.145 | 0.442 |
| r3_beta10 | `aggr3b10_1e-1` | 0.1000 | 1.0000 | 3 | 0.753 ± 0.047 | 2.745 ± 0.028 | 0.459 |
| r3_beta0p3 | `aggr3b0.3_0.0769231` | 0.0769 | 0.0231 | 3 | 0.873 ± 0.009 | 2.757 ± 0.246 | 0.425 |
| r3_beta1 | `aggr3b1_0.05` | 0.0500 | 0.0500 | 3 | 0.827 ± 0.074 | 2.857 ± 0.442 | 0.437 |
| r3_beta3 | `aggr3b3_0.025` | 0.0250 | 0.0750 | 3 | 0.613 ± 0.068 | 3.174 ± 0.025 | 0.480 |
| r3_beta30 | `aggr3b30_0.00322581` | 0.0032 | 0.0968 | 3 | 0.273 ± 0.025 | 3.178 ± 0.091 | 0.506 |
| r1_speed_only | `aggr1_1e-1` | 0.0000 | 0.1000 | 3 | 0.160 ± 0.016 | 2.848 ± 0.228 | 0.517 |

## Notes

- Error bars are population std across seeds, not across episodes.
- Every condition is a single training run, so seed spread reflects
  planning-seed variation only; training variation is unmeasured.
- `r2` expands to `scale * (r1 + 2*r0)`, so `aggr2_5e-2` carries a
  direction coefficient of 0.1 -- matched to `r0` by construction.
- Mean state distance is not monotone in success rate at the extremes:
  a model that rarely reaches the goal can still stop at a middling
  average distance, so success and distance can disagree.
