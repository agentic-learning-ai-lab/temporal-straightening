# UMaze trajectory-penalty ablation

## Scope

This study compares four trajectory penalties using the same UMaze planning protocol: DINOv2 patch features, channel projector, model epoch 20, three seeds (100, 200, 300), and 50 planning evaluations per seed (150 evaluations per condition).

- **R0 — direction only:** `0.1 × (1 − cos θ)`
- **R1 — speed only:** `0.1 × (r + 1/r − 2)`
- **R2 — full matched:** `0.05 × (r + 1/r − 2 cos θ)`; the 0.05 coefficient gives the same effective 0.1 direction weight as R0.
- **R3 — blended:** `0.1 × (R0 + R1)`

R0 was imported from the completed `~/ts-cls/umaze_reproduction` run. Its metadata identifies it as DINOv2 patch 14×14×8 + channel projector + straightening ON (aggcos, λ=0.1), with the same seeds and evaluation count.

## Planning results

| Condition | Seed 100 | Seed 200 | Seed 300 | Mean success | Population SD |
|---|---:|---:|---:|---:|---:|
| R0 direction only | 96% | 98% | 94% | **96.00%** | 1.63 pp |
| R1 speed only | 16% | 18% | 14% | **16.00%** | 1.63 pp |
| R2 full matched | 98% | 96% | 94% | **96.00%** | 1.63 pp |
| R3 blended | 98% | 96% | 90% | **94.67%** | 3.40 pp |

All 600 matched evaluations completed: 150 per condition.

## Final-state distance metrics

Lower is better. Values are means across the three seed aggregates; parentheses give population SD across seeds.

| Condition | State distance | Visual distance | Proprioceptive distance |
|---|---:|---:|---:|
| R0 direction only | 2.7301 (0.1639) | **0.3952** (0.0133) | 0.9023 (0.0555) |
| R1 speed only | 2.8481 (0.2278) | 0.5175 (0.0055) | 1.1419 (0.0985) |
| R2 full matched | **2.4121** (0.1119) | 0.4044 (0.0190) | **0.8108** (0.0480) |
| R3 blended | 2.6553 (0.0707) | 0.4075 (0.0113) | 0.8895 (0.0292) |

## Comparison against R0

- **R2 is the strongest overall result.** It exactly matches R0's 96.0% success while reducing mean state distance by 11.6% and proprioceptive distance by 10.1%. Its visual distance is 2.3% higher than R0.
- **R3 remains competitive but does not beat R0 on success.** It is 1.33 percentage points lower in success, while improving state distance by 2.7% and proprioceptive distance by 1.4%. Visual distance is 3.1% higher.
- **R1 fails decisively.** Speed-only regularization loses 80 percentage points of success relative to R0 and worsens all three final-distance metrics. Directional consistency is therefore essential on UMaze.
- The useful speed term is **complementary**, not sufficient alone: it helps most when included in R2 with the direction component and matched effective direction weight.

## Training endpoint

| Condition | Epoch | Final validation loss |
|---|---:|---:|
| R0 | 20 | Not retained in the imported planning artifact |
| R1 | 20 | 0.0048 |
| R2 | 20 | 0.0914 |
| R3 | 20 | 0.1067 |

These losses are not directly comparable across conditions because each objective contains a differently scaled regularizer. Planning success and matched final-state metrics are the primary comparison.

## Artifact layout

- `comparison.json`: four-condition seed aggregation and paired comparisons against R0.
- `comparison.stdout`: aggregation output.
- `r0_direction_only/seed_{100,200,300}/aggregate.json`: normalized R0 seed aggregates.
- `r0_direction_only/seed_{100,200,300}/source_logs.json`: original 50-evaluation R0 logs.
- `r0_direction_only/source_metadata.json` and `source_result.txt`: imported R0 provenance and original summary.
- `r1_speed_only/seed_{100,200,300}/aggregate.json`: R1 seed aggregates.
- `r2_full_matched/seed_{100,200,300}/aggregate.json`: R2 seed aggregates.
- `r3_beta1/seed_{100,200,300}/aggregate.json`: R3 seed aggregates.
- Each R1–R3 seed directory also retains its five 10-evaluation planning chunks and raw logs.

## Conclusion

The ablation supports retaining direction straightening. R2 adds speed consistency without sacrificing R0's success rate and produces substantially better state and proprioceptive terminal accuracy. R1 shows that speed regularization without direction control is not a viable replacement.
