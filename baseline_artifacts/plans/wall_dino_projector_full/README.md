# Full Wall DINOv2 Patch + Projector Reproduction

This directory records a full Wall planning comparison using the authors'
`wall_single` dataset rather than the earlier 50-episode generated dataset.
The dataset contains 1,920 trajectories of 50 frames (96,000 frames total).

## Compared conditions

Both conditions use DINOv2 patch features, the learned channel projector, a
14x14 feature grid, and the epoch-20 checkpoint.

- `off`: no straightening loss
  (`wall_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06`)
- `on`: cosine straightening enabled with coefficient 0.1
  (`wall_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05`)

Training was run for 20 epochs on 8 A100 GPUs. Planning used seeds 100, 200,
and 300. Each seed contains 50 evaluations, split into five deterministic
10-evaluation chunks at offsets 0, 10, 20, 30, and 40. This gives 150 planning
evaluations per condition and 300 evaluations total.

## Planning results

| Condition | Seed 100 | Seed 200 | Seed 300 | Mean | Population SD |
|---|---:|---:|---:|---:|---:|
| Straightening OFF | 84% | 74% | 84% | 80.67% | 4.71 pp |
| Straightening ON | 100% | 92% | 88% | 93.33% | 4.99 pp |

The paired ON-minus-OFF success-rate difference is **+12.67 percentage
points** (population SD 6.18 pp; standard error 3.57 pp across the three
paired seeds).

The complete metric set, including state, visual, proprioceptive, and latent
divergence metrics, is stored in `comparison.json`. Each `seed_*/aggregate.json`
combines its five chunk logs into a 50-evaluation result. Chunk aggregation
uses evaluation-count-weighted means for distance and success metrics. The
historical `mean_div_*` fields are full-batch norms in the evaluator, so their
chunk values are combined by root-sum-of-squares.

## Artifact layout

- `off/` and `on/`: condition directories
- `seed_100/`, `seed_200/`, `seed_300/`: paired planning seeds
- `chunk_0/` through `chunk_40/`: raw 10-evaluation planner outputs
- `logs.json`: raw JSONL metrics for a chunk
- `runner.log`: full planner stdout/stderr for a chunk
- `aggregate.json`: validated 50-evaluation seed summary
- `comparison.json`: three-seed condition summary and paired deltas

The full directory contains 30 raw metric logs, six seed aggregates, 60 PNg
planning figures, and 600 MP4 rollouts. Training logs, Hydra configurations,
exact watcher commands, and epoch-20 checkpoints live under the sibling
`baseline_artifacts/logs/` and `baseline_artifacts/checkpoints/` directories.

## Validation

- Both epoch-20 checkpoints are non-empty and training logs record completed
  epoch-20 validation.
- All 30 planner chunks exited with return code 0.
- All 30 `logs.json` files contain `final_eval/success_rate`.
- Every seed aggregate has `status: complete` and `n_evals: 50`.
- `test_aggregate_plan_chunks.py` and
  `test_aggregate_condition_seeds.py` pass.

This full-data result supersedes the repository's earlier small-data Wall
experiment, whose 50 random-policy episodes were insufficient to measure the
effect of straightening.
