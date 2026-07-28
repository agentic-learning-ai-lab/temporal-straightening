# Medium Modal Run Evidence

- Run id: `medium-full-20260702-01`
- Environment: `point_maze_medium`
- Dataset: 50 random-policy episodes x 100 steps, saved in Modal at `/mnt/ts/datasets/medium_random_50x100/point_maze_medium`
- Training: 20 epochs per variant
- Planner: corrected gradient-descent planner, 50 evals, `goal_H=25`, logs under `plans_fixed`

## Modal Runs
- training_full_run: https://modal.com/apps/usharma123/main/ap-K4JvRZKJhWtAPf2a3rd1W2
- authors_fixed_planner: https://modal.com/apps/usharma123/main/ap-zKWH3GRDt5zYMUJLD1qFXc
- dino_patch_fixed_planner: https://modal.com/apps/usharma123/main/ap-QsoLtRwFuUf34QKvJHTqBX
- dino_cls_fixed_planner: https://modal.com/apps/usharma123/main/ap-uM635flofsixaWomdWe9cE

## Results
| Variant | Train loss | Val loss | Success rate | Mean state dist | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Authors approach baseline | 0.2552 | 0.2385 | 0.14 | 4.4943 | `modal_evidence/medium-full-20260702-01/plans_fixed/authors_dino_patch_straightened/logs.json` |
| DINO patch, no straightening | 0.1195 | 0.1025 | 0.14 | 4.4943 | `modal_evidence/medium-full-20260702-01/plans_fixed/dino_patch_no_straightening/logs.json` |
| DINO CLS, no straightening | 0.0785 | 0.0691 | 0.22 | 3.5456 | `modal_evidence/medium-full-20260702-01/plans_fixed/dino_cls_no_straightening/logs.json` |

## Latent Diagnostics
`latent_analysis.json` compares directional straightness against speed constancy on 50 Medium rollouts. Lower is better for all three diagnostic columns below.

| Variant | Cosine curvature | Latent speed CV | Speed p95/p05 | Relative speed jump |
| --- | ---: | ---: | ---: | ---: |
| Authors approach baseline | 1.3402 | 0.3973 | 4.7802 | 0.3791 |
| DINO patch, no straightening | 1.3402 | 0.3973 | 4.7802 | 0.3791 |
| DINO CLS, no straightening | 1.3533 | 0.3377 | 3.9247 | 0.3636 |

The representative validation and rollout PNGs are visually near-identical because they show the same Medium maze layout and same sampled batches. In the checked epoch-20 files, authors and DINO patch are pixel-identical; DINO CLS differs slightly. The stronger experimental signal is in the planner and latent diagnostics: DINO CLS improves success rate and reduces speed variation, while not improving cosine curvature.

## Local Evidence Files
- `combined_results.json`: machine-readable merged results
- `summary.json`: original full training run summary; includes stale planner override errors after successful training
- `summary_plans_fixed_latest.json`: latest single-variant corrected planner summary from Modal
- `logs/*/train.log`: per-variant training logs
- `logs/*/plan_gd_fixed.log`: corrected planner stdout/stderr
- `plans_fixed/*/logs.json`: corrected planner metrics as JSONL
- `latent_analysis.json`: latent straightness and speed constancy diagnostics
- `logs/generate_medium_dataset.log`: dataset generation log
- `full_artifacts/runs/medium-full-20260702-01`: full local mirror of Modal run artifacts
- `full_artifacts/datasets/medium_random_50x100/point_maze_medium`: local copy of the generated Medium dataset
- `archives/medium-full-20260702-01-media-evidence.tar`: Modal-built tar archive for evidence/media artifacts
- `archives/medium-full-20260702-01-media-evidence-manifest.json`: manifest for the archive
- `local_artifact_manifest.json`: local artifact counts after extraction and checkpoint pull
- `epoch20_validation_contact_sheet.png`: quick visual sheet for epoch-20 validation images

## Image And Checkpoint Artifacts
- Local artifact mirror size: about 30 GB extracted, 31 GB including archive.
- Total mirrored files: 2,632.
- Image/media files: 2,520 PNGs.
- Training/validation PNGs: 120.
- Rollout plot PNGs: 2,400.
- Checkpoint files: 63 `.pth` snapshots, including every epoch plus `model_latest.pth` for all three variants.
- Dataset files: 53, including 50 episode observation tensors plus `states.pth`, `actions.pth`, and `seq_lengths.pth`.
- Main image locations:
  - `full_artifacts/runs/medium-full-20260702-01/checkpoints/*/test/*/train/`
  - `full_artifacts/runs/medium-full-20260702-01/checkpoints/*/test/*/valid/`
  - `full_artifacts/runs/medium-full-20260702-01/checkpoints/*/test/*/rollout_plots/`

## Important Note
The original full-run planner commands failed due stale Hydra overrides (`planner.horizon` not in the structured planner config). The corrected `plan-existing` passes use `planner.sub_planner.*` and are the planner metrics reported above.
