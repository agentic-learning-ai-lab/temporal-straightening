# UMaze · DINOv2 (CLS) · no straightening — reproduction

Reproduces **Table 1, row 1** of Wang et al., *Temporal Straightening for Latent
Planning*, ICML 2026 (arXiv:2603.12231v2): the `DINOv2 (CLS) | 1×384 | Lcurv OFF`
row on **PointMaze-UMaze**. This is an unshaded row — a DINO-WM baseline
(Zhou et al. 2025) as reported by Wang et al., not their straightening
contribution. There is no CLS + straightening row in the paper.

One frozen checkpoint (`model_20.pth`, DINOv2 ViT-S/14 CLS token, world model
trained 20 epochs) underlies every number here. Only the planner and the data
sampling seed vary.

## Headline

| Setting | Ours | Paper (Table 1) |
|---|---|---|
| Open-loop GD, 10 seeds | **23.80 ± 8.32 %** | 25.33 ± 0.94 % (n=3) |
| MPC (closed-loop GD), 3 seeds | **74.0 ± 1.6 %** | 82.67 ± 9.98 % |

## What's measured vs. what's argued

**Measured** (parsed from `plan.py` logs; see the `RESULT_*.txt` files):

- Open-loop mean reproduces to within ~1.5 points of the paper.
- MPC lifts the mean by ~57 points over open-loop on the same checkpoint.
- On the *same three seeds* (100/200/300): open-loop = 28/8/14, MPC = 74/76/72.
  Replanning raised the mean **and** collapsed the seed spread (σ 8.3 → 1.6).

**Argued** (interpretation, not a measurement):

- The paper's open-loop σ of 0.94 (n=3) understates seed sensitivity. Our 10
  seeds span 8–36 % with σ = 8.32. The paper's own CLS row supports this: every
  *other* cell reports σ 8–13 (Wall 12.68, Medium 8.16, PushT 8.22), so their
  UMaze 0.94 is the outlier within their own results.
- We cannot explain *why* their three seeds clustered — lucky draw is simplest,
  but their seeds are unnamed (Section 5.3), so it is not checkable from the PDF.

## A note on our first result

Our first three seeds (100/200/300) gave **16.67 ± 8.38**, which looked like a
failed reproduction. It was a bad draw: 8 and 14 are two of the three lowest
values in a 10-seed sample. Seven more seeds (5 min of planning, no retraining)
moved the mean to 23.80 and dissolved the "gap." The lesson generalizes — treat
any 3-seed number on this benchmark as directional, and sweep before concluding.

## Layout

```
results/umaze_cls_RESULT_50.txt   original 3-seed open-loop run (16.67 — superseded; see seed_sweep/)
plans/seed_{100,200,300}/         open-loop rollouts + decoded videos (decode_for_viz=true)
seed_sweep/                       open-loop, seeds 400–1000; RESULT_SEEDS.txt has the 10-seed aggregate
mpc/                              MPC, seeds 100/200/300; RESULT_MPC.txt
hydra.yaml                        frozen config the checkpoint was planned under
```

The 434 MB `model_20.pth` is not committed (matches the other evidence dirs).

## Reproduce

```
# open-loop, any seed:
python plan.py --config-name plan_gd.yaml     ckpt_base_path=<ckpt> model_epoch=20 n_evals=50 seed=<S>
# MPC, any seed:
python plan.py --config-name plan_gd_mpc.yaml ckpt_base_path=<ckpt> model_epoch=20 n_evals=50 seed=<S>
```

Hardware: 1× A100-SXM4-40GB. CLS is light (3 tokens/frame), so 50 evals fit
un-chunked; patch-token rows do not. Render backend is OSMesa via conda-forge
`mesalib=23.3.*` (EGL compiles but crashes at runtime on this GPU).
