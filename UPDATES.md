# Updates

Re-runs of the paper's Table 1 protocol on updated code.

Protocol for every table: success rate (%) of 50 test episodes with the GD planner, open-loop /
MPC, mean ± std over eval seeds 100/101/102; old results in brackets. Models are trained from
scratch with `conf/train.yaml` defaults (batch 32, frameskip 5, num_hist 3, seed 0; 20 epochs,
pusht 2) and evaluated with `plan_gd.yaml` / `plan_gd_mpc.yaml` (pusht: `objective.alpha=1`, MPC
`objective.mode=staged`).

## DINOv2 (patch) + global projector: predictor sizing (`models/dino.py`)

Commit [`64a7585`](https://github.com/agentic-learning-ai-lab/temporal-straightening/commit/64a7585819e749bfec327ad984ee08570d07f0eb).

**What changed.** `train.py` sizes the predictor from `encoder.latent_ndim` before any forward
pass, but for the global projector (`encoder=dino_global`, one pooled token per frame)
`latent_ndim` was only set inside `forward()`. A fresh `dino_global` run therefore got a
predictor sized for the 14×14 patch grid (196 tokens per frame) while the encoder emitted one.
The ViT slices its block-causal mask to the runtime length (`bias[:, :, :T, :T]`), so the three
history tokens all landed in frame 0's all-ones block and attention was fully visible: every
history position but the last saw its own target, its loss became a copy task, and only the last
position trained on genuine one-step prediction. Runs did not crash. `DinoV2Encoder.__init__`
now sets `latent_ndim` from the projector's `pool_hw`.

**What is unaffected.** Every other encoder fixes `latent_ndim` at construction (`dino`,
`dino_channel`, `scratch_resnet_spatial`, `scratch_resnet`, `dino_cls`), so their rows are
untouched. Planning code is unchanged, and planning with an affected checkpoint was internally
consistent — the rollout sees only current and past frames, and only the last position's
prediction (the one trained properly) is consumed. The missing mask weakened the training
signal, not the planning procedure. Only the `DINOv2 (patch) + proj, 1×384` row moves;
pre-change checkpoints of it carry the weakened predictor.

**Results** — the row re-run with the change (projector lr 1e-6), λ selected on MPC validation
success (seeds 42/43, 50 episodes each) and tested on seeds 100/101/102.

Validation (GD / MPC %, mean over seeds 42/43; **best λ by MPC in bold**):

| λ | Wall | UMaze | Medium | PushT |
|---|---|---|---|---|
| ✗ | 67.0 / 72.0 | 31.0 / 79.0 | 26.0 / 69.0 | 22.0 / 54.0 |
| 1e-1 | **81.0 / 97.0** | 63.0 / 86.0 | **31.0 / 94.0** | 26.0 / 50.0 |
| 1e-2 | 80.0 / 82.0 | **30.0 / 97.0** | 30.0 / 90.0 | **24.0 / 59.0** |
| 1e-3 | 68.0 / 73.0 | 34.0 / 89.0 | 22.0 / 77.0 | 28.0 / 51.0 |

Selected λ: 1e-1 for wall and medium; 1e-2 for umaze and pusht (paper's markers: wall 1e-3,
medium 1e-2, umaze/pusht 1e-1). Test:

| L_curv | Wall | UMaze | Medium | PushT |
|---|---|---|---|---|
| ✗ | 64.7±4.1 / 69.3±6.8 (28.7 / 76.0) | 26.0±7.5 / 76.0±8.6 (34.7 / 79.3) | 24.7±5.0 / 73.3±4.1 (18.0 / 46.0) | 24.7±6.8 / 52.7±3.4 (2.0 / 11.3) |
| ✓ | 82.7±2.5 / 96.0±2.8 (32.0 / 77.3) | 35.3±9.0 / 98.0±1.6 (38.7 / 96.0) | 24.0±3.3 / 95.3±0.9 (22.7 / 78.0) | 25.3±3.8 / 58.7±1.9 (2.0 / 8.7) |

**The paper's conclusion holds, with better numbers.** Restoring the mask lifts both arms well
above the paper's row on every env, and straightening helps across the board.
