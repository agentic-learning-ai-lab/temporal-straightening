# Training — train.py

**Source:** `train.py` (960 lines). Entry point: `python train.py --config-name train.yaml env=point_maze ...`

One big `Trainer` class driven entirely by [Hydra Configs](Hydra%20Configs.md). If you strip away the infrastructure, the core loop is simply: *load a batch of 4-frame windows → `VWorldModel.forward()` → backprop → repeat* ([The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md)). Everything else is scaffolding, and the scaffolding is most of the file:

## What the Trainer actually manages

- **Model assembly (`init_models`)** — Hydra `_target_` instantiation of encoder/predictor/decoder/proprio/action encoders from their config groups, then wiring them into `VWorldModel`. Also handles resuming from `model_latest.pth`.
- **Selective trainability (`_configure_encoder_trainability`)** — this is where "frozen DINO backbone but trainable projector" is enforced: base DINO weights get `requires_grad=False`, projector/agg parameters stay trainable. Worth reading if you're studying the adapter ablation.
- **Per-module optimizers (`init_optimizers`)** — separate learning rates for encoder (1e-5), predictor (5e-4), decoder (3e-4), action encoder (5e-4), from `conf/train.yaml`.
- **Multi-GPU** — HuggingFace `Accelerate` wraps everything (bf16 mixed precision by default); a SLURM/multirun branch sets up NCCL DDP process groups for cluster sweeps. See also [Supporting Code](Supporting%20Code.md) (`distributed_fn/`).
- **Checkpointing (`save_ckpt`)** — writes `model_latest.pth` plus per-epoch `model_<N>.pth` into a Hydra-generated directory whose *name encodes the whole experiment* (env, straighten flag, projector, dims, stop_grad, lr — built by the resolvers in `custom_resolvers.py`). Downstream, [Planning - plan.py](Planning%20-%20plan.py.md) parses this.
- **Validation (`val`) and `openloop_rollout`** — beyond val-set loss, it dreams forward from validation prefixes using `rollout()` and compares imagined frames against reality with the image metrics (SSIM/PSNR/LPIPS — see [Supporting Code](Supporting%20Code.md)), logging reconstruction grids (`plot_samples`) to wandb. These grids are the ancestors of the contact sheets in [Evidence Packs](Evidence%20Packs.md).
- **`err_eval`** — measures prediction error separately for visual and proprio token slices, so logs can distinguish "can't predict the scene" from "can't predict the robot state."
- **Mid-training planning evals (`monitor_jobs` + `plan_settings` in the config)** — optionally submits real planning jobs (via submitit/SLURM) on saved checkpoints *while training continues*, then folds their success rates back into the logs. Disabled by default (`plan_cfg_path: null`); the reproduction ran planning separately instead.
- **`decoder_start_epoch`** — lets the decoder join late so early noisy latents don't waste decoder capacity.

## The reading path

To understand training deeply, read in this order:
1. `Trainer.run()` — the epoch skeleton
2. `train()` — the batch loop (note `loss_components` fan-out into logs)
3. `init_models()` + `_configure_encoder_trainability()` — what's trainable
4. `openloop_rollout()` — how "does the dream match reality?" is measured

Details of what the loss contains live in [The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md) and [The Straightening Loss](The%20Straightening%20Loss.md); the data feeding it comes from [Datasets and Data Loading](Datasets%20and%20Data%20Loading.md).
