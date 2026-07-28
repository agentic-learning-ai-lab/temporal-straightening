# Supporting Code

The connective tissue: small modules every pipeline stage leans on.

## `preprocessor.py` — one normalization to rule them all

A tiny class holding the dataset's action/state/proprio mean & std plus the image transform. Its job is consistency: **planning-time observations must be normalized exactly like training data**, or the encoder sees out-of-distribution inputs. It travels with the checkpoint (stats are computed by the loaders in [Datasets and Data Loading](Datasets%20and%20Data%20Loading.md)) and provides `normalize_/denormalize_actions` (the planner optimizes in normalized action space, then denormalizes to execute — see [Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md)) and `transform_obs` (uint8 HWC images → normalized float CHW tensors).

## `utils.py` — the trajectory-dict toolbox

Observations flow through the code as dicts (`{"visual": ..., "proprio": ...}`), so generic helpers do dict-wise ops: `slice_trajdict_with_t` (time-slice every entry), `concat_trajdict` / `aggregate_dct`, `move_to_device`, `sample_tensors`, `cfg_to_dict`, and `seed()` (seeds python/numpy/torch for reproducibility). Also RAM introspection helpers used for logging.

## `custom_resolvers.py` — two string helpers

Registers `replace_slash` and `replace_substring` as OmegaConf resolvers so [Hydra Configs](Hydra%20Configs.md) can build path-safe, self-documenting checkpoint directory names. Imported for its side effect (`import custom_resolvers  # noqa`) at the top of `train.py`.

## `metrics/` — image similarity for eval

- `image_metrics.py` — SSIM, PSNR, L1/L2: standard "how similar are these two images" measures, used by [Training - train.py](Training%20-%20train.py.md)'s open-loop rollout eval to grade dreamed frames against real ones.
- `lpipsPyTorch/` — vendored LPIPS ("perceptual distance": compares deep-network features instead of pixels, matching human judgments better). Third-party code; don't deep-dive.

## `distributed_fn/` — multi-GPU plumbing

`distributed.py` (rank/world-size helpers, `all_gather`, `reduce_dict`) and `launch.py` (spawn one worker per GPU). Standard PyTorch DDP boilerplate supporting [Training - train.py](Training%20-%20train.py.md)'s cluster mode; nothing project-specific.

## `env/venv.py` + `env/serial_vector_env.py` — parallel simulators

`SubprocVectorEnv` (adapted from the Tianshou RL library, ~1000 lines) runs N environment copies in separate processes with shared-memory observation buffers, so [Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md) can execute 50 planned rollouts simultaneously. `SerialVectorEnv` is the same interface without subprocesses (debugging / constrained environments like the Modal containers). Treat both as infrastructure: the interface (`reset`, `step`, per-env `render`) matters, the internals don't.

## Odds and ends

- `LICENSE` — MIT (inherited from DINO-WM).
- `.gitignore` — excludes checkpoints, datasets, wandb logs, and other bulky run artifacts.
- `assets/architecture.png` — the paper's architecture/teaser figure, embedded by `README.md`.
