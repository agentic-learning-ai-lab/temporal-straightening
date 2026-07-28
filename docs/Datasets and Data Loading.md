# Datasets and Data Loading

**Source:** `datasets/` — `traj_dset.py` (the machinery), plus one loader per environment.

## The two-level design

Recorded data comes as **whole episodes** (a 100-frame video plus per-step states/actions), but training wants **tiny windows** (3 history frames + 1 future frame). The `datasets/` package cleanly separates the two concerns:

1. **`TrajDataset`** (abstract, in `traj_dset.py`) — "give me episode *i*": returns `(obs_dict, act, state, ...)` for a full trajectory. Each environment implements one subclass that knows its file format.
2. **`TrajSlicerDataset`** — wraps any TrajDataset and enumerates every valid sliding window of length `num_frames × frameskip` across every episode. It's the thing the DataLoader actually iterates. Frameskip is applied here: a window takes every 5th frame, and the 5 skipped raw actions get **concatenated into one "macro-action"** — so the model's action space is (5 × action_dim) per model-step. This is the same convention the planner must respect ([Planning - plan.py](Planning%20-%20plan.py.md)).

`random_split_traj` / `split_traj_datasets` split **by episode** (95/5 by default, fixed seed 42) — so no frames from a training episode ever leak into validation.

## The per-environment loaders

Each follows the same pattern — load `states.pth`, `actions.pth`, `seq_lengths.pth` tensors plus per-episode observation files, compute normalization statistics, expose `get_frames(idx, frame_range)`:

| File | Environment | Quirks worth knowing |
|---|---|---|
| `point_maze_dset.py` | PointMaze (UMaze & Medium) | proprio = position+velocity from the state; supports `use_preprocessed`/`use_frame_files` for per-frame `.pth` loading (an I/O optimization for HPC — the README explains why: thousands of tiny files vs one huge one) |
| `pusht_dset.py` | PushT | richer format (zarr-derived); state includes T-block pose |
| `wall_dset.py` | Wall | **contains the reproduction's compatibility fix** — the legacy import expected a `WallDataset` symbol; the Wall run initially crashed without it (see [The Reproduction Study](The%20Reproduction%20Study.md)) |
| `deformable_env_dset.py` | Rope / Granular | loads particle-sim data, has its own yaml-driven config |

Each module also exports a `load_<env>_slice_train_val()` convenience that stacks loader + splitter + slicer — this is the function name referenced by `conf/env/*.yaml` ([Hydra Configs](Hydra%20Configs.md)).

## Normalization and transforms

Loaders compute action/state/proprio mean & std, which travel with the model checkpoint inside the `Preprocessor` ([Supporting Code](Supporting%20Code.md)) — so at planning time observations are normalized *exactly* as during training. `img_transforms.py` is minimal: resize to 224 (no augmentation — a world model must predict what the camera actually sees; random crops would corrupt the dynamics).
