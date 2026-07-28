# Setup and Dependencies

**Sources:** `environment.yaml`, `reqs310.txt`, `setup.sh`

Three dependency files because the code ran in three different places:

## `environment.yaml` — the paper's conda env (Layer 1)

The original authors' full conda specification (`conda env create -f environment.yaml`, env name `ts`). Pinned scientific stack: PyTorch, mujoco-py, gym 0.23, d4rl, hydra 1.2, accelerate, wandb, einops, etc. This is what the README's install section uses.

## `reqs310.txt` — the reproduction's pip list (Layer 2)

A flat pip requirements file for Python 3.10, distilled by the reproduction for fast rebuilds on ephemeral cloud machines (conda solves are slow; pip installs into a bare python 3.10 env are fast). Same core libraries, pip-pinned.

## `setup.sh` — bootstrap for a Lambda GPU instance (Layer 2)

A `source`-able script that makes a **fresh, wiped-on-restart cloud machine** ready to train, revealing the project's actual working setup:

- Persistent storage lives on NFS (`/lambda/nfs/temporal-straightening`); local disk is disposable, so the conda env, MuJoCo, and GL libraries are rebuilt each session while datasets and `reqs310.txt` persist on NFS.
- **Headless rendering** is the fiddly part: MuJoCo renders through EGL with no display attached, requiring GLEW/OSMesa plus the NVIDIA EGL vendor library matched to the driver version (the script literally greps `nvidia-smi` for it). This is what lets [Dataset Generators](Dataset%20Generators.md) render 224×224 frames on a display-less server.
- MuJoCo 210 is downloaded to `~/.mujoco` (the pre-DeepMind-acquisition version d4rl needs), with `LD_LIBRARY_PATH` exports.
- One-time dataset copy to NFS; `DATASET_DIR` env var points loaders at it ([Datasets and Data Loading](Datasets%20and%20Data%20Loading.md)); `WANDB_MODE=disabled` — the reproduction logged to files, not wandb.

## The third environment: Modal

[Modal Runner](Modal%20Runner.md) duplicates all of this *again* as a container image definition (`modal.Image` with apt + pip layers pinned inside `modal_medium_runner.py`) — the cloud-function equivalent of setup.sh. If you ever wonder "which env actually produced the evidence," it's the Modal image, not the conda env.

## Practical gotchas recorded here

- gym is pinned at 0.23 (old API: `step` returns 4 values) — modern gymnasium would break the wrappers.
- `setuptools<81` is force-installed because wandb still imports `pkg_resources`.
- The README's LD_LIBRARY_PATH instructions are duplicated in setup.sh; on any new machine, MuJoCo-can't-find-GL errors are almost always this.
