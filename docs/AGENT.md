# AGENT.md — orientation for this repo

Context file for anyone (human or AI agent) landing in this repository. For the full guided tour, start at [docs/README.md](README.md).

## What this repository is

Two projects stacked on top of each other:

1. **The paper's code** — *Temporal Straightening for Latent Planning* (Wang et al. 2026), a fork of NYU's [DINO-WM](https://github.com/gaoyuezhou/dino_wm). A visual world model: encode camera images into latents, learn to predict how latents evolve under actions, then plan by gradient-descending on actions inside that learned imagination. The paper's contribution is a **curvature regularizer** that trains latent trajectories to be locally straight, which is meant to make gradient-based planning better-conditioned.

2. **A reproduction study** (July 2026, run on Modal cloud GPUs) that re-ran the experiments and tested a critique: the cosine straightening loss fixes the *direction* of latent motion but ignores its *speed*, even though the theory assumes constant speed.

## The two-layer map (the fastest way to orient)

**Layer 1 — paper code:** `train.py`, `plan.py`, `preprocessor.py`, `utils.py`, `custom_resolvers.py`, `models/`, `planning/`, `datasets/`, `env/`, `conf/`, `metrics/`, `distributed_fn/`, `README.md`, `environment.yaml`, `assets/`.

**Layer 2 — reproduction study:** `report.md`, `research_note.tex/.pdf`, `modal_medium_runner.py`, `modal_evidence/`, `generate_point_maze_medium.py`, `generate_wall_dataset.py`, `setup.sh`, `reqs310.txt`. Also the `speed`/`aggspeed` loss terms inside `models/visual_world_model.py` are a Layer-2 addition to a Layer-1 file.

## Key facts

- **The core loss lives in `models/visual_world_model.py`** (`VWorldModel`). `training.straighten` is a string DSL parsed in `__init__`: `+`-joined tokens like `cos1e-1+aggspeed1e-1`. `cos`/`aggcos` = curvature (per-patch / pooled); `speed`/`aggspeed` = speed constancy. Curvature = `1 − cos(z_{t+1}−z_t, z_{t+2}−z_{t+1})`.
- **Prediction uses stop-grad on targets** (`training.stop_grad=True`) to prevent representation collapse.
- **The planner optimizes actions, not weights** (`planning/gd.py`): actions are a `requires_grad` tensor; loss backprops through `VWorldModel.rollout()`.
- **Encoder variants → conf files**: `conf/encoder/dino.yaml` (frozen patch), `dino_cls.yaml` (CLS token), `dino_channel.yaml` (frozen backbone + trainable adapter), `dino_global.yaml`, `scratch_resnet*.yaml`, `r3m.yaml`.
- **Frozen vs trainable matters**: with a fully frozen encoder there are no trainable params between pixels and the measured latents, so straightening losses can't reshape geometry — this is why the reproduction's adapter ablation (`dino_channel`) was the meaningful test.
- **Vendored / low-value for deep dives**: `models/encoder/r3m/`, `metrics/lpipsPyTorch/`, `env/venv.py`, `env/deformable_env/src/sim/assets/`, `models/vqvae.py`.
- **Checkpoint dir names encode the whole config** (see `conf/train.yaml` `hydra.run.dir`); `custom_resolvers.py` exists to make those strings path-safe.
- **Headline results** (`report.md`): Medium — straightening (0.14) didn't beat no-straightening (0.14); DINO CLS won (0.22). Adapter ablation — cosine straightens direction but worsens speed variation; combined gives best geometry but not best planner success. Wall — both variants 0.02.

## docs/ conventions

These notes were authored in an Obsidian vault and published here. Internal links are standard relative markdown links (URL-encoded spaces). Each note leads with plain-English intuition before mechanism. `docs/README.md` is the map of content.
