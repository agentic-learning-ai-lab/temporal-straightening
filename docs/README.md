# Home — Map of the Repo

This wiki explains everything in **this repository**, which contains **two projects stacked on top of each other**:

1. **The paper's code** — *Temporal Straightening for Latent Planning* (Wang et al. 2026), a fork of NYU's [DINO-WM](https://github.com/gaoyuezhou/dino_wm). A robot watches the world through a camera, compresses each image into a compact vector (a "latent"), learns to predict how that vector changes when it acts, and then **plans by doing gradient descent on its own imagination**. The paper's twist: training the latents so that trajectories through latent space are *straight lines*, which should make that gradient descent easier.

2. **A reproduction study** (July 2026, run on Modal cloud GPUs) that re-ran the paper's experiments and tested a critique: the paper's straightening loss fixes the *direction* of latent motion but ignores its *speed* — even though the theory assumes constant speed.

Start with [The Big Idea](The%20Big%20Idea.md) if you're new. Every note links back here.

## 1 · The Idea

- [The Big Idea](The%20Big%20Idea.md) — what temporal straightening is and why anyone would want it
- [The Straightening Loss](The%20Straightening%20Loss.md) — the actual math, and the string-DSL that switches it on
- [The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md) — the hole the reproduction study poked in it

## 2 · The Pipeline

- [Pipeline Overview](Pipeline%20Overview.md) — the whole journey: pixels → latents → predictions → plans
- [Training - train.py](Training%20-%20train.py.md) — how the world model is trained
- [Planning - plan.py](Planning%20-%20plan.py.md) — how a trained model is used to reach goals
- [Datasets and Data Loading](Datasets%20and%20Data%20Loading.md) — how recorded trajectories become training batches

## 3 · The Model

- [The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md) — the central class where everything (including the losses) lives
- [Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md) — the eyes: DINO patch/CLS, projectors/adapters, ResNets, R3M
- [Predictor and Decoders](Predictor%20and%20Decoders.md) — the imagination (ViT) and the "show me" modules (VQ-VAE)

## 4 · Planning

- [The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md) — three ways to search for good actions
- [Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md) — what "close to the goal" means, and how success is judged

## 5 · The Worlds

- [Environments](Environments.md) — PointMaze, PushT, Wall, deformables — the four simulated worlds
- [Dataset Generators](Dataset%20Generators.md) — the reproduction's scripts for creating fresh training data

## 6 · Configuration & Support

- [Hydra Configs](Hydra%20Configs.md) — the `conf/` tree that wires every experiment together
- [Supporting Code](Supporting%20Code.md) — preprocessing, utils, metrics, distributed training, vectorized envs
- [Setup and Dependencies](Setup%20and%20Dependencies.md) — environment.yaml vs reqs310.txt vs setup.sh

## 7 · The Reproduction Study

- [The Reproduction Study](The%20Reproduction%20Study.md) — what was tested, what was found
- [Results Analysis](Results%20Analysis.md) — an independent read of the raw result files, and why most "no difference" results are degenerate
- [Modal Runner](Modal%20Runner.md) — `modal_medium_runner.py`, the 1,200-line cloud orchestrator
- [Evidence Packs](Evidence%20Packs.md) — the `modal_evidence/` folders, decoded

## File-to-note index

| Path in repo | Covered in |
|---|---|
| `README.md` | [The Big Idea](The%20Big%20Idea.md) |
| `report.md`, `research_note.tex/.pdf` | [The Reproduction Study](The%20Reproduction%20Study.md) |
| `modal_evidence/**/*.json` (raw results) | [Results Analysis](Results%20Analysis.md) |
| `train.py` | [Training - train.py](Training%20-%20train.py.md) |
| `plan.py` | [Planning - plan.py](Planning%20-%20plan.py.md) |
| `preprocessor.py`, `utils.py`, `custom_resolvers.py` | [Supporting Code](Supporting%20Code.md) |
| `models/visual_world_model.py` | [The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md) |
| `models/dino.py`, `models/encoder/` | [Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md) |
| `models/vit.py`, `models/vqvae.py`, `models/decoder/`, `models/proprio.py`, `models/dummy.py` | [Predictor and Decoders](Predictor%20and%20Decoders.md) |
| `planning/gd.py`, `cem.py`, `mpc.py`, `base_planner.py` | [The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md) |
| `planning/objectives.py`, `planning/evaluator.py` | [Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md) |
| `datasets/` | [Datasets and Data Loading](Datasets%20and%20Data%20Loading.md) |
| `env/` | [Environments](Environments.md) |
| `env/venv.py`, `env/serial_vector_env.py` | [Supporting Code](Supporting%20Code.md) |
| `conf/` | [Hydra Configs](Hydra%20Configs.md) |
| `metrics/`, `distributed_fn/` | [Supporting Code](Supporting%20Code.md) |
| `generate_point_maze_medium.py`, `generate_wall_dataset.py` | [Dataset Generators](Dataset%20Generators.md) |
| `modal_medium_runner.py` | [Modal Runner](Modal%20Runner.md) |
| `modal_evidence/` | [Evidence Packs](Evidence%20Packs.md) |
| `setup.sh`, `environment.yaml`, `reqs310.txt` | [Setup and Dependencies](Setup%20and%20Dependencies.md) |
| `assets/architecture.png` | [The Big Idea](The%20Big%20Idea.md) |
| `LICENSE`, `.gitignore` | MIT-style license; ignores checkpoints/data |
