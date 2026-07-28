# Modal Runner

**Source:** `modal_medium_runner.py` (1,209 lines) — the reproduction study's mission control. A Layer-2 addition; the paper's code knows nothing about it.

[Modal](https://modal.com) is a serverless GPU platform: you define a container image and functions in Python, and `modal run` executes them on rented cloud GPUs with a persistent network volume. This one file encodes the *entire* reproduction so any experiment is a single reproducible command (the exact commands are recorded at the bottom of `report.md`).

## Anatomy

- **The image** — a pinned `pytorch:2.3.0-cuda12.1` base plus apt GL/EGL packages and ~40 pinned pip deps. The cloud twin of `setup.sh` ([Setup and Dependencies](Setup%20and%20Dependencies.md)).
- **The volume** — a persistent disk (`temporal-straightening-medium`) mounted at `/mnt/ts`, holding generated datasets, checkpoints, plan targets, and logs across runs. [Evidence Packs](Evidence%20Packs.md) are what got *pulled down* from this volume.
- **Variant registry (`_variant_specs`)** — the single source of truth mapping human-readable variant names to config overrides, e.g. `dino_channel_cos_plus_speed` → `encoder=dino_channel training.straighten=cos1e-1+speed1e-1`. The variant names in every results table come from here.
- **Pipeline steps** — `_ensure_dataset` (runs [Dataset Generators](Dataset%20Generators.md) on the volume if data is missing), `_train_and_plan_variant`, `_plan_variant` (invoking `train.py` / `plan.py` as subprocesses), plus log-scraping helpers (`_parse_epoch_losses`, `_read_plan_logs`) that turn raw stdout into the JSON summaries.

## The `--action` subcommands

| Action | What it does |
|---|---|
| `smoke` | Tiny end-to-end sanity run |
| `run` | Full pipeline for one variant: dataset → train → plan → summarize |
| `plan-existing` | Re-run planning on an already-trained checkpoint (used when the planner was fixed after training, and after the spend-limit interruption) |
| `analyze-latents` | The diagnostics engine — see below |
| `package-artifacts` | Bundle results/manifests for download |

## `analyze-latents` — where the geometry numbers come from

`_latent_metrics_from_z` encodes real validation rollouts with a trained checkpoint and computes, per trajectory, the four metrics used throughout [The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md): mean cosine curvature, path-to-endpoint ratio, speed coefficient-of-variation, and relative speed jump (plus p95/p05 speed ratio). `_analyze_variant_latents` aggregates them across rollouts into `latent_analysis.json`. This is the measurement layer that revealed the frozen-encoder ablation was inert (identical numbers across all loss settings) — arguably the most consequential code in the reproduction.

## Reading tips

- The file is long but flat: helpers up top, one Modal function per action, `main()` dispatching CLI flags at the bottom.
- Run IDs like `medium-adapter-cos-20260702-01` are chosen by the caller and become directory names on the volume *and* the names of [Evidence Packs](Evidence%20Packs.md) — the ID is the join key between report, evidence, and Modal dashboard URLs (some are recorded in the evidence READMEs).
