# UMaze linear probing

Train linear probes on latent activations to ask: **at this layer, is position / speed / direction linearly readable?** Then use hook knockout or mean-replacement to test whether that layer is causally involved.

This is a minimal prototype you can run locally before wiring interventions into planning.

---

## What you are probing

| Target | Ground truth | Source |
|---|---|---|
| **Position** | `(x, y)` | `states[..., 0:2]` — raw qpos, **not** normalized proprio |
| **Speed** | scalar | `‖qvel‖` from `states[..., 2:4]` |
| **Direction** | angle θ | `atan2(vy, vx)`; probe predicts **cos θ** and **sin θ** separately |

Frames with speed below `1e-4` get NaN direction labels and are skipped for direction probes.

Always align labels with the same `frameskip` used at training (default **5**).

---

## Method (3 steps)

### Step 1 — Cache activations

For each validation rollout:

1. Subsample frames: `0, frameskip, 2*frameskip, ...`
2. Load visual + proprio; read raw `state` for labels
3. Forward through the world model encoder
4. Save a feature vector per frame

**Default readout (`post_projector`):** mean-pool patch tokens after the channel projector → shape `(8,)` per frame. This is what planning mostly sees.

Other readouts:

| Name | Shape | Matches |
|---|---|---|
| `post_projector` | 8 | Pooled visual tokens after projector |
| `agg_mlp` | 128 | Straightening loss aggregation head |
| `flatten` | 1568 | All 196×8 patches flattened |

Hook sites (optional, saved in `activations.pt` under `hook_features`):

- `dino.block.{0,5,11}` — early / mid / late ViT blocks (frozen)
- `projector` — channel projector output
- `encoder` — full encoder output
- `predictor` — ViT predictor (when you run through full encode + predict)

### Step 2 — Train linear probes

- **Model:** Ridge regression (`sklearn`, α=1.0)
- **Features:** standardized across training frames
- **Split:** by **episode**, not by frame (80/20 default) — avoids leakage
- **Optional location holdout:** train in one maze half, test in the other
- **Metrics:** validation R² and RMSE per target; direction also reports angular MAE in degrees

A high val R² means the concept is **linearly decodable** at that readout. It does not by itself mean the model "uses" it for planning — that needs interventions (step 3).

### Location holdout (mentor check)

Episode split can still look good if the probe is secretly a location probe, because both train and val episodes visit similar maze regions. Location holdout asks:

> If I train the speed/direction probe only on the left (or bottom) half of the maze, does it still work on the right (or top) half?

For each of `x` and `y`, we run both directions (low→high and high→low), cut at the median coordinate by default.

**How to read it**

| Pattern | Interpretation |
|---|---|
| Episode-split speed R² high, location-holdout speed R² collapses | Likely region-specific position confounding |
| Holdout R² stays high **and** beats the `(x,y)→speed` baseline | Better evidence of motion information beyond location |
| Holdout R² stays high but matches the position baseline | Features may still be reading a global position→speed map |
| Position probes stay high under holdout | Expected if the representation encodes absolute location globally |

This still does **not** fully decorrelate position/speed/direction (that needs factorial data). It is the cheap Tier-1 confound check.

### Step 3 — Hook interventions (smoke test)

`ActivationHookManager` registers forward hooks and supports:

| Mode | Effect |
|---|---|
| **knockout** | Zero activations at hook site |
| **mean** | Replace with batch mean activation |

The smoke test runs knockout/mean on each hook site for 5 frames and reports how much probe predictions shift. Next step (not in this script yet): replay the same intervention during **planning** and measure Δsuccess.

---

## Quick start

```bash
# From repo root; needs DATASET_DIR or --data-path
export DATASET_DIR=/path/to/data

# Sanity-check labels (no checkpoint)
python probing/run_umaze_probes.py --labels-only

# Mentor location holdout on post_projector (single-frame)
# writes probing/speed_holdout/<condition>/
python probing/run_umaze_probes.py \
  --model-dir baseline_artifacts/checkpoints/umaze_physics_layer_ablations/r0_direction_only \
  --epoch 20 \
  --max-rollouts 80 \
  --location-holdout \
  --skip-interventions

# Daniel-style DINO mid-block feature diffs + same location holdout
# writes probing/dino5_diff_holdout/<condition>/
python probing/run_umaze_probes.py \
  --from-cache probing/speed_holdout/r0_direction_only/activations.pt \
  --probe-source dino.block.5 \
  --feature-mode diff \
  --location-holdout

# Roll up all conditions into one table
python probing/summarize_holdout.py --root probing
```

Omit `--output` to use the defaults above (derived from `--probe-source` / `--feature-mode` and the condition name). Pass `--output` only to override.

**Layout:**

```text
probing/
  speed_holdout/<condition>/          # readout=post_projector, feature_mode=raw
    activations.pt
    probe_results.json
    location_holdout.json
  dino5_diff_holdout/<condition>/     # probe_source=dino.block.5, feature_mode=diff
    probe_results.json
    location_holdout.json
  holdout_summary.json                # from summarize_holdout.py
```

**Per-run outputs:**

- `activations.pt` — features, labels, episode_ids, optional hook_features
- `probe_results.json` — per-target R² / RMSE (episode split)
- `location_holdout.json` — spatial generalization R² (if `--location-holdout`)
- `intervention_smoke.json` — probe prediction deltas under knockout/mean

---

## Comparing conditions (paper figure)

Run the same command on four checkpoints:

| Condition | Expected probe pattern |
|---|---|
| **R0** direction only | High direction R² at post_projector |
| **R1** speed only | High speed R², **low** direction R² → explains 16% planning |
| **R2** direction + speed | High on both |
| **Off** no straightening | Baseline decodability |

Main figure: heatmap of **val R²** × (position, speed, direction) × (readout or layer) × (condition).

---

## Code map

```
probing/
  labels.py           # state → position / speed / direction targets
  hooks.py            # ActivationHookManager + knockout/mean
  linear_probe.py     # Ridge probes + episode split
  run_umaze_probes.py # end-to-end CLI
```

Key repo dependencies:

- Checkpoint loading: `plan.load_model`
- Dataset: `datasets.point_maze_dset.PointMazeDataset`
- Encoder forward: `VWorldModel.encode_obs` → `visual` tokens `(B, T, 196, 8)`

---

## Pitfalls

1. **Use raw `states`, not normalized `proprio`** for labels.
2. **Split by episode**, not random frames. Use `--location-holdout` before trusting speed/direction R².
3. **Frozen DINO blocks** may show similar probe R² across conditions; focus on **projector** and **agg_mlp** where training actually writes.
4. Direction probes on stationary frames are noisy — low-speed frames are masked automatically.
5. High probe R² ≠ causal for planning — follow up with knockout during `plan.py`.

---

## Next steps

1. Run probes on R0 / R1 / R2 / off — plot heatmap
2. Wire knockout/mean into planning eval loop
3. Correlate post_projector direction R² with seed-level planning success
