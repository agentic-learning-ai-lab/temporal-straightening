# The Straightening Loss

**Source:** `models/visual_world_model.py` (`total_curvature`, `_cos_curvature`, `total_speed_constancy`, and the string parser in `VWorldModel.__init__`)

## The intuition

Take a training video and encode every frame into a latent vector. You now have a dotted path through latent space — one dot per frame. Between consecutive dots there's a "velocity" vector:

```
v_t = z_{t+1} − z_t        (the step the trajectory just took)
```

If the path is a straight line, every step points the same direction as the last one. So "how bent is this path?" reduces to "how misaligned are consecutive steps?" — which cosine similarity measures perfectly:

```
curvature loss = 1 − cos(v_t, v_{t+1})
```

- Steps perfectly aligned → cos = 1 → loss 0. Straight.
- Steps at right angles → cos = 0 → loss 1. Bent.
- Path doubles back → cos = −1 → loss 2. Hairpin.

Averaged over all consecutive step pairs in a batch, this is the entire straightening idea. Adding it (scaled by λ) to the training loss pressures the *encoder's projector* to lay trajectories out straight. Tiny steps (norm below `1e-6`) are masked out so noise-direction on near-stationary frames doesn't dominate (`_cos_curvature`).

## The string DSL: `training.straighten`

Which loss runs is controlled by a single config string parsed in `VWorldModel.__init__`. Tokens are joined with `+`, and each token is a mode name followed by a scale λ:

| Token | Loss | Applied to |
|---|---|---|
| `cos1e-1` | curvature | every patch token independently (196 mini-trajectories per video) |
| `aggcos1e-1` | curvature | one pooled vector per frame (via `encoder.agg()` — mean, flatten, or MLP pooling) |
| `speed1e-1` | speed constancy | per-patch |
| `aggspeed1e-1` | speed constancy | pooled |
| `r0_1e-1` / `aggr0_1e-1` | R0 direction penalty | per-patch / pooled |
| `r1_1e-1` / `aggr1_1e-1` | R1 adjacent speed-ratio penalty | per-patch / pooled |
| `r2_5e-2` / `aggr2_5e-2` | R2 direction + speed penalty | per-patch / pooled |
| `r3b1_1e-1` / `aggr3b1_1e-1` | R3 blend with beta = 1 | per-patch / pooled |
| `r4_1e-1` / `aggr4_1e-1` | R4 unnormalized acceleration | per-patch / pooled |
| `False` | nothing | — |
| `cos1e-1+aggspeed1e-1` | both, added together | mix and match |

So the paper's headline setting is `cos1e-1`: patch-wise curvature with λ = 0.1. The `speed`/`aggspeed` and R0--R4 tokens are **not from the paper** — they were added by the reproduction study (see [The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md)).

**Patch vs pooled, intuitively:** DINO's patch features give you a 14×14 grid of vectors per image — one per image region. `cos` says "each region's feature should move in a straight line over time." `aggcos` first squashes the grid into a single per-frame vector and straightens *that* — the trajectory the planner actually cares about when features get pooled.

## The speed-constancy loss (the reproduction's addition)

`total_speed_constancy()` penalizes variation in step *length* rather than step *direction*:

```
speed_t = ‖z_{t+1} − z_t‖
loss    = mean over t of ((speed_t − mean_speed) / mean_speed)²
```

i.e., the squared *relative* deviation of each step's length from the trajectory's average step length. A path that alternates lunge–creep–lunge gets punished even if it's ruler-straight. Why this matters is the whole story of [The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md).

## The R0--R4 trajectory penalties

The Wall ablation uses adjacent velocities `v1 = z[t+1] - z[t]` and
`v2 = z[t+2] - z[t+1]`. Let `c` be their cosine similarity and let
`r = ||v2|| / ||v1||` be the adjacent speed ratio:

```
R0 = 1 - c
R1 = r + 1/r - 2
R2 = ||v2 - v1||² / (||v1|| ||v2||) = R1 + 2 R0
R3 = R0 + beta R1
R4 = ||v2 - v1||²
```

R0 measures direction only. R1 measures pairwise speed change and is symmetric
under swapping `v1` and `v2`. R2 is the normalized full penalty. R3 exposes an
explicit blend coefficient in its token (`r3bBETA_SCALE`). R0--R3 are invariant
to a global rescaling of the latent space; R4 deliberately is not. Pairs where
either step has norm at most `1e-6` are excluded from the average.

The matched Wall comparison uses pooled features and these settings:

- R0 baseline: `aggcos1e-1` (equivalent to `aggr0_1e-1`)
- R1 speed-only: `aggr1_1e-1`
- R2 full penalty: `aggr2_5e-2`; because `R2 = R1 + 2 R0`, its effective R0
  coefficient is 0.1
- R3 direction + speed: `aggr3b1_1e-1`

The older `speed`/`aggspeed` loss compares every speed with the trajectory-wide
mean. R1 is a different, local penalty comparing each adjacent speed pair.

## Where it plugs in

In `VWorldModel.forward()`, after the main prediction loss is computed, both regularizers operate on the **visual part of the encoded latents only** (`visual_only(z)` strips off the action/proprio dimensions that get concatenated in — see [The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md)):

```
loss = prediction_loss
     + λ_cos   · curvature_loss        (if straighten)
     + λ_speed · speed_constancy_loss  (if speed_constancy)
     + λ_R     · trajectory_penalty    (if an R0--R4 token is configured)
     + VICReg terms                    (if vcreg, off by default)
```

One subtlety worth internalizing: with the default frozen-DINO encoder (`encoder=dino`, no projector), **there are no trainable parameters between pixels and the measured latents** — the curvature loss can push gradients into the predictor's inputs but cannot actually reshape the latent geometry. This is exactly the trap the reproduction's first ablation fell into, and why the adapter ablation (`encoder=dino_channel`, which has a trainable projector) was the meaningful test. Details in [The Reproduction Study](The%20Reproduction%20Study.md).
