# The Speed-Constancy Critique

**Sources:** `report.md` (Methodology), `research_note.tex` (Motivation)

## The gap between the theory and the loss

The paper's theoretical story goes: *if latent trajectories are straight, then Euclidean distance ≈ geodesic distance, and gradient planning is well-conditioned.* But hiding inside that story is an assumption of **constant latent speed** — the trajectory should move through latent space in even steps.

Now look at what the cosine loss actually does (see [The Straightening Loss](The%20Straightening%20Loss.md)):

```
1 − cos(z_{t+1} − z_t,  z_{t+2} − z_{t+1})
```

Cosine similarity **normalizes both vectors**. It only compares directions. The magnitudes — how *far* each step travels — are divided away entirely.

## Why that matters: the highway analogy

Picture two drives down a perfectly straight highway:

- **Drive A:** steady 100 km/h. Position over time: evenly spaced dots.
- **Drive B:** floor it for a minute, park for a minute, repeat. Same straight road, but dots come in clusters with big gaps.

Both drives have **zero cosine curvature** — every step points due north. But for a planner using "distance between dots" as its cost signal, Drive B is a nightmare: equal time-steps correspond to wildly unequal latent distances, so the gradient of "distance to goal" with respect to "action at time t" is huge at some steps and vanishing at others. The nice conditioning the theory promises quietly assumed Drive A.

So the critique is precise: **the cosine loss can produce trajectories that are directionally straight yet lurch between fast and slow, and nothing in the training objective prevents it.** Worse, the paper's theory *needs* the constant-speed property that its loss doesn't enforce.

## The proposed patch

The reproduction study added a complementary penalty — speed constancy — that punishes variation in step lengths (relative to the trajectory's mean speed, so slow and fast trajectories are treated fairly). Combined:

- `cos` handles **direction** (straightness),
- `speed` handles **magnitude** (even pacing),
- together they should approximate the "constant-velocity straight line" the theory actually assumes.

## Did it hold up?

Partly. The adapter ablation in [The Reproduction Study](The%20Reproduction%20Study.md) confirmed the *diagnosis* cleanly:

- Cosine-only training made paths straighter **but made speed variation worse** (speed CV rose from 0.46 to 0.58 — the loss was trading pacing away for direction).
- Speed-only training evened out pacing but left paths just as bent.
- Cosine + speed achieved the best geometry on both axes.

But the *treatment* didn't pay off where it counts: the best-geometry model did **not** get the best planner success rate. In these small runs (50 episodes, 20 epochs, one seed), latent geometry and planning success simply didn't move together. The honest conclusion, quoted from the research note: "more seeds and larger datasets would be needed before claiming the speed penalty is a robust improvement."

## Diagnostics vocabulary

The evidence packs measure geometry with four numbers (computed in `_latent_metrics_from_z` in [Modal Runner](Modal%20Runner.md)):

| Metric | Question it answers |
|---|---|
| **Curvature** | Average `1 − cos` between consecutive steps — how bent? |
| **Path/endpoint ratio** | Total path length ÷ straight-line start-to-end distance — how much wandering? (1.0 = perfectly direct) |
| **Speed CV** | Std ÷ mean of step lengths — how uneven is the pacing? |
| **Speed jump** | Average relative change between consecutive step lengths — how jerky? |
