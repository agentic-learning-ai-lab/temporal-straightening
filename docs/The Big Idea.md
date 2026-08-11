# The Big Idea

**Sources:** `README.md`, `assets/architecture.png`

## Planning inside your head

Imagine you want to push a cup across a table without touching it — you can only imagine outcomes. You picture the cup, mentally simulate "if I push here, it slides there," and pick the push that gets it where you want. That's a **world model**: a learned simulator that runs in your head instead of in reality.

This codebase does exactly that for robots:

1. **Encode** — a neural network (the encoder) compresses each camera image into a compact vector, a *latent state* `z`. Think of it as the model's mental snapshot of the world.
2. **Predict** — another network (the predictor) learns dynamics: "given the last few mental snapshots and an action, what's the next snapshot?"
3. **Plan** — to reach a goal image, encode it into `z_goal`, then search for the action sequence whose *imagined* rollout ends closest to `z_goal`. Crucially, the search is **gradient descent on the actions themselves** — the whole imagination is differentiable, so you can ask "which way should I nudge my actions to end up closer to the goal?"

The base system is [DINO-WM](https://github.com/gaoyuezhou/dino_wm): it uses a frozen DINOv2 vision transformer as the encoder (features pretrained on internet images, never fine-tuned) and only trains the predictor.

## The paper's twist: straighten the path

Here's the problem the paper attacks. Gradient-based planning treats *Euclidean distance in latent space* as "how far am I from the goal?" But that's only meaningful if latent space is well-behaved. If the latent trajectory of a real episode is a wildly curling scribble, then two points can be close in straight-line distance while being far apart along any *achievable* path — like two hairpin turns on a mountain road that are 50 m apart as the crow flies but 5 km apart by car.

The fix is inspired by neuroscience's **perceptual straightening hypothesis**: human visual cortex appears to transform curved pixel-space video trajectories into straighter neural trajectories, presumably because straight paths make prediction trivial (just keep going the same direction).

So the paper adds a **curvature penalty** during training: consecutive "velocity" vectors in latent space (`z_{t+1} − z_t` and `z_{t+2} − z_{t+1}`) should point the same way. If they do, the trajectory is locally a straight line. The claims:

- Straight latent trajectories make Euclidean distance a good proxy for *geodesic* (actually-traversable) distance.
- The planning objective becomes better-conditioned — gradient descent on actions stops getting stuck.
- Empirically: higher goal-reaching success rates.

The math and the config switches live in [The Straightening Loss](The%20Straightening%20Loss.md). Whether it actually works is examined in [The Reproduction Study](The%20Reproduction%20Study.md) — spoiler: in the small-scale reproduction, it didn't beat the un-straightened baseline, and there's a principled critique in [The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md).

## What `README.md` tells you

The README is the paper repo's front door: install steps (conda env + MuJoCo 210), where to get the DINO-WM datasets (OSF download, pointed to by `$DATASET_DIR`), the training command matrix (which `encoder=` + `training.straighten=` combos reproduce which paper variant), and the three planning configs. The teaser figure `assets/architecture.png` shows the encode → predict → plan architecture.

## Where to go next

- How the pieces fit: [Pipeline Overview](Pipeline%20Overview.md)
- The class that implements everything: [The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md)
- The citation: Wang, Bounou, Zhou, Balestriero, Rudner, LeCun, Ren — *Temporal Straightening for Latent Planning*, arXiv:2603.12231, 2026.
