# Pipeline Overview

The whole system, end to end. Each stage has its own note.

```
 ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
 │ Simulators   │    │ Recorded     │    │ Training     │    │ Planning    │
 │ (env/)       │───▶│ trajectories │───▶│ (train.py)   │───▶│ (plan.py)   │
 │ maze, pusht, │    │ (datasets/)  │    │ learn encoder│    │ gradient-   │
 │ wall, rope   │    │ .pth files   │    │ + predictor  │    │ descend the │
 └─────────────┘    └──────────────┘    └──────────────┘    │ actions     │
                                                             └─────────────┘
```

## Stage 1 — Worlds and data

Four simulated environments ([Environments](Environments.md)) produce episodes: sequences of camera images, low-dimensional states (e.g. the agent's true x,y — used only for evaluation, never shown to the model), proprioception, and actions. The paper uses pre-recorded DINO-WM datasets; the reproduction generated its own with [Dataset Generators](Dataset%20Generators.md).

Loaders in `datasets/` ([Datasets and Data Loading](Datasets%20and%20Data%20Loading.md)) slice long episodes into short training windows: `num_hist = 3` context frames plus `num_pred = 1` future frame, sampled every `frameskip = 5` sim steps. Frameskip matters: one "model step" equals five real steps, so latent velocities are computed over meaningful motion, not sub-pixel jitter.

## Stage 2 — Training ([Training - train.py](Training%20-%20train.py.md))

Every batch flows through [The World Model - VWorldModel](The%20World%20Model%20-%20VWorldModel.md):

1. **Encode** each frame with the vision encoder ([Encoders - DINO and Friends](Encoders%20-%20DINO%20and%20Friends.md)) → latent tokens; embed proprio and actions and concatenate them onto the latents.
2. **Predict**: the ViT predictor ([Predictor and Decoders](Predictor%20and%20Decoders.md)) reads the 3 history frames' tokens and predicts the next frame's tokens.
3. **Loss** = MSE(predicted tokens, actual next tokens) — with a stop-gradient on the target to prevent the encoder from collapsing everything to a constant (a constant would give zero prediction error!) — plus optionally [The Straightening Loss](The%20Straightening%20Loss.md).
4. A separate **decoder** is trained (on detached latents, so it can't influence them) purely so humans can look at reconstructed/predicted images and check the latents contain real information.

## Stage 3 — Planning ([Planning - plan.py](Planning%20-%20plan.py.md))

Given a start image and a goal image:

1. Encode both. The goal becomes a fixed target `z_goal`.
2. Initialize a sequence of candidate actions (random or zeros).
3. **Imagine**: `VWorldModel.rollout()` unrolls the predictor from the start latent using the candidate actions — no simulator involved.
4. **Score**: distance between the imagined final latent and `z_goal` ([Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md)).
5. **Improve**: because everything is differentiable, backprop reaches the *actions themselves*; an Adam optimizer nudges them downhill ([The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md)). Alternatives: CEM (sampling-based, gradient-free) and MPC (replan as you go).
6. **Verify**: the evaluator executes the final actions in the *real* simulator and checks whether the true state got within the success threshold of the goal.

## The experiment, in one sentence

The paper claims step 5 works better when training adds a straightness penalty to the latents; the reproduction ([The Reproduction Study](The%20Reproduction%20Study.md)) tested that claim and its blind spot ([The Speed-Constancy Critique](The%20Speed-Constancy%20Critique.md)) across ~14 trained variants, packaging all results in [Evidence Packs](Evidence%20Packs.md).
