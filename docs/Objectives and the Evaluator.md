# Objectives and the Evaluator

**Sources:** `planning/objectives.py`, `planning/evaluator.py`

Two halves of "how do we know a plan is good?": the **objective** scores plans *inside the dream* (differentiably, for the optimizer), the **evaluator** scores them *in reality* (the number that goes in the paper).

## Objectives (`objectives.py`)

`create_objective_fn(alpha, base, mode)` returns a loss over imagined latent trajectories vs the goal latent. All variants are MSE in latent space, with `alpha` weighting the proprioceptive part against the visual part:

```
loss = MSE(z_visual, z_goal_visual) + alpha · MSE(z_proprio, z_goal_proprio)
```

Three modes:

- **`last`** (default) — only the final imagined frame must match the goal. The purest "get there" objective; the path taken is unconstrained.
- **`all`** — every imagined frame is compared to the goal, weighted by `base^t` (normalized). With base > 1 later frames dominate; it softly encourages making progress *throughout* the trajectory, giving gradient signal even when the endpoint is hopelessly far.
- **`staged`** — curriculum: first optimize only the endpoint (`last`), then switch to the full-horizon objective (`all`) in later optimization steps. The README prescribes this for PushT with GD-MPC — contact-rich pushing has nastier optimization landscapes.

Note the loop closure: this objective is *Euclidean distance in latent space* — precisely the quantity [The Straightening Loss](The%20Straightening%20Loss.md) is supposed to make meaningful.

## PlanEvaluator (`evaluator.py`)

The reality check. Takes the planner's current actions and:

1. Executes them in the **real** parallel simulators (`SubprocVectorEnv`, see [Supporting Code](Supporting%20Code.md)) — denormalizing actions and expanding frameskip back to raw env steps.
2. Asks each environment's own `eval_state(goal_state, cur_state)` whether the final true state is within the success threshold (each env defines its own — see [Environments](Environments.md)).
3. Produces the metrics that appear in every results table of [The Reproduction Study](The%20Reproduction%20Study.md):
   - **success rate** — fraction of the n_evals rollouts that reached the goal,
   - **state distance** — mean true-state gap to the goal (success is binary; this shows *how* close failures got),
   - **visual/proprio distances and embedding divergences** — pixel-space and latent-space gaps, useful for diagnosing *where* things went wrong.
4. Optionally renders side-by-side rollout images/GIFs — the ancestors of the contact sheets in [Evidence Packs](Evidence%20Packs.md).

The separation matters: a planner can only see latents, so it can be *confidently wrong* — imagined success, real failure — whenever the predictor's dream drifts from physics. Comparing objective loss (dream) against evaluator success (reality) is how you detect that drift.
