# The Planners — GD, CEM, MPC

**Sources:** `planning/base_planner.py`, `gd.py`, `cem.py`, `mpc.py`

All planners answer the same question — *which action sequence makes the imagined future end at the goal?* — and share the `BasePlanner` interface: they hold the world model, an objective function ([Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md)), a preprocessor, and an evaluator, and implement `plan(obs_0, obs_g)`.

## GDPlanner (`gd.py`) — the star of the show

Treats the action sequence itself as the thing to optimize, exactly like training a network — except the "weights" are the actions:

```
actions = tensor(n_evals, horizon, action_dim, requires_grad=True)
for step in range(opt_steps):
    imagined = wm.rollout(obs_0, actions)         # dream
    loss     = distance(imagined_end, z_goal)     # score
    loss.backward(); optimizer.step()             # nudge actions downhill
    actions += gaussian_noise                     # jiggle out of local minima
```

Concretely: Adam (or SGD/AdamW) on the actions, optional cosine LR schedule, Gaussian exploration noise after each step, early exit when the evaluator reports all rollouts succeeded. Initialization is random-normal or normalized zero actions (`init_actions`).

**Why this planner is the paper's protagonist:** its entire signal is `∂(latent distance)/∂(actions)`, flowing backward through the dreamed trajectory. If latent space is a crumpled scribble, that gradient is noisy and ill-conditioned; if trajectories are straight and evenly paced, following the gradient is like walking downhill on a smooth ramp. GD planning success is therefore the paper's chosen behavioral measure of representation quality — the whole straightening bet ([The Big Idea](The%20Big%20Idea.md)) is that better latent geometry shows up *here*.

## CEMPlanner (`cem.py`) — the gradient-free control

The Cross-Entropy Method, a "guess-and-check with statistics" loop:

1. Sample many action sequences from a Gaussian (mean μ, std σ).
2. Dream them all in parallel, score each.
3. Keep the top-k elites; refit μ, σ to them.
4. Repeat — the distribution tightens around what works.

No gradients touch the latent space, so CEM cares much less about geometric conditioning. That contrast is the point: if straightening helps GD but not CEM, the effect is genuinely about gradient conditioning, not general representation quality.

## MPCPlanner (`mpc.py`) — closed-loop wrapper

Model Predictive Control: plan with a sub-planner (GD or CEM), **execute only the first `n_taken_actions` chunk in the real environment**, observe where you actually ended up, and replan from there. Like driving with GPS rerouting instead of memorizing all directions before leaving. It tolerates imagination error far better than open-loop planning, at the cost of many more planning rounds. Config: `plan_gd_mpc.yaml` (see [Planning - plan.py](Planning%20-%20plan.py.md)).

## Config mapping

`conf/planner/gd.yaml`, `cem.yaml`, `mpc_gd.yaml`, `mpc_cem.yaml` hold each planner's knobs (horizon, opt_steps/iterations, samples, top-k, lr, noise). The planner is instantiated by Hydra from the chosen `plan_*.yaml` ([Hydra Configs](Hydra%20Configs.md)).
