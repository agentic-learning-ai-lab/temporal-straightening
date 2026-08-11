# Planning — plan.py

**Source:** `plan.py` (534 lines). Entry point: `python plan.py --config-name plan_gd.yaml ckpt_base_path=<...> model_name=<...>`

Takes a trained checkpoint and answers the only question that matters: **can this world model actually get a robot to a goal?**

## The flow (`planning_main` → `PlanWorkspace`)

1. **Load the checkpoint** (`load_ckpt`/`load_model`): restores all five sub-networks (`ALL_MODEL_KEYS`: encoder, predictor, decoder, proprio_encoder, action_encoder) *and the training config that was saved with them* — so planning automatically knows the frameskip, history length, and normalization stats the model was trained with.
2. **Frameskip arithmetic**: since the model thinks in 5-step strides, planning horizons given in env steps get divided by frameskip, and each "model action" is actually 5 concatenated env actions (this is why the README warns to keep horizons divisible by frameskip).
3. **Prepare goal tasks (`prepare_targets`)**: builds `n_evals` (start, goal) pairs. `goal_source` picks how:
   - `dset` — take a real trajectory segment from the dataset; use its first frame as start, last as goal (guaranteed reachable in `goal_H` steps),
   - `random_state` / `random_action` — sample fresh states or roll random actions in the env,
   - or reload previously dumped targets (`prepare_targets_from_file` / `dump_targets` → `plan_targets.pkl`) so different models face *identical* tasks — essential for fair comparison, and the reason that pickle keeps appearing in [Evidence Packs](Evidence%20Packs.md).
4. **Spin up real envs**: a `SubprocVectorEnv` ([Supporting Code](Supporting%20Code.md)) runs `n_evals` simulator copies in parallel for ground-truth evaluation.
5. **Plan (`perform_planning`)**: hand everything to the configured planner ([The Planners - GD, CEM, MPC](The%20Planners%20-%20GD,%20CEM,%20MPC.md)) with the objective from [Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md); the planner iterates imagination + optimization, periodically asking the evaluator to run current actions in the real envs.
6. **Report**: success rate and distance metrics land in `logs.json` / wandb (or a `DummyWandbRun` no-op when logging is off).

## Cluster mode

`build_plan_cfg_dicts` + `launch_plan_jobs` build a grid of planning configs (planner × goal_source × goal_H × alpha) and submit each as a SLURM job via submitit — this is what [Training - train.py](Training%20-%20train.py.md) calls for mid-training evals. Local runs just call `planning_main` directly.

## The three planning configs

| Config | Planner | Character |
|---|---|---|
| `plan_gd.yaml` | gradient descent, open loop | plan once, execute blindly — the paper's main setting, most sensitive to latent geometry |
| `plan_cem.yaml` | CEM, open loop | gradient-free sampling baseline |
| `plan_gd_mpc.yaml` | MPC wrapping GD | replan after each executed chunk — closed loop |

"Open loop" = commit to the entire action sequence before touching the real world; any imagination error compounds uncorrected. That makes open-loop GD the sharpest test of whether straightened latents genuinely help — which is exactly why the reproduction used it ([The Reproduction Study](The%20Reproduction%20Study.md)).
