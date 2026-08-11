# Environments

**Source:** `env/` — one subfolder per world, plus vector-env infrastructure (covered in [Supporting Code](Supporting%20Code.md)).

Four simulated worlds, deliberately spanning easy → hard. Each provides gym-style step/reset, offscreen camera rendering (the model only ever sees pixels), a ground-truth `eval_state` for judging planner success, and a wrapper exposing the uniform interface [Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md) expects.

## PointMaze (`env/pointmaze/`) — the workhorse

A ball rolling through a 2-D maze (MuJoCo physics, from D4RL lineage). Two layouts: **UMaze** (a simple U) and **Medium** (a proper multi-corridor maze — the reproduction's main testbed, config `point_maze_medium.yaml`).

- `maze_model.py` — builds MuJoCo XML from ASCII maze maps (`MEDIUM_MAZE` is literally a string of `#` and space characters); `dynamic_mjc.py` helps generate the XML.
- `point_maze_wrapper.py` — the standard wrapper: rendering, state get/set, `eval_state`.
- `waypoint_controller.py` + `q_iteration.py` — a scripted expert: value-iteration over the maze grid produces waypoints, a PD controller chases them. Used by [Dataset Generators](Dataset%20Generators.md) to produce sensible (non-random-walk) trajectories.
- `gridcraft/` — a small vendored grid-world library the value iteration runs on.

Why mazes are the perfect testbed for straightening: pixel/latent distance and *task* distance genuinely disagree here — two points on opposite sides of a wall are visually near but behaviorally far. Exactly the geodesic-vs-Euclidean gap [The Big Idea](The%20Big%20Idea.md) cares about.

## PushT (`env/pusht/`)

A 2-D fingertip must push a T-shaped block onto a target outline (pymunk physics, from the Diffusion Policy benchmark). Contact-rich: you can't pull, only push, so plans must approach from the correct side. Hardest optimization landscape of the four — hence the README's special objective settings (`alpha=1`, staged mode; see [Objectives and the Evaluator](Objectives%20and%20the%20Evaluator.md)).

## Wall (`env/wall/`)

A minimalist 2-D world: an agent, a wall with a doorway, goals on the other side. Almost a unit test for planning — the *only* obstacle is the wall, so success hinges on whether latent space understands "you must route through the door." Contains its own `envs/` + `data/` mini-package and `wall_env_wrapper.py`. Note from [The Reproduction Study](The%20Reproduction%20Study.md): both straightened and un-straightened variants scored only 0.02 success here, suggesting this setup (50 episodes, 20 epochs) was too small for *anything* to plan well on Wall.

## Deformables (`env/deformable_env/`)

Rope and granular-material manipulation with a simulated xArm robot (NVIDIA FleX particle physics). The bulkiest folder by disk size — `src/sim/assets/` is all URDF robot descriptions and meshes (vendored, skip). `FlexEnvWrapper.py` adapts it to the common interface; `src/sim/sim_env/` holds scenes/cameras/robot glue. Requires special hardware/deps; the reproduction didn't touch it, but the paper's rope/granular results come from here (configs `rope.yaml`, `granular.yaml`).

## The shared contract

Every wrapper exposes: `reset` / `step`, `render` (offscreen RGB), `update_env`/state-set (so the evaluator can teleport to arbitrary start states), `sample_random_init_goal_states`, and `eval_state(goal, cur)` → success bool + distance. That contract is what lets one `plan.py` serve four totally different physics engines.
