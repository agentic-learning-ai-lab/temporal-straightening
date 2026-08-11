"""Generate a PointMaze-Medium dataset in the same format as data/point_maze (UMaze).

Format (verified against the UMaze dataset):
  states.pth       torch.float64  (N, T, 4)   [qpos_x, qpos_y, qvel_x, qvel_y]
  actions.pth      torch.float64  (N, T, 2)   waypoint-controller output in [-1, 1]
  seq_lengths.pth  torch.int64    (N,)        all T
  obses/episode_XXX.pth  torch.uint8  (T, 224, 224, 3) rendered frames

Alignment: actions[t] is the action computed at states[t]; only the first T-1
actions are executed (the state resulting from actions[T-1] is not stored).
Verified by replaying UMaze actions through the env with zero state error.

Run inside the ts310 conda env with mujoco210 on LD_LIBRARY_PATH (see setup.sh).
Rendering is headless via mujoco_py's EGL backend (needs libEGL_nvidia /
libnvidia-gl matching the driver on a headless box).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from env.pointmaze.maze_model import MEDIUM_MAZE
from env.pointmaze.point_maze_wrapper import PointMazeWrapper
from env.pointmaze.waypoint_controller import WaypointController


def generate_episode(env, seed, episode_length, policy="waypoint"):
    """Roll out one episode. Returns (obses, states, actions, info).

    policy='waypoint': goal-directed waypoint controller.
    policy='random': i.i.d. uniform actions in [-1, 1] — this is what the
    original UMaze dataset used (its action histogram is exactly flat with
    zero lag-1 autocorrelation).
    """
    # The waypoint controller jitters waypoints via the *global* np.random;
    # seed it so every episode is fully reproducible (needed for clean resume).
    np.random.seed(seed)
    init_state, goal_state = env.sample_random_init_goal_states(seed)
    # Fresh controller per episode: it only replans when the *target* changes,
    # so reusing one across episodes with the same goal would keep stale waypoints.
    controller = WaypointController(MEDIUM_MAZE)
    action_rng = np.random.RandomState(seed)

    obs, state = env.prepare(seed, init_state)

    obses, states, actions = [], [], []
    for t in range(episode_length):
        obses.append(obs["visual"])
        states.append(state)
        if policy == "random":
            action = action_rng.uniform(-1.0, 1.0, size=2)
        else:
            action, solved = controller.get_action(state[:2], state[2:4], goal_state[:2])
        actions.append(action)
        if t < episode_length - 1:
            obs, _reward, _done, step_info = env.step(action)
            state = step_info["state"]

    info = {
        "init": init_state[:2],
        "goal": goal_state[:2],
        "final": states[-1][:2],
        "final_dist": float(np.linalg.norm(states[-1][:2] - goal_state[:2])),
    }
    return (
        torch.from_numpy(np.stack(obses)),            # (T, 224, 224, 3) uint8
        torch.from_numpy(np.stack(states).astype(np.float64)),
        torch.from_numpy(np.stack(actions).astype(np.float64)),
        info,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--episode_length", type=int, default=100)
    parser.add_argument(
        "--output_dir", type=str,
        default="/lambda/nfs/temporal-straightening/data/point_maze_medium_test",
    )
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--policy", choices=["waypoint", "random"], default="waypoint",
                        help="'random' matches the original UMaze dataset's action policy")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "obses").mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pth"

    # return_value defaults to 'state' so gym can build the observation space;
    # env.prepare() switches it to 'obs' (rendered frames) via prepare_for_render().
    env = PointMazeWrapper(maze_spec=MEDIUM_MAZE, reward_type="sparse", reset_target=False)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path)
        all_states, all_actions, seed = ckpt["states"], ckpt["actions"], ckpt["next_seed"]
        episode_idx = len(all_states)
        print(f"resuming from episode {episode_idx} (seed {seed})", flush=True)
    else:
        all_states, all_actions = [], []
        episode_idx = 0
        seed = args.seed_offset

    def save_checkpoint():
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"states": all_states, "actions": all_actions, "next_seed": seed}, tmp)
        tmp.rename(ckpt_path)

    n_success = 0
    dist_sum = 0.0
    t0 = time.time()
    start_idx = episode_idx
    while episode_idx < args.n_episodes:
        try:
            obses, states, actions, info = generate_episode(env, seed, args.episode_length, args.policy)
        except (AssertionError, IndexError) as e:
            # Rare waypoint-controller edge case (grid rollout not converging);
            # skip this seed and draw a new episode.
            print(f"seed {seed}: controller failed ({e!r}), resampling", flush=True)
            seed += 1
            continue

        torch.save(obses, out / "obses" / f"episode_{episode_idx:03d}.pth")
        all_states.append(states)
        all_actions.append(actions)
        n_success += info["final_dist"] < 0.5
        dist_sum += info["final_dist"]
        episode_idx += 1
        seed += 1

        if episode_idx % args.checkpoint_every == 0 or episode_idx == args.n_episodes:
            save_checkpoint()
            done_this_run = episode_idx - start_idx
            rate = done_this_run / (time.time() - t0)
            eta = (args.n_episodes - episode_idx) / rate
            print(
                f"[{episode_idx}/{args.n_episodes}] {rate:.1f} eps/s, "
                f"ETA {eta / 60:.1f} min | last {done_this_run} eps: "
                f"success {n_success / done_this_run:.1%}, "
                f"mean final dist {dist_sum / done_this_run:.3f}",
                flush=True,
            )

    torch.save(torch.stack(all_states), out / "states.pth")
    torch.save(torch.stack(all_actions), out / "actions.pth")
    torch.save(
        torch.full((args.n_episodes,), args.episode_length, dtype=torch.int64),
        out / "seq_lengths.pth",
    )
    ckpt_path.unlink(missing_ok=True)
    print(f"saved {args.n_episodes} episodes to {out}", flush=True)


if __name__ == "__main__":
    main()
