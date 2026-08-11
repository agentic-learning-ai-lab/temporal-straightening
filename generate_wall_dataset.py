"""Generate a wall_single dataset compatible with conf/env/wall.yaml."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from env.wall.wall_env_wrapper import WallEnvWrapper


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _visual_to_uint8_hwc(visual):
    arr = _to_numpy(visual)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0, 255).astype(np.uint8)


def _canonical_visual(env: WallEnvWrapper, visual):
    """Return visual observations as fixed-size HWC tensors."""
    if not isinstance(visual, torch.Tensor):
        visual = torch.as_tensor(visual)

    if visual.ndim != 3:
        raise ValueError(f"expected 3D visual observation, got shape {tuple(visual.shape)}")

    if visual.shape[0] == 3:
        return env.transform(visual).permute(1, 2, 0)

    if visual.shape[-1] == 3:
        if visual.shape[0] == env.transform.size[0] and visual.shape[1] == env.transform.size[1]:
            return visual
        return env.transform(visual.permute(2, 0, 1)).permute(1, 2, 0)

    raise ValueError(f"unsupported visual observation shape {tuple(visual.shape)}")


def _controller_action(state, goal, wall_x, door_y, wall_width):
    state_xy = np.asarray(state[:2], dtype=np.float32)
    goal_xy = np.asarray(goal[:2], dtype=np.float32)
    half_width = wall_width // 2
    margin = 3.0

    left_to_right = state_xy[0] < wall_x
    if left_to_right:
        pre_door = np.array([wall_x - half_width - margin, door_y], dtype=np.float32)
        post_door = np.array([wall_x + half_width + margin, door_y], dtype=np.float32)
    else:
        pre_door = np.array([wall_x + half_width + margin, door_y], dtype=np.float32)
        post_door = np.array([wall_x - half_width - margin, door_y], dtype=np.float32)

    if np.linalg.norm(goal_xy - post_door) < np.linalg.norm(goal_xy - pre_door):
        waypoints = [goal_xy]
    else:
        waypoints = [pre_door, post_door, goal_xy]

    target = goal_xy
    for waypoint in waypoints:
        if np.linalg.norm(waypoint - state_xy) > 1.5:
            target = waypoint
            break

    delta = target - state_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return np.zeros(2, dtype=np.float32)

    # DotWall applies location += action * 2, so unit-norm action moves 2 px.
    action = delta / 2.0
    norm = float(np.linalg.norm(action))
    if norm > 1.0:
        action = action / norm
    return action.astype(np.float32)


def generate_episode(env: WallEnvWrapper, seed: int, episode_length: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    init_state, goal_state = env.sample_random_init_goal_states(seed)
    obs, state = env.prepare(seed, init_state)

    obses = []
    states = []
    actions = []
    wall_locations = []
    door_locations = []

    for t in range(episode_length):
        state_np = _to_numpy(state).astype(np.float32)
        obses.append(_visual_to_uint8_hwc(_canonical_visual(env, obs["visual"])))
        states.append(state_np)
        wall_locations.append(float(env.wall_x.detach().cpu().item()))
        door_locations.append(float(env.hole_y.detach().cpu().item()))

        action = _controller_action(
            state_np,
            goal_state,
            wall_x=wall_locations[-1],
            door_y=door_locations[-1],
            wall_width=env.wall_config.wall_width,
        )
        actions.append(action)
        if t < episode_length - 1:
            obs, _reward, _done, info = env.step(torch.as_tensor(action))
            state = info["state"]

    final_dist = float(np.linalg.norm(states[-1][:2] - np.asarray(goal_state[:2])))
    return (
        torch.from_numpy(np.stack(obses)),
        torch.from_numpy(np.stack(states).astype(np.float32)),
        torch.from_numpy(np.stack(actions).astype(np.float32)),
        torch.tensor(door_locations, dtype=torch.float32),
        torch.tensor(wall_locations, dtype=torch.float32),
        {"init": np.asarray(init_state).tolist(), "goal": np.asarray(goal_state).tolist(), "final_dist": final_dist},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--episode_length", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/wall_single")
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "obses").mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pth"

    env = WallEnvWrapper(device="cpu")

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path)
        all_states = ckpt["states"]
        all_actions = ckpt["actions"]
        all_doors = ckpt["door_locations"]
        all_walls = ckpt["wall_locations"]
        metadata = ckpt["metadata"]
        episode_idx = len(all_states)
        seed = ckpt["next_seed"]
        print(f"resuming from episode {episode_idx} (seed {seed})", flush=True)
    else:
        all_states, all_actions, all_doors, all_walls = [], [], [], []
        metadata = []
        episode_idx = 0
        seed = args.seed_offset

    def save_checkpoint():
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                "states": all_states,
                "actions": all_actions,
                "door_locations": all_doors,
                "wall_locations": all_walls,
                "metadata": metadata,
                "next_seed": seed,
            },
            tmp,
        )
        tmp.rename(ckpt_path)

    t0 = time.time()
    start_idx = episode_idx
    while episode_idx < args.n_episodes:
        obses, states, actions, doors, walls, info = generate_episode(
            env, seed, args.episode_length
        )
        torch.save(obses, out / "obses" / f"episode_{episode_idx:03d}.pth")
        all_states.append(states)
        all_actions.append(actions)
        all_doors.append(doors)
        all_walls.append(walls)
        metadata.append(info)
        episode_idx += 1
        seed += 1

        if episode_idx % args.checkpoint_every == 0 or episode_idx == args.n_episodes:
            save_checkpoint()
            done_this_run = episode_idx - start_idx
            rate = done_this_run / max(time.time() - t0, 1e-6)
            eta = (args.n_episodes - episode_idx) / max(rate, 1e-6)
            mean_final_dist = np.mean([item["final_dist"] for item in metadata[-done_this_run:]])
            print(
                f"[{episode_idx}/{args.n_episodes}] {rate:.1f} eps/s, "
                f"ETA {eta / 60:.1f} min | mean final dist {mean_final_dist:.3f}",
                flush=True,
            )

    torch.save(torch.stack(all_states), out / "states.pth")
    torch.save(torch.stack(all_actions), out / "actions.pth")
    torch.save(torch.stack(all_doors), out / "door_locations.pth")
    torch.save(torch.stack(all_walls), out / "wall_locations.pth")
    torch.save(
        torch.full((args.n_episodes,), args.episode_length, dtype=torch.int64),
        out / "seq_lengths.pth",
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ckpt_path.unlink(missing_ok=True)
    print(f"saved {args.n_episodes} episodes to {out}", flush=True)


if __name__ == "__main__":
    main()
