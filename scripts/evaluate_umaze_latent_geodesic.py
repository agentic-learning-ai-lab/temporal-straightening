#!/usr/bin/env python3
"""Compare UMaze latent distance with true shortest-path distance.

This is the missing "middle-link" test for the trajectory-penalty study:

    trajectory regularity -> honest latent distances -> better planning

The script renders a dense grid of valid UMaze states, computes the visual
latent distance from every state to the canonical goal, and compares that
distance with A* step-count using Spearman correlation.  It evaluates two
checkpoints on exactly the same rendered observations and writes paired
artifacts (CSV, JSON, Markdown, and plots).

Example:
    python scripts/evaluate_umaze_latent_geodesic.py \
      --r0-checkpoint /path/to/r0_direction_only \
      --r2-checkpoint /path/to/r2_full_matched \
      --output-dir baseline_artifacts/analysis/umaze_latent_geodesic_r0_vs_r2
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr


GridNode = Tuple[int, int]


@dataclass(frozen=True)
class GridState:
    ix: int
    iy: int
    x: float
    y: float
    astar_steps: int


@dataclass(frozen=True)
class CorrelationResult:
    spearman_rho: float
    p_value: float
    bootstrap_ci_95: Tuple[float, float]


def umaze_is_free(x: float, y: float) -> bool:
    """Continuous free-space used by PointMazeWrapper's UMaze sampler."""
    in_left_or_right_arm = 0.5 <= x <= 1.1 or 2.5 <= x <= 3.1
    in_vertical_extent = 0.5 <= y <= 3.1
    in_top_bridge = 1.1 < x < 2.5 and 2.5 <= y <= 3.1
    return (in_left_or_right_arm and in_vertical_extent) or in_top_bridge


def make_occupancy_grid(
    lower: float = 0.5,
    upper: float = 3.1,
    spacing: float = 0.1,
    is_free: Callable[[float, float], bool] = umaze_is_free,
) -> Tuple[np.ndarray, Dict[GridNode, Tuple[float, float]]]:
    """Return coordinate values and valid integer grid nodes.

    Integer nodes avoid floating-point equality problems in A*.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    n_steps = int(round((upper - lower) / spacing))
    coords = lower + np.arange(n_steps + 1, dtype=np.float64) * spacing
    coords = np.round(coords, 10)
    valid: Dict[GridNode, Tuple[float, float]] = {}
    for ix, x in enumerate(coords):
        for iy, y in enumerate(coords):
            if is_free(float(x), float(y)):
                valid[(ix, iy)] = (float(x), float(y))
    if not valid:
        raise ValueError("occupancy grid contains no valid states")
    return coords, valid


def nearest_valid_node(
    xy: Tuple[float, float],
    valid: Mapping[GridNode, Tuple[float, float]],
) -> GridNode:
    target = np.asarray(xy, dtype=np.float64)
    return min(
        valid,
        key=lambda node: float(
            np.square(np.asarray(valid[node], dtype=np.float64) - target).sum()
        ),
    )


def astar_step_counts(
    valid: Mapping[GridNode, Tuple[float, float]],
    goal: GridNode,
) -> Dict[GridNode, int]:
    """Compute shortest 4-neighbor step-count from every node to ``goal``.

    A* is run independently for clarity.  The heuristic is Manhattan distance,
    which is admissible and consistent on this unit-cost grid.
    """
    if goal not in valid:
        raise ValueError(f"goal node {goal} is not in free space")

    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def distance_from(start: GridNode) -> int:
        if start == goal:
            return 0
        frontier: List[Tuple[int, int, GridNode]] = []
        counter = 0
        h0 = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        heapq.heappush(frontier, (h0, counter, start))
        best_g: Dict[GridNode, int] = {start: 0}

        while frontier:
            _, _, node = heapq.heappop(frontier)
            g = best_g[node]
            if node == goal:
                return g
            for dx, dy in offsets:
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor not in valid:
                    continue
                new_g = g + 1
                if new_g >= best_g.get(neighbor, math.inf):
                    continue
                best_g[neighbor] = new_g
                h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                counter += 1
                heapq.heappush(frontier, (new_g + h, counter, neighbor))
        raise RuntimeError(f"free-space node {start} cannot reach goal {goal}")

    return {node: distance_from(node) for node in valid}


def build_grid_states(
    lower: float,
    upper: float,
    spacing: float,
    goal_xy: Tuple[float, float],
) -> Tuple[List[GridState], GridNode]:
    _, valid = make_occupancy_grid(lower, upper, spacing)
    goal = nearest_valid_node(goal_xy, valid)
    steps = astar_step_counts(valid, goal)
    states = [
        GridState(ix=node[0], iy=node[1], x=xy[0], y=xy[1], astar_steps=steps[node])
        for node, xy in sorted(valid.items())
    ]
    return states, goal


def resolve_checkpoint(path: Path) -> Path:
    """Accept either a model file or a training/checkpoint directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = [
        path / "checkpoints" / "model_20.pth",
        path / "model_20.pth",
        path / "checkpoints" / "model_latest.pth",
        path / "model_latest.pth",
    ]
    candidates.extend(sorted(path.glob("checkpoints/model_*.pth"), reverse=True))
    candidates.extend(sorted(path.glob("model_*.pth"), reverse=True))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no model checkpoint found under {path}")


def load_encoder(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Load the serialized trained encoder using the repository's checkpoint path."""
    # Torch Hub's DINO class must be registered before unpickling these checkpoints.
    from models.dino import DinoV2Encoder

    _ = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    # Keep unused predictor/decoder payloads off the GPU; only the encoder moves.
    payload = torch.load(checkpoint, map_location="cpu")
    if "encoder" not in payload:
        raise KeyError(f"{checkpoint} does not contain an 'encoder' entry")
    encoder = payload["encoder"].to(device)
    del payload
    encoder.eval()
    return encoder


def render_grid_observations(
    states: Sequence[GridState],
    seed: int,
) -> np.ndarray:
    """Render the agent at each state with velocity fixed to zero."""
    # Importing env registers point_maze with Gym.
    import env  # noqa: F401
    import gym

    wrapped = gym.make("point_maze")
    maze = wrapped.unwrapped
    maze.prepare_for_render()
    frames: List[np.ndarray] = []
    try:
        for index, state in enumerate(states):
            init_state = np.array([state.x, state.y, 0.0, 0.0], dtype=np.float32)
            # prepare() reinitializes the render context on every call.  The
            # camera is already initialized above, so use its remaining steps
            # directly to make the dense-grid pass substantially faster.
            maze.seed(seed + index)
            maze.set_init_state(init_state)
            obs, _ = maze.reset()
            frames.append(np.asarray(obs["visual"], dtype=np.uint8))
    finally:
        wrapped.close()
    return np.stack(frames, axis=0)


def encode_frames(
    frames: np.ndarray,
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode HWC uint8 frames and flatten all visual patch dimensions."""
    from datasets.img_transforms import default_transform

    encoder = load_encoder(checkpoint, device)
    transform = default_transform(img_size=224)
    embeddings: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch_np = frames[start : start + batch_size]
            batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
            batch = transform(batch).to(device)
            latent = encoder(batch)
            latent = latent.reshape(latent.shape[0], -1)
            embeddings.append(latent.float().cpu().numpy())
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def latent_distances_to_goal(
    embeddings: np.ndarray,
    states: Sequence[GridState],
    goal_node: GridNode,
) -> np.ndarray:
    goal_index = next(
        index
        for index, state in enumerate(states)
        if (state.ix, state.iy) == goal_node
    )
    delta = embeddings - embeddings[goal_index]
    return np.linalg.norm(delta, axis=1)


def bootstrap_spearman_ci(
    true_steps: np.ndarray,
    latent_distances: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(true_steps)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        estimates[index] = spearmanr(
            true_steps[sample], latent_distances[sample]
        ).statistic
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return (float("nan"), float("nan"))
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def paired_bootstrap_delta_ci(
    true_steps: np.ndarray,
    r0_distances: np.ndarray,
    r2_distances: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    """Paired state bootstrap for rho(R2) - rho(R0)."""
    rng = np.random.default_rng(seed)
    n = len(true_steps)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        r0_rho = spearmanr(true_steps[sample], r0_distances[sample]).statistic
        r2_rho = spearmanr(true_steps[sample], r2_distances[sample]).statistic
        deltas[index] = r2_rho - r0_rho
    finite = deltas[np.isfinite(deltas)]
    if not len(finite):
        return (float("nan"), float("nan"))
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def summarize_correlation(
    true_steps: np.ndarray,
    latent_distances: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> CorrelationResult:
    result = spearmanr(true_steps, latent_distances)
    return CorrelationResult(
        spearman_rho=float(result.statistic),
        p_value=float(result.pvalue),
        bootstrap_ci_95=bootstrap_spearman_ci(
            true_steps,
            latent_distances,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
    )


def write_csv(
    path: Path,
    states: Sequence[GridState],
    r0_distances: np.ndarray,
    r2_distances: np.ndarray,
) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ix",
                "iy",
                "x",
                "y",
                "astar_steps",
                "r0_latent_distance",
                "r2_latent_distance",
            ]
        )
        for state, r0, r2 in zip(states, r0_distances, r2_distances):
            writer.writerow(
                [state.ix, state.iy, state.x, state.y, state.astar_steps, r0, r2]
            )


def write_plots(
    output_dir: Path,
    states: Sequence[GridState],
    true_steps: np.ndarray,
    r0_distances: np.ndarray,
    r2_distances: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(true_steps, r0_distances, s=12, alpha=0.45, label="R0 direction-only")
    ax.scatter(true_steps, r2_distances, s=12, alpha=0.45, label="R2 full penalty")
    ax.set_xlabel("True A* distance to goal (grid steps)")
    ax.set_ylabel("Visual latent Euclidean distance to goal")
    ax.set_title("UMaze latent distance vs. true shortest-path distance")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_vs_astar_scatter.png", dpi=180)
    plt.close(fig)

    x = np.array([state.x for state in states])
    y = np.array([state.y for state in states])
    values = [true_steps, r0_distances, r2_distances]
    titles = ["A* step-count", "R0 latent distance", "R2 latent distance"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, value, title in zip(axes, values, titles):
        scatter = ax.scatter(x, y, c=value, s=28, cmap="viridis", marker="s")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(scatter, ax=ax, shrink=0.82)
    fig.suptitle("UMaze distance fields to the canonical goal")
    fig.savefig(output_dir / "distance_fields.png", dpi=180)
    plt.close(fig)


def write_markdown(
    path: Path,
    results: Mapping[str, object],
) -> None:
    r0 = results["r0"]
    r2 = results["r2"]
    delta = results["delta_r2_minus_r0"]
    delta_ci = results["paired_bootstrap_delta_ci_95"]
    interpretation = (
        "supports"
        if delta > 0 and delta_ci[0] > 0
        else "is directionally consistent with"
        if delta > 0
        else "does not support"
    )
    text = f"""# UMaze latent-geodesic correlation: R0 vs R2

This analysis tests the missing middle link in the hypothesis:

> steadier latent pace → more honest latent distances → better planning

The visual latent Euclidean distance from each valid grid state to the
canonical UMaze goal was compared with true A* step-count. The same rendered
observations were passed through both checkpoints. Agent velocity was fixed at
zero and the planning-matched visual representation was used (`alpha=0`).

| Condition | Spearman ρ | 95% bootstrap CI | p-value |
|---|---:|---:|---:|
| R0 direction-only | {r0['spearman_rho']:.4f} | [{r0['bootstrap_ci_95'][0]:.4f}, {r0['bootstrap_ci_95'][1]:.4f}] | {r0['p_value']:.3g} |
| R2 full penalty | {r2['spearman_rho']:.4f} | [{r2['bootstrap_ci_95'][0]:.4f}, {r2['bootstrap_ci_95'][1]:.4f}] | {r2['p_value']:.3g} |

Paired difference, `ρ(R2) - ρ(R0)`: **{delta:.4f}**, with a paired state
bootstrap 95% CI of **[{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]**.

This result **{interpretation}** the proposed middle link. A positive,
well-separated difference means R2 orders states by true path distance more
faithfully than R0. This is a geometry diagnostic, not by itself evidence that
the improvement causes planning success.

Artifacts:

- `grid_latent_distances.csv`: paired per-state measurements
- `results.json`: machine-readable statistics and run settings
- `latent_vs_astar_scatter.png`: correlation view
- `distance_fields.png`: true and learned distance maps
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-checkpoint", type=Path, required=True)
    parser.add_argument("--r2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "baseline_artifacts/analysis/umaze_latent_geodesic_r0_vs_r2"
        ),
    )
    parser.add_argument("--grid-spacing", type=float, default=0.1)
    parser.add_argument("--grid-lower", type=float, default=0.5)
    parser.add_argument("--grid-upper", type=float, default=3.1)
    parser.add_argument("--goal-x", type=float, default=1.0)
    parser.add_argument("--goal-y", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    r0_checkpoint = resolve_checkpoint(args.r0_checkpoint)
    r2_checkpoint = resolve_checkpoint(args.r2_checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    states, goal_node = build_grid_states(
        args.grid_lower,
        args.grid_upper,
        args.grid_spacing,
        (args.goal_x, args.goal_y),
    )
    print(f"Rendering {len(states)} valid grid states; goal node={goal_node}")
    frames = render_grid_observations(states, seed=args.seed)

    print(f"Encoding R0 observations with {r0_checkpoint}")
    r0_embeddings = encode_frames(frames, r0_checkpoint, args.batch_size, device)
    print(f"Encoding R2 observations with {r2_checkpoint}")
    r2_embeddings = encode_frames(frames, r2_checkpoint, args.batch_size, device)

    r0_distances = latent_distances_to_goal(r0_embeddings, states, goal_node)
    r2_distances = latent_distances_to_goal(r2_embeddings, states, goal_node)
    true_steps = np.asarray([state.astar_steps for state in states], dtype=np.float64)

    r0_result = summarize_correlation(
        true_steps,
        r0_distances,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed,
    )
    r2_result = summarize_correlation(
        true_steps,
        r2_distances,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    delta = r2_result.spearman_rho - r0_result.spearman_rho
    delta_ci = paired_bootstrap_delta_ci(
        true_steps,
        r0_distances,
        r2_distances,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 2,
    )

    results = {
        "n_states": len(states),
        "grid_spacing": args.grid_spacing,
        "grid_bounds": [args.grid_lower, args.grid_upper],
        "goal_xy_requested": [args.goal_x, args.goal_y],
        "goal_node": list(goal_node),
        "goal_xy_evaluated": [
            next(state.x for state in states if (state.ix, state.iy) == goal_node),
            next(state.y for state in states if (state.ix, state.iy) == goal_node),
        ],
        "r0_checkpoint": str(r0_checkpoint),
        "r2_checkpoint": str(r2_checkpoint),
        "r0": asdict(r0_result),
        "r2": asdict(r2_result),
        "delta_r2_minus_r0": delta,
        "paired_bootstrap_delta_ci_95": delta_ci,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "device": str(device),
    }

    write_csv(output_dir / "grid_latent_distances.csv", states, r0_distances, r2_distances)
    write_plots(output_dir, states, true_steps, r0_distances, r2_distances)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_markdown(output_dir / "README.md", results)

    print(json.dumps(results, indent=2))
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
