#!/usr/bin/env python3
"""
Collect UMaze activations, train linear probes, and smoke-test hook interventions.

Examples
--------
# 1) Verify labels only (no checkpoint / GPU needed beyond dataset):
python probing/run_umaze_probes.py --labels-only --data-path $DATASET_DIR/point_maze

# 2) Full probe run on an R0 checkpoint:
python probing/run_umaze_probes.py \\
  --model-dir baseline_artifacts/checkpoints/umaze_speed_ablations/r0_direction_only \\
  --epoch 20 \\
  --max-rollouts 80 \\
  --output probing/out/r0

# 3) Compare readouts:
python probing/run_umaze_probes.py --model-dir ... --readout agg_mlp --output probing/out/r0_agg

# 4) Location holdout (train left/bottom, test right/top, and swaps):
python probing/run_umaze_probes.py \\
  --model-dir .../r0_direction_only --epoch 20 \\
  --location-holdout --output probing/out/r0

# 5) Re-run holdout from an existing activations cache (no GPU encode):
python probing/run_umaze_probes.py \\
  --from-cache probing/out/r0/activations.pt --location-holdout \\
  --output probing/out/r0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.hooks import ActivationHookManager, flatten_activation
from probing.labels import state_tensor_to_probe_labels
from probing.linear_probe import (
    LinearProbeSuite,
    episode_train_val_split,
    run_location_holdout,
    tensors_to_numpy,
)


def _load_model_from_ckpt(model_ckpt: Path, train_cfg, num_action_repeat: int, device):
    """Load VWorldModel without importing plan.py (avoids mujoco_py / env side effects)."""
    import hydra

    from models.dino import DinoV2Encoder

    model_keys = (
        "encoder",
        "predictor",
        "decoder",
        "proprio_encoder",
        "action_encoder",
    )
    # Touch encoder class so torch.hub dinov2 is importable when unpickling.
    _ = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    with model_ckpt.open("rb") as f:
        payload = torch.load(f, map_location=device)
    result = {k: payload[k].to(device) for k in model_keys if k in payload}
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")
    if not train_cfg.has_decoder:
        result["decoder"] = None
    elif "decoder" not in result:
        raise ValueError("Decoder missing from checkpoint and has_decoder=True")

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


def _load_model_and_dataset(model_dir: Path, epoch: int, data_path: str | None):
    import hydra
    from omegaconf import OmegaConf

    hydra_yaml = model_dir / "hydra.yaml"
    if not hydra_yaml.exists():
        raise FileNotFoundError(f"Missing {hydra_yaml}")

    model_cfg = OmegaConf.load(hydra_yaml)
    if data_path is not None:
        model_cfg.env.dataset.data_path = data_path

    ckpt = model_dir / "checkpoints" / f"model_{epoch}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_ckpt(ckpt, model_cfg, model_cfg.num_action_repeat, device)
    model.eval()

    _, traj_dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    base_dset = traj_dset["train"].dataset
    return model, base_dset, model_cfg, device


def _load_point_maze_only(data_path: str, max_rollouts: int | None):
    from datasets.img_transforms import default_transform
    from datasets.point_maze_dset import PointMazeDataset

    dset = PointMazeDataset(
        data_path=data_path,
        normalize_action=True,
        transform=default_transform(img_size=224),
    )
    n = len(dset) if max_rollouts is None else min(max_rollouts, len(dset))
    return dset, n


@torch.no_grad()
def collect_activations(
    model,
    base_dset,
    *,
    device: torch.device,
    frameskip: int,
    max_rollouts: int,
    readout: str,
    hook_manager: ActivationHookManager | None = None,
) -> dict:
    feature_rows: list[torch.Tensor] = []
    label_rows: dict[str, list[torch.Tensor]] = {
        k: []
        for k in [
            "position_x",
            "position_y",
            "speed",
            "direction_cos",
            "direction_sin",
        ]
    }
    episode_rows: list[torch.Tensor] = []

    hook_names = []
    if hook_manager is not None:
        hook_names = hook_manager.register_umaze_defaults()
        hook_site_rows: dict[str, list[torch.Tensor]] = {n: [] for n in hook_names}

    n_rollouts = min(max_rollouts, len(base_dset))
    for rollout_idx in range(n_rollouts):
        seq_len = int(base_dset.get_seq_length(rollout_idx))
        frame_idx = list(range(0, seq_len, frameskip))
        if len(frame_idx) < 2:
            continue

        visual = base_dset.load_visual_frames(rollout_idx, frame_idx)
        proprio = base_dset.proprios[rollout_idx, frame_idx]
        state = base_dset.states[rollout_idx, frame_idx]

        obs = {
            "visual": visual.unsqueeze(0).to(device),
            "proprio": proprio.unsqueeze(0).to(device),
        }

        if hook_manager is not None:
            hook_manager.activations.clear()
            features = hook_manager.capture_encode_obs(obs, readout=readout)
            for name in hook_names:
                if name in hook_manager.activations:
                    flat = flatten_activation(name, hook_manager.activations[name])
                    hook_site_rows[name].append(flat)
        else:
            features = model.encode_obs(obs)["visual"].mean(dim=2)

        labels = state_tensor_to_probe_labels(state)

        t = features.shape[1]
        feature_rows.append(features.reshape(t, -1))
        episode_rows.append(torch.full((t,), rollout_idx, dtype=torch.long))
        for key in label_rows:
            label_rows[key].append(labels[key])

    if not feature_rows:
        raise RuntimeError("No frames collected; check dataset path and frameskip.")

    out = {
        "features": torch.cat(feature_rows, dim=0),
        "labels": {k: torch.cat(v, dim=0) for k, v in label_rows.items()},
        "episode_ids": torch.cat(episode_rows, dim=0),
        "readout": readout,
        "n_rollouts": n_rollouts,
        "n_frames": int(torch.cat(episode_rows, dim=0).shape[0]),
    }
    if hook_manager is not None:
        out["hook_features"] = {
            name: torch.cat(rows, dim=0) for name, rows in hook_site_rows.items() if rows
        }
    return out


def run_labels_only(data_path: str, max_rollouts: int, frameskip: int) -> None:
    dset, n = _load_point_maze_only(data_path, max_rollouts)
    speeds: list[float] = []
    for idx in range(n):
        seq_len = int(dset.get_seq_length(idx))
        frame_idx = list(range(0, seq_len, frameskip))
        state = dset.states[idx, frame_idx]
        labels = state_tensor_to_probe_labels(state)
        speeds.extend(labels["speed"].tolist())

    print(f"Loaded {n} UMaze rollouts from {data_path}")
    print(
        f"Speed stats (||velocity||): mean={np.mean(speeds):.4f}, "
        f"std={np.std(speeds):.4f}, min={np.min(speeds):.4f}, max={np.max(speeds):.4f}"
    )
    print("Label columns: position_x/y from qpos; speed/direction from qvel.")


def train_probes_from_cache(cache: dict, *, val_fraction: float, seed: int) -> LinearProbeSuite:
    x, y, episode_ids = tensors_to_numpy(
        cache["features"], cache["labels"], cache["episode_ids"]
    )
    train_mask, val_mask = episode_train_val_split(
        episode_ids, val_fraction=val_fraction, seed=seed
    )
    suite = LinearProbeSuite()
    suite.fit(x[train_mask], {k: v[train_mask] for k, v in y.items()},
              x[val_mask], {k: v[val_mask] for k, v in y.items()})
    return suite


def location_holdout_from_cache(
    cache: dict,
    *,
    axes: tuple[str, ...] = ("x", "y"),
    threshold_mode: str = "median",
) -> dict:
    """Train in one maze half, evaluate in the other (mentor location-generalization check)."""
    x, y, _ = tensors_to_numpy(cache["features"], cache["labels"], cache["episode_ids"])
    return run_location_holdout(
        x,
        y,
        axes=axes,
        threshold_mode=threshold_mode,
    )


def _print_location_holdout(report: dict) -> None:
    print("\nLocation holdout (train in one region, test in the complementary region):")
    print(f"  threshold_mode={report['threshold_mode']}")
    for split in report["splits"]:
        tag = f"{split['axis']}:{split['train_side']}→other"
        if "skipped" in split:
            print(f"  {tag:18s}  skipped ({split['skipped']})")
            continue
        by_target = {r["target"]: r for r in split["results"]}
        speed = by_target.get("speed", {})
        baseline = {
            r["target"]: r for r in split.get("position_baseline", [])
        }.get("speed", {})
        base_txt = ""
        if baseline:
            base_txt = f"  pos→speed={baseline.get('heldout_r2', float('nan')):.3f}"
        print(
            f"  {tag:18s}  thr={split['threshold']:.3f}  "
            f"n_train={split['n_train']} n_test={split['n_test']}  "
            f"speed heldout_r2={speed.get('heldout_r2', float('nan')):.3f}{base_txt}"
        )
    print("\n  Mean held-out R² across splits:")
    for target, stats in report["summary"].items():
        base = stats.get("position_baseline_mean_heldout_r2")
        base_txt = f"  pos_baseline={base:.3f}" if base is not None else ""
        print(
            f"    {target:16s}  mean={stats['mean_heldout_r2']:6.3f}  "
            f"min={stats['min_heldout_r2']:6.3f}  (n={stats['n_splits']}){base_txt}"
        )


@torch.no_grad()
def smoke_test_interventions(
    model,
    base_dset,
    hook_manager: ActivationHookManager,
    *,
    device: torch.device,
    frameskip: int,
    suite: LinearProbeSuite,
) -> dict:
    """Knockout / mean-replace one hook site on a single frame; report probe delta."""
    rollout_idx = 0
    seq_len = int(base_dset.get_seq_length(rollout_idx))
    frame_idx = list(range(0, seq_len, frameskip))[:5]
    visual = base_dset.load_visual_frames(rollout_idx, frame_idx)
    proprio = base_dset.proprios[rollout_idx, frame_idx]
    state = base_dset.states[rollout_idx, frame_idx]
    obs = {
        "visual": visual.unsqueeze(0).to(device),
        "proprio": proprio.unsqueeze(0).to(device),
    }

    hook_manager.clear_interventions()
    baseline = hook_manager.capture_encode_obs(obs, readout="post_projector")
    x_base = baseline.reshape(-1, baseline.shape[-1]).cpu().numpy()
    y = state_tensor_to_probe_labels(state)
    y_np = {k: v.cpu().numpy() for k, v in y.items()}
    pred_base = suite.predict(x_base)

    if not hook_manager.registered_names:
        hook_manager.register_umaze_defaults()

    reports = {}
    for hook_name in hook_manager.registered_names:
        if hook_name == "encoder":
            continue
        for mode in ("knockout", "mean"):
            hook_manager.clear_interventions()
            if mode == "mean" and hook_name not in hook_manager.activations:
                hook_manager.capture_encode_obs(obs, readout="post_projector")
            if mode == "mean":
                ref = hook_manager.activations.get(hook_name)
                if ref is None:
                    continue
                hook_manager.set_intervention(hook_name, "mean", ref.mean(dim=0, keepdim=True))
            else:
                hook_manager.set_intervention(hook_name, "knockout")

            perturbed = hook_manager.capture_encode_obs(obs, readout="post_projector")
            x_pert = perturbed.reshape(-1, perturbed.shape[-1]).cpu().numpy()
            pred_pert = suite.predict(x_pert)
            delta = {
                target: float(np.mean(np.abs(pred_pert[target] - pred_base[target])))
                for target in pred_base
            }
            reports[f"{hook_name}:{mode}"] = delta

    hook_manager.clear_interventions()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="UMaze linear probing prototype")
    parser.add_argument("--model-dir", type=Path, default=None, help="Hydra run dir with hydra.yaml + checkpoints/")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--data-path", type=str, default=None, help="Override point_maze dataset path")
    parser.add_argument("--max-rollouts", type=int, default=80)
    parser.add_argument("--frameskip", type=int, default=None)
    parser.add_argument(
        "--readout",
        choices=["post_projector", "agg_mlp", "flatten"],
        default="post_projector",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("probing/out/umaze"))
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--skip-interventions", action="store_true")
    parser.add_argument(
        "--location-holdout",
        action="store_true",
        help="Also train/eval probes with a spatial split (mentor location-generalization check)",
    )
    parser.add_argument(
        "--holdout-axes",
        type=str,
        default="x,y",
        help="Comma-separated axes for location holdout, e.g. x or x,y",
    )
    parser.add_argument(
        "--holdout-threshold",
        choices=["median", "midpoint"],
        default="median",
        help="How to choose the spatial cut for location holdout",
    )
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="Skip encoding; load activations.pt and (re)run probe + optional location holdout",
    )
    args = parser.parse_args()

    data_path = args.data_path or f"{Path.home()}/data/point_maze"
    if args.data_path is None:
        import os

        data_path = os.environ.get("DATASET_DIR", str(REPO_ROOT / "data"))
        data_path = str(Path(data_path) / "point_maze")

    if args.labels_only:
        frameskip = args.frameskip or 5
        run_labels_only(data_path, args.max_rollouts, frameskip)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    holdout_axes = tuple(a.strip() for a in args.holdout_axes.split(",") if a.strip())

    if args.from_cache is not None:
        cache = torch.load(args.from_cache, map_location="cpu", weights_only=False)
        suite = train_probes_from_cache(
            cache, val_fraction=args.val_fraction, seed=args.seed
        )
        suite.save_report(args.output / "probe_results.json")
        print(f"\nLoaded cache {args.from_cache}")
        print(f"Frames={cache['n_frames']}  feature_dim={cache['features'].shape[-1]}")
        print("\nEpisode-split validation R²:")
        for result in suite.results:
            print(
                f"  {result.target:16s}  val_r2={result.val_r2:6.3f}  "
                f"val_rmse={result.val_rmse:.4f}"
            )
        if args.location_holdout:
            holdout = location_holdout_from_cache(
                cache, axes=holdout_axes, threshold_mode=args.holdout_threshold
            )
            (args.output / "location_holdout.json").write_text(
                json.dumps(holdout, indent=2)
            )
            _print_location_holdout(holdout)
            print(f"\nWrote {args.output}/location_holdout.json")
        else:
            print("\nTip: pass --location-holdout to run the spatial generalization check.")
        return

    if args.model_dir is None:
        parser.error("--model-dir is required unless --labels-only or --from-cache is set")

    model, base_dset, model_cfg, device = _load_model_and_dataset(
        args.model_dir, args.epoch, args.data_path
    )
    frameskip = args.frameskip or int(model_cfg.frameskip)

    hook_manager = ActivationHookManager(model)
    cache = collect_activations(
        model,
        base_dset,
        device=device,
        frameskip=frameskip,
        max_rollouts=args.max_rollouts,
        readout=args.readout,
        hook_manager=hook_manager,
    )

    suite = train_probes_from_cache(
        cache, val_fraction=args.val_fraction, seed=args.seed
    )

    torch.save(cache, args.output / "activations.pt")
    suite.save_report(args.output / "probe_results.json")

    print(f"\nCollected {cache['n_frames']} frames from {cache['n_rollouts']} rollouts")
    print(f"Readout: {args.readout}  feature_dim={cache['features'].shape[-1]}")
    print("\nEpisode-split validation R²:")
    for result in suite.results:
        extra = ""
        if "direction_mae_deg" in result.extra:
            extra = f", direction MAE={result.extra['direction_mae_deg']:.1f}°"
        print(
            f"  {result.target:16s}  val_r2={result.val_r2:6.3f}  "
            f"val_rmse={result.val_rmse:.4f}{extra}"
        )

    if args.location_holdout:
        holdout = location_holdout_from_cache(
            cache, axes=holdout_axes, threshold_mode=args.holdout_threshold
        )
        (args.output / "location_holdout.json").write_text(json.dumps(holdout, indent=2))
        _print_location_holdout(holdout)

    if not args.skip_interventions:
        intervention_report = smoke_test_interventions(
            model,
            base_dset,
            hook_manager,
            device=device,
            frameskip=frameskip,
            suite=suite,
        )
        (args.output / "intervention_smoke.json").write_text(
            json.dumps(intervention_report, indent=2)
        )
        print("\nIntervention smoke test (mean |Δ probe pred| on 5 frames):")
        for site, deltas in intervention_report.items():
            direction_delta = deltas.get("direction_cos", 0.0) + deltas.get(
                "direction_sin", 0.0
            )
            print(f"  {site:22s}  directionΔ≈{direction_delta:.4f}")

    print(f"\nWrote {args.output}/activations.pt and probe_results.json")
    if args.location_holdout:
        print(f"Wrote {args.output}/location_holdout.json")


if __name__ == "__main__":
    main()
