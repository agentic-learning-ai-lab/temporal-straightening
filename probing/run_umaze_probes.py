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
  --output probing/speed_holdout/r0_direction_only

# 3) Compare readouts:
python probing/run_umaze_probes.py --model-dir ... --readout agg_mlp --output probing/speed_holdout/r0_agg

# 4) Location holdout (train left/bottom, test right/top, and swaps):
python probing/run_umaze_probes.py \\
  --model-dir .../r0_direction_only --epoch 20 \\
  --location-holdout --output probing/speed_holdout/r0_direction_only

# 5) Re-run holdout from an existing activations cache (no GPU encode):
python probing/run_umaze_probes.py \\
  --from-cache probing/speed_holdout/r0_direction_only/activations.pt --location-holdout \\
  --output probing/speed_holdout/r0_direction_only

# 6) Daniel-style DINO feature-diff + location holdout (auto path if --output omitted):
python probing/run_umaze_probes.py \\
  --from-cache probing/speed_holdout/r0_direction_only/activations.pt \\
  --probe-source dino.block.5 --feature-mode diff --location-holdout
# writes probing/dino5_diff_holdout/r0_direction_only/

# 7) Predictor location holdout (needs a fresh encode that fills predictor_pack):
python probing/run_umaze_probes.py \\
  --model-dir .../r0_direction_only --epoch 20 \\
  --location-holdout --skip-interventions --probe-source predictor
# writes probing/predictor_holdout/r0_direction_only/
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
    DEFAULT_TARGETS,
    LinearProbeSuite,
    episode_train_val_split,
    run_location_holdout,
    tensors_to_numpy,
)


def resolve_output_dir(output: Path | None, *, probe_source: str, feature_mode: str, model_dir: Path | None, from_cache: Path | None) -> Path:
    """Map probe settings to probing/<experiment>/<condition>/ unless --output is set."""
    if output is not None:
        return output
    if probe_source == "predictor":
        experiment = "predictor_holdout"
    elif probe_source == "dino.block.5" and feature_mode == "diff":
        experiment = "dino5_diff_holdout"
    elif feature_mode == "diff":
        experiment = f"{probe_source.replace('.', '_')}_diff_holdout"
    else:
        experiment = "speed_holdout"
    if model_dir is not None:
        condition = model_dir.name
    elif from_cache is not None:
        # caches live under speed_holdout/<cond>/; keep the condition name
        condition = from_cache.parent.name
    else:
        condition = "umaze"
    return Path("probing") / experiment / condition


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

    dset = PointMazeDataset(data_path=data_path, normalize_action=True, transform=default_transform(img_size=224))
    n = len(dset) if max_rollouts is None else min(max_rollouts, len(dset))
    return dset, n


@torch.no_grad()
def collect_activations(
    model, base_dset, *, device: torch.device, frameskip: int, max_rollouts: int,
    readout: str, hook_manager: ActivationHookManager | None = None,
) -> dict:
    feature_rows: list[torch.Tensor] = []
    label_rows: dict[str, list[torch.Tensor]] = {k: [] for k in DEFAULT_TARGETS}
    episode_rows: list[torch.Tensor] = []

    hook_names: list[str] = []
    hook_site_rows: dict[str, list[torch.Tensor]] = {}
    predictor_rows: list[torch.Tensor] = []
    predictor_label_rows: dict[str, list[torch.Tensor]] = {k: [] for k in DEFAULT_TARGETS}
    predictor_episode_rows: list[torch.Tensor] = []
    if hook_manager is not None:
        hook_names = hook_manager.register_umaze_defaults()
        hook_site_rows = {n: [] for n in hook_names}

    n_rollouts = min(max_rollouts, len(base_dset))
    for rollout_idx in range(n_rollouts):
        seq_len = int(base_dset.get_seq_length(rollout_idx))
        # Match TrajSlicerDataset: only steps with a full frameskip action block.
        frame_idx = list(range(0, seq_len - frameskip + 1, frameskip))
        if len(frame_idx) < 2:
            continue

        visual = base_dset.load_visual_frames(rollout_idx, frame_idx)
        proprio = base_dset.proprios[rollout_idx, frame_idx]
        state = base_dset.states[rollout_idx, frame_idx]
        # Concatenate frameskip env actions per model step: (T, frameskip * d).
        act_rows = []
        for start in frame_idx:
            chunk = base_dset.actions[rollout_idx, start : start + frameskip]
            act_rows.append(chunk.reshape(-1))
        action = torch.stack(act_rows, dim=0)

        obs = {
            "visual": visual.unsqueeze(0).to(device),
            "proprio": proprio.unsqueeze(0).to(device),
        }
        act = action.unsqueeze(0).to(device)

        if hook_manager is not None:
            hook_manager.activations.clear()
            features = hook_manager.capture_encode_obs(obs, readout=readout)
            for name in hook_names:
                if name in hook_manager.activations:
                    flat = flatten_activation(name, hook_manager.activations[name])
                    if name != "predictor":
                        hook_site_rows[name].append(flat)

            pred_feat, keep = hook_manager.capture_predictor_features(obs, act)
            if keep:
                predictor_rows.append(pred_feat)
                labels_full = state_tensor_to_probe_labels(state)
                for key in predictor_label_rows:
                    predictor_label_rows[key].append(labels_full[key][keep])
                predictor_episode_rows.append(
                    torch.full((len(keep),), rollout_idx, dtype=torch.long)
                )
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

    episode_ids = torch.cat(episode_rows, dim=0)
    out = {
        "features": torch.cat(feature_rows, dim=0),
        "labels": {k: torch.cat(v, dim=0) for k, v in label_rows.items()},
        "episode_ids": episode_ids,
        "readout": readout,
        "n_rollouts": n_rollouts,
        "n_frames": int(episode_ids.shape[0]),
    }
    if hook_manager is not None:
        out["hook_features"] = {
            name: torch.cat(rows, dim=0) for name, rows in hook_site_rows.items() if rows
        }
        if predictor_rows:
            out["predictor_pack"] = {
                "features": torch.cat(predictor_rows, dim=0),
                "labels": {k: torch.cat(v, dim=0) for k, v in predictor_label_rows.items()},
                "episode_ids": torch.cat(predictor_episode_rows, dim=0),
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


def select_probe_cache(cache: dict, *, probe_source: str = "readout", feature_mode: str = "raw") -> dict:
    """Select readout/hook/predictor features; optional within-episode diffs."""
    if probe_source == "predictor":
        pack = cache.get("predictor_pack")
        if pack is None:
            raise KeyError(
                "probe source 'predictor' needs predictor_pack in the cache; "
                "re-encode with the updated collect_activations (encode+predict)."
            )
        features = pack["features"]
        labels = pack["labels"]
        episode_ids = pack["episode_ids"]
    elif probe_source == "readout":
        features = cache["features"]
        labels = cache["labels"]
        episode_ids = cache["episode_ids"]
    else:
        hooks = cache.get("hook_features") or {}
        if probe_source not in hooks:
            available = ", ".join(sorted(hooks)) or "(none)"
            raise KeyError(f"probe source {probe_source!r} not in hook_features; available: {available}")
        features = hooks[probe_source]
        labels = cache["labels"]
        episode_ids = cache["episode_ids"]

    if feature_mode == "raw":
        return {
            **cache,
            "features": features,
            "labels": labels,
            "episode_ids": episode_ids,
            "n_frames": int(features.shape[0]),
            "probe_source": probe_source,
            "feature_mode": feature_mode,
        }
    if feature_mode != "diff":
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    feat_rows, ep_rows = [], []
    label_rows = {k: [] for k in labels}
    ids = episode_ids.cpu().numpy() if torch.is_tensor(episode_ids) else np.asarray(episode_ids)
    for ep in np.unique(ids):
        idx = np.where(ids == ep)[0]
        if len(idx) < 2:
            continue
        f = features[idx]
        feat_rows.append(f[1:] - f[:-1])
        for k, v in labels.items():
            label_rows[k].append(v[idx][1:])
        ep_rows.append(episode_ids[idx][1:])

    if not feat_rows:
        raise RuntimeError("No within-episode pairs available for feature_mode=diff")

    return {
        **cache,
        "features": torch.cat(feat_rows, dim=0),
        "labels": {k: torch.cat(v, dim=0) for k, v in label_rows.items()},
        "episode_ids": torch.cat(ep_rows, dim=0),
        "n_frames": int(sum(t.shape[0] for t in feat_rows)),
        "probe_source": probe_source,
        "feature_mode": feature_mode,
    }


def location_holdout_from_cache(cache: dict, *, axes: tuple[str, ...] = ("x", "y"), threshold_mode: str = "median") -> dict:
    """Train in one maze half, evaluate in the other."""
    x, y, _ = tensors_to_numpy(cache["features"], cache["labels"], cache["episode_ids"])
    return run_location_holdout(x, y, axes=axes, threshold_mode=threshold_mode)


def _print_location_holdout(report: dict) -> None:
    print("\nLocation holdout (train in one region, test in the complementary region):")
    print(f"  threshold_mode={report['threshold_mode']}")
    for split in report["splits"]:
        tag = f"{split['axis']}:{split['train_side']}->other"
        if "skipped" in split:
            print(f"  {tag:18s}  skipped ({split['skipped']})")
            continue
        by_target = {r["target"]: r for r in split["results"]}
        speed = by_target.get("speed", {})
        baseline_by_target = {r["target"]: r for r in split.get("position_baseline", [])}
        baseline = baseline_by_target.get("speed", {})
        base_txt = ""
        if baseline:
            base_txt = f"  pos->speed={baseline.get('heldout_r2', float('nan')):.3f}"
        print(
            f"  {tag:18s}  thr={split['threshold']:.3f}  "
            f"n_train={split['n_train']} n_test={split['n_test']}  "
            f"speed heldout_r2={speed.get('heldout_r2', float('nan')):.3f}{base_txt}"
        )
    print("\n  Mean held-out R2 across splits:")
    for target, stats in report["summary"].items():
        base = stats.get("position_baseline_mean_heldout_r2")
        base_txt = f"  pos_baseline={base:.3f}" if base is not None else ""
        print(
            f"    {target:16s}  mean={stats['mean_heldout_r2']:6.3f}  "
            f"min={stats['min_heldout_r2']:6.3f}  (n={stats['n_splits']}){base_txt}"
        )


@torch.no_grad()
def smoke_test_interventions(model, base_dset, hook_manager: ActivationHookManager, *, device: torch.device, frameskip: int, suite: LinearProbeSuite) -> dict:
    """Knockout / mean-replace one hook site; report probe delta."""
    rollout_idx = 0
    seq_len = int(base_dset.get_seq_length(rollout_idx))
    frame_idx = list(range(0, seq_len, frameskip))[:5]
    visual = base_dset.load_visual_frames(rollout_idx, frame_idx)
    proprio = base_dset.proprios[rollout_idx, frame_idx]
    obs = {
        "visual": visual.unsqueeze(0).to(device),
        "proprio": proprio.unsqueeze(0).to(device),
    }

    hook_manager.clear_interventions()
    baseline = hook_manager.capture_encode_obs(obs, readout="post_projector")
    x_base = baseline.reshape(-1, baseline.shape[-1]).cpu().numpy()
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
            reports[f"{hook_name}:{mode}"] = {
                target: float(np.mean(np.abs(pred_pert[target] - pred_base[target])))
                for target in pred_base
            }

    hook_manager.clear_interventions()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="UMaze linear probing prototype")
    parser.add_argument("--model-dir", type=Path, default=None, help="Hydra run dir with hydra.yaml + checkpoints/")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--data-path", type=str, default=None, help="Override point_maze dataset path")
    parser.add_argument("--max-rollouts", type=int, default=80)
    parser.add_argument("--frameskip", type=int, default=None)
    parser.add_argument("--readout", choices=["post_projector", "agg_mlp", "flatten"], default="post_projector")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Output dir (default: probing/speed_holdout/<cond> or probing/dino5_diff_holdout/<cond>)")
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--skip-interventions", action="store_true")
    parser.add_argument("--location-holdout", action="store_true", help="Spatial train/test split")
    parser.add_argument("--holdout-axes", type=str, default="x,y", help="e.g. x or x,y")
    parser.add_argument("--holdout-threshold", choices=["median", "midpoint"], default="median")
    parser.add_argument("--from-cache", type=Path, default=None, help="Load activations.pt; skip encode")
    parser.add_argument("--probe-source", type=str, default="readout", help="readout or hook name")
    parser.add_argument("--feature-mode", choices=["raw", "diff"], default="raw", help="raw or within-ep diff")
    args = parser.parse_args()

    if args.data_path is not None:
        data_path = args.data_path
    else:
        import os
        data_path = str(Path(os.environ.get("DATASET_DIR", str(REPO_ROOT / "data"))) / "point_maze")

    if args.labels_only:
        run_labels_only(data_path, args.max_rollouts, args.frameskip or 5)
        return

    args.output = resolve_output_dir(
        args.output,
        probe_source=args.probe_source,
        feature_mode=args.feature_mode,
        model_dir=args.model_dir,
        from_cache=args.from_cache,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.output}")
    holdout_axes = tuple(a.strip() for a in args.holdout_axes.split(",") if a.strip())

    def _run_probe_and_holdout(cache: dict, *, save_activations: bool) -> LinearProbeSuite:
        probe_cache = select_probe_cache(
            cache,
            probe_source=args.probe_source,
            feature_mode=args.feature_mode,
        )
        suite = train_probes_from_cache(
            probe_cache,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        if save_activations:
            torch.save(cache, args.output / "activations.pt")
        suite.save_report(args.output / "probe_results.json")
        print(
            f"\nProbe source={args.probe_source}  feature_mode={args.feature_mode}  "
            f"frames={probe_cache['n_frames']}  dim={probe_cache['features'].shape[-1]}"
        )
        print("\nEpisode-split validation R2:")
        for result in suite.results:
            extra = ""
            if "direction_mae_deg" in result.extra:
                extra = f", direction MAE={result.extra['direction_mae_deg']:.1f} deg"
            print(
                f"  {result.target:16s}  val_r2={result.val_r2:6.3f}  "
                f"val_rmse={result.val_rmse:.4f}{extra}"
            )
        if args.location_holdout:
            holdout = location_holdout_from_cache(
                probe_cache,
                axes=holdout_axes,
                threshold_mode=args.holdout_threshold,
            )
            (args.output / "location_holdout.json").write_text(json.dumps(holdout, indent=2))
            _print_location_holdout(holdout)
            print(f"\nWrote {args.output}/location_holdout.json")
        return suite

    if args.from_cache is not None:
        cache = torch.load(args.from_cache, map_location="cpu", weights_only=False)
        print(f"\nLoaded cache {args.from_cache}")
        if cache.get("hook_features"):
            print("Available hooks:", ", ".join(sorted(cache["hook_features"])))
        _run_probe_and_holdout(cache, save_activations=False)
        return

    if args.model_dir is None:
        parser.error("--model-dir is required unless --labels-only or --from-cache is set")

    model, base_dset, model_cfg, device = _load_model_and_dataset(
        args.model_dir, args.epoch, data_path
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
    print(f"\nCollected {cache['n_frames']} frames from {cache['n_rollouts']} rollouts")
    print(f"Readout: {args.readout}  feature_dim={cache['features'].shape[-1]}")
    if cache.get("hook_features"):
        print("Hook sites:", ", ".join(sorted(cache["hook_features"])))

    suite = _run_probe_and_holdout(cache, save_activations=True)

    if not args.skip_interventions:
        # Interventions always use the single-frame readout probe (not hook/diff).
        intervention_suite = suite
        if args.probe_source != "readout" or args.feature_mode != "raw":
            intervention_suite = train_probes_from_cache(
                select_probe_cache(
                    cache,
                    probe_source="readout",
                    feature_mode="raw",
                ),
                val_fraction=args.val_fraction,
                seed=args.seed,
            )
        intervention_report = smoke_test_interventions(
            model,
            base_dset,
            hook_manager,
            device=device,
            frameskip=frameskip,
            suite=intervention_suite,
        )
        (args.output / "intervention_smoke.json").write_text(json.dumps(intervention_report, indent=2))
        print("\nIntervention smoke test (mean |d probe pred| on 5 frames):")
        for site, deltas in intervention_report.items():
            direction_delta = deltas.get("direction_cos", 0.0) + deltas.get("direction_sin", 0.0)
            print(f"  {site:22s}  direction_d~={direction_delta:.4f}")

    print(f"\nWrote {args.output}/activations.pt and probe_results.json")
    if args.location_holdout:
        print(f"Wrote {args.output}/location_holdout.json")


if __name__ == "__main__":
    main()
