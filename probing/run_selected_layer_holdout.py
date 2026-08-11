#!/usr/bin/env python3
"""Region holdout on selected-layer tensors (dino/6 + predictor/1).

Features match ``layer_probes/selected_layers.json``:
  - dino/6/pooled_patches  (raw and/or Δ across consecutive window frames)
  - predictor/1/pooled_visual (Δ, same as the layer-probe speed/direction setup)

Split is median maze-half location holdout (mentor region check).

Examples
--------
# One condition (GPU):
python probing/run_selected_layer_holdout.py \\
  --checkpoint baseline_artifacts/checkpoints/umaze_physics_layer_ablations/r2_full_matched \\
  --data-dir $DATASET_DIR/point_maze \\
  --output-root probing/selected_layer_holdout

# All five ablation conditions:
for cond in r0_direction_only r2_full_matched calibrated_speed factorized layer_aware_factorized; do
  python probing/run_selected_layer_holdout.py \\
    --checkpoint baseline_artifacts/checkpoints/umaze_physics_layer_ablations/$cond \\
    --data-dir $DATASET_DIR/point_maze \\
    --output-root probing/selected_layer_holdout
done

python probing/summarize_holdout.py --root probing/selected_layer_holdout
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

from datasets.img_transforms import default_transform
from datasets.point_maze_dset import PointMazeDataset
from probing.selected_layer_features import (
    DEFAULT_DINO_LAYER,
    DEFAULT_PRED_LAYER,
    collect_selected_activations,
    load_checkpoint,
    pack_probe_matrix,
    sample_windows,
)
from probing.linear_probe import run_location_holdout


PROBES = (
    ("dino_pooled_patches_raw", "dino", False),
    ("dino_pooled_patches_diff", "dino", True),
    ("predictor_pooled_visual_diff", "predictor", True),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("probing/selected_layer_holdout"))
    p.add_argument("--condition-name", type=str, default=None, help="defaults to checkpoint directory name")
    p.add_argument("--max-windows", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--ridge", type=float, default=10.0, help="Daniel default ridge α")
    p.add_argument("--dino-layer", type=int, default=DEFAULT_DINO_LAYER)
    p.add_argument("--predictor-layer", type=int, default=DEFAULT_PRED_LAYER)
    p.add_argument(
        "--probes",
        nargs="+",
        default=[name for name, _, _ in PROBES],
        choices=[name for name, _, _ in PROBES],
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    condition = args.condition_name or args.checkpoint.name
    print(f"Loading {args.checkpoint} on {device}", flush=True)
    modules = load_checkpoint(args.checkpoint, device)

    use_frame_files = (args.data_dir / "obses" / "episode_000_frame_000.pth").exists()
    dataset = PointMazeDataset(
        data_path=str(args.data_dir),
        transform=default_transform(224),
        normalize_action=True,
        use_frame_files=use_frame_files,
    )
    choices = sample_windows(
        dataset, args.max_windows, args.frameskip, args.num_frames, args.seed
    )
    print(f"Sampled {len(choices)} windows (frameskip={args.frameskip}, nframes={args.num_frames})", flush=True)

    reps, states, _actions = collect_selected_activations(
        modules,
        dataset,
        choices,
        batch_size=args.batch_size,
        frameskip=args.frameskip,
        nframes=args.num_frames,
        device=device,
        dino_layer=args.dino_layer,
        predictor_layer=args.predictor_layer,
    )
    dino_key = f"dino/{args.dino_layer}/pooled_patches"
    pred_key = f"predictor/{args.predictor_layer}/pooled_visual"
    key_by_family = {"dino": dino_key, "predictor": pred_key}
    for key in key_by_family.values():
        if key not in reps:
            raise KeyError(f"missing {key}; have {sorted(reps)}")

    meta = {
        "condition": condition,
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "max_windows": args.max_windows,
        "n_windows": len(choices),
        "frameskip": args.frameskip,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "ridge": args.ridge,
        "dino_layer": args.dino_layer,
        "dino_representation": "pooled_patches",
        "predictor_layer": args.predictor_layer,
        "predictor_representation": "pooled_visual",
        "split": "location_median_halves",
        "feature_source": "selected_layers",
    }

    for probe_name, family, transition in PROBES:
        if probe_name not in args.probes:
            continue
        rep = reps[key_by_family[family]]
        features, labels = pack_probe_matrix(rep, states, transition=transition)
        report = run_location_holdout(features, labels, alpha=args.ridge)
        report["meta"] = {
            **meta,
            "probe": probe_name,
            "representation": key_by_family[family],
            "transition": transition,
            "n_rows": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
        }
        out_dir = args.output_root / probe_name / condition
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "location_holdout.json").write_text(json.dumps(report, indent=2))
        speed = report["summary"].get("speed", {})
        print(
            f"{probe_name}/{condition}: speed mean_heldout_r2="
            f"{speed.get('mean_heldout_r2')}  "
            f"pos_base={speed.get('position_baseline_mean_heldout_r2')}  "
            f"n={features.shape[0]} d={features.shape[1]}",
            flush=True,
        )

    # Lightweight numpy cache for re-summarizing without GPU.
    cache_dir = args.output_root / "_cache" / condition
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / "windows.npz",
        states=states,
        **{k.replace("/", "__"): v for k, v in reps.items()},
    )
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote holdouts under {args.output_root} and cache {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
