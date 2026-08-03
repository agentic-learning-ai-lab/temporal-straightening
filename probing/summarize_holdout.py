#!/usr/bin/env python3
"""Collapse per-run location_holdout.json files into one summary table/JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGETS = ["position_x", "position_y", "speed", "direction_cos", "direction_sin"]


def summarize_run(path: Path) -> dict:
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    condition = path.parent.name
    experiment = path.parent.parent.name
    row = {
        "experiment": experiment,
        "condition": condition,
        "run": f"{experiment}/{condition}",
        "path": str(path),
        "threshold_mode": data.get("threshold_mode"),
        "n_splits": len(data.get("splits", [])),
        "targets": {},
    }
    for target in TARGETS:
        stats = summary.get(target, {})
        row["targets"][target] = {
            "mean_heldout_r2": stats.get("mean_heldout_r2"),
            "min_heldout_r2": stats.get("min_heldout_r2"),
            "n_splits": stats.get("n_splits"),
            "position_baseline_mean_heldout_r2": stats.get(
                "position_baseline_mean_heldout_r2"
            ),
        }
    return row


def print_table(rows: list[dict]) -> None:
    print(f"{'run':48s}  {'speed':>8s}  {'pos_base':>8s}  {'pos_x':>8s}  {'pos_y':>8s}")
    for row in rows:
        t = row["targets"]
        speed = t["speed"].get("mean_heldout_r2")
        base = t["speed"].get("position_baseline_mean_heldout_r2")
        px = t["position_x"].get("mean_heldout_r2")
        py = t["position_y"].get("mean_heldout_r2")
        print(
            f"{row['run']:48s}  "
            f"{speed if speed is not None else float('nan'):8.3f}  "
            f"{base if base is not None else float('nan'):8.3f}  "
            f"{px if px is not None else float('nan'):8.3f}  "
            f"{py if py is not None else float('nan'):8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize location-holdout probe runs")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("probing"),
        help="Root to search (finds */*/location_holdout.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path (default: <root>/holdout_summary.json)",
    )
    args = parser.parse_args()

    paths = sorted(args.root.glob("*/**/location_holdout.json"))
    if not paths:
        paths = sorted(args.root.glob("*/location_holdout.json"))
    if not paths:
        raise SystemExit(f"No location_holdout.json under {args.root}")

    rows = [summarize_run(p) for p in paths]
    out_path = args.output or (args.root / "holdout_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": rows}, indent=2))
    print_table(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
