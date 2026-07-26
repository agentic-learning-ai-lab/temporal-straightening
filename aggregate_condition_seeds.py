#!/usr/bin/env python3
"""Aggregate completed per-seed planning summaries across conditions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


def parse_condition(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("condition must have the form NAME=PATH") from exc
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must have the form NAME=PATH")
    return name, Path(path)


def load_seed_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise ValueError(f"seed summary is not complete: {path}")
    final_metrics = data.get("metrics", {}).get("final_eval")
    if not isinstance(final_metrics, dict) or not final_metrics:
        raise ValueError(f"seed summary has no final_eval metrics: {path}")
    return data


def aggregate_condition(path: Path) -> dict[str, Any]:
    seed_files = sorted(path.glob("seed_*/aggregate.json"))
    if not seed_files:
        raise FileNotFoundError(f"no seed_*/aggregate.json files under {path}")

    summaries = [load_seed_summary(seed_file) for seed_file in seed_files]
    seeds = [int(summary["seed"]) for summary in summaries]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds under {path}: {seeds}")

    metric_keys = set(summaries[0]["metrics"]["final_eval"])
    for summary in summaries[1:]:
        if set(summary["metrics"]["final_eval"]) != metric_keys:
            raise ValueError(f"seed metric keys do not match under {path}")

    aggregate: dict[str, dict[str, float]] = {}
    for key in sorted(metric_keys):
        values = [float(summary["metrics"]["final_eval"][key]) for summary in summaries]
        aggregate[key] = {
            "mean": fmean(values),
            "population_std": pstdev(values),
        }

    return {
        "status": "complete",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "n_evals_per_seed": [int(summary["n_evals"]) for summary in summaries],
        "seed_summaries": [str(seed_file) for seed_file in seed_files],
        "per_seed": {
            str(summary["seed"]): summary["metrics"]["final_eval"]
            for summary in summaries
        },
        "aggregate": aggregate,
    }


def paired_delta(
    baseline: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, dict[str, float]]:
    if baseline["seeds"] != treatment["seeds"]:
        raise ValueError("paired conditions must contain the same ordered seeds")
    keys = set(baseline["aggregate"])
    if set(treatment["aggregate"]) != keys:
        raise ValueError("paired conditions must contain the same metric keys")

    result: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        deltas = [
            float(treatment["per_seed"][str(seed)][key])
            - float(baseline["per_seed"][str(seed)][key])
            for seed in baseline["seeds"]
        ]
        result[key] = {
            "mean": fmean(deltas),
            "population_std": pstdev(deltas),
            "standard_error": pstdev(deltas) / math.sqrt(len(deltas)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        type=parse_condition,
        metavar="NAME=PATH",
    )
    parser.add_argument("--baseline")
    parser.add_argument(
        "--treatment",
        action="append",
        help=(
            "condition to compare with --baseline; may be repeated. If omitted "
            "while --baseline is set, every non-baseline condition is compared."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conditions = dict(args.condition)
    if len(conditions) != len(args.condition):
        raise ValueError("condition names must be unique")
    result: dict[str, Any] = {
        "status": "complete",
        "conditions": {
            name: aggregate_condition(path) for name, path in conditions.items()
        },
    }
    treatments = args.treatment or []
    if treatments and not args.baseline:
        raise ValueError("--treatment requires --baseline")
    if args.baseline:
        if args.baseline not in conditions:
            raise ValueError("--baseline must match a --condition name")
        if not treatments:
            treatments = [name for name in conditions if name != args.baseline]
        if len(treatments) != len(set(treatments)):
            raise ValueError("--treatment names must be unique")
        if args.baseline in treatments or any(
            treatment not in conditions for treatment in treatments
        ):
            raise ValueError("paired condition names must match --condition names")

        result["paired_deltas"] = {}
        for treatment in treatments:
            comparison = {
                "treatment_minus_baseline": f"{treatment}-{args.baseline}",
                "metrics": paired_delta(
                    result["conditions"][args.baseline],
                    result["conditions"][treatment],
                ),
            }
            result["paired_deltas"][treatment] = comparison

        # Preserve the original single-comparison field for callers that pass
        # exactly one treatment.
        if len(treatments) == 1:
            result["paired_delta"] = result["paired_deltas"][treatments[0]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
