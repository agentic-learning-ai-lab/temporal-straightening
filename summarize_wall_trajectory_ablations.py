#!/usr/bin/env python3
"""Render a trajectory-penalty ablation report from validated artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional


CONDITIONS = {
    "r0": {
        "label": "R0 direction-only baseline",
        "objective": "0.1 × R0",
        "checkpoint_dir": None,
    },
    "r1": {
        "label": "R1 speed-only",
        "objective": "0.1 × R1",
        "checkpoint_dir": "r1_speed_only",
    },
    "r2": {
        "label": "R2 full penalty",
        "objective": "0.05 × R2",
        "checkpoint_dir": "r2_full_matched",
    },
    "r3": {
        "label": "R3 direction + speed",
        "objective": "0.1 × (R0 + R1)",
        "checkpoint_dir": "r3_beta1",
    },
}

FINAL_LOSS_RE = re.compile(
    r"Epoch\s+20\s+Training loss:\s*([0-9.eE+-]+)\s+"
    r"Validation loss:\s*([0-9.eE+-]+)"
)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise ValueError(f"comparison is not complete: {path}")
    missing = set(CONDITIONS) - set(data.get("conditions", {}))
    if missing:
        raise ValueError(f"comparison is missing conditions: {sorted(missing)}")
    return data


def final_losses(path: Path) -> Optional[tuple[float, float]]:
    if not path.is_file():
        return None
    matches = FINAL_LOSS_RE.findall(path.read_text(encoding="utf-8"))
    if not matches:
        return None
    training, validation = matches[-1]
    return float(training), float(validation)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def render_report(
    comparison: dict[str, Any],
    checkpoint_root: Path,
    r0_train_log: Optional[Path] = None,
    environment: str = "Wall",
) -> str:
    seeds = comparison["conditions"]["r0"]["seeds"]
    seed_headers = " | ".join(f"Seed {seed}" for seed in seeds)
    separator = " | ".join("---:" for _ in seeds)
    lines = [
        f"# {environment} trajectory-penalty ablation",
        "",
        "This study compares speed-sensitive latent trajectory penalties with the",
        f"existing direction-only {environment} baseline. Every condition uses DINOv2 patch",
        "features, the learned channel projector, and the epoch-20 checkpoint.",
        "",
        "## Objectives",
        "",
        "- `R0 = 1 - cos(theta)`",
        "- `R1 = r + 1/r - 2`",
        "- `R2 = R1 + 2*R0`",
        "- `R3 = R0 + R1`",
        "",
        "R2 uses coefficient 0.05 so its effective direction coefficient is 0.1,",
        "matching the R0 baseline. R1 and R3 use coefficient 0.1.",
        "",
        "## Planning results",
        "",
        f"| Condition | Objective | {seed_headers} | Mean | Population SD | Delta vs R0 |",
        f"|---|---|{separator}|---:|---:|---:|",
    ]

    paired = comparison.get("paired_deltas", {})
    for name, metadata in CONDITIONS.items():
        condition = comparison["conditions"][name]
        metric = "final_eval/success_rate"
        per_seed = [pct(float(condition["per_seed"][str(seed)][metric])) for seed in seeds]
        aggregate = condition["aggregate"][metric]
        delta = "baseline" if name == "r0" else pp(
            float(paired[name]["metrics"][metric]["mean"])
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    metadata["label"],
                    f"`{metadata['objective']}`",
                    *per_seed,
                    pct(float(aggregate["mean"])),
                    f"{100.0 * float(aggregate['population_std']):.2f} pp",
                    delta,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Planning uses seeds 100, 200, and 300. Each seed contains 50",
            "evaluations split into five deterministic 10-evaluation chunks, for",
            "150 evaluations per condition.",
            "",
            "## Final training losses",
            "",
            "| Condition | Training loss | Validation loss |",
            "|---|---:|---:|",
        ]
    )
    for name in CONDITIONS:
        metadata = CONDITIONS[name]
        if name == "r0":
            log_path = r0_train_log or Path("__missing_r0_train_log__")
        else:
            log_path = checkpoint_root / str(metadata["checkpoint_dir"]) / "train.log"
        losses = final_losses(log_path)
        if losses is None:
            train_text, validation_text = "unavailable", "unavailable"
        else:
            train_text, validation_text = (f"{losses[0]:.4f}", f"{losses[1]:.4f}")
        lines.append(
            f"| {metadata['label']} | {train_text} | {validation_text} |"
        )

    lines.extend(
        [
            "",
            "## Artifact layout",
            "",
            "- `r1_speed_only/`, `r2_full_matched/`, and `r3_beta1/`: raw planner chunks",
            "- `seed_*/aggregate.json`: validated 50-evaluation seed summaries",
            "- `comparison.json`: aggregate metrics and paired deltas against R0",
            "- `comparison.stdout`: human-readable aggregation output",
            "",
            "## Validation",
            "",
            "- All epoch-20 checkpoints are non-empty.",
            "- Every planner chunk contains `final_eval/success_rate`.",
            "- Every seed aggregate has `status: complete` and `n_evals: 50`.",
            "- Paired comparisons use identical planning seeds across conditions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--r0-train-log", type=Path)
    parser.add_argument("--environment", default="Wall")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = render_report(
        load_json(args.comparison),
        args.checkpoint_root,
        args.r0_train_log,
        args.environment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
