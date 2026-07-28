#!/usr/bin/env python3
"""Aggregate deterministic planning chunks into one full-run result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MEAN_SUFFIXES = (
    "/success_rate",
    "/mean_state_dist",
    "/mean_visual_dist",
    "/mean_proprio_dist",
)
NORM_SUFFIXES = (
    "/mean_div_visual_emb",
    "/mean_div_proprio_emb",
)


def parse_chunk(value: str) -> tuple[int, int, Path]:
    try:
        offset_text, count_text, path_text = value.split(":", 2)
        offset, count = int(offset_text), int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "chunk must have the form OFFSET:COUNT:PATH"
        ) from exc
    if offset < 0 or count <= 0:
        raise argparse.ArgumentTypeError("OFFSET must be >= 0 and COUNT must be > 0")
    return offset, count, Path(path_text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "logs.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f"planning log is empty: {path}")
    return entries


def record_name(record: dict[str, Any]) -> str:
    prefixes = {key.split("/", 1)[0] for key in record if "/" in key}
    if len(prefixes) != 1:
        raise ValueError(f"record has ambiguous metric prefixes: {sorted(prefixes)}")
    return prefixes.pop()


def aggregate_records(
    records: list[tuple[int, dict[str, Any]]], expected_evals: int
) -> dict[str, float]:
    if sum(count for count, _ in records) != expected_evals:
        raise ValueError("record counts do not add up to --expected-evals")
    metric_keys = set(records[0][1]) - {"step"}
    for _, record in records[1:]:
        if set(record) - {"step"} != metric_keys:
            raise ValueError("chunk metric keys do not match")

    result: dict[str, float] = {}
    for key in sorted(metric_keys):
        values = [(count, float(record[key])) for count, record in records]
        if key.endswith(MEAN_SUFFIXES):
            result[key] = sum(count * value for count, value in values) / expected_evals
        elif key.endswith(NORM_SUFFIXES):
            # Evaluator._compute_rollout_metrics uses torch.norm over the full
            # batch for these fields, despite their historical ``mean_`` names.
            result[key] = math.sqrt(sum(value * value for _, value in values))
        else:
            raise ValueError(f"unknown aggregation rule for metric: {key}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk",
        action="append",
        required=True,
        type=parse_chunk,
        metavar="OFFSET:COUNT:PATH",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--expected-evals", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chunks = sorted(args.chunk)
    cursor = 0
    by_record: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    sources = []
    for offset, count, path in chunks:
        if offset != cursor:
            raise ValueError(f"expected chunk offset {cursor}, got {offset}")
        entries = read_jsonl(path)
        names_seen: set[str] = set()
        for entry in entries:
            name = record_name(entry)
            if name in names_seen:
                raise ValueError(f"duplicate {name!r} record in {path}")
            names_seen.add(name)
            by_record.setdefault(name, []).append((count, entry))
        sources.append(
            {
                "eval_start_index": offset,
                "n_evals": count,
                "path": str(path),
                "first_seed": args.seed * offset + 1,
                "last_seed": args.seed * (offset + count - 1) + 1,
            }
        )
        cursor += count

    if cursor != args.expected_evals:
        raise ValueError(f"chunks cover {cursor} evaluations, expected {args.expected_evals}")
    record_names = set(by_record)
    if "final_eval" not in record_names:
        raise ValueError("chunks do not contain final_eval records")
    if any(len(records) != len(chunks) for records in by_record.values()):
        raise ValueError("record types are not present in every chunk")

    result = {
        "status": "complete",
        "seed": args.seed,
        "n_evals": args.expected_evals,
        "eval_seeds": [args.seed * n + 1 for n in range(args.expected_evals)],
        "chunks": sources,
        "metrics": {
            name: aggregate_records(records, args.expected_evals)
            for name, records in sorted(by_record.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
