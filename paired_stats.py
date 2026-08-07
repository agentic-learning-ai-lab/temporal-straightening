#!/usr/bin/env python3
"""Paired per-episode statistics for the R0-R3 trajectory-penalty ablations.

Every condition within an environment is planned on the same evaluation
episodes (same sampling seeds, same eval_start_index offsets), so the R1/R2/R3
vs R0 comparison is paired at the episode level rather than only at the
seed level. This script recovers the per-episode success vector from the
planner logs and reports an exact two-sided McNemar test, which is a much
tighter statement than "the deficit is consistent in sign across three seeds".

Per-episode outcomes live in different places per environment:
  PushT       pusht_ablations/<cond>/plan_logs/gd_seed<S>_chunk<OO>.log
  Wall/UMaze  baseline_artifacts/plans/.../<cond>/seed_<S>/chunk_<O>/runner.log

UMaze R0 is imported from an unchunked 50-evaluation reproduction run that
kept only aggregate metrics, so no per-episode vector exists for it and the
UMaze rows are reported without a McNemar test.

Run:  python paired_stats.py
"""
from __future__ import annotations

import math
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (100, 200, 300)
PLANS = os.path.join("baseline_artifacts", "plans")

SUCCESS_RE = re.compile(r"'success':\s*array\(\[(.*?)\]\)", re.S)
STATE_RE = re.compile(r"'state_dist':\s*array\(\[(.*?)\]", re.S)
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?")

# (env label, condition -> log-directory template keyed by seed/offset)
ENVS = {
    "PushT": {
        "chunks": ("00", "10", "20", "30", "40"),
        "path": lambda base, s, o: os.path.join(
            HERE, base, "plan_logs", f"gd_seed{s}_chunk{o}.log"
        ),
        "conds": {
            "R0": "pusht_ablations/r0_direction_only",
            "R1": "pusht_ablations/r1_speed_only",
            "R2": "pusht_ablations/r2_full_matched",
            "R3": "pusht_ablations/r3_beta1",
        },
    },
    "Wall": {
        "chunks": ("0", "10", "20", "30", "40"),
        "path": lambda base, s, o: os.path.join(
            HERE, base, f"seed_{s}", f"chunk_{o}", "runner.log"
        ),
        "conds": {
            "R0": f"{PLANS}/wall_dino_projector_full/on",
            "R1": f"{PLANS}/wall_trajectory_penalty_ablations/r1_speed_only",
            "R2": f"{PLANS}/wall_trajectory_penalty_ablations/r2_full_matched",
            "R3": f"{PLANS}/wall_trajectory_penalty_ablations/r3_beta1",
        },
    },
    "UMaze": {
        "chunks": ("0", "10", "20", "30", "40"),
        "path": lambda base, s, o: os.path.join(
            HERE, base, f"seed_{s}", f"chunk_{o}", "runner.log"
        ),
        "conds": {
            "R0": f"{PLANS}/umaze_trajectory_penalty_ablations/r0_direction_only",
            "R1": f"{PLANS}/umaze_trajectory_penalty_ablations/r1_speed_only",
            "R2": f"{PLANS}/umaze_trajectory_penalty_ablations/r2_full_matched",
            "R3": f"{PLANS}/umaze_trajectory_penalty_ablations/r3_beta1",
        },
    },
}


def read_episodes(env: dict, base: str):
    """Return (success bools, state distances, eval-seed signature) or None."""
    success, state, signature = [], [], []
    for seed in SEEDS:
        for offset in env["chunks"]:
            path = env["path"](base, seed, offset)
            if not os.path.isfile(path):
                return None
            text = open(path, errors="ignore").read()
            hits = SUCCESS_RE.findall(text)
            if not hits:
                return None
            success += [tok == "True" for tok in re.findall(r"True|False", hits[-1])]
            dists = STATE_RE.findall(text)
            if dists:
                state += [float(x) for x in NUM_RE.findall(dists[-1])]
            seeds = re.search(r"eval_seed:\s*\[([^\]]*)\]", text)
            signature.append(seeds.group(1).strip() if seeds else "")
    return success, state, tuple(signature)


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs only."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_t(delta: list[float]) -> tuple[float, float, float]:
    mean = st.mean(delta)
    se = st.stdev(delta) / math.sqrt(len(delta))
    return mean, se, mean / se if se else float("nan")


def main() -> None:
    for name, env in ENVS.items():
        print(f"===== {name} =====")
        loaded = {c: read_episodes(env, b) for c, b in env["conds"].items()}
        available = {c: v for c, v in loaded.items() if v is not None}
        for cond in env["conds"]:
            if cond not in available:
                print(f"  {cond}: per-episode outcomes unavailable (aggregate metrics only)")

        signatures = {v[2] for v in available.values() if any(v[2])}
        print(f"  evaluation episodes identical across conditions: {len(signatures) <= 1}")

        if "R0" not in available:
            print("  no R0 per-episode vector; skipping paired tests\n")
            continue

        r0_success, r0_state, _ = available["R0"]
        n = len(r0_success)
        print(f"  R0  {sum(r0_success)}/{n} = {100 * sum(r0_success) / n:.2f}%"
              + (f"   mean state dist {st.mean(r0_state):.3f}" if r0_state else ""))

        for cond in ("R1", "R2", "R3"):
            if cond not in available:
                continue
            success, state, _ = available[cond]
            b = sum(1 for a, x in zip(r0_success, success) if a and not x)
            c = sum(1 for a, x in zip(r0_success, success) if not a and x)
            delta = 100 * (sum(success) - sum(r0_success)) / n
            line = (f"  {cond}  {sum(success)}/{n} = {100 * sum(success) / n:.2f}%"
                    f"   {delta:+.2f} pp   discordant {b}/{c}"
                    f"   exact McNemar p={exact_mcnemar(b, c):.3g}")
            if state and r0_state:
                mean, se, t = paired_t([x - a for x, a in zip(state, r0_state)])
                line += (f"\n        state dist {st.mean(state):.3f}"
                         f"   paired {mean:+.3f} +/- {se:.3f}"
                         f"  ({100 * mean / st.mean(r0_state):+.1f}%, t={t:+.2f})")
            print(line)
        print()


if __name__ == "__main__":
    main()
