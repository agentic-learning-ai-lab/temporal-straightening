"""Pool chunked planning results for the R3 beta sweep into one table.

Planning runs in chunks (see run_umaze_r3_beta_planning.sh), so a seed's
metrics are the equal-weighted mean over its chunks, and an arm's metrics are
the mean over seeds with the spread across seeds as the error bar.

Usage:
    python scripts/summarize_umaze_r3_beta_planning.py \
        [--plan-root /opt/dlami/nvme/$USER/plans/umaze_r3_beta_sweep]
"""
import argparse
import json
import os
import re
import statistics as st
from pathlib import Path

# Directory names are r3_beta<value> with dots written as 'p'.
BETA_RE = re.compile(r"^r3_beta(.+)$")
METRICS = [
    "success_rate",
    "mean_state_dist",
    "mean_visual_dist",
    "mean_proprio_dist",
]


def parse_beta(name):
    m = BETA_RE.match(name)
    if not m:
        return None
    try:
        return float(m.group(1).replace("p", "."))
    except ValueError:
        return None


def read_final_eval(logs_json):
    """Last final_eval record in a chunk's logs.json (one JSON object per line)."""
    best = None
    with open(logs_json) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(k.startswith("final_eval/") for k in rec):
                best = rec
    if best is None:
        return None
    return {
        k.removeprefix("final_eval/"): v
        for k, v in best.items()
        if k.startswith("final_eval/")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plan-root",
        default=f"/opt/dlami/nvme/{os.environ.get('USER','')}/plans/umaze_r3_beta_sweep",
    )
    args = ap.parse_args()
    root = Path(args.plan_root)
    if not root.is_dir():
        raise SystemExit(f"No such plan root: {root}")

    rows = []
    for arm_dir in sorted(root.iterdir()):
        if not arm_dir.is_dir():
            continue
        beta = parse_beta(arm_dir.name)
        per_seed = {}
        for seed_dir in sorted(arm_dir.glob("plan_seed_*")):
            if not seed_dir.is_dir():
                continue
            chunks = []
            for chunk_dir in sorted(seed_dir.glob("chunk_*")):
                logs_json = chunk_dir / "logs.json"
                if logs_json.is_file():
                    rec = read_final_eval(logs_json)
                    if rec:
                        chunks.append(rec)
            if chunks:
                # Equal-weighted: every chunk evaluates the same number of episodes.
                per_seed[seed_dir.name] = {
                    m: st.mean([c[m] for c in chunks if m in c])
                    for m in METRICS
                    if any(m in c for c in chunks)
                }
        if per_seed:
            rows.append((beta, arm_dir.name, per_seed))

    if not rows:
        raise SystemExit(f"No completed chunks found under {root}")

    rows.sort(key=lambda r: (r[0] is None, r[0]))

    w = 0.1  # total penalty budget, for the mixing-weight column
    print(f"{'beta':>8} {'w(speed)':>9} {'seeds':>6} "
          f"{'success_rate':>22} {'state_dist':>22} {'visual_dist':>14}")
    print("-" * 88)
    for beta, name, per_seed in rows:
        n = len(per_seed)

        def agg(metric):
            vals = [s[metric] for s in per_seed.values() if metric in s]
            if not vals:
                return "n/a", None
            mean = st.mean(vals)
            spread = st.stdev(vals) if len(vals) > 1 else 0.0
            return f"{mean:.4f} +/- {spread:.4f}", mean

        sr, _ = agg("success_rate")
        sd, _ = agg("mean_state_dist")
        vd, _ = agg("mean_visual_dist")
        mix = f"{beta/(1+beta):.3f}" if beta not in (None,) else "?"
        print(f"{beta if beta is not None else '?':>8} {mix:>9} {n:>6} "
              f"{sr:>22} {sd:>22} {vd:>14}")

    print()
    print("w(speed) = beta/(1+beta): fraction of the fixed penalty budget on the")
    print("speed-constancy term. w=0 is r0_direction_only, w=1 is r1_speed_only.")
    print("+/- is the spread across seeds, not across episodes.")


if __name__ == "__main__":
    main()
