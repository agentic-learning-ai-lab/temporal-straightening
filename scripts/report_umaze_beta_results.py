"""Build a shareable summary of every UMaze straightening-penalty condition.

Pools chunked planning runs, pulls in the committed baseline artifacts, and
reports everything against the two coefficients that actually enter the loss:

    penalty = scale * (r0 + beta * r1)
        direction coefficient = scale
        speed coefficient     = scale * beta

Conditions are ordered by direction coefficient, because that is the axis the
results separate along. Coefficients are derived from each run's own
training.straighten token (read from its hydra.yaml), not hardcoded, so a
mislabelled directory cannot silently misreport what was trained.

Usage:
    python scripts/report_umaze_beta_results.py > umaze_beta_report.md
    python scripts/report_umaze_beta_results.py \
        --plan-root /path/plans/umaze_r3_beta_sweep:/path/plans/umaze_r3_beta_fixed \
        --ckpt-root /path/ckpt/umaze_r3_beta_sweep:/path/ckpt/umaze_r3_beta_fixed
"""
import argparse
import json
import os
import statistics as st
from pathlib import Path

METRICS = ["success_rate", "mean_state_dist", "mean_visual_dist", "mean_proprio_dist"]

# Committed reproductions, same planner/seeds/evals as the new runs.
BASELINE_SUMMARY = "baseline_artifacts/plans/umaze_trajectory_penalty_ablations/condition_summary.json"
BASELINE_COMPARISON = "baseline_artifacts/plans/umaze_trajectory_penalty_ablations/comparison.json"

# The two committed files name the same conditions differently
# (comparison.json uses r0/r1/r2/r3), so fold both onto one canonical name.
BASELINE_ALIASES = {
    "r0": "r0_direction_only",
    "r1": "r1_speed_only",
    "r2": "r2_full_matched",
    "r3": "r3_beta1",
}

# Tokens for the committed conditions, from docs/The Straightening Loss.md.
BASELINE_TOKENS = {
    "r0_direction_only": "aggcos1e-1",
    "r1_speed_only": "aggr1_1e-1",
    "r2_full_matched": "aggr2_5e-2",
    "r3_beta1": "aggr3b1_1e-1",
}


def token_coefficients(token):
    """(direction, speed) coefficients for a training.straighten token.

    r2 expands as scale*(r1 + 2*r0), so its direction coefficient is 2*scale.
    """
    if not token or token == "False":
        return 0.0, 0.0
    t = token[3:] if token.startswith("agg") else token
    if t.startswith("cos"):
        s = float(t[3:]) if t[3:] else 1.0
        return s, 0.0
    if t.startswith("speed"):
        s = float(t[5:]) if t[5:] else 1.0
        return 0.0, s
    if not t.startswith("r") or len(t) < 2:
        return None
    mode, rest = t[1], t[2:]
    if mode == "3":
        if not rest.startswith("b") or "_" not in rest:
            return None
        beta_text, scale_text = rest[1:].split("_", 1)
        beta, scale = float(beta_text), float(scale_text)
        return scale, scale * beta
    scale = float(rest.removeprefix("_")) if rest.removeprefix("_") else 1.0
    if mode == "0":
        return scale, 0.0
    if mode == "1":
        return 0.0, scale
    if mode == "2":
        return 2.0 * scale, scale
    return None


def read_final_eval(logs_json):
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
    return {k.removeprefix("final_eval/"): v for k, v in best.items()
            if k.startswith("final_eval/")}


def collect_planned(plan_root, ckpt_root):
    """Pool chunks within a seed, then seeds, for each arm under plan_root."""
    out = []
    plan_root, ckpt_root = Path(plan_root), Path(ckpt_root)
    if not plan_root.is_dir():
        return out
    for arm_dir in sorted(plan_root.iterdir()):
        if not arm_dir.is_dir():
            continue
        token = None
        hydra = ckpt_root / arm_dir.name / "hydra.yaml"
        if hydra.is_file():
            for line in hydra.read_text().splitlines():
                if "straighten:" in line:
                    token = line.split("straighten:", 1)[1].strip().strip("'\"")
                    break
        per_seed = {}
        for seed_dir in sorted(arm_dir.glob("plan_seed_*")):
            if not seed_dir.is_dir():
                continue
            chunks = [read_final_eval(c / "logs.json")
                      for c in sorted(seed_dir.glob("chunk_*"))
                      if (c / "logs.json").is_file()]
            chunks = [c for c in chunks if c]
            if not chunks:
                single = seed_dir / "logs.json"
                if single.is_file():
                    rec = read_final_eval(single)
                    if rec:
                        chunks = [rec]
            if chunks:
                per_seed[seed_dir.name] = {
                    m: st.mean([c[m] for c in chunks if m in c])
                    for m in METRICS if any(m in c for c in chunks)
                }
        if per_seed:
            out.append({"name": arm_dir.name, "token": token,
                        "per_seed": per_seed, "source": "new run"})
    return out


def collect_baselines(repo):
    out, seen = [], set()
    for path in (BASELINE_SUMMARY, BASELINE_COMPARISON):
        p = Path(repo) / path
        if not p.is_file():
            continue
        for raw, cond in json.load(open(p)).get("conditions", {}).items():
            name = BASELINE_ALIASES.get(raw, raw)
            if name in seen:
                continue
            seen.add(name)
            per_seed = {s: {k.removeprefix("final_eval/"): v for k, v in vals.items()}
                        for s, vals in cond.get("per_seed", {}).items()}
            if per_seed:
                out.append({"name": name, "token": BASELINE_TOKENS.get(name),
                            "per_seed": per_seed, "source": "committed"})
    return out


def agg(per_seed, metric):
    vals = [s[metric] for s in per_seed.values() if metric in s]
    if not vals:
        return None, None
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    user = os.environ.get("USER", "")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--plan-root",
                    default=f"/opt/dlami/nvme/{user}/plans/umaze_r3_beta_sweep:"
                            f"/opt/dlami/nvme/{user}/plans/umaze_r3_beta_fixed")
    ap.add_argument("--ckpt-root",
                    default=f"/opt/dlami/nvme/{user}/ckpt/umaze_r3_beta_sweep:"
                            f"/opt/dlami/nvme/{user}/ckpt/umaze_r3_beta_fixed")
    args = ap.parse_args()

    rows = collect_baselines(args.repo)
    plans, ckpts = args.plan_root.split(":"), args.ckpt_root.split(":")
    for i, plan_root in enumerate(plans):
        ckpt_root = ckpts[i] if i < len(ckpts) else ckpts[-1]
        rows += collect_planned(plan_root, ckpt_root)

    for r in rows:
        coef = token_coefficients(r["token"]) if r["token"] else None
        r["dir_coef"], r["speed_coef"] = coef if coef else (None, None)

    # Unknown coefficients sort last rather than crashing the comparison.
    rows.sort(key=lambda r: (r["dir_coef"] is None,
                             -(r["dir_coef"] or 0.0),
                             r["speed_coef"] or 0.0))

    print("# UMaze straightening-penalty results\n")
    print("All conditions: open-loop GD planning, `plan_gd.yaml`, seeds 100/200/300,")
    print("50 evaluations per seed, epoch-20 checkpoints.\n")
    print("The penalty is `scale * (r0 + beta * r1)`, so each condition is two")
    print("coefficients: **direction** (`scale`) and **speed** (`scale * beta`).")
    print("Rows are ordered by direction coefficient.\n")
    print("| condition | token | direction | speed | seeds | success rate | state dist | visual dist |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        sr, sr_sd = agg(r["per_seed"], "success_rate")
        sd, sd_sd = agg(r["per_seed"], "mean_state_dist")
        vd, _ = agg(r["per_seed"], "mean_visual_dist")
        d = f"{r['dir_coef']:.4f}" if r["dir_coef"] is not None else "?"
        s = f"{r['speed_coef']:.4f}" if r["speed_coef"] is not None else "?"
        print(f"| {r['name']} | `{r['token'] or '?'}` | {d} | {s} | {len(r['per_seed'])} | "
              f"{sr:.3f} ± {sr_sd:.3f} | {sd:.3f} ± {sd_sd:.3f} | {vd:.3f} |"
              if sr is not None and sd is not None and vd is not None else
              f"| {r['name']} | `{r['token'] or '?'}` | {d} | {s} | {len(r['per_seed'])} | n/a | n/a | n/a |")

    print("\n## Notes\n")
    print("- Error bars are population std across seeds, not across episodes.")
    print("- Every condition is a single training run, so seed spread reflects")
    print("  planning-seed variation only; training variation is unmeasured.")
    print("- `r2` expands to `scale * (r1 + 2*r0)`, so `aggr2_5e-2` carries a")
    print("  direction coefficient of 0.1 -- matched to `r0` by construction.")
    print("- Mean state distance is not monotone in success rate at the extremes:")
    print("  a model that rarely reaches the goal can still stop at a middling")
    print("  average distance, so success and distance can disagree.")


if __name__ == "__main__":
    main()
