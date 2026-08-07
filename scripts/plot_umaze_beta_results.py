"""Plot UMaze planning success against the two penalty coefficients.

Two panels, one shared success axis, because the finding is a contrast:

  left  - budget-matched arms (direction + speed always sum to 0.1):
          success rises smoothly with the direction coefficient
  right - fixed-scale arms (direction pinned at 0.1, speed varying):
          success is flat, so the speed coefficient carries no signal

Reuses the collection logic in report_umaze_beta_results.py, so both the
table and the figure derive coefficients from the same training tokens.

Usage:
    python scripts/plot_umaze_beta_results.py
    python scripts/plot_umaze_beta_results.py --out pngs/umaze_beta_curve.png
"""
import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "report_umaze", HERE / "report_umaze_beta_results.py")
_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_report)

# Categorical slots 1 and 2 of the validated default palette.
SERIES_1 = "#2a78d6"   # blue  - budget-matched
SERIES_2 = "#eb6834"   # orange - fixed scale
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"

BUDGET_TOTAL = 0.1
TOL = 1e-6


def collect(repo, plan_roots, ckpt_roots):
    rows = _report.collect_baselines(repo)
    for i, plan_root in enumerate(plan_roots):
        ckpt_root = ckpt_roots[i] if i < len(ckpt_roots) else ckpt_roots[-1]
        rows += _report.collect_planned(plan_root, ckpt_root)
    out = []
    for r in rows:
        coef = _report.token_coefficients(r["token"]) if r["token"] else None
        if not coef:
            continue
        sr, sd = _report.agg(r["per_seed"], "success_rate")
        if sr is None:
            continue
        out.append({"name": r["name"], "dir": coef[0], "speed": coef[1],
                    "success": sr, "err": sd or 0.0})
    return out


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
    ap.add_argument("--out", default="pngs/umaze_beta_curve.png")
    args = ap.parse_args()

    rows = collect(args.repo, args.plan_root.split(":"), args.ckpt_root.split(":"))
    if not rows:
        raise SystemExit("No conditions with derivable coefficients found.")

    # Budget-matched: the two coefficients sum to the fixed budget. That set
    # includes r0 and r1, which sit at the ends of the same 0.1 budget.
    budget = sorted((r for r in rows if abs(r["dir"] + r["speed"] - BUDGET_TOTAL) < TOL),
                    key=lambda r: r["dir"])
    fixed = sorted((r for r in rows if abs(r["dir"] - BUDGET_TOTAL) < TOL),
                   key=lambda r: r["speed"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.set_ylim(0, 1.05)

    # Left: the curve. Success against direction weight, budget held constant.
    xs = [r["dir"] for r in budget]
    ys = [r["success"] for r in budget]
    es = [r["err"] for r in budget]
    ax1.plot(xs, ys, "-", color=SERIES_1, linewidth=2, zorder=2)
    ax1.errorbar(xs, ys, yerr=es, fmt="o", color=SERIES_1, markersize=8,
                 markeredgecolor=SURFACE, markeredgewidth=2,
                 ecolor=SERIES_1, elinewidth=1.5, capsize=3, zorder=3)
    for r in budget:  # label the ends only; a number on every point is noise
        if r is budget[0] or r is budget[-1]:
            ax1.annotate(f"{r['success']:.2f}", (r["dir"], r["success"]),
                         textcoords="offset points", xytext=(0, 12),
                         ha="center", fontsize=9, color=INK)
    ax1.set_xlabel("direction coefficient", fontsize=10, color=INK_MUTED)
    ax1.set_ylabel("planning success rate", fontsize=10, color=INK_MUTED)
    ax1.set_title("Budget held at 0.1: success tracks direction",
                  fontsize=11, color=INK, pad=10, loc="left")

    # Right: the non-relationship. Direction pinned, speed swept.
    xs2 = [r["speed"] for r in fixed]
    ys2 = [r["success"] for r in fixed]
    es2 = [r["err"] for r in fixed]
    ax2.plot(xs2, ys2, "-", color=SERIES_2, linewidth=2, zorder=2)
    ax2.errorbar(xs2, ys2, yerr=es2, fmt="o", color=SERIES_2, markersize=8,
                 markeredgecolor=SURFACE, markeredgewidth=2,
                 ecolor=SERIES_2, elinewidth=1.5, capsize=3, zorder=3)
    for r in (fixed[0], fixed[-1]) if len(fixed) > 1 else fixed:
        ax2.annotate(f"{r['success']:.2f}", (r["speed"], r["success"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color=INK)
    ax2.set_xlabel("speed coefficient", fontsize=10, color=INK_MUTED)
    ax2.set_title("Direction pinned at 0.1: speed changes nothing",
                  fontsize=11, color=INK, pad=10, loc="left")

    fig.suptitle("UMaze planning success vs. penalty coefficients",
                 fontsize=13, color=INK, x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.015,
             f"3 planning seeds x 50 evaluations; bars are spread across seeds.  "
             f"Left: {len(budget)} conditions.  Right: {len(fixed)} conditions.",
             fontsize=8, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"  budget-matched ({len(budget)}): "
          + ", ".join(f"dir={r['dir']:.4f}->{r['success']:.3f}" for r in budget))
    print(f"  fixed-scale ({len(fixed)}): "
          + ", ".join(f"speed={r['speed']:.4f}->{r['success']:.3f}" for r in fixed))


if __name__ == "__main__":
    main()
