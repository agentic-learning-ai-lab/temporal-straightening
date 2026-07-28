#!/usr/bin/env bash
# run_row6_table5_probe.sh — re-plan the EXISTING row 6 checkpoint under the
#   repo's harder GD settings, to test whether Table 5's higher patch-384 number
#   is reachable by stronger gradient-based optimization.
#
#   DINOv2 (patch) | 14x14x384 | Lcurv OFF | PointMaze-UMaze | open-loop GD
#   NO RETRAINING. Reuses the same checkpoint run_row6_a100.sh trained.
#
# ── WHY THIS RUN ─────────────────────────────────────────────────────────────
#   The patch-384 UMaze cell is the ONLY cell that disagrees between Table 1 and
#   Table 5 (35.33 vs 63.33). Your documented-recipe reproduction landed at
#   ~39.2% over 10 seeds (range 32-48) -- Table-1-like, and 63% is >4 std away,
#   so seed variance alone cannot reach it. The remaining hypothesis is that
#   Table 5 used a HARDER GD optimizer.
#
#   The upstream repo carries exactly such a config, but only as the (disabled)
#   in-training eval hook, conf/planner/gd.yaml:
#       lr 1  |  sample_type randn  |  action_noise 0.003  |  opt_steps 1000
#   vs the documented EVAL runfile conf/plan_gd.yaml (== run_row6_a100.sh plan):
#       lr 0.1|  sample_type zero   |  action_noise 0      |  opt_steps 100
#
#   NOTHING in the paper or code says this hook produced Table 5. This run tests
#   it directly.
#
# ── WHAT IS AND ISN'T CHANGED ────────────────────────────────────────────────
#   Changed  : ONLY the 4 optimizer knobs above (via CLI overrides).
#   Unchanged: horizon (25 -> /frameskip 5 == effective 5), n_taken_actions=25,
#              objective.mode=last, max_iter=1 -- i.e. the SAME open-loop
#              protocol as `run_row6_a100.sh plan`. So any delta is optimizer
#              effort, not protocol. (Override HORIZON=5 if you also want to
#              mimic the hook's raw horizon; see note at the override block.)
#
# ── READING THE RESULT ───────────────────────────────────────────────────────
#   near 63 +/- 8  -> Table 5's patch-384 == harder GD. Discrepancy reproduced
#                     from BOTH sides (39 with documented, 63 with this probe).
#   near 39/35     -> optimization effort is NOT the cause. The gap stays
#                     unexplained by anything in the repo (different checkpoint
#                     or data the repo does not record).
#   between        -> partial; sweep the 4 knobs individually (they are env vars).
#
# ── COST ─────────────────────────────────────────────────────────────────────
#   opt_steps 1000 == 10x the fwd+bwd passes of the documented recipe -> ~10x
#   slower wall-clock. Peak memory is UNCHANGED (opt_steps is sequential; memory
#   is set by the per-step 588-token rollout), so CHUNK=10 still fits 40GB.
#   10 seeds x 5 chunks x 1000 steps is long -- nohup it or trim SEEDS first.
#
# ── USAGE (JupyterLab terminal) ──────────────────────────────────────────────
#   ZIP=~/ts_data/point_maze.zip DATA_ROOT=~/ts_data/data source setup_a100.sh
#   bash run_row6_table5_probe.sh            # 10 seeds, harder GD, pooled
#   # quick look first:
#   SEEDS="100 200 300" bash run_row6_table5_probe.sh
#   # isolate one knob (e.g. only more steps, keep zero-init):
#   SAMPLE_TYPE=zero ACTION_NOISE=0 LR=0.1 bash run_row6_table5_probe.sh
#
# ── PARALLEL (one seed per GPU, ~30 min each) ────────────────────────────────
#   The hydra.yaml patch is now atomic+idempotent, so NO stagger is needed. Two
#   rules only: pin CUDA_VISIBLE_DEVICES per job, and give each a distinct OUT.
#   Then pool the per-seed OUT dirs and re-summarize (see 'pooling' note below).
#     CUDA_VISIBLE_DEVICES=0 OUT=~/probe_s100 SEEDS=100 bash run_row6_table5_probe.sh &
#     CUDA_VISIBLE_DEVICES=1 OUT=~/probe_s200 SEEDS=200 bash run_row6_table5_probe.sh &
#     CUDA_VISIBLE_DEVICES=2 OUT=~/probe_s300 SEEDS=300 bash run_row6_table5_probe.sh &
#     wait
#   pooling: each job prints its own 1-seed mean; for a combined mean+std across
#   seeds, point ONE run at all the logs -- copy the per-seed logs into a single
#   OUT and re-run the summary, or just average the per-seed means by hand.
#   (One process per GPU: a single chunk needs the whole 40GB card.)
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ts_ckpts_row6}"   # SAME root as run_row6_a100.sh
DATA="${DATA:-${DATASET_DIR:-$HOME/data}/point_maze}"
OUT="${OUT:-$HOME/row6_table5_probe_out}"       # separate -> never clobbers documented-recipe artifacts
SEEDS="${SEEDS:-100 200 300 400 500 600 700 800 900 1000}"   # same 10 seeds as your row6 run
NEVALS="${NEVALS:-50}"
CHUNK="${CHUNK:-10}"
EPOCHS="${EPOCHS:-20}"
DECODE="${DECODE:-false}"   # quantitative test; skip decode/viz for speed (set true to eyeball trajectories)

# ── the harder optimizer knobs (conf/planner/gd.yaml). Overridable to sweep. ──
LR="${LR:-1}"
SAMPLE_TYPE="${SAMPLE_TYPE:-randn}"
ACTION_NOISE="${ACTION_NOISE:-0.003}"
OPT_STEPS="${OPT_STEPS:-1000}"
# HORIZON: leave at plan_gd.yaml's 25 (-> effective 5 after /frameskip) so this
# is an apples-to-apples open-loop plan. The hook's raw horizon is 5; only set
# HORIZON=5 if you deliberately want to also change the planning horizon.
HORIZON="${HORIZON:-25}"

# The two published numbers for this exact config.
PAPER_T1="35.33"; PAPER_T1_STD="4.11"    # Table 1, open-loop (documented recipe -> your ~39.2)
PAPER_T5="63.33"; PAPER_T5_STD="8.22"    # Table 5, GD        (the number under test)

[ -f "$REPO/train.py" ] || { echo "!! run from the repo root (train.py not found)"; exit 1; }
[ -d "$DATA" ]          || { echo "!! dataset dir missing: $DATA  (source setup_a100.sh)"; exit 1; }

export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

if [ -z "${PYTHONPATH:-}" ] || ! echo "${PYTHONPATH:-}" | grep -q facebookresearch_dinov2; then
  HUB_DINO="$(ls -d "$HOME/.cache/torch/hub/facebookresearch_dinov2"* 2>/dev/null | head -1)"
  [ -n "$HUB_DINO" ] && export PYTHONPATH="$HUB_DINO:${PYTHONPATH:-}"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IDLEST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${IDLEST:-0}"
  echo "[gpu] pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Same run-dir signature as run_row6_a100.sh: patch, no projector, dim384.
find_run_dir() {
  find "$CKPT_ROOT/test" -maxdepth 1 -type d -name 'umaze_False_*projnone_dim384*' 2>/dev/null | head -1
}

RUN="$(find_run_dir)"
[ -n "$RUN" ] || { echo "!! no row6 run dir under $CKPT_ROOT/test -- train it first (run_row6_a100.sh train)"; exit 1; }
CKPT="$RUN/checkpoints/model_${EPOCHS}.pth"
[ -f "$RUN/hydra.yaml" ] || { echo "!! MISSING $RUN/hydra.yaml"; exit 1; }
[ -f "$CKPT" ]           || { echo "!! MISSING $CKPT (training not finished?)"; exit 1; }
case "$RUN" in /*) ;; *) echo "!! run dir must be absolute (plan.py:467)"; exit 1 ;; esac
echo "BUNDLE=$RUN"
echo "[recipe] HARDER GD (table5 probe): lr=$LR sample_type=$SAMPLE_TYPE action_noise=$ACTION_NOISE opt_steps=$OPT_STEPS horizon=$HORIZON"
echo "[recipe] (documented recipe for contrast: lr=0.1 sample_type=zero action_noise=0 opt_steps=100)"

echo "[setup] verifying mujoco_py imports (OSMesa, prebuilt)..."
python -c "import mujoco_py" 2>/dev/null && echo "[setup] mujoco_py OK" \
  || { echo "!! mujoco_py import FAILED -- run: python -c 'import mujoco_py'"; exit 1; }

# plan.py reads env.dataset from the FROZEN hydra.yaml; CLI overrides do not reach it.
# This is the ONE file shared by parallel seed jobs (plan.py:472 reads it; nothing
# else writes into $RUN). Verified: plan.py:38 os.chdir's into the per-chunk run
# dir, so plan_targets.pkl / logs.json / viz are all per-OUT and never collide.
# The patch below is IDEMPOTENT (skips the write once correct) and ATOMIC (temp +
# os.replace), so concurrent jobs never see a half-written hydra.yaml -> safe to
# launch in parallel with NO stagger.
python - "$RUN/hydra.yaml" "$DATA" <<'PY'
import sys, os, tempfile
from omegaconf import OmegaConf
p, data = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(p)
ds = cfg.env.dataset
need = (ds.get("data_path") != data
        or ds.get("use_frame_files") is not False
        or ds.get("use_preprocessed") is not False)
if not need:
    print(f"[config] {p} already correct -- skipping write (parallel-safe)")
else:
    ds.data_path = data
    ds.use_frame_files = False
    ds.use_preprocessed = False
    # Render to a temp file in the SAME dir (same filesystem), then os.replace():
    # atomic on POSIX, so a concurrent reader always sees a COMPLETE hydra.yaml
    # -- the old one or the new one, never a truncated file. Even if two jobs
    # both write, the content is identical, so the result is well-defined.
    d = os.path.dirname(p) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".hydra.tmp"); os.close(fd)
    try:
        OmegaConf.save(cfg, tmp)
        os.replace(tmp, p)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise
    print(f"[config] set data_path={data}, use_frame_files=False in {p} (atomic write)")
PY

mkdir -p "$OUT"; cd "$REPO"
OFFSETS=$(python -c "print(' '.join(str(o) for o in range(0, $NEVALS, $CHUNK)))")
echo "===== CHUNKED TABLE-5 PROBE PLAN: $NEVALS evals/seed as $CHUNK-eval passes at offsets: $OFFSETS ====="
echo "(opt_steps=$OPT_STEPS is ~$(python -c "print(round($OPT_STEPS/100))")x the documented 100 -> expect ~that much slower)"
for S in $SEEDS; do
  for O in $OFFSETS; do
    OO=$(printf "%02d" "$O")
    echo "----- seed $S chunk $OO (n_evals=$CHUNK, harder GD) -----"
    python plan.py --config-name plan_gd.yaml \
      ckpt_base_path="$RUN" \
      model_epoch="$EPOCHS" \
      n_evals="$CHUNK" \
      +eval_start_index="$O" \
      seed="$S" \
      decode_for_viz="$DECODE" \
      planner.sub_planner.lr="$LR" \
      planner.sub_planner.sample_type="$SAMPLE_TYPE" \
      planner.sub_planner.action_noise="$ACTION_NOISE" \
      planner.sub_planner.opt_steps="$OPT_STEPS" \
      planner.sub_planner.horizon="$HORIZON" \
      hydra.run.dir="$OUT/plan_seed_${S}/chunk_${OO}" \
      2>&1 | tee "$OUT/plan_seed_${S}_chunk${OO}.log"
  done
done

python - "$OUT" "$NEVALS" "$CHUNK" "$PAPER_T1" "$PAPER_T1_STD" "$PAPER_T5" "$PAPER_T5_STD" \
        "$LR" "$SAMPLE_TYPE" "$ACTION_NOISE" "$OPT_STEPS" "$HORIZON" $SEEDS <<'PY'
import sys, re, os, statistics as st
out, nevals, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t1, t1s, t5, t5s = sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
lr, samp, an, opt, hor = sys.argv[8:13]
seeds = sys.argv[13:]
offsets = list(range(0, nevals, chunk))
rates = []
for s in seeds:
    chunk_rates = []
    for o in offsets:
        log = os.path.join(out, f"plan_seed_{s}_chunk{o:02d}.log")
        vals = re.findall(r"Success rate:\s*([0-9.]+)", open(log).read())
        if not vals:
            print(f"!! no 'Success rate' in {log}"); sys.exit(1)
        chunk_rates.append(float(vals[-1]) * 100.0)
    frac = sum(chunk_rates) / len(chunk_rates)   # equal chunks -> mean == pooled successes/nevals
    rates.append(frac)
    print(f"seed {s}: {frac:.2f}%  ({round(frac/100*nevals)}/{nevals})   chunks: "
          + ", ".join(f"{c:.0f}" for c in chunk_rates))
mean = sum(rates)/len(rates)
pop_std = st.pstdev(rates) if len(rates) > 1 else 0.0
d1, d5 = abs(mean - float(t1)), abs(mean - float(t5))
# Verdict framed around the hypothesis under test.
if d5 <= float(t5s):
    verdict = (f"REACHES Table 5 ({t5} +/- {t5s}). Harder GD reproduces the higher number\n"
               f"  -> Table 5's patch-384 is consistent with STRONGER GD optimization.")
elif d1 <= max(float(t1s), 5.0):
    verdict = (f"STAYS at Table 1 ({t1} +/- {t1s}). Harder GD does NOT lift this cell ->\n"
               f"  optimization effort is not the cause; the gap stays unexplained by the repo.")
else:
    verdict = ("lands BETWEEN the two tables -> partial. Sweep the 4 knobs individually\n"
               "  (LR / SAMPLE_TYPE / ACTION_NOISE / OPT_STEPS are env vars) to localize it.")
res = os.path.join(out, "RESULT_ROW6_TABLE5_PROBE.txt")
with open(res, "w") as f:
    f.write("===== UMAZE / DINOv2 (patch) 14x14x384 / NO STRAIGHTENING / TABLE-5 PROBE (harder GD) =====\n")
    f.write("config: encoder=dino, frozen patch tokens, no projector, straightening OFF\n")
    f.write(f"planner: lr={lr} sample_type={samp} action_noise={an} opt_steps={opt} horizon={hor}\n")
    f.write("         (documented recipe for contrast: lr=0.1 sample_type=zero action_noise=0 opt_steps=100)\n")
    f.write("SAME checkpoint as run_row6_a100.sh -- NO retraining; only the optimizer changed.\n")
    f.write(f"n_evals={nevals}/seed as {len(offsets)}x{chunk}-eval chunks, seeds={' '.join(seeds)}\n")
    f.write("per-seed success (%): " + ", ".join(f"{r:.2f}" for r in rates) + "\n")
    f.write(f"OURS (harder GD) = {mean:.2f} +/- {pop_std:.2f} %  (mean +/- pop-std, n={len(rates)})\n")
    f.write("\n--- THE TWO PUBLISHED NUMBERS FOR THIS CONFIG ---\n")
    f.write(f"  Table 1 (open-loop, documented recipe) = {t1} +/- {t1s} %   (your documented run: ~39.2)\n")
    f.write(f"  Table 5 (GD, the number under test)     = {t5} +/- {t5s} %\n")
    f.write(f"  |ours - Table1| = {d1:.2f}    |ours - Table5| = {d5:.2f}\n")
    f.write(f"  -> {verdict}\n")
    f.write("\nCAVEAT: this tests ONE alternative (the repo's in-training GD hook). A null\n")
    f.write("result rules out optimizer effort, not every possible protocol difference.\n")
print("\n" + open(res).read())
print(f"[written] {res}")
print("Compare directly against RESULT_ROW6.txt (documented recipe) from run_row6_a100.sh.")
PY