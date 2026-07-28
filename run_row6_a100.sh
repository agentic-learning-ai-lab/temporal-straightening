#!/usr/bin/env bash
# run_row6_a100.sh — reproduce Table 1 row 6:
#   DINOv2 (patch) | 14x14x384 | Lcurv = OFF | PointMaze-UMaze | open-loop GD
#   PAPER TARGET: ambiguous -- and that is the point of this run.
#
# Wang et al., "Temporal Straightening for Latent Planning", ICML 2026
# (arXiv:2603.12231v2). A DINO-WM baseline row (Zhou et al. 2025), unshaded.
#
# ⚠️  THE PAPER CONTRADICTS ITSELF ON THIS ROW.
#
#     Config `DINOv2 (patch) | 14x14x384 | Lcurv OFF` appears in BOTH tables:
#
#       Env      Table 1 (open-loop)   Table 5 (GD)
#       Wall     52.67 +/- 5.73        73.33 +/- 3.40
#       UMaze    35.33 +/- 4.11        63.33 +/- 8.22     <-- 28 points apart
#       Medium   40.83 +/- 10.07       70.00 +/- 4.08
#       PushT    56.00 +/- 4.32        62.67 +/- 4.11
#
#     Table 1's caption: "using the GD planner". Table 5's: "in open-loop
#     planning ... We compare GD and CEM planners." Same config, same stated
#     protocol, all four environments disagree.
#
#     And it is THIS ROW SPECIFICALLY -- Table 5's other four rows match Table 1
#     exactly on UMaze:
#       patch+proj 14x14x8 OFF -> 44.00 +/- 7.12  in both
#       patch+proj 14x14x8 ON  -> 94.00 +/- 1.63  in both
#       ResNet     14x14x8 OFF -> 14.67 +/- 4.99  in both
#       ResNet     14x14x8 ON  -> 64.67 +/- 8.38  in both
#
#     So one of {35.33, 63.33} is wrong, or the two runs differ in a way the
#     paper does not state. Reproducing it adjudicates, and any outcome is a
#     finding: near 35 -> Table 1 stands; near 63 -> Table 5 does; near neither
#     -> the row is not reproducible as specified.
#
# CONFIG: frozen DINOv2 patch tokens, NO projector -> 196 tokens x 384 dim.
#   encoder=dino  (dino.yaml differs from dino_cls.yaml ONLY in feature_key)
#   training.straighten=False
#
# NOTE ON encoder_lr: Table 3's footnote ("lr = 1e-6 for no straightening")
#   governs the PROJECTOR/ResNet lr. dino.yaml has no projector and no agg head
#   (agg_type defaults to "flatten"), so the encoder has ZERO trainable params
#   and encoder_lr is a NO-OP here -- exactly as with dino_cls. It is set to 1e-6
#   anyway for consistency with row 7, and because it is the ONLY thing keeping
#   this run's directory distinct from the CLS run's:
#       CLS  : umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05
#       row 6: umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-06
#   feature_key is not in the dir template, so patch-no-proj and CLS agree on
#   every other field. CKPT_ROOT is separate too. Do NOT point this at the CLS root.
#
# CHUNKED PLANNING: a PATCH model at emb_dim 404 -- the heaviest UMaze row.
#   vit.py:69 materializes (n_evals, 16, 588, 588) per layer per rollout step.
#   plan_50_runpod.sh:3-6 wants >=48 GB; this box has 40. So NEVALS=10 x 5
#   offsets, pooled -- as baseline_artifacts/results/wall_baseline.json does.
#
# STAGES (JupyterLab terminal; this box is JupyterHub, not ssh):
#   ZIP=~/ts_data/point_maze.zip DATA_ROOT=~/ts_data/data source setup_a100.sh
#   bash run_row6_a100.sh smoke     # 1 epoch -- verify (1, 588, 404)
#   bash run_row6_a100.sh train     # 20 epochs, ~3-7 hr, nohup'd
#   bash run_row6_a100.sh status
#   bash run_row6_a100.sh plan      # 3 seeds x 5 chunks x 10 evals, pooled + viz
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ts_ckpts_row6}"   # separate root -- see the dir-collision note
DATA="${DATA:-${DATASET_DIR:-$HOME/data}/point_maze}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
OUT="${OUT:-$HOME/row6_plan_out}"
SEEDS="${SEEDS:-100 200 300}"
NEVALS="${NEVALS:-50}"
CHUNK="${CHUNK:-10}"
EPOCHS="${EPOCHS:-20}"
ENCODER_LR="${ENCODER_LR:-1e-6}"   # no-op (no trainable encoder params); disambiguates the run dir
# decode_for_viz renders output_final.png / plan0.png and per-eval mp4s.
# The viz only renders the first 10 evals of a plan call, so at CHUNK=10 every
# chunk renders ALL of its evals -> all 50 per seed, vs 10 for an un-chunked run.
DECODE="${DECODE:-true}"
STAGE="${1:-}"

# Deliberately NOT one number: the paper gives two. Both get printed.
PAPER_T1="35.33"; PAPER_T1_STD="4.11"    # Table 1, open-loop
PAPER_T5="63.33"; PAPER_T5_STD="8.22"    # Table 5, GD

mkdir -p "$LOG_DIR"
usage() { echo "usage: bash run_row6_a100.sh {smoke|train|plan|status}"; exit 1; }
[ -n "$STAGE" ] || usage

[ -f "$REPO/train.py" ] || { echo "!! run from the repo root (train.py not found)"; exit 1; }
[ -d "$DATA" ]          || { echo "!! dataset dir missing: $DATA  (source setup_a100.sh)"; exit 1; }

export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"   # presence-tested -> OSMesa, not EGL
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

enable_fast_loader() {
  python - "$REPO/conf/env/point_maze.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
p = sys.argv[1]
cfg = OmegaConf.load(p)
if cfg.dataset.use_frame_files is not True or cfg.dataset.use_preprocessed is not False:
    cfg.dataset.use_frame_files = True
    cfg.dataset.use_preprocessed = False
    OmegaConf.save(cfg, p)
    print(f"[config] enabled fast loader in {p}")
else:
    print("[config] fast loader already enabled")
PY
  # find -print -quit, not a glob: 200k frame files exceed ARG_MAX
  if ! find "$DATA/obses" -maxdepth 1 -name '*_frame_*' -print -quit 2>/dev/null | grep -q .; then
    echo "!! WARNING: no per-frame files under $DATA/obses -- training will be ~5x slower."
    echo "!!   bash run_cls_a100.sh prep     (shared across rows; one-time)"
    echo "!! Continuing in 10s (Ctrl-C to abort)..."
    sleep 10
  fi
}

# umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-06
# The lr suffix is the ONLY field distinguishing this from the CLS run dir.
find_run_dir() {
  find "$CKPT_ROOT/test" -maxdepth 1 -type d -name 'umaze_False_*projnone_dim384*' 2>/dev/null | head -1
}

TRAIN_FLAGS=(--config-name train.yaml
             env=point_maze
             encoder=dino
             training.straighten=False
             training.encoder_lr="$ENCODER_LR"
             ckpt_base_path="$CKPT_ROOT")

case "$STAGE" in

smoke)
  enable_fast_loader
  echo "===== ROW 6 SMOKE: 1 epoch (expect ~10-20 min) ====="
  echo "encoder=dino (patch tokens, no projector)  straighten=False"
  echo ""
  echo "VERIFY in the log:  pos_embedding shape=(1, 588, 404)"
  echo "   3 frames x 196 patches = 588 tokens; emb 384 + 10 proprio + 10 action"
  echo "   (1, 3, 404)  -> CLS. Wrong row."
  echo "   (1, 588, 28) -> channel projector is on. That is row 7."
  echo ""
  echo "EXPECTED and harmless: 'Unknown projector none for patch tokens'."
  echo "   models/dino.py:144-147 fires for patch tokens without a projector,"
  echo "   which is exactly what dino.yaml is. On the CLS run this same warning"
  echo "   came from plan.py:372's throwaway encoder; here it is the real one."
  python train.py "${TRAIN_FLAGS[@]}" training.epochs=1 2>&1 | tee "$LOG_DIR/row6_smoke.log"
  echo ""
  echo "Run dir: $(find_run_dir)"
  echo "Expect ..._lr1e-06. If it says lr1e-05 you are about to collide with the CLS run."
  ;;

train)
  enable_fast_loader
  echo "===== ROW 6 TRAIN: $EPOCHS epochs, backgrounded, GPU $CUDA_VISIBLE_DEVICES ====="
  nohup python train.py "${TRAIN_FLAGS[@]}" training.epochs="$EPOCHS" \
    > "$LOG_DIR/train_row6.log" 2>&1 &
  echo "pid $! -> $LOG_DIR/train_row6.log"
  echo "tail:   tail -3 $LOG_DIR/train_row6.log"
  echo ""
  echo "Closing the browser is fine (nohup). A JupyterHub cull kills it, but files"
  echo "survive and train.py resumes from model_latest.pth -- just re-run 'train'."
  echo "NOTE: on resume the loop runs epochs N+1..N+20 (train.py:475-476), overshooting"
  echo "      past 20. model_20.pth is written on the way and is what 'plan' uses."
  ;;

status)
  echo "--- last 3 lines ---"; tail -3 "$LOG_DIR/train_row6.log" 2>/dev/null || echo "(no log yet)"
  echo "--- completed epochs ---"; grep -c "Training loss" "$LOG_DIR/train_row6.log" 2>/dev/null || echo 0
  echo "--- checkpoints ---"
  RUN="$(find_run_dir)"; [ -n "$RUN" ] && ls -la "$RUN/checkpoints/" 2>/dev/null || echo "(no run dir yet)"
  ;;

plan)
  RUN="$(find_run_dir)"
  [ -n "$RUN" ] || { echo "!! no row6 run dir under $CKPT_ROOT/test -- train first"; exit 1; }
  CKPT="$RUN/checkpoints/model_${EPOCHS}.pth"
  [ -f "$RUN/hydra.yaml" ] || { echo "!! MISSING $RUN/hydra.yaml"; exit 1; }
  [ -f "$CKPT" ]           || { echo "!! MISSING $CKPT (training not finished?)"; exit 1; }
  case "$RUN" in /*) ;; *) echo "!! run dir must be absolute (plan.py:467)"; exit 1 ;; esac
  echo "BUNDLE=$RUN"

  echo "[setup] verifying mujoco_py imports (OSMesa, prebuilt)..."
  python -c "import mujoco_py" 2>/dev/null && echo "[setup] mujoco_py OK" \
    || { echo "!! mujoco_py import FAILED -- run: python -c 'import mujoco_py'"; exit 1; }

  # plan.py reads env.dataset from the FROZEN hydra.yaml; CLI overrides do not reach it.
  python - "$RUN/hydra.yaml" "$DATA" <<'PY'
import sys
from omegaconf import OmegaConf
p, data = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(p)
cfg.env.dataset.data_path = data
cfg.env.dataset.use_frame_files = False
cfg.env.dataset.use_preprocessed = False
OmegaConf.save(cfg, p)
print(f"[config] set data_path={data}, use_frame_files=False in {p}")
PY

  mkdir -p "$OUT"; cd "$REPO"
  OFFSETS=$(python -c "print(' '.join(str(o) for o in range(0, $NEVALS, $CHUNK)))")
  echo "===== CHUNKED PLAN: $NEVALS evals/seed as $CHUNK-eval passes at offsets: $OFFSETS ====="
  echo "(40GB cannot hold a 50-eval 588-token GD batch; see header)"
  for S in $SEEDS; do
    for O in $OFFSETS; do
      OO=$(printf "%02d" "$O")
      echo "----- seed $S chunk $OO (n_evals=$CHUNK) -----"
      python plan.py --config-name plan_gd.yaml \
        ckpt_base_path="$RUN" \
        model_epoch="$EPOCHS" \
        n_evals="$CHUNK" \
        +eval_start_index="$O" \
        seed="$S" \
        decode_for_viz="$DECODE" \
        hydra.run.dir="$OUT/plan_seed_${S}/chunk_${OO}" \
        2>&1 | tee "$OUT/plan_seed_${S}_chunk${OO}.log"
    done
  done

  if [ "$DECODE" = "true" ]; then
    echo ""
    echo "[viz] rendered artifacts:"
    find "$OUT" -name '*.mp4' | wc -l | xargs echo "  mp4 files:"
    find "$OUT" -name 'output_final.png' | wc -l | xargs echo "  contact sheets:"
  fi

  python - "$OUT" "$NEVALS" "$CHUNK" "$PAPER_T1" "$PAPER_T1_STD" "$PAPER_T5" "$PAPER_T5_STD" $SEEDS <<'PY'
import sys, re, os, statistics as st
out, nevals, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t1, t1s, t5, t5s = sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
seeds = sys.argv[8:]
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
    # Equal-sized chunks -> equal-weighted mean == pooled successes / nevals.
    # Same aggregation as baseline_artifacts/results/wall_baseline.json.
    frac = sum(chunk_rates) / len(chunk_rates)
    rates.append(frac)
    print(f"seed {s}: {frac:.2f}%  ({round(frac/100*nevals)}/{nevals})   chunks: "
          + ", ".join(f"{c:.0f}" for c in chunk_rates))
mean = sum(rates)/len(rates)
pop_std = st.pstdev(rates) if len(rates) > 1 else 0.0
d1, d5 = abs(mean - float(t1)), abs(mean - float(t5))
if min(d1, d5) > 10:
    verdict = "supports NEITHER table (>10 points from both)"
elif d1 < d5:
    verdict = f"is closer to TABLE 1 ({t1})"
else:
    verdict = f"is closer to TABLE 5 ({t5})"
res = os.path.join(out, "RESULT_ROW6.txt")
with open(res, "w") as f:
    f.write("===== UMAZE / DINOv2 (patch) 14x14x384 / NO STRAIGHTENING (open-loop GD) =====\n")
    f.write("config: encoder=dino, frozen patch tokens, no projector, straightening OFF\n")
    f.write(f"n_evals={nevals}/seed as {len(offsets)}x{chunk}-eval chunks, seeds={' '.join(seeds)}\n")
    f.write("hardware: 1x A100-SXM4-40GB (of 8), CHUNKED (40GB < the >=48GB a 588-token batch needs)\n")
    f.write("aggregation: equal-weighted mean across chunks per seed (as wall_baseline.json)\n")
    f.write("per-seed success (%): " + ", ".join(f"{r:.2f}" for r in rates) + "\n")
    f.write(f"OURS  = {mean:.2f} +/- {pop_std:.2f} %  (mean +/- pop-std, n={len(rates)})\n")
    f.write("\n--- THE PAPER GIVES TWO NUMBERS FOR THIS CONFIG ---\n")
    f.write(f"  Table 1 (open-loop) = {t1} +/- {t1s} %\n")
    f.write(f"  Table 5 (GD)        = {t5} +/- {t5s} %\n")
    f.write("  Both claim open-loop GD, 50 test samples, three data seeds.\n")
    f.write("  Table 5's other four UMaze rows match Table 1 exactly (44.00, 94.00,\n")
    f.write("  14.67, 64.67), so the discrepancy is specific to this row.\n")
    f.write(f"  |ours - Table1| = {d1:.2f}    |ours - Table5| = {d5:.2f}\n")
    f.write(f"  -> This run {verdict}.\n")
    f.write("\nCAVEAT: n=3 seeds, so SE ~= std/sqrt(3). Treat the verdict as directional,\n")
    f.write("not decisive. More seeds are cheap once the checkpoint exists.\n")
print("\n" + open(res).read())
print(f"[written] {res}")
PY
  ;;

mpc)
  # Closed-loop MPC (plan_gd_mpc.yaml: max_iter=20, n_taken_actions=5). Same
  # checkpoint as open-loop; captures videos/images (decode_for_viz=$DECODE).
  # Peak memory == open-loop (MPC just does more sequential GD solves), so the
  # same CHUNK=10 applies. Separate output dir so it does NOT overwrite the
  # open-loop artifacts. RUN OPEN-LOOP ('plan') FIRST -- this is the follow-up.
  RUN="$(find_run_dir)"
  [ -n "$RUN" ] || { echo "!! no row6 run dir under $CKPT_ROOT/test -- train first"; exit 1; }
  CKPT="$RUN/checkpoints/model_${EPOCHS}.pth"
  [ -f "$RUN/hydra.yaml" ] || { echo "!! MISSING $RUN/hydra.yaml"; exit 1; }
  [ -f "$CKPT" ]           || { echo "!! MISSING $CKPT (training not finished?)"; exit 1; }
  case "$RUN" in /*) ;; *) echo "!! run dir must be absolute (plan.py:467)"; exit 1 ;; esac
  echo "BUNDLE=$RUN"

  python -c "import mujoco_py" 2>/dev/null && echo "[setup] mujoco_py OK" \
    || { echo "!! mujoco_py import FAILED -- run: python -c 'import mujoco_py'"; exit 1; }

  # Idempotent: ensures data_path is set even if you run mpc without plan first.
  python - "$RUN/hydra.yaml" "$DATA" <<'PY'
import sys
from omegaconf import OmegaConf
p, data = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(p)
cfg.env.dataset.data_path = data
cfg.env.dataset.use_frame_files = False
cfg.env.dataset.use_preprocessed = False
OmegaConf.save(cfg, p)
print(f"[config] set data_path={data}, use_frame_files=False in {p}")
PY

  MPC_OUT="${MPC_OUT:-${OUT}_mpc}"
  MPC_MEAN="80.67"; MPC_STD="6.18"   # Table 1 row 6, UMaze MPC
  mkdir -p "$MPC_OUT"; cd "$REPO"
  OFFSETS=$(python -c "print(' '.join(str(o) for o in range(0, $NEVALS, $CHUNK)))")
  echo "===== CHUNKED MPC: $NEVALS evals/seed as $CHUNK-eval passes, offsets: $OFFSETS ====="
  echo "(MPC replans 20x/eval -> slower than open-loop; decode_for_viz=$DECODE)"
  for S in $SEEDS; do
    for O in $OFFSETS; do
      OO=$(printf "%02d" "$O")
      echo "----- seed $S chunk $OO (MPC, n_evals=$CHUNK) -----"
      python plan.py --config-name plan_gd_mpc.yaml \
        ckpt_base_path="$RUN" \
        model_epoch="$EPOCHS" \
        n_evals="$CHUNK" \
        +eval_start_index="$O" \
        seed="$S" \
        decode_for_viz="$DECODE" \
        hydra.run.dir="$MPC_OUT/plan_seed_${S}/chunk_${OO}" \
        2>&1 | tee "$MPC_OUT/plan_seed_${S}_chunk${OO}.log"
    done
  done

  if [ "$DECODE" = "true" ]; then
    echo ""; echo "[viz] MPC rendered artifacts:"
    find "$MPC_OUT" -name '*.mp4' | wc -l | xargs echo "  mp4 files:"
    find "$MPC_OUT" -name 'output_final.png' | wc -l | xargs echo "  contact sheets:"
  fi

  python - "$MPC_OUT" "$NEVALS" "$CHUNK" "$MPC_MEAN" "$MPC_STD" $SEEDS <<'PY'
import sys, re, os, statistics as st
out, nevals, chunk, pmean, pstd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
seeds = sys.argv[6:]; offsets = list(range(0, nevals, chunk)); rates = []
for s in seeds:
    cr = []
    for o in offsets:
        v = re.findall(r"Success rate:\s*([0-9.]+)", open(os.path.join(out, f"plan_seed_{s}_chunk{o:02d}.log")).read())
        if not v: print(f"!! no rate seed {s} chunk {o}"); sys.exit(1)
        cr.append(float(v[-1])*100.0)   # last print = final MPC iter over the chunk
    rates.append(sum(cr)/len(cr)); print(f"seed {s}: {rates[-1]:.2f}%")
m = sum(rates)/len(rates); sd = st.pstdev(rates) if len(rates)>1 else 0.0
res = os.path.join(out, "RESULT_ROW6_MPC.txt")
open(res,"w").write(
    "===== UMAZE / DINOv2 (patch) 14x14x384 / NO STRAIGHTENING (MPC / closed-loop GD) =====\n"
    f"n_evals={nevals}/seed, {len(offsets)}x{chunk} chunks, seeds={' '.join(seeds)}\n"
    "per-seed success (%): " + ", ".join(f"{r:.2f}" for r in rates) + "\n"
    f"OURS  = {m:.2f} +/- {sd:.2f} %  (n={len(rates)})\n"
    f"PAPER = {pmean} +/- {pstd} %  (Table 1 row 6, UMaze MPC)\n"
    "Compare against this row's OPEN-LOOP result (RESULT_ROW6.txt).\n")
print("\n"+open(res).read()); print(f"[written] {res}")
PY
  ;;

*) usage ;;
esac
