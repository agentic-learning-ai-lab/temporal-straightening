#!/usr/bin/env bash
# run_row7_a100.sh — reproduce Table 1 row 7:
#   DINOv2 (patch) + proj | 14x14x8 | Lcurv = OFF | PointMaze-UMaze | open-loop GD
#   PAPER TARGET: 44.00 +/- 7.12 %
#
# Wang et al., "Temporal Straightening for Latent Planning", ICML 2026
# (arXiv:2603.12231v2), Table 1 row 7, PointMaze-UMaze / Open-loop.
# Cross-checked: Table 5 lists the same config at 44.00 +/- 7.12 GD. They agree.
#
# WHY THIS ROW MATTERS MORE THAN ANY OTHER
#   This is the CONTROL for umaze_reproduction's 94.00. Same architecture, one
#   flag flipped. It is the paper's headline sentence (Section 5.3):
#       "UMaze's open-loop success rate is improved from 44% to 94% with the
#        projector"
#   You have reproduced the 94; without the 44 you have not tested the claim.
#   proposal_simple.tex currently lists this row as "--- & 44.0+/-7.1" -- paper
#   number, no reproduction.
#
# DIFFERS FROM umaze_reproduction BY EXACTLY TWO FLAGS:
#   training.straighten=aggcos1e-1 -> False
#   training.encoder_lr=1e-5       -> 1e-6      <-- DO NOT SKIP, see below
#
# ⚠️  encoder_lr=1e-6 IS MANDATORY HERE. Table 3's footnote:
#       "We observe severe performance degradation when training without
#        straightening and decreasing the learning rate helps. We thus use
#        lr = 1e-6 for no straightening."
#     That footnote governs the PROJECTOR lr. dino_channel HAS a trainable
#     projector (ChannelProjector conv layers) AND a trainable agg head (MLP,
#     agg_out_dim=128), so encoder_lr is live -- unlike dino_cls/dino, where the
#     encoder has zero trainable params and the footnote is a no-op.
#     Run this at 1e-5 and you are running a config the paper says degrades.
#
# CHUNKED PLANNING IS THE DEFAULT HERE (unlike run_cls_a100.sh).
#   This is a PATCH model: 3 frames x 196 patches = 588 tokens, so vit.py:69
#   materializes (n_evals, 16, 588, 588) per layer per rollout step -- ~1.1 GB
#   each at n_evals=50, x6 layers x5 steps ~= 33 GB retained before the backward
#   graph. plan_50_runpod.sh:3-6 wants >=48 GB for exactly this reason, and this
#   box has 40 GB. umaze_reproduction got away un-chunked only because it ran on
#   an A100-80GB. So: NEVALS=10 x 5 offsets, pooled -- the same shape
#   baseline_artifacts/results/wall_baseline.json uses
#   ("chunk_offsets": [0,10,20,30,40]). plan.py:294 replays the RNG draws so a
#   chunk matches the corresponding slice of a monolithic run.
#   If you later run on an 80GB card, set CHUNK=50 for a single un-chunked pass.
#
# STAGES (run from a JupyterLab terminal; this box is JupyterHub, not ssh):
#   ZIP=~/ts_data/point_maze.zip DATA_ROOT=~/ts_data/data source setup_a100.sh
#   bash run_row7_a100.sh smoke     # 1 epoch  -- verify (1, 588, 28)
#   bash run_row7_a100.sh train     # 20 epochs, ~3-7 hr, nohup'd
#   bash run_row7_a100.sh status
#   bash run_row7_a100.sh plan      # 3 seeds x 5 chunks x 10 evals, pooled
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ts_ckpts_row7}"     # separate root: do NOT collide with cls/baseline
DATA="${DATA:-${DATASET_DIR:-$HOME/data}/point_maze}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
OUT="${OUT:-$HOME/row7_plan_out}"
SEEDS="${SEEDS:-100 200 300}"
NEVALS="${NEVALS:-50}"          # total evals per seed (the paper's protocol)
CHUNK="${CHUNK:-10}"            # per-pass evals; 40GB cannot fit 50 for a patch model
EPOCHS="${EPOCHS:-20}"
ENCODER_LR="${ENCODER_LR:-1e-6}"
# decode_for_viz=true renders output_final.png / plan0.png and the per-eval
# *_success.mp4 / *_failure.mp4, matching umaze_reproduction's artifact set.
# Nice side effect of chunking: the viz renders only the first 10 evals of a
# plan call, so at CHUNK=10 every chunk renders ALL of its evals -> you get all
# 50 per seed, where an un-chunked 50-eval run would only render 10.
# Set DECODE=false if you hit memory pressure; planning is unaffected either way.
DECODE="${DECODE:-true}"
STAGE="${1:-}"

PAPER_MEAN="44.00"
PAPER_STD="7.12"

mkdir -p "$LOG_DIR"
usage() { echo "usage: bash run_row7_a100.sh {smoke|train|plan|status}"; exit 1; }
[ -n "$STAGE" ] || usage

[ -f "$REPO/train.py" ] || { echo "!! run from the repo root (train.py not found)"; exit 1; }
[ -d "$DATA" ]          || { echo "!! dataset dir missing: $DATA  (source setup_a100.sh)"; exit 1; }

# OSMesa, not EGL. EGL compiles here but corrupts the heap in the render workers
# at runtime -- see run_cls_a100.sh's header. Value is irrelevant, presence is what
# mujoco_py tests, so this must be SET.
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

# The checkpoint pickles the torch.hub-loaded encoder, so torch.load needs
# `import dinov2`. setup_a100.sh normally sets this; belt and braces.
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
    echo "!!   bash run_cls_a100.sh prep     (shared across all rows; one-time)"
    echo "!! Continuing in 10s (Ctrl-C to abort)..."
    sleep 10
  fi
}

# Resolves to: umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
#   straighten=False -> "False" (replace_substring finds no "agg" in "False")
#   projector=channel, projector_out_dim=8, projector_target_hw=14, stop_grad=True
find_run_dir() {
  find "$CKPT_ROOT/test" -maxdepth 1 -type d -name 'umaze_False_*projchannel_dim8*' 2>/dev/null | head -1
}

TRAIN_FLAGS=(--config-name train.yaml
             env=point_maze
             encoder=dino_channel
             training.straighten=False
             training.encoder_lr="$ENCODER_LR"
             ckpt_base_path="$CKPT_ROOT")

case "$STAGE" in

smoke)
  enable_fast_loader
  echo "===== ROW 7 SMOKE: 1 epoch (expect ~10-20 min) ====="
  echo "encoder=dino_channel  straighten=False  encoder_lr=$ENCODER_LR"
  echo "VERIFY in the log:"
  echo "  pos_embedding shape=(1, 588, 28)   <- 3 frames x 196 patches, emb 8+10+10"
  echo "  If it says (1, 3, ...) you are on CLS. If (1, 588, 404) the projector is off."
  python train.py "${TRAIN_FLAGS[@]}" training.epochs=1 2>&1 | tee "$LOG_DIR/row7_smoke.log"
  echo ""
  echo "Run dir: $(find_run_dir)"
  echo "Expect  ..._lr1e-06 in that name. If it says lr1e-05, encoder_lr did not take."
  ;;

train)
  enable_fast_loader
  echo "===== ROW 7 TRAIN: $EPOCHS epochs, backgrounded, GPU $CUDA_VISIBLE_DEVICES ====="
  nohup python train.py "${TRAIN_FLAGS[@]}" training.epochs="$EPOCHS" \
    > "$LOG_DIR/train_row7.log" 2>&1 &
  echo "pid $! -> $LOG_DIR/train_row7.log"
  echo "tail:   tail -3 $LOG_DIR/train_row7.log"
  echo ""
  echo "Closing the browser is fine (nohup). A JupyterHub cull kills it, but files"
  echo "survive and train.py resumes from model_latest.pth -- just re-run 'train'."
  echo "NOTE: resuming makes the loop run epochs N+1..N+20 (train.py:475-476), so it"
  echo "      overshoots past 20. model_20.pth is still written on the way; that is"
  echo "      the one 'plan' uses. Harmless."
  ;;

status)
  echo "--- last 3 lines ---"; tail -3 "$LOG_DIR/train_row7.log" 2>/dev/null || echo "(no log yet)"
  echo "--- completed epochs ---"; grep -c "Training loss" "$LOG_DIR/train_row7.log" 2>/dev/null || echo 0
  echo "--- checkpoints ---"
  RUN="$(find_run_dir)"; [ -n "$RUN" ] && ls -la "$RUN/checkpoints/" 2>/dev/null || echo "(no run dir yet)"
  ;;

plan)
  RUN="$(find_run_dir)"
  [ -n "$RUN" ] || { echo "!! no row7 run dir under $CKPT_ROOT/test -- train first"; exit 1; }
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
  echo "(40GB cannot hold a 50-eval patch-model GD batch; see header)"
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
    echo "[viz] rendered artifacts per chunk:"
    find "$OUT" -name '*.mp4' | wc -l | xargs echo "  mp4 files:"
    find "$OUT" -name 'output_final.png' | wc -l | xargs echo "  contact sheets:"
  fi

  python - "$OUT" "$NEVALS" "$CHUNK" "$PAPER_MEAN" "$PAPER_STD" $SEEDS <<'PY'
import sys, re, os, statistics as st
out, nevals, chunk, pmean, pstd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
seeds = sys.argv[6:]
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
    # equal-sized chunks -> equal-weighted mean == pooled count / nevals.
    # Same aggregation as baseline_artifacts/results/wall_baseline.json.
    frac = sum(chunk_rates) / len(chunk_rates)
    rates.append(frac)
    print(f"seed {s}: {frac:.2f}%  ({round(frac/100*nevals)}/{nevals})   chunks: "
          + ", ".join(f"{c:.0f}" for c in chunk_rates))
mean = sum(rates)/len(rates)
pop_std = st.pstdev(rates) if len(rates) > 1 else 0.0
res = os.path.join(out, "RESULT_ROW7.txt")
with open(res, "w") as f:
    f.write("===== UMAZE / DINOv2 (patch) + proj 14x14x8 / NO STRAIGHTENING (open-loop GD) =====\n")
    f.write("config: encoder=dino_channel, projector=channel dim 8, straightening OFF,\n")
    f.write(f"        training.encoder_lr={os.environ.get('ENCODER_LR','1e-6')} (Table 3 footnote: 1e-6 for no straightening)\n")
    f.write(f"n_evals={nevals}/seed as {len(offsets)}x{chunk}-eval chunks, seeds={' '.join(seeds)}\n")
    f.write("hardware: 1x A100-SXM4-40GB (of 8), CHUNKED (40GB < the >=48GB a 50-eval patch batch needs)\n")
    f.write("aggregation: equal-weighted mean across chunks per seed (as wall_baseline.json)\n")
    f.write("per-seed success (%): " + ", ".join(f"{r:.2f}" for r in rates) + "\n")
    f.write(f"OURS  = {mean:.2f} +/- {pop_std:.2f} %  (mean +/- pop-std, n={len(rates)})\n")
    f.write(f"PAPER = {pmean} +/- {pstd} %  (Wang et al. ICML 2026, arXiv:2603.12231v2,\n")
    f.write("        Table 1 row 7: DINOv2 (patch)+proj, 14x14x8, Lcurv OFF, UMaze, Open-loop.\n")
    f.write("        Table 5 lists the same config at 44.00 +/- 7.12 GD -- the tables agree.)\n")
    f.write("\nTHIS IS THE CONTROL FOR umaze_reproduction's 94.00. The paper's claim is\n")
    f.write("44 -> 94 with straightening (Section 5.3). Compare the two directly.\n")
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
  [ -n "$RUN" ] || { echo "!! no row7 run dir under $CKPT_ROOT/test -- train first"; exit 1; }
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
  MPC_MEAN="81.33"; MPC_STD="6.80"   # Table 1 row 7, UMaze MPC
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
res = os.path.join(out, "RESULT_ROW7_MPC.txt")
open(res,"w").write(
    "===== UMAZE / DINOv2 (patch)+proj 14x14x8 / NO STRAIGHTENING (MPC / closed-loop GD) =====\n"
    f"n_evals={nevals}/seed, {len(offsets)}x{chunk} chunks, seeds={' '.join(seeds)}\n"
    "per-seed success (%): " + ", ".join(f"{r:.2f}" for r in rates) + "\n"
    f"OURS  = {m:.2f} +/- {sd:.2f} %  (n={len(rates)})\n"
    f"PAPER = {pmean} +/- {pstd} %  (Table 1 row 7, UMaze MPC)\n"
    "Compare against this row's OPEN-LOOP result (RESULT_ROW7.txt).\n")
print("\n"+open(res).read()); print(f"[written] {res}")
PY
  ;;

*) usage ;;
esac
