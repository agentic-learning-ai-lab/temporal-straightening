#!/usr/bin/env bash
# run_cls_a100.sh — reproduce Table 1 row:
#   DINOv2 (CLS) | dim 1x384 | Lcurv = OFF | PointMaze-UMaze | open-loop GD
#   PAPER TARGET: 25.33 +/- 0.94 %
#
# VERIFIED against Wang et al., "Temporal Straightening for Latent Planning",
# ICML 2026 (arXiv:2603.12231v2), Table 1, row 1, PointMaze-UMaze / Open-loop.
# Table 1's caption: "The shaded rows are ours while the rest is DINO-WM
# (Zhou et al., 2025)." This row is UNSHADED -> a DINO-WM baseline row as
# reported by Wang et al. It does NOT test straightening; it validates the
# pipeline across encoders.
#
# 25.33 +/- 0.94 is consistent with 3 seeds x 50 evals at 24 / 26 / 26 %
# (12/50, 13/50, 13/50): mean = 25.33, population std = 0.94.
#
# Differs from the straightening baseline by exactly two flags:
#   encoder=dino_channel -> dino_cls        (CLS token, 1x384, no projector)
#   training.straighten=aggcos1e-1 -> False
#
# CONFIG MATCHES THE PAPER (checked):
#   Table 3: batch 32, history 3, frameskip 5, predictor lr 5e-4, act enc lr 5e-4
#   Table 4: horizon 25, 25 executed actions, Adam, zero init, lr 0.1, 100 steps
#   A.2:     2,000 UMaze trajectories x 100 steps, 20 epochs
#   5.3:     50 test samples, 3 data seeds, goals reachable within 25 steps
#
#   NOTE on Table 3's footnote ("lr = 1e-6 for no straightening"): that governs
#   the PROJECTOR/ResNet lr. dino_cls has no projector and no agg head, so the
#   encoder has ZERO trainable params and training.encoder_lr is a no-op here.
#   (The `lr1e-05` in the run-dir name is cosmetic.) If you ever run the
#   DINOv2(patch)+proj Lcurv=OFF rows, you MUST set training.encoder_lr=1e-6.
#
# JUPYTERHUB / A100 PORT of run_cls_aws.sh. Changes, and only these:
#   - sudo apt-get      -> REMOVED (no root here; setup_a100.sh conda-installs
#                          mesalib/glew/glfw/patchelf instead)
#   - added BACKUP_DIR  -> $HOME is ephemeral; sync checkpoints out or lose them
#   - added `watch`     -> background backup loop (see the survival note below)
#   - GPU pin respected -> shared 8-GPU box; setup_a100.sh picks the idlest
# The train/plan flags are byte-identical to the original. Do not "tidy" them.
#
# RUN THIS FROM A JUPYTERLAB TERMINAL (this box is JupyterHub, not ssh).
#
# WHAT SURVIVES WHAT (nohup is NOT the whole story here):
#   closing the browser tab      -> training SURVIVES (nohup detaches it)
#   JupyterHub culls your server -> training DIES, but files on disk SURVIVE.
#                                   Recovery is cheap: re-run `train`. train.py:264-267
#                                   auto-resumes from checkpoints/model_latest.pth,
#                                   and save_every_x_epoch=1 (conf/train.yaml:36),
#                                   so you lose at most one epoch (~10-20 min).
#   the instance stops/rotates   -> EVERYTHING GONE, checkpoints included.
#                                   $HOME is lv_ephemeral. This is what BACKUP_DIR
#                                   is for, and what happened to jupyter-deuk-b329.
#
# ON 40GB AND n_evals=50 (why this is NOT chunked):
#   The ">=48GB" floor in plan_50_runpod.sh was measured on the PATCH model
#   (dino / dino_channel), whose predictor sees 3 frames x 196 patches = 588
#   tokens -- attention is (n_evals, 16, 588, 588) per layer per step.
#   dino_cls has ONE token per frame: pos_embedding is (1, 3, 404), not
#   (1, 588, 404) -- confirmed in modal_evidence/medium-full-20260702-01/logs/.
#   So attention is (n_evals, 16, 3, 3): ~38,000x smaller. Also plan.py:189-192
#   divides horizon by frameskip, so the rollout is 5 model steps, not 25
#   (matching the paper: "we only need to roll out the world model for H = 5").
#   50 evals should fit 40GB with room to spare. If it OOMs anyway, chunk --
#   see the CHUNKED fallback in the `plan` stage below.
#
# STAGES:
#   BACKUP_DIR=<persistent> source setup_a100.sh   # once per session
#   bash run_cls_a100.sh prep         # one-time, CPU, ~40 min, ~30GB
#   bash run_cls_a100.sh smoke        # 1 epoch sanity check   -- DO THIS FIRST
#   bash run_cls_a100.sh train        # 20 epochs, ~3-7 hr, nohup'd
#   bash run_cls_a100.sh watch        # 2nd terminal: back up every epoch, forever
#   bash run_cls_a100.sh status       # progress + backup state
#   bash run_cls_a100.sh plan         # 3 seeds x 50 evals, ~15 min, aggregates
#   bash run_cls_a100.sh backup       # force a checkpoint sync right now
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ts_ckpts_cls}"
DATA="${DATA:-${DATASET_DIR:-$HOME/data}/point_maze}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
OUT="${OUT:-$HOME/cls_plan_out}"
BACKUP_DIR="${BACKUP_DIR:-}"
SEEDS="${SEEDS:-100 200 300}"
NEVALS="${NEVALS:-50}"
EPOCHS="${EPOCHS:-20}"
STAGE="${1:-}"

PAPER_MEAN="25.33"
PAPER_STD="0.94"

mkdir -p "$LOG_DIR"

usage() { echo "usage: bash run_cls_a100.sh {prep|smoke|train|plan|status|backup}"; exit 1; }
[ -n "$STAGE" ] || usage

# --- shared checks ------------------------------------------------------------
[ -f "$REPO/train.py" ] || { echo "!! run from the repo root (train.py not found)"; exit 1; }
[ -d "$DATA" ]          || { echo "!! dataset dir missing: $DATA  (source setup_a100.sh)"; exit 1; }

# RENDER BACKEND: OSMesa. plan_50_runpod.sh:36-38 is right and we proved it the
# hard way -- do not "fix" this to EGL.
#
#   EGL COMPILES on this box (glew + xorg-libx11 + xorg-xproto satisfy it) but
#   CRASHES AT RUNTIME: the render workers die with
#       malloc_consolidate(): unaligned fastbin chunk detected
#   followed by EOFError in venv.py:878 when the parent reads from the dead
#   subprocess. Exactly the failure plan_50_runpod.sh documents for headless
#   cloud GPUs. Compiling is not the same as working.
#
#   THE GOTCHA THAT COSTS AN HOUR: conda-forge `mesalib` must be pinned to
#   23.3.*. Version 26.0.3 ships NO libOSMesa and NO GL/osmesa.h, so the OSMesa
#   backend cannot even compile against it (`find $CONDA_PREFIX -iname '*osmesa*'`
#   returns only Python files). 23.3 pulls libllvm17 (the llvmpipe software
#   rasterizer OSMesa needs) and provides the header. setup_a100.sh pins it.
#
# MUJOCO_PY_FORCE_CPU selects LinuxCPUExtensionBuilder (OSMesa) over
# LinuxGPUExtensionBuilder (EGL). Its VALUE is irrelevant -- mujoco_py tests
# `'MUJOCO_PY_FORCE_CPU' in os.environ` -- so it must be set, and to unset it is
# to choose EGL.
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

# Shared 8-GPU box. If setup_a100.sh didn't pin one, pick the idlest now.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IDLEST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${IDLEST:-0}"
  echo "[gpu] pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# --- backup: $HOME is ephemeral ----------------------------------------------
# jupyter-deuk-b329's home is already gone (see the Wall plan.log). Don't be next.
do_backup() {
  local src="$1"
  if [ -z "$BACKUP_DIR" ]; then
    echo "!! BACKUP_DIR unset -- NOT backing up $src"
    echo "!! \$HOME is on lv_ephemeral; this is destroyed when the box stops."
    return 0
  fi
  [ -e "$src" ] || { echo "[backup] nothing at $src yet"; return 0; }
  echo "[backup] $src -> $BACKUP_DIR"
  case "$BACKUP_DIR" in
    s3://*) aws s3 cp --recursive --only-show-errors "$src" "$BACKUP_DIR/$(basename "$src")" \
              || echo "!! s3 cp FAILED -- checkpoint is NOT safe" ;;
    *)      mkdir -p "$BACKUP_DIR" && rsync -a "$src" "$BACKUP_DIR/" \
              || echo "!! rsync FAILED -- checkpoint is NOT safe" ;;
  esac
}

enable_fast_loader() {
  python - "$REPO/conf/env/point_maze.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
p = sys.argv[1]
cfg = OmegaConf.load(p)
if cfg.dataset.use_frame_files is not True or cfg.dataset.use_preprocessed is not False:
    cfg.dataset.use_frame_files = True     # per-frame files (preprocess_frames.py default mode)
    cfg.dataset.use_preprocessed = False   # raw uint8 HWC; loader does /255 + transform
    OmegaConf.save(cfg, p)
    print(f"[config] enabled fast loader in {p}")
else:
    print(f"[config] fast loader already enabled")
PY
  # NOTE: do NOT use `ls "$DATA"/obses/*_frame_*` here. After prep there are
  # 200,000 matching files and the glob expansion exceeds ARG_MAX, so ls fails
  # and the check reports "no frame files" exactly when they all exist.
  # find -print -quit stops at the first hit and never expands the list.
  if ! find "$DATA/obses" -maxdepth 1 -name '*_frame_*' -print -quit 2>/dev/null | grep -q .; then
    echo ""
    echo "!! WARNING: no per-frame files under $DATA/obses"
    echo "!!   bash run_cls_a100.sh prep"
    echo "!! Without them training is ~5x slower (~90 min/epoch)."
    echo "!! Continuing in 10s (Ctrl-C to abort)..."
    sleep 10
  fi
}

find_run_dir() {
  find "$CKPT_ROOT/test" -maxdepth 1 -type d -name 'umaze_False_*projnone_dim384*' 2>/dev/null | head -1
}

TRAIN_FLAGS=(--config-name train.yaml
             env=point_maze
             encoder=dino_cls
             training.straighten=False
             ckpt_base_path="$CKPT_ROOT")

case "$STAGE" in

prep)
  # Default mode = raw uint8 [H,W,C], ~30GB. Do NOT pass --preprocessed
  # (float32, ~117GB). You have 6.2T free so it would fit, but it buys nothing:
  # the loader's /255 + transform is cheap at 224x224.
  # find -print -quit, not a glob: 200k files would exceed ARG_MAX (see enable_fast_loader)
  if find "$DATA/obses" -maxdepth 1 -name '*_frame_*' -print -quit 2>/dev/null | grep -q .; then
    echo "[prep] per-frame files already present (preprocess is resumable; re-running is a no-op scan)"
  else
    echo "===== PREPROCESS FRAMES: one-time, CPU only, ~40 min, ~30GB ====="
    df -h "$DATA" | tail -1
  fi
  python preprocess_frames.py --data_path "$DATA" 2>&1 | tee "$LOG_DIR/prep.log"
  enable_fast_loader
  echo ">> next: bash run_cls_a100.sh smoke"
  ;;

smoke)
  enable_fast_loader
  echo "===== SMOKE TEST: 1 epoch (expect ~10-20 min, NOT ~90) ====="
  echo "Watch for: predictor pos_embedding = (1, 3, 404) -- 3 frames x 1 CLS token."
  echo "If you see (1, 588, 404) you are running the PATCH encoder, not CLS. Stop."
  python train.py "${TRAIN_FLAGS[@]}" training.epochs=1 2>&1 | tee "$LOG_DIR/cls_smoke.log"
  echo ""
  echo "Run dir: $(find_run_dir)"
  ;;

train)
  enable_fast_loader
  if [ -z "$BACKUP_DIR" ]; then
    echo ""
    echo "⚠️  BACKUP_DIR is unset and \$HOME is on EPHEMERAL storage."
    echo "⚠️  This is a 3-7 hour run whose only output is a checkpoint."
    echo "⚠️  If the box stops, you do it all again. Ctrl-C and re-source with"
    echo "⚠️  BACKUP_DIR=<persistent path> if you have one."
    echo "⚠️  Continuing in 15s..."
    sleep 15
  fi
  echo "===== FULL TRAIN: $EPOCHS epochs, backgrounded, GPU $CUDA_VISIBLE_DEVICES ====="
  nohup python train.py "${TRAIN_FLAGS[@]}" training.epochs="$EPOCHS" \
    > "$LOG_DIR/train_cls.log" 2>&1 &
  echo "pid $! -> $LOG_DIR/train_cls.log"
  echo "tail:   tail -f $LOG_DIR/train_cls.log"
  echo ""
  echo "Closing the browser tab is fine -- nohup detaches it from your terminal."
  echo "But if JupyterHub culls your server, this DIES. Files survive, so just"
  echo "re-run 'train': train.py auto-resumes from model_latest.pth and"
  echo "save_every_x_epoch=1, so you lose at most one epoch."
  echo ""
  if [ -n "$BACKUP_DIR" ]; then
    echo ">> NOW, IN A SECOND TERMINAL:  bash run_cls_a100.sh watch"
    echo "   (backs up each new checkpoint; the box itself is ephemeral)"
  else
    echo ">> WHEN IT FINISHES:  bash run_cls_a100.sh backup"
  fi
  ;;

watch)
  # Checkpoints land every epoch (save_every_x_epoch=1, ~10-20 min apart) and
  # live on ephemeral disk. This loop copies each one out as it appears.
  # Run it in a SECOND JupyterLab terminal alongside `train`.
  [ -n "$BACKUP_DIR" ] || { echo "!! BACKUP_DIR unset -- nothing to watch for. Set it and re-source setup_a100.sh"; exit 1; }
  INTERVAL="${INTERVAL:-300}"
  echo "===== BACKUP WATCH: every ${INTERVAL}s -> $BACKUP_DIR  (Ctrl-C to stop) ====="
  LAST=""
  while true; do
    RUN="$(find_run_dir)"
    if [ -n "$RUN" ] && [ -f "$RUN/checkpoints/model_latest.pth" ]; then
      STAMP=$(stat -c %Y "$RUN/checkpoints/model_latest.pth" 2>/dev/null || echo 0)
      if [ "$STAMP" != "$LAST" ]; then
        do_backup "$RUN"
        LAST="$STAMP"
      fi
    fi
    sleep "$INTERVAL"
  done
  ;;

backup)
  RUN="$(find_run_dir)"
  [ -n "$RUN" ] || { echo "!! no run dir under $CKPT_ROOT/test yet"; exit 1; }
  do_backup "$RUN"
  do_backup "$OUT"
  ;;

status)
  echo "--- last 15 lines of $LOG_DIR/train_cls.log ---"
  tail -15 "$LOG_DIR/train_cls.log" 2>/dev/null || echo "(no log yet)"
  echo "--- checkpoints ---"
  RUN="$(find_run_dir)"
  [ -n "$RUN" ] && ls -la "$RUN/checkpoints/" 2>/dev/null || echo "(no run dir yet)"
  echo "--- backup ---"
  if [ -z "$BACKUP_DIR" ]; then
    echo "⚠️  BACKUP_DIR unset. Checkpoints live ONLY on ephemeral disk."
  else
    echo "BACKUP_DIR=$BACKUP_DIR"
  fi
  ;;

plan)
  RUN="$(find_run_dir)"
  [ -n "$RUN" ] || { echo "!! no CLS run dir under $CKPT_ROOT/test — train first"; exit 1; }
  CKPT="$RUN/checkpoints/model_${EPOCHS}.pth"
  [ -f "$RUN/hydra.yaml" ] || { echo "!! MISSING $RUN/hydra.yaml"; exit 1; }
  [ -f "$CKPT" ]           || { echo "!! MISSING $CKPT (training not finished?)"; exit 1; }
  echo "BUNDLE=$RUN"

  # plan.py uses ckpt_base_path directly only if it is ABSOLUTE (plan.py:467)
  case "$RUN" in /*) ;; *) echo "!! run dir must be absolute"; exit 1 ;; esac

  # NO apt-get here: no root on this box. setup_a100.sh conda-installs the GL
  # deps and puts $CONDA_PREFIX/lib first on LD_LIBRARY_PATH.
  #
  # DO NOT clear _pyxbld* here. The original did, to force a rebuild against the
  # conda GL libs -- but the EGL extension takes minutes to compile and deleting
  # it means recompiling on every single plan run. Worse, a failed rebuild leaves
  # you with nothing. mujoco_py rebuilds by itself when cymj.pyx changes, so the
  # cache is safe to keep. If you ever DO need a clean rebuild:
  #   find "$(python -c 'import site;print(site.getsitepackages()[0])')/mujoco_py/generated" \
  #        -maxdepth 1 -name '_pyxbld*' -exec rm -rf {} +
  echo "[setup] verifying mujoco_py imports (EGL backend, prebuilt)..."
  python -c "import mujoco_py" 2>/dev/null \
    && echo "[setup] mujoco_py OK" \
    || { echo "!! mujoco_py import FAILED -- run: python -c 'import mujoco_py' to see why"; exit 1; }

  # plan.py reads env.dataset from the FROZEN hydra.yaml on disk; CLI overrides
  # do not reach it. Planning touches few frames, so use raw episode files.
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

  mkdir -p "$OUT"
  cd "$REPO"
  # n_evals=50 un-chunked. See the header for why 40GB is enough for CLS.
  # CHUNKED FALLBACK if this OOMs -- replace the loop below with:
  #   for S in $SEEDS; do for O in 0 10 20 30 40; do
  #     python plan.py --config-name plan_gd.yaml ckpt_base_path="$RUN" \
  #       model_epoch="$EPOCHS" n_evals=10 +eval_start_index=$O seed="$S" \
  #       decode_for_viz=false hydra.run.dir="$OUT/plan_seed_${S}/chunk_$(printf %02d $O)" \
  #       2>&1 | tee "$OUT/plan_seed_${S}_chunk$(printf %02d $O).log"
  #   done; done
  # then pool the per-chunk counts equal-weighted, as
  # baseline_artifacts/results/wall_baseline.json does ("chunk_offsets":[0,10,20,30,40]).
  # plan.py:294 replays the RNG draws so a chunk matches the monolithic slice.
  for S in $SEEDS; do
    echo "===== seed $S (n_evals=$NEVALS) ====="
    python plan.py --config-name plan_gd.yaml \
      ckpt_base_path="$RUN" \
      model_epoch="$EPOCHS" \
      n_evals="$NEVALS" \
      seed="$S" \
      decode_for_viz=false \
      hydra.run.dir="$OUT/plan_seed_$S" \
      2>&1 | tee "$OUT/plan_seed_$S.log"
  done

  python - "$OUT" "$NEVALS" "$PAPER_MEAN" "$PAPER_STD" $SEEDS <<'PY'
import sys, re, os, statistics as st
out, nevals, pmean, pstd = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
seeds = sys.argv[5:]
rates = []
for s in seeds:
    log = os.path.join(out, f"plan_seed_{s}.log")
    vals = re.findall(r"Success rate:\s*([0-9.]+)", open(log).read())
    if not vals:
        print(f"!! no 'Success rate' in {log}"); sys.exit(1)
    frac = float(vals[-1]) * 100.0        # last print = final eval over all n_evals
    rates.append(frac)
    print(f"seed {s}: {frac:.2f}%  ({round(frac/100*nevals)}/{nevals})")
mean = sum(rates)/len(rates)
pop_std = st.pstdev(rates) if len(rates) > 1 else 0.0
res = os.path.join(out, "RESULT_CLS.txt")
with open(res, "w") as f:
    f.write("===== UMAZE / DINOv2 (CLS) / NO STRAIGHTENING (open-loop GD) =====\n")
    f.write("config: DINOv2 CLS token, dim 1x384, no projector, straightening OFF\n")
    f.write(f"n_evals={nevals}/seed, seeds={' '.join(seeds)} -> {nevals*len(seeds)} evals total\n")
    f.write("hardware: 1x A100-SXM4-40GB (of 8), un-chunked\n")
    f.write("per-seed success (%): " + ", ".join(f"{r:.2f}" for r in rates) + "\n")
    f.write(f"OURS  = {mean:.2f} +/- {pop_std:.2f} %  (mean +/- pop-std, n={len(rates)})\n")
    f.write(f"PAPER = {pmean} +/- {pstd} %  (Wang et al. ICML 2026, arXiv:2603.12231v2,\n")
    f.write("        Table 1 row 1: DINOv2 (CLS), 1x384, Lcurv OFF, PointMaze-UMaze, Open-loop)\n")
    f.write("        Unshaded row => a DINO-WM baseline (Zhou et al. 2025) as reported by Wang et al.\n")
    f.write("NOTE: Table 1 has no CLS + straightening row. This validates the\n")
    f.write("      pipeline across encoders; it does NOT test straightening.\n")
print("\n" + open(res).read())
print(f"[written] {res}")
PY
  do_backup "$OUT"
  ;;

*) usage ;;
esac
