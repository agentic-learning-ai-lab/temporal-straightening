#!/usr/bin/env bash
# run_pusht_ablation_a100.sh — PushT trajectory-penalty ablation (R0–R3) on JupyterHub A100.
#
# Mirrors the wall/umaze ablation workflow (origin/agent/umaze-trajectory-penalty-ablation)
# with PushT-specific settings: 2 training epochs, objective.alpha=1 at planning, dataset
# at $DATASET_DIR/pusht_noise.
#
# CONDITIONS (dino_channel + trainable projector):
#   r0_direction_only  aggcos1e-1    0.1 × R0   <- reuse straightening_reproduction (no retrain)
#   r1_speed_only      aggr1_1e-1    0.1 × R1   <- train
#   r2_full_matched    aggr2_5e-2    0.05 × R2  <- train
#   r3_beta1           aggr3b1_1e-1  0.1 × R3   <- train
#
# PREREQUISITE — merge Daniel's trajectory-penalty code first:
#   git fetch origin
#   git merge origin/agent/umaze-trajectory-penalty-ablation
#   # needs models/visual_world_model.py r0..r4 / aggr token parser
#   bash run_pusht_ablation_a100.sh check-deps
#
# SETUP (once per JupyterHub session):
#   source setup_a100.sh                    # conda env, MuJoCo, WANDB disabled
#   export DATASET_DIR=~/data               # must contain pusht_noise/
#   export PUSHT_ROOT=~/pusht-reproduction  # where your Table-1 runs live (for R0)
#
# WORKFLOW:
#   bash run_pusht_ablation_a100.sh check-deps
#   bash run_pusht_ablation_a100.sh train              # R1/R2/R3, one GPU each, detached
#   bash run_pusht_ablation_a100.sh status
#   bash run_pusht_ablation_a100.sh plan               # all 4 conditions, chunked GD
#   bash run_pusht_ablation_a100.sh plan-status
#   bash run_pusht_ablation_a100.sh results              # regenerate pusht_ablations/RESULTS.txt
#
# Results land in pusht_ablations/ (same layout as pusht_reproduction/):
#   pusht_ablations/r{0,1,2,3}_*/plan_logs/gd_seed*_chunk*.log
#
# NOTE: $HOME on the team A100 box is EPHEMERAL — set BACKUP_DIR in setup_a100.sh or copy
# baseline_artifacts/checkpoints/pusht_speed_ablations/ and pusht_ablations/ when done.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
DATA="${DATA:-${DATASET_DIR:-$HOME/data}/pusht_noise}"
PUSHT_ROOT="${PUSHT_ROOT:-$HOME/pusht-reproduction}"
ABLATION_ROOT="${ABLATION_ROOT:-$REPO/pusht_ablations}"
CKPT_ROOT="${CKPT_ROOT:-$REPO/baseline_artifacts/checkpoints/pusht_speed_ablations}"
R0_ROOT="${R0_ROOT:-$PUSHT_ROOT/straightening_reproduction}"
EPOCHS="${EPOCHS:-2}"
FREE_MIB="${FREE_MIB:-2000}"
SEEDS="${SEEDS:-100 200 300}"
NEVALS="${NEVALS:-50}"
CHUNK="${CHUNK:-10}"
PLAN_DECODE="${PLAN_DECODE:-false}"
STAGE="${1:-}"; shift || true

TRAIN_CONDS="r1_speed_only r2_full_matched r3_beta1"
# R0 plan_logs are pre-populated from pusht_reproduction/straightening; plan R1–R3 by default.
PLAN_CONDS="r1_speed_only r2_full_matched r3_beta1"
ALL_PLAN_CONDS="r0_direction_only r1_speed_only r2_full_matched r3_beta1"

usage() {
  cat <<EOF
usage: bash run_pusht_ablation_a100.sh STAGE [conds...]

stages:
  check-deps    verify trajectory-penalty code + dataset
  train         train R1/R2/R3 (default: all three)
  status        training progress
  plan          GD planning for R1–R3 (default); pass r0_direction_only to re-plan R0
  plan-status   planning progress
  results       regenerate pusht_ablations/RESULTS.txt via make_results.py

train conds: $TRAIN_CONDS
plan  conds: $PLAN_CONDS  (all four: $ALL_PLAN_CONDS; r0 ckpt from \$R0_ROOT)
EOF
  exit 1
}

[ -n "$STAGE" ] || usage
[ -f "$REPO/train.py" ] || { echo "!! run from repo root (train.py not found)"; exit 1; }

# --- shared env (matches run_pusht_a100.sh) ---------------------------------
if [ -z "${PYTHONPATH:-}" ] || ! echo "${PYTHONPATH:-}" | grep -q facebookresearch_dinov2; then
  HUB_DINO="$(ls -d "$HOME/.cache/torch/hub/facebookresearch_dinov2"* 2>/dev/null | head -1)"
  [ -n "$HUB_DINO" ] && export PYTHONPATH="$HUB_DINO:${PYTHONPATH:-}"
fi
export WANDB_MODE="${WANDB_MODE:-disabled}"
if [ -d "$HOME/.mujoco/mujoco210/bin" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$HOME/.mujoco/mujoco210/bin:"*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin" ;;
  esac
fi
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

PYTHON="${PYTHON:-python}"
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  PYTHON="$(which python)"
fi

cond_token() {
  case "$1" in
    r1_speed_only)   echo "aggr1_1e-1" ;;
    r2_full_matched) echo "aggr2_5e-2" ;;
    r3_beta1)        echo "aggr3b1_1e-1" ;;
    *) echo "!! unknown train cond '$1'" >&2; return 1 ;;
  esac
}

train_dir() { echo "$CKPT_ROOT/$1"; }

find_ablation_run_dir() {
  local out="$1"
  if [ -f "$out/checkpoints/model_${EPOCHS}.pth" ]; then
    echo "$out"; return
  fi
  find "$out/test" -maxdepth 1 -type d -name 'pusht_*' 2>/dev/null | head -1
}

find_r0_run_dir() {
  find "$R0_ROOT/test" -maxdepth 1 -type d -name 'pusht_*' 2>/dev/null | head -1
}

plan_ckpt_path() {
  case "$1" in
    r0_direction_only)
      RUN="$(find_r0_run_dir)"
      [ -n "$RUN" ] || { echo "!! R0: no run under $R0_ROOT/test — run run_pusht_a100.sh train channel_on first"; exit 1; }
      echo "$RUN"
      ;;
    *)
      RUN="$(find_ablation_run_dir "$(train_dir "$1")")"
      [ -n "$RUN" ] || { echo "!! $1: no checkpoint — train first"; exit 1; }
      echo "$RUN"
      ;;
  esac
}

patch_hydra_data_path() {
  local hydra_yaml="$1"
  "$PYTHON" - "$hydra_yaml" "$DATA" <<'PY'
import sys
from omegaconf import OmegaConf
p, data = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(p)
cfg.env.dataset.data_path = data
OmegaConf.save(cfg, p)
print(f"[config] set data_path={data}")
PY
}

pick_free_gpus() {
  if [ -n "${GPUS:-}" ]; then
    echo "$GPUS" | tr ',' ' '
    return
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v t="$FREE_MIB" '($2+0)<t {gsub(/ /,"",$1); print $1}'
}

case "$STAGE" in

check-deps)
  echo "=== trajectory-penalty code ==="
  if grep -q 'trajectory_penalty_mode' "$REPO/models/visual_world_model.py" 2>/dev/null; then
    echo "  OK  visual_world_model.py has trajectory_penalty support"
  else
    echo "  FAIL  merge origin/agent/umaze-trajectory-penalty-ablation first"
    exit 1
  fi
  echo "=== config ==="
  [ -f "$REPO/conf/pusht_ablation_base.yaml" ] && echo "  OK  conf/pusht_ablation_base.yaml" \
    || { echo "  FAIL  missing pusht_ablation_base.yaml"; exit 1; }
  echo "=== dataset ==="
  [ -d "$DATA" ] && echo "  OK  $DATA" \
    || { echo "  FAIL  dataset missing — set DATASET_DIR so \$DATASET_DIR/pusht_noise exists"; exit 1; }
  echo "=== R0 baseline (optional until plan) ==="
  R0="$(find_r0_run_dir || true)"
  if [ -n "$R0" ] && [ -f "$R0/checkpoints/model_${EPOCHS}.pth" ]; then
    echo "  OK  R0 checkpoint $R0/checkpoints/model_${EPOCHS}.pth"
  else
    echo "  WARN  no R0 yet — train with: bash run_pusht_a100.sh train channel_on"
  fi
  echo "=== ready ==="
  ;;

train)
  CONDS="${*:-$TRAIN_CONDS}"
  for c in $CONDS; do cond_token "$c" >/dev/null || usage; done
  [ -d "$DATA" ] || { echo "!! dataset missing: $DATA"; exit 1; }
  grep -q 'trajectory_penalty_mode' "$REPO/models/visual_world_model.py" \
    || { echo "!! run check-deps first — trajectory penalty code not merged"; exit 1; }

  read -r -a GPU_ARR <<< "$(pick_free_gpus)"
  N_GPU=${#GPU_ARR[@]}
  [ "$N_GPU" -gt 0 ] || { echo "!! no free GPU (used < ${FREE_MIB} MiB). Set GPUS=2 to force."; exit 1; }
  echo "[gpu] assigned: ${GPU_ARR[*]}"
  mkdir -p "$CKPT_ROOT" "$REPO/logs"

  i=0
  for c in $CONDS; do
    if [ "$i" -ge "$N_GPU" ]; then
      echo "!! only $N_GPU GPU(s) — skipping '$c'. Re-run: bash $0 train $c"
      break
    fi
    g="${GPU_ARR[$i]}"; i=$((i+1))
    out="$(train_dir "$c")"; tok="$(cond_token "$c")"
    log="$out/train.log"; mkdir -p "$out"
    if [ -f "$out/checkpoints/model_${EPOCHS}.pth" ] \
       && grep -qE "Epoch[[:space:]]+$EPOCHS[[:space:]]+Training loss:" "$log" 2>/dev/null; then
      echo "----- skip '$c' (epoch $EPOCHS done) -----"
      continue
    fi
    echo "----- launch '$c' token=$tok on GPU $g -> $out -----"
    CUDA_VISIBLE_DEVICES="$g" setsid nohup "$PYTHON" train.py --config-name pusht_ablation_base \
      "training.straighten=$tok" \
      "training.epochs=$EPOCHS" \
      "ckpt_base_path=$out" \
      "hydra.run.dir=$out" \
      > "$log" 2>&1 &
    echo "  pid $!   log: $log"
  done
  echo ""
  echo "Detached (setsid+nohup). Monitor:  bash $0 status"
  echo "Re-run the same command to resume from model_latest.pth after interruptions."
  ;;

status)
  CONDS="${*:-$TRAIN_CONDS}"
  for c in $CONDS; do
    out="$(train_dir "$c")"; log="$out/train.log"
    ep=$(grep -cE "Epoch[[:space:]]+[0-9]+[[:space:]]+Training loss:" "$log" 2>/dev/null || echo 0)
    echo "===== $c  (epochs logged: $ep / $EPOCHS) ====="
    [ -f "$log" ] && tail -2 "$log" | sed 's/^/  /' || echo "  (no log)"
    RUN="$(find_ablation_run_dir "$out" || true)"
    if [ -n "$RUN" ]; then
      ls -1 "$RUN/checkpoints/" 2>/dev/null | sed 's/^/  ckpt: /' || true
    else
      echo "  (no checkpoint dir yet)"
    fi
  done
  ;;

plan)
  CONDS="${*:-$PLAN_CONDS}"
  [ -d "$DATA" ] || { echo "!! dataset missing: $DATA"; exit 1; }

  for c in $CONDS; do
    CKPT="$(plan_ckpt_path "$c")"
    [ -f "$CKPT/checkpoints/model_${EPOCHS}.pth" ] \
      || { echo "!! $c: missing model_${EPOCHS}.pth under $CKPT"; exit 1; }
  done

  read -r -a GPU_ARR <<< "$(pick_free_gpus)"
  N_GPU=${#GPU_ARR[@]}
  [ "$N_GPU" -gt 0 ] || { echo "!! no free GPU. Set GPUS=0,1,2 to force."; exit 1; }
  echo "[gpu] planning GPUs: ${GPU_ARR[*]}"
  mkdir -p "$ABLATION_ROOT"

  i=0
  for c in $CONDS; do
    if [ "$i" -ge "$N_GPU" ]; then
      echo "!! only $N_GPU GPU(s) — skipping '$c'. Re-run: bash $0 plan $c"
      break
    fi
    g="${GPU_ARR[$i]}"; i=$((i+1))
    cond_dir="$ABLATION_ROOT/$c"
    mkdir -p "$cond_dir/plan_logs"
    echo "----- plan '$c' on GPU $g -> $cond_dir/plan_logs -----"
    CUDA_VISIBLE_DEVICES="$g" setsid nohup bash "$0" _plan_one "$c" > "$cond_dir/plan_logs/driver.log" 2>&1 &
    echo "  pid $!"
  done
  echo ""
  echo "Detached. Monitor:  bash $0 plan-status"
  echo "Then:             bash $0 results"
  ;;

_plan_one)
  c="${1:-}"; [ -n "$c" ] || exit 1
  CKPT="$(plan_ckpt_path "$c")"; CKPT="$(cd "$CKPT" && pwd)"
  cond_dir="$ABLATION_ROOT/$c"
  logs="$cond_dir/plan_logs"
  hydra="$cond_dir/hydra"
  mkdir -p "$logs" "$hydra"
  patch_hydra_data_path "$CKPT/hydra.yaml" "$DATA"

  OFFSETS=$("$PYTHON" -c "print(' '.join(str(o) for o in range(0, $NEVALS, $CHUNK)))")
  for S in $SEEDS; do
    for O in $OFFSETS; do
      OO=$(printf "%02d" "$O")
      log="$logs/gd_seed${S}_chunk${OO}.log"
      out="$hydra/seed_${S}/chunk_${OO}"
      if [ -s "$log" ] && grep -q 'Success rate:' "$log"; then
        echo "skip $c seed=$S chunk=$OO (plan_logs done)"
        continue
      fi
      mkdir -p "$out"
      echo "===== $c seed $S chunk $OO ====="
      "$PYTHON" plan.py --config-name plan_gd.yaml \
        "ckpt_base_path=$CKPT" \
        "model_epoch=$EPOCHS" \
        "n_evals=$CHUNK" \
        "+eval_start_index=$O" \
        "seed=$S" \
        "objective.alpha=1" \
        "decode_for_viz=$PLAN_DECODE" \
        "+wandb_logging=false" \
        "hydra.run.dir=$out" \
        2>&1 | tee "$log"
    done
  done
  echo "PLAN_DONE cond=$c"
  ;;

plan-status)
  CONDS="${*:-$ALL_PLAN_CONDS}"
  chunks_per_seed=$("$PYTHON" -c "print(len(range(0, $NEVALS, $CHUNK)))")
  total=$((3 * chunks_per_seed))
  for c in $CONDS; do
    logs="$ABLATION_ROOT/$c/plan_logs"
    done=$(grep -l 'Success rate:' "$logs"/gd_seed*_chunk*.log 2>/dev/null | wc -l | tr -d ' ' || true)
    echo "===== $c   $done/$total plan_logs ($total = 3 seeds x $chunks_per_seed) ====="
    [ -f "$logs/driver.log" ] && tail -1 "$logs/driver.log" | sed 's/^/  /' || echo "  (not started)"
  done
  ;;

results)
  "$PYTHON" "$ABLATION_ROOT/make_results.py"
  ;;

*) usage ;;
esac
