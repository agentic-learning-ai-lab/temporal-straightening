#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/temporal-straightening}"
DATASET_DIR="${DATASET_DIR:-$HOME/data}"
DATASET="$DATASET_DIR/point_maze"
CKPT_ROOT="${CKPT_ROOT:-$REPO_DIR/baseline_artifacts/checkpoints/umaze_q1_retrain}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$REPO_DIR/baseline_artifacts/analysis/umaze_latent_geodesic_r0_vs_r2}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/baseline_artifacts/logs}"
TARGET_EPOCHS="${TARGET_EPOCHS:-20}"
PYTHON="${PYTHON:-$HOME/.conda/envs/ts310/bin/python}"
AWS_CLI="${AWS_CLI:-$HOME/.conda/envs/ts310/bin/aws}"

R2_PROFILE="${R2_PROFILE:-r2}"
R2_BUCKET="${R2_BUCKET:-temporal-straightening}"
R2_PREFIX="${R2_PREFIX:-umaze_q1_retrain}"
R2_ENDPOINT="${R2_ENDPOINT:-https://2914c19ff6db6db0ee4a54ff30e02f9c.r2.cloudflarestorage.com}"

mkdir -p "$CKPT_ROOT" "$ANALYSIS_DIR" "$LOG_DIR"
cd "$REPO_DIR"

export DATASET_DIR
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

status_log="$LOG_DIR/umaze_q1_retrain.status"

status() {
  echo "$(date -Is) $*" | tee -a "$status_log"
}

r2_cp() {
  local src="$1"
  local dst="$2"
  "$AWS_CLI" --profile "$R2_PROFILE" s3 cp "$src" "s3://$R2_BUCKET/$R2_PREFIX/$dst" \
    --region auto \
    --endpoint-url "$R2_ENDPOINT" \
    --only-show-errors
}

r2_sync() {
  local src="$1"
  local dst="$2"
  "$AWS_CLI" --profile "$R2_PROFILE" s3 sync "$src" "s3://$R2_BUCKET/$R2_PREFIX/$dst" \
    --region auto \
    --endpoint-url "$R2_ENDPOINT" \
    --only-show-errors
}

completed_epoch() {
  local out="$1"
  local checkpoint_file checkpoint_name checkpoint_epoch highest=0
  shopt -s nullglob
  for checkpoint_file in "$out"/checkpoints/model_[0-9]*.pth; do
    checkpoint_name="${checkpoint_file##*/}"
    checkpoint_epoch="${checkpoint_name#model_}"
    checkpoint_epoch="${checkpoint_epoch%.pth}"
    if [[ "$checkpoint_epoch" =~ ^[0-9]+$ ]] && (( checkpoint_epoch > highest )); then
      highest="$checkpoint_epoch"
    fi
  done
  shopt -u nullglob
  echo "$highest"
}

validate_dataset() {
  "$PYTHON" - "$DATASET" <<'PY'
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
states = torch.load(root / "states.pth", map_location="cpu")
actions = torch.load(root / "actions.pth", map_location="cpu")
lengths = torch.load(root / "seq_lengths.pth", map_location="cpu")
assert states.shape[:2] == (2000, 100), states.shape
assert actions.shape[:2] == (2000, 100), actions.shape
assert tuple(lengths.shape) == (2000,), lengths.shape
assert bool((lengths == 100).all()), torch.unique(lengths)
assert len(list((root / "obses").glob("episode_*.pth"))) >= 2000
print(
    f"DATASET_OK states={tuple(states.shape)} actions={tuple(actions.shape)} "
    f"episodes={len(lengths)}"
)
PY
}

wait_for_dataset() {
  status "WAITING_FOR_DATASET path=$DATASET"
  while [ -f "$DATASET/checkpoint.pth" ] || \
        [ ! -f "$DATASET/states.pth" ] || \
        [ ! -f "$DATASET/actions.pth" ] || \
        [ ! -f "$DATASET/seq_lengths.pth" ]; do
    sleep 30
  done
  validate_dataset
  status "DATASET_VALIDATED"
}

prepare_fast_loader() {
  local sentinel="$DATASET/obses/episode_000_frame_000.pth"
  if [ -f "$sentinel" ]; then
    status "FAST_LOADER_ALREADY_READY"
    return
  fi
  status "FAST_LOADER_PREPROCESS_START"
  "$PYTHON" preprocess_frames.py --data_path "$DATASET" \
    > "$LOG_DIR/umaze_preprocess_frames.log" 2>&1
  [ -f "$sentinel" ] || {
    status "FAST_LOADER_PREPROCESS_FAILED"
    return 1
  }
  status "FAST_LOADER_PREPROCESS_COMPLETE"
}

prepare_dino_cache() {
  status "DINO_CACHE_PREWARM_START"
  "$PYTHON" - <<'PY'
from models.dino import DinoV2Encoder

DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
print("DINO_CACHE_OK")
PY
  status "DINO_CACHE_PREWARM_COMPLETE"
}

wait_for_gpus() {
  local busy
  while true; do
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' | sort -u)"
    if [ -z "$busy" ]; then
      status "GPU_READY devices=$CUDA_VISIBLE_DEVICES"
      return
    fi
    status "GPU_WAIT busy_pids=$(echo "$busy" | paste -sd, -)"
    sleep 300
  done
}

backup_watcher() {
  declare -A last_stamp=()
  local name out latest stamp
  while true; do
    for name in r0_direction_only r2_full_matched; do
      out="$CKPT_ROOT/$name"
      latest="$out/checkpoints/model_latest.pth"
      [ -f "$latest" ] || continue
      stamp="$(stat -c %Y "$latest" 2>/dev/null || echo 0)"
      if [ "${last_stamp[$name]:-}" != "$stamp" ]; then
        status "BACKUP_START condition=$name stamp=$stamp"
        if r2_cp "$latest" "$name/checkpoints/model_latest.pth"; then
          [ -f "$out/hydra.yaml" ] && r2_cp "$out/hydra.yaml" "$name/hydra.yaml"
          [ -f "$out/train.log" ] && r2_cp "$out/train.log" "$name/train.log"
          last_stamp[$name]="$stamp"
          status "BACKUP_COMPLETE condition=$name stamp=$stamp"
        else
          status "BACKUP_FAILED condition=$name stamp=$stamp"
        fi
      fi
    done
    sleep 120
  done
}

run_condition() {
  local token="$1"
  local name="$2"
  local port="$3"
  local out="$CKPT_ROOT/$name"
  local done_epochs remaining

  mkdir -p "$out"
  done_epochs="$(completed_epoch "$out")"
  if (( done_epochs >= TARGET_EPOCHS )); then
    status "TRAIN_SKIP condition=$name completed_epoch=$done_epochs"
    return
  fi
  remaining=$(( TARGET_EPOCHS - done_epochs ))
  status "TRAIN_START condition=$name token=$token completed_epoch=$done_epochs remaining=$remaining"

  wait_for_gpus
  "$HOME/.conda/envs/ts310/bin/accelerate" launch \
    --multi_gpu \
    --num_processes 8 \
    --main_process_port "$port" \
    train.py \
    --config-name umaze_ablation_base \
    "training.straighten=$token" \
    "training.epochs=$remaining" \
    "env.dataset.use_frame_files=true" \
    "ckpt_base_path=$out" \
    "hydra.run.dir=$out" \
    > "$out/train.log" 2>&1

  if [ ! -f "$out/checkpoints/model_${TARGET_EPOCHS}.pth" ] || \
     ! grep -qE "Epoch[[:space:]]+${TARGET_EPOCHS}[[:space:]]+Training loss:" "$out/train.log"; then
    status "TRAIN_INVALID condition=$name"
    return 1
  fi

  sha256sum "$out/checkpoints/model_${TARGET_EPOCHS}.pth" \
    > "$out/checkpoints/model_${TARGET_EPOCHS}.pth.sha256"
  r2_cp "$out/checkpoints/model_${TARGET_EPOCHS}.pth" \
    "$name/checkpoints/model_${TARGET_EPOCHS}.pth"
  r2_cp "$out/checkpoints/model_${TARGET_EPOCHS}.pth.sha256" \
    "$name/checkpoints/model_${TARGET_EPOCHS}.pth.sha256"
  r2_cp "$out/hydra.yaml" "$name/hydra.yaml"
  r2_cp "$out/train.log" "$name/train.log"
  status "TRAIN_COMPLETE condition=$name"
}

main() {
  "$AWS_CLI" --profile "$R2_PROFILE" s3api head-bucket \
    --bucket "$R2_BUCKET" \
    --region auto \
    --endpoint-url "$R2_ENDPOINT" >/dev/null
  status "R2_VALIDATED bucket=$R2_BUCKET prefix=$R2_PREFIX"

  wait_for_dataset
  prepare_fast_loader
  prepare_dino_cache

  backup_watcher &
  watcher_pid=$!
  trap 'kill "$watcher_pid" 2>/dev/null || true' EXIT INT TERM

  run_condition aggcos1e-1 r0_direction_only 29740
  run_condition aggr2_5e-2 r2_full_matched 29741

  status "Q1_ANALYSIS_START"
  "$PYTHON" scripts/evaluate_umaze_latent_geodesic.py \
    --r0-checkpoint "$CKPT_ROOT/r0_direction_only" \
    --r2-checkpoint "$CKPT_ROOT/r2_full_matched" \
    --output-dir "$ANALYSIS_DIR" \
    --device cuda:0 \
    > "$ANALYSIS_DIR/analysis.log" 2>&1
  r2_sync "$ANALYSIS_DIR" analysis
  status "ALL_COMPLETE analysis=$ANALYSIS_DIR"
}

main "$@"
