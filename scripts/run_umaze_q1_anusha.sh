#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/temporal-straightening}"
DATASET_DIR="${DATASET_DIR:-$HOME/data}"
DATASET="$DATASET_DIR/point_maze"
CKPT_ROOT="${CKPT_ROOT:-$REPO_DIR/baseline_artifacts/checkpoints/umaze_q1_retrain}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$REPO_DIR/baseline_artifacts/analysis/umaze_latent_geodesic_r0_vs_r2}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/baseline_artifacts/logs}"
TARGET_EPOCHS="${TARGET_EPOCHS:-20}"
GPU_CSV="${GPU_CSV:-1,2,3,4}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PYTHON="${PYTHON:-$HOME/.conda/envs/ts310/bin/python}"
ACCELERATE="${ACCELERATE:-$HOME/.conda/envs/ts310/bin/accelerate}"

mkdir -p "$CKPT_ROOT" "$ANALYSIS_DIR" "$LOG_DIR" "$DATASET_DIR"
cd "$REPO_DIR"

export DATASET_DIR
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export MUJOCO_PY_FORCE_CPU="${MUJOCO_PY_FORCE_CPU:-1}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="$HOME/.conda/envs/ts310/lib:${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

status_log="$LOG_DIR/umaze_q1_anusha.status"

status() {
  echo "$(date -Is) $*" | tee -a "$status_log"
}

wait_for_setup() {
  local setup_pid=""
  [ -f "$HOME/umaze_setup.pid" ] && setup_pid="$(cat "$HOME/umaze_setup.pid")"
  while [ ! -f "$HOME/umaze_setup.exit" ]; do
    if [ -n "$setup_pid" ] && ! kill -0 "$setup_pid" 2>/dev/null; then
      status "SETUP_DIED pid=$setup_pid"
      return 1
    fi
    status "WAITING_FOR_SETUP pid=${setup_pid:-unknown}"
    sleep 60
  done

  [ -x "$PYTHON" ] || {
    status "SETUP_INVALID missing=$PYTHON"
    return 1
  }
  "$PYTHON" - <<'PY'
import accelerate
import gym
import hydra
import numpy
import scipy
import torch
print("PYTHON_ENV_OK", torch.__version__, torch.cuda.is_available())
PY
  status "SETUP_VALIDATED"
}

prepare_generator() {
  if [ ! -f scripts/generate_umaze_dataset.py ]; then
    cp generate_point_maze_medium.py scripts/generate_umaze_dataset.py
    sed -i 's/MEDIUM_MAZE/U_MAZE/g' scripts/generate_umaze_dataset.py
    sed -i 's/point_maze_medium_test/point_maze/g' scripts/generate_umaze_dataset.py
  fi
}

generate_dataset() {
  if [ -f "$DATASET/states.pth" ] &&
     [ -f "$DATASET/actions.pth" ] &&
     [ -f "$DATASET/seq_lengths.pth" ] &&
     [ ! -f "$DATASET/checkpoint.pth" ]; then
    status "DATASET_ALREADY_COMPLETE"
    return
  fi

  status "DATASET_GENERATION_START path=$DATASET"
  "$PYTHON" scripts/generate_umaze_dataset.py \
    --n_episodes 2000 \
    --episode_length 100 \
    --output_dir "$DATASET" \
    --policy random \
    --checkpoint_every 100 \
    > "$LOG_DIR/umaze_dataset_generation.log" 2>&1
  status "DATASET_GENERATION_COMPLETE"
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
episodes = list((root / "obses").glob("episode_*.pth"))
assert states.shape[:2] == (2000, 100), states.shape
assert actions.shape[:2] == (2000, 100), actions.shape
assert tuple(lengths.shape) == (2000,), lengths.shape
assert bool((lengths == 100).all()), torch.unique(lengths)
assert len(episodes) >= 2000, len(episodes)
print("DATASET_OK", states.shape, actions.shape, len(episodes))
PY
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
  [ -f "$sentinel" ]
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

completed_epoch() {
  local out="$1" checkpoint_file checkpoint_name checkpoint_epoch highest=0
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

wait_for_gpus() {
  local busy gpu pids
  while true; do
    busy=""
    IFS=',' read -r -a gpu_ids <<< "$GPU_CSV"
    for gpu in "${gpu_ids[@]}"; do
      pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
      [ -z "$pids" ] || busy="${busy} gpu${gpu}:$(echo "$pids" | paste -sd, -)"
    done
    if [ -z "$busy" ]; then
      status "GPU_READY devices=$GPU_CSV"
      return
    fi
    status "GPU_WAIT$busy"
    sleep 300
  done
}

run_condition() {
  local token="$1" name="$2" port="$3"
  local out="$CKPT_ROOT/$name" done_epochs remaining
  mkdir -p "$out"
  done_epochs="$(completed_epoch "$out")"
  if (( done_epochs >= TARGET_EPOCHS )); then
    status "TRAIN_SKIP condition=$name completed_epoch=$done_epochs"
    return
  fi
  remaining=$((TARGET_EPOCHS - done_epochs))
  wait_for_gpus
  status "TRAIN_START condition=$name token=$token gpus=$GPU_CSV completed_epoch=$done_epochs remaining=$remaining"

  CUDA_VISIBLE_DEVICES="$GPU_CSV" "$ACCELERATE" launch \
    --multi_gpu \
    --num_processes "$NUM_PROCESSES" \
    --main_process_port "$port" \
    train.py \
    --config-name umaze_ablation_base \
    "training.straighten=$token" \
    "training.epochs=$remaining" \
    "env.dataset.use_frame_files=true" \
    "ckpt_base_path=$out" \
    "hydra.run.dir=$out" \
    > "$out/launcher.log" 2>&1

  if [ ! -s "$out/checkpoints/model_${TARGET_EPOCHS}.pth" ]; then
    status "TRAIN_INVALID condition=$name"
    return 1
  fi
  sha256sum "$out/checkpoints/model_${TARGET_EPOCHS}.pth" \
    > "$out/checkpoints/model_${TARGET_EPOCHS}.pth.sha256"
  status "TRAIN_COMPLETE condition=$name"
}

main() {
  status "PIPELINE_START gpus=$GPU_CSV processes=$NUM_PROCESSES"
  wait_for_setup
  prepare_generator
  generate_dataset
  validate_dataset
  prepare_fast_loader
  prepare_dino_cache

  run_condition aggcos1e-1 r0_direction_only 29841
  run_condition aggr2_5e-2 r2_full_matched 29842

  status "Q1_ANALYSIS_START"
  CUDA_VISIBLE_DEVICES="$GPU_CSV" "$PYTHON" scripts/evaluate_umaze_latent_geodesic.py \
    --r0-checkpoint "$CKPT_ROOT/r0_direction_only" \
    --r2-checkpoint "$CKPT_ROOT/r2_full_matched" \
    --output-dir "$ANALYSIS_DIR" \
    --device cuda:0 \
    > "$ANALYSIS_DIR/analysis.log" 2>&1
  status "ALL_COMPLETE analysis=$ANALYSIS_DIR"
}

main "$@"
