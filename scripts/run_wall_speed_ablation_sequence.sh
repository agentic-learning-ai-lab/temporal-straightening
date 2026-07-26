#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-$HOME/temporal-straightening}"

base="$PWD/baseline_artifacts/checkpoints/wall_speed_ablations"
status_log="$PWD/logs/wall_speed_sequence.status"
r3_pid="${R3_PID:-355403}"
r3_output="${R3_OUTPUT:-$PWD/outputs/2026-07-17/04-59-42}"

mkdir -p "$base" "$PWD/logs"

echo "$(date -Is) WAITING_FOR_R3 pid=$r3_pid" >> "$status_log"
while kill -0 "$r3_pid" 2>/dev/null; do
  sleep 30
done

if [[ -d "$r3_output" && ! -e "$base/r3_beta1" ]]; then
  mv "$r3_output" "$base/r3_beta1"
fi
echo "$(date -Is) R3_COMPLETE" >> "$status_log"

run_one() {
  local token="$1"
  local name="$2"
  local out="$base/$name"
  local target_epochs="${TARGET_EPOCHS:-20}"
  local completed_epoch=0
  local remaining_epochs checkpoint_file checkpoint_name checkpoint_epoch

  shopt -s nullglob
  for checkpoint_file in "$out"/checkpoints/model_[0-9]*.pth; do
    checkpoint_name="${checkpoint_file##*/}"
    checkpoint_epoch="${checkpoint_name#model_}"
    checkpoint_epoch="${checkpoint_epoch%.pth}"
    if [[ "$checkpoint_epoch" =~ ^[0-9]+$ ]] && (( checkpoint_epoch > completed_epoch )); then
      completed_epoch="$checkpoint_epoch"
    fi
  done
  shopt -u nullglob

  if (( completed_epoch >= target_epochs )); then
    echo "$(date -Is) SKIP $name completed_epoch=$completed_epoch target_epochs=$target_epochs" >> "$status_log"
    return
  fi
  remaining_epochs=$((target_epochs - completed_epoch))

  echo "$(date -Is) START $name $token completed_epoch=$completed_epoch remaining_epochs=$remaining_epochs" >> "$status_log"
  env \
    CPATH="$HOME/.conda/envs/ts/include" \
    LIBRARY_PATH="$HOME/.conda/envs/ts/lib" \
    LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:$HOME/.conda/envs/ts/lib:/usr/lib/nvidia" \
    MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210" \
    DATASET_DIR="$HOME/ts_data/data" \
    WANDB_MODE=disabled \
    HYDRA_FULL_ERROR=1 \
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    "$HOME/.conda/envs/ts/bin/accelerate" launch \
      --num_processes 4 \
      --main_process_port "${MAIN_PROCESS_PORT:-29610}" \
      train.py \
      --config-name wall_ablation_base \
      "training.straighten=$token" \
      "training.epochs=$remaining_epochs" \
      "ckpt_base_path=$out" \
      "hydra.run.dir=$out" \
      > "$PWD/logs/wall_speed_$name.log" 2>&1
  echo "$(date -Is) END $name rc=0" >> "$status_log"
}

run_one aggr1_1e-1 r1_speed_only
run_one aggr2_5e-2 r2_full_matched
echo "$(date -Is) ALL_TRAINING_COMPLETE" >> "$status_log"
