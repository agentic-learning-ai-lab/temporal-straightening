#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-$HOME/temporal-straightening-umaze-speed}"

checkpoint_root="$PWD/baseline_artifacts/checkpoints/umaze_speed_ablations"
status_log="$PWD/logs/umaze_speed_sequence.status"
lock_file="$PWD/logs/umaze_speed_sequence.lock"
target_epochs="${TARGET_EPOCHS:-20}"
gpu_max_used_mib="${TRAINING_GPU_MAX_USED_MIB:-1024}"

mkdir -p "$checkpoint_root" "$PWD/logs"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "$(date -Is) SEQUENCE_ALREADY_RUNNING" >> "$status_log"
  exit 0
fi

common_env=(
  "CPATH=$HOME/.conda/envs/ts/include"
  "LIBRARY_PATH=$HOME/.conda/envs/ts/lib"
  "LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$HOME/.conda/envs/ts/lib:/usr/lib/nvidia"
  "MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210"
  "DATASET_DIR=$HOME/ts_data/data"
  "WANDB_MODE=disabled"
  "HYDRA_FULL_ERROR=1"
)

cleanup_children() {
  local child_pid
  while read -r child_pid; do
    kill "$child_pid" 2>/dev/null || true
  done < <(jobs -pr)
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_gpus() {
  local gpu_csv="$1"
  local -a requested memory_used busy
  IFS=',' read -r -a requested <<< "$gpu_csv"
  while true; do
    mapfile -t memory_used < <(
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    )
    busy=()
    for gpu_id in "${requested[@]}"; do
      if (( ${memory_used[$gpu_id]:-999999} > gpu_max_used_mib )); then
        busy+=("$gpu_id:${memory_used[$gpu_id]:-unknown}MiB")
      fi
    done
    if (( ${#busy[@]} == 0 )); then
      return
    fi
    echo "$(date -Is) WAITING_FOR_GPUS requested=$gpu_csv busy=${busy[*]}" >> "$status_log"
    sleep 60
  done
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

run_one() {
  local token="$1" name="$2" gpu_csv="$3" port="$4"
  local out="$checkpoint_root/$name" done_epochs remaining rc num_processes
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
  num_processes="${#gpu_ids[@]}"
  mkdir -p "$out"
  done_epochs="$(completed_epoch "$out")"
  if (( done_epochs >= target_epochs )) && grep -qE "Epoch[[:space:]]+$target_epochs[[:space:]]+Training loss:" "$out/train.log" 2>/dev/null; then
    echo "$(date -Is) SKIP condition=$name completed_epoch=$done_epochs" >> "$status_log"
    return
  fi
  remaining=$((target_epochs - done_epochs))
  wait_for_gpus "$gpu_csv"
  echo "$(date -Is) START condition=$name token=$token gpus=$gpu_csv num_processes=$num_processes completed_epoch=$done_epochs remaining_epochs=$remaining" >> "$status_log"
  set +e
  env "${common_env[@]}" CUDA_VISIBLE_DEVICES="$gpu_csv" \
    "$HOME/.conda/envs/ts/bin/accelerate" launch \
      --num_processes "$num_processes" \
      --main_process_port "$port" \
      train.py \
      --config-name umaze_ablation_base \
      "training.straighten=$token" \
      "training.epochs=$remaining" \
      "ckpt_base_path=$out" \
      "hydra.run.dir=$out" \
      > "$out/launcher.log" 2>&1
  rc=$?
  set -e
  echo "$(date -Is) END condition=$name rc=$rc" >> "$status_log"
  if (( rc != 0 )); then
    return "$rc"
  fi
  if [[ ! -s "$out/checkpoints/model_${target_epochs}.pth" ]] || ! grep -qE "Epoch[[:space:]]+$target_epochs[[:space:]]+Training loss:" "$out/train.log"; then
    echo "$(date -Is) INVALID_COMPLETION condition=$name" >> "$status_log"
    return 1
  fi
  sha256sum "$out/checkpoints/model_${target_epochs}.pth" > "$out/model_${target_epochs}.pth.sha256"
}

run_wave() {
  local -a pids=() names=()
  while (( $# )); do
    local token="$1" name="$2" gpu_csv="$3" port="$4"
    shift 4
    run_one "$token" "$name" "$gpu_csv" "$port" &
    pids+=("$!")
    names+=("$name")
  done
  local index rc=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "$(date -Is) WAVE_FAILURE condition=${names[$index]}" >> "$status_log"
      rc=1
    fi
  done
  return "$rc"
}

# Two conditions run concurrently with the known-stable four-GPU/effective-batch-32 layout.
run_wave \
  aggr3b1_1e-1 r3_beta1 0,1,2,3 29710 \
  aggr1_1e-1 r1_speed_only 4,5,6,7 29720

# R2 runs after the first wave, so use the full node while preserving the
# configured effective batch size (train.py divides it across all processes).
run_wave aggr2_5e-2 r2_full_matched 0,1,2,3,4,5,6,7 29730

echo "$(date -Is) ALL_TRAINING_COMPLETE" >> "$status_log"
