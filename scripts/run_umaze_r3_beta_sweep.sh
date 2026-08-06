#!/usr/bin/env bash
# R3 beta sweep on PointMaze UMaze.
#
# R3 is the combined trajectory penalty defined in models/visual_world_model.py:
#
#     r0 = 1 - cos(v1, v2)                  direction / straightness
#     r1 = (sqrt(s2/s1) - sqrt(s1/s2))^2    speed constancy
#     r3 = r0 + beta * r1
#
# beta sets how much the speed-constancy term counts relative to the
# straightness term. This script trains one model per beta value with the
# penalty scale held fixed, so beta is the only quantity that varies.
#
# beta is not a config field -- it is encoded in the training.straighten
# token as aggr3b<BETA>_<SCALE> (see visual_world_model.py:89-112).
#
# Usage:
#   ./scripts/run_umaze_r3_beta_sweep.sh              # full ladder, 8 arms
#   BETAS="0.05 20" ./scripts/run_umaze_r3_beta_sweep.sh   # cheap 2-arm probe
#
# On a shared node, raise TRAINING_GPU_MAX_USED_MIB above whatever resident
# inference servers hold, or wait_for_gpus will block forever.

set -euo pipefail

cd "${REPO_DIR:-$HOME/temporal-straightening}"

conda_env="${CONDA_ENV:-$HOME/.conda/envs/ts}"
checkpoint_root="${CKPT_ROOT:-$PWD/baseline_artifacts/checkpoints/umaze_r3_beta_sweep}"
status_log="$PWD/logs/umaze_r3_beta_sweep.status"
lock_file="$PWD/logs/umaze_r3_beta_sweep.lock"
target_epochs="${TARGET_EPOCHS:-20}"
gpu_max_used_mib="${TRAINING_GPU_MAX_USED_MIB:-1024}"

# Penalty scale is held constant across every arm so that beta is the only
# independent variable.
penalty_scale="${PENALTY_SCALE:-1e-1}"

# The four existing umaze ablations are already points on this beta axis,
# because r2 == 2*r0 + r1 == 2*r3(beta=0.5) exactly:
#
#   aggcos1e-1    r0_direction_only  ==  beta 0
#   aggr2_5e-2    r2_full_matched    ==  beta 0.5 at scale 1e-1
#   aggr3b1_1e-1  r3_beta1           ==  beta 1
#   aggr1_1e-1    r1_speed_only      ==  beta -> infinity
#
# So the arms below deliberately skip 0.5 and 1 and instead walk half-decade
# steps out to both limits: small beta approaches pure direction, large beta
# approaches pure speed constancy. Together with the four runs above this
# gives twelve points spanning four decades.
betas="${BETAS:-0.01 0.03 0.1 0.3 3 10 30 100}"

# Two arms train concurrently on four GPUs each, matching the known-stable
# effective-batch-32 layout used by the other umaze ablations.
gpu_group_a="${GPU_GROUP_A:-0,1,2,3}"
gpu_group_b="${GPU_GROUP_B:-4,5,6,7}"
base_port="${BASE_PORT:-29750}"
min_free_gib="${MIN_FREE_GIB:-50}"

mkdir -p "$checkpoint_root" "$PWD/logs"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "$(date -Is) SWEEP_ALREADY_RUNNING" >> "$status_log"
  exit 0
fi

common_env=(
  "CPATH=$conda_env/include"
  "LIBRARY_PATH=$conda_env/lib"
  "LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$conda_env/lib:/usr/lib/nvidia"
  "MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210"
  "DATASET_DIR=${DATASET_DIR:-$HOME/ts_data/data}"
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

# Checkpoints are large and land under $checkpoint_root. Fail here with a clear
# message rather than partway through training with "No space left on device".
check_disk() {
  local free_gib
  free_gib="$(df -Pk "$checkpoint_root" | awk 'NR==2 {print int($4/1048576)}')"
  if (( free_gib < min_free_gib )); then
    echo "$(date -Is) INSUFFICIENT_DISK path=$checkpoint_root free=${free_gib}GiB required=${min_free_gib}GiB" >> "$status_log"
    echo "Only ${free_gib}GiB free on the volume holding $checkpoint_root (need ${min_free_gib}GiB)." >&2
    echo "Point CKPT_ROOT at a larger volume or free space before training." >&2
    exit 1
  fi
  echo "$(date -Is) DISK_OK path=$checkpoint_root free=${free_gib}GiB" >> "$status_log"
}

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
    "$conda_env/bin/accelerate" launch \
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

# Directory names avoid dots so that model_20.pth globbing stays unambiguous.
condition_name() {
  echo "r3_beta${1//./p}"
}

check_disk

read -r -a beta_list <<< "$betas"
echo "$(date -Is) SWEEP_START betas=$betas scale=$penalty_scale epochs=$target_epochs" >> "$status_log"

# Arms are paired two-at-a-time; an odd final arm runs alone on group A.
index=0
while (( index < ${#beta_list[@]} )); do
  beta_a="${beta_list[$index]}"
  args=(
    "aggr3b${beta_a}_${penalty_scale}"
    "$(condition_name "$beta_a")"
    "$gpu_group_a"
    "$((base_port + index * 10))"
  )
  if (( index + 1 < ${#beta_list[@]} )); then
    beta_b="${beta_list[$((index + 1))]}"
    args+=(
      "aggr3b${beta_b}_${penalty_scale}"
      "$(condition_name "$beta_b")"
      "$gpu_group_b"
      "$((base_port + (index + 1) * 10))"
    )
  fi
  run_wave "${args[@]}"
  index=$((index + 2))
done

echo "$(date -Is) SWEEP_COMPLETE betas=$betas" >> "$status_log"
