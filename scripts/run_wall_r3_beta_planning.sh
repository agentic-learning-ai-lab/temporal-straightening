#!/usr/bin/env bash
# Planning + aggregation for the wall fixed-direction R3 beta ladder.
#
# Waits for the checkpoints trained by run_wall_r3_beta_fixed_sequence.sh,
# then plans seeds 100/200/300 x 50 evals (5 chunks of 10) per condition and
# builds a comparison.json spanning the WHOLE direction=0.1 axis, reusing the
# existing wall plan outputs for the rungs that already ran:
#
#     r0_direction_only   plans/wall_dino_projector_full/on            speed 0
#     r2_full_matched     plans/wall_trajectory_penalty_ablations/...  speed 0.05
#     r3_beta1            plans/wall_trajectory_penalty_ablations/...  speed 0.1
#     r3_beta2/5/10       planned here                                 speed 0.2/0.5/1.0
#
# Safe to launch immediately after (or alongside) the training script -- it
# polls until each condition's epoch-20 checkpoint is complete.
#
# Usage (from a JupyterHub terminal):
#   nohup bash scripts/run_wall_r3_beta_planning.sh > logs/wall_r3_beta_planning.nohup 2>&1 &
#   tail -f logs/wall_r3_beta_planning.status

set -euo pipefail

cd "${REPO_DIR:-$HOME/temporal-straightening}"

checkpoint_root="${CKPT_ROOT:-$PWD/baseline_artifacts/checkpoints/wall_speed_ablations}"
output_root="${PLAN_ROOT:-$PWD/baseline_artifacts/plans/wall_r3_beta_fixed}"
status_log="$PWD/logs/wall_r3_beta_planning.status"
lock_file="$PWD/logs/wall_r3_beta_planning.lock"
max_gpu_used_mib="${PLANNING_GPU_MAX_USED_MIB:-1024}"
IFS=',' read -r -a gpu_ids <<< "${PLANNING_GPU_IDS:-1,2,3,4,5,6,7}"
gpu_count="${#gpu_ids[@]}"

conditions=(r3_beta2 r3_beta5 r3_beta10)
seeds=(100 200 300)
offsets=(0 10 20 30 40)

# Already-planned rungs on the same direction=0.1 axis, reused for comparison.
r0_plans="$PWD/baseline_artifacts/plans/wall_dino_projector_full/on"
prior_plans="$PWD/baseline_artifacts/plans/wall_trajectory_penalty_ablations"

mkdir -p "$output_root" "$PWD/logs"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "$(date -Is) WATCHER_ALREADY_RUNNING" >> "$status_log"
  exit 0
fi

cleanup_children() {
  local child_pid
  while read -r child_pid; do
    kill "$child_pid" 2>/dev/null || true
  done < <(jobs -pr)
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for condition in "${conditions[@]}"; do
  checkpoint="$checkpoint_root/$condition/checkpoints/model_20.pth"
  train_log="$checkpoint_root/$condition/train.log"
  while [[ ! -s "$checkpoint" ]] || ! grep -qE 'Epoch[[:space:]]+20[[:space:]]+Training loss:' "$train_log" 2>/dev/null; do
    echo "$(date -Is) WAITING condition=$condition checkpoint=$checkpoint epoch20_complete=false" >> "$status_log"
    sleep 60
  done
done

echo "$(date -Is) ALL_CHECKPOINTS_READY" >> "$status_log"

while true; do
  mapfile -t gpu_memory_used < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  busy_gpus=()
  for gpu_id in "${gpu_ids[@]}"; do
    if (( ${gpu_memory_used[$gpu_id]:-999999} > max_gpu_used_mib )); then
      busy_gpus+=("$gpu_id:${gpu_memory_used[$gpu_id]:-unknown}MiB")
    fi
  done
  if (( ${#busy_gpus[@]} == 0 )); then
    break
  fi
  echo "$(date -Is) WAITING_FOR_GPUS busy=${busy_gpus[*]}" >> "$status_log"
  sleep 60
done

echo "$(date -Is) PLANNING_GPUS_READY ids=${gpu_ids[*]}" >> "$status_log"

run_worker() {
  local worker_index="$1"
  local gpu_id="$2"
  local index=0
  local condition seed offset out rc

  for condition in "${conditions[@]}"; do
    for seed in "${seeds[@]}"; do
      for offset in "${offsets[@]}"; do
        if (( index % gpu_count == worker_index )); then
          out="$output_root/$condition/seed_$seed/chunk_$offset"
          mkdir -p "$out"
          if [[ -s "$out/logs.json" ]] && grep -q 'final_eval/success_rate' "$out/logs.json"; then
            echo "$(date -Is) SKIP condition=$condition seed=$seed offset=$offset" >> "$status_log"
          else
            echo "$(date -Is) START gpu=$gpu_id condition=$condition seed=$seed offset=$offset" >> "$status_log"
            set +e
            env \
              CUDA_VISIBLE_DEVICES="$gpu_id" \
              CPATH="$HOME/.conda/envs/ts/include" \
              LIBRARY_PATH="$HOME/.conda/envs/ts/lib" \
              LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:$HOME/.conda/envs/ts/lib:/usr/lib/nvidia" \
              MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210" \
              DATASET_DIR="$HOME/ts_data/data" \
              WANDB_MODE=disabled \
              HYDRA_FULL_ERROR=1 \
              "$HOME/.conda/envs/ts/bin/python" plan.py \
                --config-name plan_gd.yaml \
                "ckpt_base_path=$checkpoint_root/$condition" \
                "model_name=$condition" \
                model_epoch=20 \
                n_evals=10 \
                +wandb_logging=false \
                "seed=$seed" \
                "+eval_start_index=$offset" \
                "hydra.run.dir=$out" \
                > "$out/runner.log" 2>&1
            rc=$?
            set -e
            echo "$(date -Is) END rc=$rc gpu=$gpu_id condition=$condition seed=$seed offset=$offset" >> "$status_log"
          fi
        fi
        index=$((index + 1))
      done
    done
  done
}

for ((worker_index = 0; worker_index < gpu_count; worker_index++)); do
  run_worker "$worker_index" "${gpu_ids[$worker_index]}" &
done
wait

for condition in "${conditions[@]}"; do
  for seed in "${seeds[@]}"; do
    seed_dir="$output_root/$condition/seed_$seed"
    chunk_args=()
    for offset in "${offsets[@]}"; do
      chunk_args+=(--chunk "$offset:10:$seed_dir/chunk_$offset")
    done
    "$HOME/.conda/envs/ts/bin/python" aggregate_plan_chunks.py \
      "${chunk_args[@]}" \
      --seed "$seed" \
      --expected-evals 50 \
      --output "$seed_dir/aggregate.json" \
      > "$seed_dir/aggregate.stdout"
  done
done

"$HOME/.conda/envs/ts/bin/python" aggregate_condition_seeds.py \
  --condition "r0_direction_only=$r0_plans" \
  --condition "r2_full_matched=$prior_plans/r2_full_matched" \
  --condition "r3_beta1=$prior_plans/r3_beta1" \
  --condition "r3_beta2=$output_root/r3_beta2" \
  --condition "r3_beta5=$output_root/r3_beta5" \
  --condition "r3_beta10=$output_root/r3_beta10" \
  --baseline r0_direction_only \
  --output "$output_root/comparison.json" \
  > "$output_root/comparison.stdout"

echo "$(date -Is) PLANNING_FINISHED" >> "$status_log"
