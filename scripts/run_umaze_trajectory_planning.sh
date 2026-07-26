#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-$HOME/temporal-straightening-umaze-speed}"

checkpoint_root="$PWD/baseline_artifacts/checkpoints/umaze_speed_ablations"
output_root="$PWD/baseline_artifacts/plans/umaze_trajectory_penalty_ablations"
status_log="$PWD/logs/umaze_trajectory_planning.status"
lock_file="$PWD/logs/umaze_trajectory_planning.lock"
max_gpu_used_mib="${PLANNING_GPU_MAX_USED_MIB:-1024}"
IFS=',' read -r -a gpu_ids <<< "${PLANNING_GPU_IDS:-0,1,2,3,4,5,6,7}"
gpu_count="${#gpu_ids[@]}"

conditions=(r3_beta1 r1_speed_only r2_full_matched)
seeds=(100 200 300)
offsets=(0 10 20 30 40)

mkdir -p "$output_root" "$PWD/logs"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "$(date -Is) PLANNER_ALREADY_RUNNING" >> "$status_log"
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
    echo "$(date -Is) WAITING condition=$condition epoch20_complete=false" >> "$status_log"
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

run_worker() {
  local worker_index="$1" gpu_id="$2" index=0
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
            if (( rc != 0 )); then
              return "$rc"
            fi
          fi
        fi
        index=$((index + 1))
      done
    done
  done
}

pids=()
for ((worker_index = 0; worker_index < gpu_count; worker_index++)); do
  run_worker "$worker_index" "${gpu_ids[$worker_index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

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

# Normalize the committed complete R0 reproduction into the same per-seed schema.
for seed in "${seeds[@]}"; do
  r0_dir="$output_root/r0_direction_only/seed_$seed"
  mkdir -p "$r0_dir"
  "$HOME/.conda/envs/ts/bin/python" aggregate_plan_chunks.py \
    --chunk "0:50:$PWD/umaze_reproduction/plans/seed_$seed" \
    --seed "$seed" \
    --expected-evals 50 \
    --output "$r0_dir/aggregate.json" \
    > "$r0_dir/aggregate.stdout"
done

"$HOME/.conda/envs/ts/bin/python" aggregate_condition_seeds.py \
  --condition "r0=$output_root/r0_direction_only" \
  --condition "r1=$output_root/r1_speed_only" \
  --condition "r2=$output_root/r2_full_matched" \
  --condition "r3=$output_root/r3_beta1" \
  --baseline r0 \
  --output "$output_root/comparison.json" \
  > "$output_root/comparison.stdout"

"$HOME/.conda/envs/ts/bin/python" summarize_wall_trajectory_ablations.py \
  --environment UMaze \
  --comparison "$output_root/comparison.json" \
  --checkpoint-root "$checkpoint_root" \
  --output "$output_root/README.md" \
  > "$output_root/report.stdout"

echo "$(date -Is) PLANNING_FINISHED" >> "$status_log"
