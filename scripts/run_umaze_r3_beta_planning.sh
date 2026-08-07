#!/usr/bin/env bash
# Planning evaluation for the R3 beta sweep arms.
#
# Runs plan.py for every trained arm under CKPT_ROOT, SEEDS x N_EVALS each,
# following the protocol in run_cls_a100.sh: 3 seeds, 50 evals, gd planner.
# One arm per GPU, arms in parallel, seeds sequential within an arm.
#
# Reports per run into <arm>/plan_seed_<S>/logs.json:
#   success_rate, mean_state_dist, mean_visual_dist, mean_proprio_dist
# The per-episode success/state_dist values are printed by the evaluator and
# land in plan_seed_<S>.log, if a success-vs-epsilon curve is wanted later.
#
# NOTE: this rewrites env.dataset in each arm's hydra.yaml (data_path,
# use_frame_files=false, use_preprocessed=false). plan.py reads the dataset
# config from that frozen file and CLI overrides do not reach it. Only dataset
# loading fields change; model and training hyperparameters are untouched.
#
# Usage:
#   ./scripts/run_umaze_r3_beta_planning.sh
#   GPUS="3,4,5,7" SEEDS="100 200 300" ./scripts/run_umaze_r3_beta_planning.sh

set -euo pipefail

cd "${REPO_DIR:-$HOME/temporal-straightening}"

checkpoint_root="${CKPT_ROOT:-/opt/dlami/nvme/$USER/ckpt/umaze_r3_beta_sweep}"
plan_root="${PLAN_ROOT:-/opt/dlami/nvme/$USER/plans/umaze_r3_beta_sweep}"
dataset="${DATASET_DIR:-/opt/dlami/nvme/$USER/data}/point_maze"
status_log="$PWD/logs/umaze_r3_beta_planning.status"
lock_file="$PWD/logs/umaze_r3_beta_planning.lock"

model_epoch="${MODEL_EPOCH:-20}"
n_evals="${N_EVALS:-50}"
seeds="${SEEDS:-100 200 300}"
gpus="${GPUS:-3,4,5,7}"

mkdir -p "$plan_root" "$PWD/logs"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "$(date -Is) PLANNING_ALREADY_RUNNING" >> "$status_log"
  exit 0
fi

status() { echo "$(date -Is) $*" | tee -a "$status_log"; }

cleanup_children() {
  local child_pid
  while read -r child_pid; do
    kill "$child_pid" 2>/dev/null || true
  done < <(jobs -pr)
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Planning rolls out the real env, so mujoco_py must import. It builds an EGL
# extension on first use, which takes minutes -- do it once here rather than
# racing it across parallel arms.
if ! python -c "import mujoco_py" 2>/dev/null; then
  status "MUJOCO_PY_IMPORT_FAILED"
  echo "mujoco_py failed to import. Run 'python -c \"import mujoco_py\"' to see why." >&2
  exit 1
fi
status "MUJOCO_PY_OK"

# plan.py reads env.dataset from the frozen hydra.yaml, not from the CLI.
freeze_dataset_config() {
  python - "$1" "$dataset" <<'PY'
import sys
from omegaconf import OmegaConf
path, data = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(path)
cfg.env.dataset.data_path = data
cfg.env.dataset.use_frame_files = False
cfg.env.dataset.use_preprocessed = False
OmegaConf.save(cfg, path)
PY
}

plan_arm() {
  local arm="$1" gpu="$2"
  local run="$checkpoint_root/$arm"
  local out="$plan_root/$arm"
  mkdir -p "$out"

  freeze_dataset_config "$run/hydra.yaml"

  local seed
  for seed in $seeds; do
    if [[ -s "$out/plan_seed_$seed/logs.json" ]]; then
      status "SKIP arm=$arm seed=$seed"
      continue
    fi
    status "PLAN_START arm=$arm seed=$seed gpu=$gpu n_evals=$n_evals"
    if CUDA_VISIBLE_DEVICES="$gpu" python plan.py \
        --config-name plan_gd.yaml \
        ckpt_base_path="$run" \
        model_epoch="$model_epoch" \
        n_evals="$n_evals" \
        seed="$seed" \
        decode_for_viz=false \
        hydra.run.dir="$out/plan_seed_$seed" \
        > "$out/plan_seed_$seed.log" 2>&1
    then
      status "PLAN_DONE arm=$arm seed=$seed rate=$(
        grep -oE 'Success rate:[[:space:]]*[0-9.]+' "$out/plan_seed_$seed.log" \
          | tail -1 | grep -oE '[0-9.]+$')"
    else
      status "PLAN_FAILED arm=$arm seed=$seed (see $out/plan_seed_$seed.log)"
    fi
  done
}

# Arms with a finished checkpoint, paired with one GPU each.
IFS=',' read -r -a gpu_list <<< "$gpus"
arms=()
for d in "$checkpoint_root"/*/; do
  [[ -f "$d/checkpoints/model_${model_epoch}.pth" && -f "$d/hydra.yaml" ]] || continue
  arms+=("$(basename "$d")")
done

if (( ${#arms[@]} == 0 )); then
  status "NO_TRAINED_ARMS root=$checkpoint_root"
  exit 1
fi

status "PLANNING_START arms=${arms[*]} seeds=$seeds n_evals=$n_evals gpus=$gpus"

index=0
while (( index < ${#arms[@]} )); do
  pids=(); names=()
  for (( slot = 0; slot < ${#gpu_list[@]} && index + slot < ${#arms[@]}; slot++ )); do
    plan_arm "${arms[$((index + slot))]}" "${gpu_list[$slot]}" &
    pids+=("$!"); names+=("${arms[$((index + slot))]}")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || status "ARM_FAILED ${names[$i]}"
  done
  index=$((index + ${#gpu_list[@]}))
done

status "PLANNING_COMPLETE"
