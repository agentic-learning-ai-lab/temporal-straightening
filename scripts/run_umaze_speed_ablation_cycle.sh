#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-$HOME/temporal-straightening-umaze-speed}"

mkdir -p "$PWD/logs"
status_log="$PWD/logs/umaze_speed_cycle.status"
echo "$(date -Is) CYCLE_START" >> "$status_log"
bash scripts/run_umaze_speed_ablation_sequence.sh
echo "$(date -Is) TRAINING_FINISHED" >> "$status_log"
bash scripts/run_umaze_trajectory_planning.sh
echo "$(date -Is) CYCLE_FINISHED" >> "$status_log"
