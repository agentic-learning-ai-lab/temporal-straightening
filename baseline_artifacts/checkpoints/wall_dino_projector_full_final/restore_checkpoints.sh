#!/usr/bin/env bash
set -euo pipefail

root="baseline_artifacts/checkpoints/wall_dino_projector_full_final"

for condition in off on; do
  directory="$root/$condition"
  cat "$directory"/model_20.pth.part-* > "$directory/model_20.pth"
  (cd "$directory" && sha256sum -c model_20.pth.sha256)
done
