# Final Wall checkpoints

This bundle contains the two epoch-20 checkpoints used for the full Wall
DINOv2 patch + channel-projector planning comparison.

GitHub rejects ordinary Git objects larger than 100 MB, so each checkpoint is
stored as three numbered parts of at most 90 MiB. Run
`./restore_checkpoints.sh` from the repository root to reconstruct
`off/model_20.pth` and `on/model_20.pth` and verify their SHA-256 checksums.

- `off/`: no straightening loss
- `on/`: cosine straightening with coefficient 0.1
- `hydra.yaml`: exact training configuration
- `training.log`: complete 20-epoch training log
- `model_20.pth.sha256`: checksum for the reconstructed checkpoint

Only the final checkpoints are preserved here. Intermediate epoch checkpoints
are intentionally omitted because the two training directories total more
than 10 GB and are not needed to reproduce the reported planning results.
