"""
One-time preprocessing for PointMaze (umaze / medium) to enable the fast
data-loading path with per-frame files.

WHY: the default loader re-reads a WHOLE ~15 MB episode tensor for EVERY
training sample (datasets/point_maze_dset.py:120) -- ~2 TB of disk reads per
epoch on umaze. Writing one small file per frame lets training read only the
~4 frames it needs (~25x less I/O), which is the real epoch-time bottleneck.

TWO MODES (pick based on disk budget):

  default  -> raw uint8 frames  [H, W, C]   (~30 GB, RECOMMENDED)
             set in conf/env/point_maze.yaml:
                 use_frame_files: true
                 use_preprocessed: false
             loader applies /255 + transform at load (cheap on 224x224).

  --preprocessed -> float32 frames [C, H, W]  (~117 GB, only if disk is huge)
             set in conf/env/point_maze.yaml:
                 use_frame_files: true
                 use_preprocessed: true
             transform is baked in; nothing done at load.

Both write to <data_path>/obses/episode_XXX_frame_YYY.pth and match EXACTLY
what the default loader produces (datasets/point_maze_dset.py:117-130).
Run ONCE on CPU (no GPU needed). Resumable: skips files that already exist.

Usage (on the pod, after DATASET_DIR is set and data is on LOCAL disk):
    python preprocess_frames.py --data_path $DATASET_DIR/point_maze
    # or, if you have >120 GB of fast disk to spare:
    python preprocess_frames.py --data_path $DATASET_DIR/point_maze --preprocessed
"""
import argparse
from pathlib import Path

import torch
from datasets.img_transforms import default_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="e.g. $DATASET_DIR/point_maze")
    ap.add_argument("--img_size", type=int, default=224,
                    help="must match conf/train.yaml img_size (default 224)")
    ap.add_argument("--preprocessed", action="store_true",
                    help="bake in the transform as float32 [C,H,W] (~4x disk); "
                         "otherwise store raw uint8 [H,W,C] (recommended)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate frame files even if they already exist")
    args = ap.parse_args()

    data_path = Path(args.data_path)
    obs_dir = data_path / "obses"
    if not obs_dir.exists():
        raise SystemExit(f"No obses/ dir at {obs_dir} -- is data_path correct?")

    seq_lengths = torch.load(data_path / "seq_lengths.pth")
    n_episodes = len(seq_lengths)
    transform = default_transform(args.img_size) if args.preprocessed else None

    mode = "preprocessed float32 [C,H,W]" if args.preprocessed else "raw uint8 [H,W,C]"
    print(f">> {n_episodes} episodes  ->  {obs_dir}")
    print(f">> mode: {mode}")

    total_written = 0
    for idx in range(n_episodes):
        ep_file = obs_dir / f"episode_{idx:03d}.pth"
        if not ep_file.exists():
            raise SystemExit(f"Missing episode tensor: {ep_file}")

        # [T, H, W, C]; bound by the real (unpadded) length so we skip padding.
        episode = torch.load(ep_file, map_location="cpu")
        T = int(seq_lengths[idx])

        for t in range(T):
            out = obs_dir / f"episode_{idx:03d}_frame_{t:03d}.pth"
            if out.exists() and not args.overwrite:
                continue
            frame = episode[t]                         # [H, W, C]
            if args.preprocessed:
                frame = frame.float() / 255.0          # [H, W, C]
                frame = frame.permute(2, 0, 1)         # [C, H, W]
                frame = transform(frame)               # [C, img_size, img_size]
            # else: store raw [H, W, C] exactly as the episode tensor holds it,
            # so the loader's /255 + rearrange + transform reproduces the default.
            torch.save(frame.clone(), out)
            total_written += 1

        if (idx + 1) % 50 == 0 or idx == n_episodes - 1:
            print(f"   episode {idx + 1}/{n_episodes}  (frames written: {total_written})")

    print(f">> done. wrote {total_written} frame files.")
    print(">> now set in conf/env/point_maze.yaml:")
    print("     use_frame_files: true")
    print(f"     use_preprocessed: {str(args.preprocessed).lower()}")


if __name__ == "__main__":
    main()
