#!/usr/bin/env bash
# Per-session setup for temporal-straightening on a fresh Lambda instance.
# Run with:  source setup.sh   (must be sourced, not bash'd, so env vars stick)

NFS=/lambda/nfs/temporal-straightening
ENV=ts310

# --- conda env (lives on local disk, wiped each instance, so rebuild) ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^$ENV "; then
  echo ">> creating $ENV (python 3.10)"
  conda create -n "$ENV" python=3.10 -y
fi
conda activate "$ENV"

# --- python deps (reqs310.txt persists on NFS) ---
echo ">> installing deps"
pip install -q -r "$NFS/reqs310.txt"
pip install -q "setuptools<81"   # wandb needs pkg_resources

# --- headless rendering deps (local disk, reinstall each instance) ---
# mujoco_py builds its EGL renderer on first import (needs GLEW headers) and
# needs the NVIDIA EGL vendor lib, which the headless driver doesn't ship.
if [ ! -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  echo ">> installing headless GL/EGL deps"
  sudo apt-get update -qq
  sudo apt-get install -y -qq libglew-dev libosmesa6-dev patchelf libegl1 \
    "libnvidia-gl-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)-server"
fi

# --- MuJoCo 210 (local disk, re-download if missing) ---
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  echo ">> installing MuJoCo 210"
  mkdir -p "$HOME/.mujoco"
  wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P "$HOME/.mujoco/"
  tar -xzf "$HOME/.mujoco/mujoco210-linux-x86_64.tar.gz" -C "$HOME/.mujoco/"
fi

# --- dataset: persist to NFS once, reuse after ---
if [ ! -d "$NFS/data/point_maze" ]; then
  if [ -d "$HOME/data/point_maze" ]; then
    echo ">> copying dataset to NFS (one-time)"
    cp -r "$HOME/data" "$NFS/data"
  else
    echo "!! no dataset found. fetch with:"
    echo "   osf -p bmw48 fetch osfstorage/datasets/point_maze.zip ~/point_maze.zip"
    echo "   mkdir -p ~/data && unzip ~/point_maze.zip -d ~/data/"
  fi
fi

# --- env vars ---
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export DATASET_DIR="$NFS/data"
export WANDB_MODE=disabled

echo ">> ready. env=$ENV  DATASET_DIR=$DATASET_DIR"
