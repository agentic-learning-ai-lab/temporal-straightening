#!/usr/bin/env bash
# setup_a100.sh — setup on the team's 8x A100-40GB box (p4d.24xlarge, JupyterHub,
# no root, no SLURM). Run with:  source setup_a100.sh   (must be SOURCED)
#
# DIFFERENCES FROM setup_aws.sh (why this file exists):
#   - NO sudo on this box            -> conda-forge for GL/OSMesa, not apt-get
#   - $HOME is on lv_ephemeral       -> see the warning below. This is not paranoia.
#   - 8 shared GPUs                  -> pin CUDA_VISIBLE_DEVICES to an idle one
#
# ⚠️⚠️  $HOME IS ON EPHEMERAL STORAGE  ⚠️⚠️
#   df shows /home/jupyter-deuk-* lives on /dev/mapper/vg.01-lv_ephemeral,
#   mounted at /opt/dlami/nvme. It is DESTROYED when the instance stops.
#   This already happened once: baseline_artifacts/plans/wall_reproduction/
#   seed_100/chunk_00/plan.log records
#       /opt/dlami/nvme/home/jupyter-deuk-b329/temporal-straightening/...
#   and this box is jupyter-deuk-e40b. That box, and its home, are gone.
#   A 20-epoch run is 3-7 hours. Set BACKUP_DIR (below) to somewhere persistent
#   or accept that you may do it twice.
#
# Get the dataset here first. This box is reached via JUPYTERHUB, not ssh, so
# there is likely no scp route in. Use the JupyterLab file panel:
#   JupyterLab -> file browser -> navigate to your home dir -> drag point_maze.zip in
#   (this is exactly what RUNBOOK.md step 1 describes for RunPod)
# Then open JupyterLab -> Terminal and run this script there.
#
# Override before sourcing:
#   ZIP=~/point_maze.zip DATA_ROOT=~/data BACKUP_DIR=s3://your-bucket/ts source setup_a100.sh

# NO `set -e` HERE, deliberately. This script is SOURCED, so `set -e` applies to
# your interactive shell: any failing command kills the whole terminal before you
# can read the error. setup_runpod.sh has that bug; do not copy it back in.
# Instead each step reports its own failure via _step below and we keep going,
# so you always get to see what broke.
ENV=ts310
ZIP="${ZIP:-$HOME/point_maze.zip}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
REPO="$(pwd)"
SETUP_LOG="${SETUP_LOG:-/tmp/setup_a100.log}"
: > "$SETUP_LOG"
_FAILED=0

# Run a step, tee its output to $SETUP_LOG, and report loudly on failure
# WITHOUT exiting the shell.
_step() {
  local name="$1"; shift
  echo ">> $name"
  if "$@" >>"$SETUP_LOG" 2>&1; then
    return 0
  else
    echo "!! STEP FAILED: $name"
    echo "!! last 20 lines (full log: $SETUP_LOG):"
    tail -20 "$SETUP_LOG" | sed 's/^/!!   /'
    _FAILED=1
    return 1
  fi
}

# --- ephemeral-disk warning -------------------------------------------------
if df -h "$HOME" | tail -1 | grep -q ephemeral; then
  echo ""
  echo "⚠️  \$HOME is on EPHEMERAL storage (lv_ephemeral / /opt/dlami/nvme)."
  echo "⚠️  Checkpoints written here are LOST when this instance stops."
  if [ -z "${BACKUP_DIR:-}" ]; then
    echo "⚠️  BACKUP_DIR is NOT set. Nothing will be backed up."
    echo "⚠️  Find a persistent path (see the probe in the chat) and re-source with:"
    echo "⚠️     BACKUP_DIR=/some/persistent/path source setup_a100.sh"
    echo "⚠️     BACKUP_DIR=s3://your-bucket/ts    source setup_a100.sh"
  else
    echo ">> BACKUP_DIR=$BACKUP_DIR  (run_cls_a100.sh will sync checkpoints here)"
  fi
  echo ""
fi
export BACKUP_DIR="${BACKUP_DIR:-}"

# --- conda env --------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo ">> installing miniconda to \$HOME (no root needed)"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3" >/dev/null
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^$ENV "; then
  echo ">> creating $ENV (python 3.10)"
  # 3.10, not 3.12: mujoco_py/d4rl are unmaintained and break on newer Python.
  conda create -n "$ENV" python=3.10 -y >/dev/null
fi
conda activate "$ENV"

# --- GL headers for mujoco_py, WITHOUT root ---------------------------------
# setup_aws.sh apt-gets these; there is no sudo here, so pull them from
# conda-forge into the env instead. mujoco_py compiles against these headers.
#
# OSMesa backend deps for mujoco_py.
#
# THE VERSION PIN IS LOAD-BEARING. `mesalib` MUST be 23.3.*:
#   - mesalib 26.0.3 ships NO libOSMesa and NO GL/osmesa.h. With it, the OSMesa
#     backend cannot compile at all (osmesashim.c: 'GL/osmesa.h: No such file').
#   - mesalib 23.3.* provides the header and pulls libllvm17, the llvmpipe
#     software rasterizer OSMesa renders with.
# There is no sudo here to apt-get libosmesa6-dev, and /usr/include has no GL
# or X11 headers either, so conda is the only source.
#
# Do NOT "simplify" this to EGL. EGL compiles fine given glew+xorg-libx11+
# xorg-xproto, then CRASHES AT RUNTIME on this A100 with
# 'malloc_consolidate(): unaligned fastbin chunk detected' in the render
# workers -- the exact failure plan_50_runpod.sh:36-38 documents. The xorg/glew
# packages below are kept only because they are tiny and harmless.
_step "GL/OSMesa deps via conda-forge (no root needed, ~2-5 min)" \
  conda install -y -q -c conda-forge "mesalib=23.3.*" glew glfw patchelf xorg-libx11 xorg-libxext xorg-xproto

# --- python deps ------------------------------------------------------------
_step "installing python deps (slow step, ~10-20 min)" \
  pip install -r "$REPO/reqs310.txt"
_step "pinning setuptools<81 (wandb needs pkg_resources)" \
  pip install "setuptools<81"

# --- MuJoCo 210 -------------------------------------------------------------
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  echo ">> installing MuJoCo 210"
  mkdir -p "$HOME/.mujoco"
  wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P "$HOME/.mujoco/"
  tar -xzf "$HOME/.mujoco/mujoco210-linux-x86_64.tar.gz" -C "$HOME/.mujoco/"
fi

# --- dataset ----------------------------------------------------------------
if [ ! -d "$DATA_ROOT/point_maze" ]; then
  if [ -f "$ZIP" ]; then
    echo ">> unzipping dataset to $DATA_ROOT (one-time, ~30GB)"
    mkdir -p "$DATA_ROOT"
    unzip -q "$ZIP" -d "$DATA_ROOT"
  else
    echo "!! no dataset zip at $ZIP"
    echo "!! Drag point_maze.zip into the JupyterLab file panel (home dir), then re-source."
    echo "!! (No ssh/scp on a JupyterHub box -- the file browser IS the upload path.)"
  fi
fi

# --- env vars ---------------------------------------------------------------
# conda's lib dir first so mujoco_py links the conda mesa/glew, not a system one.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export DATASET_DIR="$DATA_ROOT"
export WANDB_MODE=disabled
# OSMesa: EGL fails at runtime on headless cloud GPUs (proven on this A100 --
# see the conda install note above). MUJOCO_PY_FORCE_CPU selects mujoco_py's
# LinuxCPUExtensionBuilder (OSMesa) over LinuxGPUExtensionBuilder (EGL); its
# VALUE is ignored, only its PRESENCE matters, so unsetting it means EGL.
export MUJOCO_PY_FORCE_CPU=1
export MUJOCO_GL=osmesa

# --- make `dinov2` importable so CHECKPOINT RESUME works ---------------------
# The checkpoint pickles the encoder object, which holds the torch.hub-loaded
# DINOv2 model, so torch.load needs `import dinov2` to succeed. That package
# only lives in torch.hub's cache dir, which is on sys.path *only* after
# torch.hub.load() runs -- but train.py:264-267 calls load_ckpt BEFORE building
# the encoder. Without this, resuming from model_latest.pth dies with
#   ModuleNotFoundError: No module named 'dinov2'
# which silently breaks the "just re-run train, it resumes" recovery path.
HUB_DINO="$(ls -d "$HOME/.cache/torch/hub/facebookresearch_dinov2"* 2>/dev/null | head -1)"
if [ -n "$HUB_DINO" ]; then
  export PYTHONPATH="$HUB_DINO:${PYTHONPATH:-}"
  echo ">> PYTHONPATH += $HUB_DINO  (so checkpoint resume can unpickle dinov2)"
else
  echo "!! torch.hub dinov2 cache not found yet -- it appears on the first train run."
  echo "!! Re-source this script after that, or resume will fail on ModuleNotFoundError."
fi

# --- pick the idlest GPU (shared box: don't land on a teammate's job) --------
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IDLEST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="${IDLEST:-0}"
fi
echo ">> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (of 8; override if you want another)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

if [ "$_FAILED" != "0" ]; then
  echo ""
  echo "!! ==================================================================="
  echo "!! ONE OR MORE STEPS FAILED. Do NOT run prep/train yet."
  echo "!! Full log:  less $SETUP_LOG"
  echo "!! ==================================================================="
else
  echo ">> ready. env=$ENV  DATASET_DIR=$DATASET_DIR  BACKUP_DIR=${BACKUP_DIR:-<unset>}"
  echo ">> next: bash run_cls_a100.sh prep"
fi
