#!/usr/bin/env bash
# run_pusht_a100.sh — train the six DINOv2 PushT rows of Table 1 on one box.
#
#   Wang et al., "Temporal Straightening for Latent Planning", ICML 2026.
#   PushT column, DINOv2 encoder rows (the ResNet rows are a separate script).
#
# THE FOUR CONDITIONS (key -> encoder / straightening / paper open-loop target):
#   cls          dino_cls      OFF          1x384    19.33 +/- 8.22
#   patch        dino          OFF        14x14x384  56.00 +/- 4.32
#   channel_off  dino_channel  OFF        14x14x8    70.00 +/- 1.63
#   channel_on   dino_channel  aggcos1e-1 14x14x8    77.33 +/- 6.18   <- best (shaded)
#
# NO SMOKE STAGE — by request. Every shape that a smoke test would confirm is
# logged in the FIRST SECONDS of the real run and stays in the log:
#   - predictor pos_embedding shape       (train.py:218  "... pos_embedding shape=(1, N, D)")
#   - "Unknown projector 'none' ..."      (models/dino.py:145, patch-no-projector rows)
# So you verify dimensions DURING the run:  bash run_pusht_a100.sh dims
#
#   Expected pos_embedding = (1, tokens, emb):
#     emb (last dim):  404 = 384 DINO features + 10 proprio + 10 action
#                       28 =   8 channel-projector features + 10 + 10
#     tokens (middle): 3          -> pooled to 1 CLS token/frame  (cls)
#                      3 x 196=588 -> channel proj, 14x14 grid    (channel_*)
#                      3 x (grid)  -> raw patch, hundreds of tokens (patch)
#   Quick read: last dim 28 => channel; ==3 middle => cls; big middle => patch.
#
# PushT vs the PointMaze scripts:
#   - Dataset is VIDEO (decord), not per-frame files. There is NO fast-loader /
#     preprocess step here — that whole dance was PointMaze-only.
#   - Data lives at $DATASET_DIR/pusht_noise (conf/env/pusht.yaml).
#   - Training needs no MuJoCo; the pusht gym env only loads at PLANNING time.
#
# OUTPUT LAYOUT — each condition is self-contained under one root ($PUSHT_ROOT,
#   default ~/pusht-reproduction). checkpoints + train.log live together:
#     pusht-reproduction/
#     ├── cls_reproduction/            (cls)          train.log + test/<run>/checkpoints/
#     ├── patch_reproduction/          (patch)        ...
#     ├── patchproj_reproduction/      (channel_off)  ...
#     └── straightening_reproduction/  (channel_on)   ...
#
# encoder_lr: Table 3's footnote is "lr=1e-6 for no straightening", so the three
#   no-straightening rows (cls, patch, channel_off) use 1e-6 and only channel_on
#   uses the 1e-5 default. (For cls/patch the encoder has no trainable params, so
#   lr is a no-op anyway; separate output folders mean no run-dir collision to
#   work around, unlike run_row6_a100.sh.)
#
# EPOCHS default is 2 to match the paper's PushT protocol (A.3). This is
#   intentional and PushT-specific — see the EPOCHS line below.
#
# PLANNING reproduces Table 1's PushT column (GD planner, open-loop + MPC). Each
#   condition plans on its own GPU, detached, no smoke test. Hyperparams are the
#   plan-config defaults (= Table 4). PushT-specific: objective.alpha=1 on both,
#   plus objective.mode=staged for MPC (README + paper 5.3). 50 samples x 3 seeds,
#   run as 5x10 chunks and pooled (40GB can't hold a 50-eval 588-token GD batch).
#
# STAGES (JupyterHub terminal):
#   export DATASET_DIR=/home/jupyter-deuk-c4e4/data     # must contain pusht_noise/
#   bash run_pusht_a100.sh train                 # all 4, one per free GPU, detached, 2 epochs
#   bash run_pusht_a100.sh train channel_on cls  # OR just the ones you name
#   bash run_pusht_a100.sh dims                  # verify shapes on the LIVE runs
#   bash run_pusht_a100.sh status                # epochs done + checkpoints
#   bash run_pusht_a100.sh plan                  # after training: all 4 plan, one per GPU, detached
#   bash run_pusht_a100.sh plan-status           # planning progress
#   bash run_pusht_a100.sh plan-summary          # success rates vs paper Table 1
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
PUSHT_ROOT="${PUSHT_ROOT:-$HOME/pusht-reproduction}"   # one root; a folder per condition
DATA="${DATA:-${DATASET_DIR:-$HOME/data}/pusht_noise}"
# PushT trains for 2 epochs, NOT 20 (paper A.3: 18500 trajectories, len 100-300).
# Wall/PointMaze use 20; PushT is the documented exception. Do not bump this to 20.
EPOCHS="${EPOCHS:-2}"
FREE_MIB="${FREE_MIB:-2000}"   # a GPU with less than this used (MiB) counts as free

# --- planning (Table 1 GD planner; hyperparams from Table 4, all plan-config defaults) ---
SEEDS="${SEEDS:-100 200 300}"   # 3 data seeds (Table 1 = mean +/- std over three)
NEVALS="${NEVALS:-50}"          # 50 test samples per seed (Table 1)
CHUNK="${CHUNK:-10}"            # 40GB can't hold a 50-eval 588-token GD batch -> 5x10, pooled
PLAN_DECODE="${PLAN_DECODE:-false}"  # success rate is unaffected; skip decode for speed/robustness
STAGE="${1:-}"; shift || true

ALL_CONDS="cls patch channel_off channel_on"

usage() { echo "usage: bash run_pusht_a100.sh {train|dims|status|plan|plan-status|plan-summary} [cond ...]"; echo "  conds: $ALL_CONDS"; exit 1; }
[ -n "$STAGE" ] || usage
[ -f "$REPO/train.py" ] || { echo "!! run from the repo root (train.py not found)"; exit 1; }

# torch.hub.load() needs the cached dinov2 repo importable.
if [ -z "${PYTHONPATH:-}" ] || ! echo "${PYTHONPATH:-}" | grep -q facebookresearch_dinov2; then
  HUB_DINO="$(ls -d "$HOME/.cache/torch/hub/facebookresearch_dinov2"* 2>/dev/null | head -1)"
  [ -n "$HUB_DINO" ] && export PYTHONPATH="$HUB_DINO:${PYTHONPATH:-}"
fi

# Detached runs have no tty, so wandb.init() would try (and fail) to log in and
# crash every run. setup_a100.sh sets this; we're standalone, so set it here too.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# cond -> "encoder=<..> training.straighten=<..> training.encoder_lr=<..>"
cond_flags() {
  case "$1" in
    cls)         echo "encoder=dino_cls     training.straighten=False      training.encoder_lr=1e-6" ;;
    patch)       echo "encoder=dino         training.straighten=False      training.encoder_lr=1e-6" ;;
    channel_off) echo "encoder=dino_channel training.straighten=False      training.encoder_lr=1e-6" ;;
    channel_on)  echo "encoder=dino_channel training.straighten=aggcos1e-1 training.encoder_lr=1e-5" ;;
    *) echo "!! unknown cond '$1' (want: $ALL_CONDS)" >&2; return 1 ;;
  esac
}

# cond -> its self-contained output folder under $PUSHT_ROOT
cond_dir() {
  case "$1" in
    cls)         echo "$PUSHT_ROOT/cls_reproduction" ;;
    patch)       echo "$PUSHT_ROOT/patch_reproduction" ;;
    channel_off) echo "$PUSHT_ROOT/patchproj_reproduction" ;;
    channel_on)  echo "$PUSHT_ROOT/straightening_reproduction" ;;
  esac
}

# each cond folder holds exactly one hydra run under test/ -> just take it
find_run_dir() {
  find "$(cond_dir "$1")/test" -maxdepth 1 -type d -name 'pusht_*' 2>/dev/null | head -1
}

CONDS="${*:-$ALL_CONDS}"
for c in $CONDS; do cond_flags "$c" >/dev/null || usage; done

case "$STAGE" in

train)
  [ -d "$DATA" ] || { echo "!! dataset dir missing: $DATA"; echo "   set DATASET_DIR so \$DATASET_DIR/pusht_noise exists (see README > Datasets)."; exit 1; }

  # Pick free GPUs unless GPUS is given explicitly (comma list, e.g. GPUS=2,3,4).
  if [ -n "${GPUS:-}" ]; then
    read -r -a GPU_ARR <<< "$(echo "$GPUS" | tr ',' ' ')"
  else
    mapfile -t GPU_ARR < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v t="$FREE_MIB" '($2+0)<t {gsub(/ /,"",$1); print $1}')
  fi
  N_GPU=${#GPU_ARR[@]}
  [ "$N_GPU" -gt 0 ] || { echo "!! no free GPU (used < ${FREE_MIB} MiB). Set GPUS=... to force."; exit 1; }
  echo "[gpu] free/assigned GPUs: ${GPU_ARR[*]}"

  i=0
  for c in $CONDS; do
    if [ "$i" -ge "$N_GPU" ]; then
      echo "!! only $N_GPU GPU(s) — skipping '$c' and any after it. Re-run 'train $c ...' when a GPU frees."
      break
    fi
    g="${GPU_ARR[$i]}"; i=$((i+1))
    dir="$(cond_dir "$c")"; mkdir -p "$dir"; log="$dir/train.log"
    read -r -a F <<< "$(cond_flags "$c")"
    echo "----- launch '$c' on GPU $g -> $dir -----"
    # setsid + nohup: new session AND SIGHUP ignored. Fully detaches from this
    # terminal, so closing the browser / logging out of JupyterLab does NOT stop
    # it. (A JupyterHub *idle cull* kills the whole cgroup regardless — see below.)
    CUDA_VISIBLE_DEVICES="$g" setsid nohup python train.py --config-name train.yaml \
      env=pusht "${F[@]}" training.epochs="$EPOCHS" ckpt_base_path="$dir" \
      > "$log" 2>&1 &
    echo "  pid $!   flags: ${F[*]}"
  done
  echo ""
  echo "All launched detached (setsid+nohup): closing the browser / logging out"
  echo "of JupyterLab does NOT stop them. The ONE thing that does is a JupyterHub"
  echo "idle cull, which kills the whole cgroup — but files persist and train.py"
  echo "resumes from model_latest.pth, so just re-run the same 'train <conds>'."
  echo "Verify shapes on the live runs:  bash run_pusht_a100.sh dims"
  ;;

dims)
  # Read the shape/projector lines straight out of the running logs.
  for c in $CONDS; do
    log="$(cond_dir "$c")/train.log"
    echo "===== $c ====="
    if [ ! -f "$log" ]; then echo "  (no log — not launched)"; continue; fi
    grep -m1 -i 'pos_embedding[^=]*shape' "$log" 2>/dev/null | sed 's/^/  /' \
      || echo "  (pos_embedding not logged yet — give it a few seconds)"
    grep -m1 -i "Unknown projector" "$log" 2>/dev/null | sed 's/^/  /' || true
    grep -m1 -iE 'Traceback|Error|CUDA out of memory|FileNotFound' "$log" 2>/dev/null | sed 's/^/  !! /' || true
  done
  echo ""
  echo "expect: cls -> (1, 3, 404)   channel_* -> (1, 588, 28)   patch -> (1, <big>, 404)"
  ;;

status)
  for c in $CONDS; do
    log="$(cond_dir "$c")/train.log"; run="$(find_run_dir "$c")"
    ep=$(grep -c "Training loss" "$log" 2>/dev/null || echo 0)
    echo "===== $c -> $(cond_dir "$c")   (epochs w/ loss logged: $ep / $EPOCHS) ====="
    [ -f "$log" ] && tail -2 "$log" | sed 's/^/  /' || echo "  (no log yet)"
    [ -n "$run" ] && ls -1 "$run/checkpoints/" 2>/dev/null | sed 's/^/  ckpt: /' || echo "  (no run dir yet)"
  done
  ;;

plan)
  # Fan out: one detached (setsid+nohup) planning job PER CONDITION, each pinned
  # to its own GPU. Each job runs open-loop GD + GD-MPC x 3 seeds x 5 chunks.
  [ -d "$DATA" ] || { echo "!! dataset dir missing: $DATA (need it for the valid split)"; exit 1; }

  # No smoke test. Just verify each condition has a finished checkpoint + frozen
  # config so a doomed job doesn't launch (file checks only -- nothing is run).
  for c in $CONDS; do
    RUN="$(find_run_dir "$c")"
    [ -n "$RUN" ]                                   || { echo "!! $c: no run dir -- train first"; exit 1; }
    [ -f "$RUN/hydra.yaml" ]                        || { echo "!! $c: missing $RUN/hydra.yaml"; exit 1; }
    [ -f "$RUN/checkpoints/model_${EPOCHS}.pth" ]   || { echo "!! $c: missing checkpoints/model_${EPOCHS}.pth"; exit 1; }
  done

  if [ -n "${GPUS:-}" ]; then
    read -r -a GPU_ARR <<< "$(echo "$GPUS" | tr ',' ' ')"
  else
    mapfile -t GPU_ARR < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F',' -v t="$FREE_MIB" '($2+0)<t {gsub(/ /,"",$1); print $1}')
  fi
  N_GPU=${#GPU_ARR[@]}
  [ "$N_GPU" -gt 0 ] || { echo "!! no free GPU (used < ${FREE_MIB} MiB). Set GPUS=... to force."; exit 1; }
  echo "[gpu] planning GPUs: ${GPU_ARR[*]}"

  i=0
  for c in $CONDS; do
    if [ "$i" -ge "$N_GPU" ]; then
      echo "!! only $N_GPU GPU(s) -- skipping '$c' and after. Re-run 'plan $c ...' when a GPU frees."
      break
    fi
    g="${GPU_ARR[$i]}"; i=$((i+1))
    drv="$(cond_dir "$c")/plan"; mkdir -p "$drv"
    echo "----- plan '$c' on GPU $g -> $drv -----"
    # Re-invoke this script's hidden per-condition worker, detached, pinned to $g.
    CUDA_VISIBLE_DEVICES="$g" setsid nohup bash "$0" _plan_one "$c" > "$drv/driver.log" 2>&1 &
    echo "  pid $!"
  done
  echo ""
  echo "Detached. Monitor:  bash run_pusht_a100.sh plan-status"
  echo "Summarize:          bash run_pusht_a100.sh plan-summary"
  ;;

_plan_one)
  # Hidden worker: full planning suite for ONE condition on the current GPU.
  c="$CONDS"
  RUN="$(find_run_dir "$c")"; RUN="$(cd "$RUN" && pwd)"   # plan.py needs an absolute path
  drv="$(cond_dir "$c")/plan"; mkdir -p "$drv"

  # plan.py reads env.dataset from the FROZEN hydra.yaml (CLI overrides don't reach it).
  # Point it at the actual data dir; pusht is video, so no frame-file flags to set.
  python - "$RUN/hydra.yaml" "$DATA" <<'PY'
import sys
from omegaconf import OmegaConf
p, data = sys.argv[1], sys.argv[2]
cfg = OmegaConf.load(p)
cfg.env.dataset.data_path = data
OmegaConf.save(cfg, p)
print(f"[config] set data_path={data} in {p}")
PY

  OFFSETS=$(python -c "print(' '.join(str(o) for o in range(0, $NEVALS, $CHUNK)))")
  # tag | plan-config | PushT-specific objective overrides (README + paper 5.3)
  for spec in \
      "gd|plan_gd.yaml|objective.alpha=1" \
      "mpc|plan_gd_mpc.yaml|objective.alpha=1 objective.mode=staged"; do
    tag="${spec%%|*}"; rest="${spec#*|}"; cfg="${rest%%|*}"; extra="${rest#*|}"
    for S in $SEEDS; do
      for O in $OFFSETS; do
        OO=$(printf "%02d" "$O")
        echo "===== $c $tag seed $S chunk $OO (n_evals=$CHUNK) ====="
        python plan.py --config-name "$cfg" \
          ckpt_base_path="$RUN" \
          model_epoch="$EPOCHS" \
          n_evals="$CHUNK" \
          +eval_start_index="$O" \
          seed="$S" \
          decode_for_viz="$PLAN_DECODE" \
          $extra \
          hydra.run.dir="$drv/${tag}_seed_${S}/chunk_${OO}" \
          2>&1 | tee "$drv/${tag}_seed${S}_chunk${OO}.log"
      done
    done
  done
  echo "PLAN_DONE cond=$c"
  ;;

plan-status)
  for c in $CONDS; do
    drv="$(cond_dir "$c")/plan"
    gd=$(grep -l "Success rate" "$drv"/gd_seed*_chunk*.log 2>/dev/null | wc -l | tr -d ' ')
    mpc=$(grep -l "Success rate" "$drv"/mpc_seed*_chunk*.log 2>/dev/null | wc -l | tr -d ' ')
    echo "===== $c   gd $gd/15   mpc $mpc/15   (15 = 3 seeds x 5 chunks) ====="
    [ -f "$drv/driver.log" ] && tail -1 "$drv/driver.log" | sed 's/^/  /' || echo "  (not started)"
  done
  ;;

plan-summary)
  python - "$PUSHT_ROOT" "$NEVALS" "$CHUNK" "$CONDS" <<'PY'
import sys, os, re, glob, statistics as st
root, nevals, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
conds = sys.argv[4].split()
folder = {"cls":"cls_reproduction","patch":"patch_reproduction",
          "channel_off":"patchproj_reproduction","channel_on":"straightening_reproduction"}
# Paper Table 1, PushT, GD planner: (open-loop, MPC)
paper = {"cls":("19.33 +/- 8.22","28.00 +/- 1.63"),
         "patch":("56.00 +/- 4.32","66.00 +/- 4.90"),
         "channel_off":("70.00 +/- 1.63","78.67 +/- 0.94"),
         "channel_on":("77.33 +/- 6.18","85.33 +/- 4.99")}
offsets = list(range(0, nevals, chunk))
def rate(cond, tag):
    drv = os.path.join(root, folder[cond], "plan")
    seed_means = []
    for s in ("100","200","300"):
        chunks = []
        for o in offsets:
            log = os.path.join(drv, f"{tag}_seed{s}_chunk{o:02d}.log")
            if not os.path.exists(log): return None
            vals = re.findall(r"Success rate:\s*([0-9.]+)", open(log, errors="ignore").read())
            if not vals: return None
            chunks.append(float(vals[-1]) * 100.0)
        seed_means.append(sum(chunks)/len(chunks))   # equal chunks -> pooled 50-sample rate
    m = sum(seed_means)/len(seed_means)
    sd = st.pstdev(seed_means) if len(seed_means) > 1 else 0.0
    return m, sd, seed_means
print(f"\nPushT GD planner — OURS vs paper Table 1 (50 samples x 3 seeds)\n")
print(f"{'condition':<13} {'open-loop (ours)':>20}  {'paper':>16}   {'MPC (ours)':>18}  {'paper':>16}")
for c in conds:
    ol, mpc = rate(c,"gd"), rate(c,"mpc")
    ol_s  = f"{ol[0]:.2f} +/- {ol[1]:.2f}"  if ol  else "(pending)"
    mpc_s = f"{mpc[0]:.2f} +/- {mpc[1]:.2f}" if mpc else "(pending)"
    print(f"{c:<13} {ol_s:>20}  {paper[c][0]:>16}   {mpc_s:>18}  {paper[c][1]:>16}")
print("\n(open-loop = plan_gd.yaml objective.alpha=1; MPC = plan_gd_mpc.yaml alpha=1 mode=staged)")
PY
  ;;

*) usage ;;
esac
