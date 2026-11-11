#!/usr/bin/env bash
# Source inside SLURM jobs. All paths resolve under $WS (your workspace).
#
# Point WS at any directory you own with ~40 GB free on a filesystem the GPU nodes can see:
#   export WS=/research/<group>/<user>/workspaces/<name>   BIND_ROOT=/research
#   export WS=$HOME/csci5540                              BIND_ROOT=$HOME
set -euo pipefail

: "${WS:=${CYFOK3_WORKSPACE:-/research/d7/fyp26/lkchau3/workspaces/cyfok3}}"  # override for your account
export WS
export CYFOK3_WORKSPACE="$WS"  # legacy alias used by the older sbatch templates
: "${BIND_ROOT:=/research}"    # filesystem root that contains $WS; mount it into the container
export BIND_ROOT
export PROJ_ROOT="${PROJ_ROOT:-$WS/proj/csci5540}"
export BIGMAC_ROOT="${BIGMAC_ROOT:-$WS/third_party/BigMac}"
export MODEL_DIR="${MODEL_DIR:-$WS/models/Qwen2-VL-2B-Instruct}"
export HF_HOME="${HF_HOME:-$WS/caches/hf}"
export RUN_ROOT="${RUN_ROOT:-$WS/runs/track_b}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$RUN_ROOT"
cd "$PROJ_ROOT"
echo "[env] workspace=$WS bind_root=$BIND_ROOT proj=$PROJ_ROOT model=$MODEL_DIR use_pp_executor=${USE_PP_EXECUTOR:-unset}" >&2
git -C "$PROJ_ROOT" rev-parse --short HEAD >&2 || true
git -C "$BIGMAC_ROOT" rev-parse --short HEAD >&2 || true
singularity --version >&2 || true
python --version >&2 || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2 || true
