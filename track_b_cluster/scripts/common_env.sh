#!/usr/bin/env bash
# Source inside SLURM jobs. All paths must resolve under CYFOK3_WORKSPACE.
set -euo pipefail

: "${CYFOK3_WORKSPACE:=/research/d7/fyp26/lkchau3/workspaces/cyfok3}"  # CONFIRM on cluster
export CYFOK3_WORKSPACE
export PROJ_ROOT="${PROJ_ROOT:-$CYFOK3_WORKSPACE/proj/csci5540}"
export BIGMAC_ROOT="${BIGMAC_ROOT:-$CYFOK3_WORKSPACE/third_party/BigMac}"
export MODEL_DIR="${MODEL_DIR:-$CYFOK3_WORKSPACE/models/Qwen2-VL-2B-Instruct}"
export HF_HOME="${HF_HOME:-$CYFOK3_WORKSPACE/caches/hf}"
export RUN_ROOT="${RUN_ROOT:-$CYFOK3_WORKSPACE/runs/track_b}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$RUN_ROOT"
cd "$PROJ_ROOT"
echo "[env] workspace=$CYFOK3_WORKSPACE proj=$PROJ_ROOT model=$MODEL_DIR use_pp_executor=${USE_PP_EXECUTOR:-unset}" >&2
git -C "$PROJ_ROOT" rev-parse --short HEAD >&2 || true
git -C "$BIGMAC_ROOT" rev-parse --short HEAD >&2 || true
singularity --version >&2 || true
python --version >&2 || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2 || true
