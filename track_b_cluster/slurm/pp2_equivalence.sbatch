#!/bin/bash
#SBATCH --job-name=q2vl-pp2-equiv
#SBATCH --account=gpu
#SBATCH -p gpu_2h
#SBATCH --gres=gpu:2
#SBATCH --constraint=rtx3090
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu54,projgpu33
# PP=2 equivalence: native schedule vs pp_executor, identical seed/weights/data.
set -euo pipefail
: "${WS:=/research/d7/fyp26/lkchau3/workspaces/cyfok3}"   # override for your account
: "${BIND_ROOT:=/research}"                         # filesystem root that contains $WS
PROJ=$WS/proj/csci5540
MODE=${MODE:-0}
: "${NUM_LAYERS:=28}"
: "${NUM_MICROBATCHES:=2}"
NPROC=${PP:-2}
RUN_TAG=${RUN_TAG:-}
# Private writable paths also work when the node has no mounted login home.
mkdir -p "$WS/containers" "$WS/container-home" "$WS/libcuda"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$WS/libcuda/libcuda.so"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$WS/libcuda/libcuda.so.1"
LOG=$WS/containers/pp2-equiv-${MODE}${RUN_TAG:+-$RUN_TAG}.log
exec > "$LOG" 2>&1
set -x

SIF=$WS/containers/ngc-pytorch-25.03-py3.sif
PMP=$WS/third_party/BigMac/third_party/Pai-Megatron-Patch
BACKEND=$PMP/backends/megatron/Megatron-LM-250624
HF=$WS/models/Qwen2-VL-2B-Instruct
OUT=$PROJ/runs/track_b_n2_pilot_measured_v1/pp2_equivalence_mode${MODE}${RUN_TAG:+-$RUN_TAG}.json

export PYTHONPATH="$BACKEND:$PMP/examples:$PMP/examples/qwen2_vl:$PMP:$WS/third_party/BigMac:$WS/venvs/trackb_pydeps"
export SINGULARITYENV_LD_LIBRARY_PATH="$WS/libcuda:/.singularity.d/libs:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
export SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_HF_HOME=$WS/caches/hf
export SINGULARITYENV_TRITON_LIBCUDA_PATH="$WS/libcuda"
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_USE_PP_EXECUTOR=$MODE
export SINGULARITYENV_PROJ_ROOT=$PROJ
export SINGULARITYENV_MANIFEST="${MANIFEST:-$PROJ/runs/track_b_n2_pilot_measured_v1/pilot_manifest.json}"
export SINGULARITYENV_QWEN2VL_HF_DIR=$HF
export SINGULARITYENV_NUM_LAYERS=$NUM_LAYERS
export SINGULARITYENV_NUM_MICROBATCHES=$NUM_MICROBATCHES
export SINGULARITYENV_PP_SIZE=$NPROC

singularity exec --nv --home "$WS/container-home" \
  --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  -B "$BIND_ROOT:$BIND_ROOT" \
  "$SIF" \
  torchrun --standalone --tee 3 --nproc_per_node "$NPROC" --master_port 12666 \
  "$PROJ/track_b_cluster/bridge/pp2_equivalence.py" "$OUT"
echo "PP2_RC=$?"
