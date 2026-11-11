#!/bin/bash
#SBATCH --job-name=q2vl-pp2-equiv
#SBATCH --account=gpu
#SBATCH -p gpu_2h
#SBATCH --gres=gpu:2
#SBATCH --constraint=rtx3090
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu54
# PP=2 equivalence: native schedule vs pp_executor, identical seed/weights/data.
set -euo pipefail
: "${WS:=/research/d7/fyp26/lkchau3/workspaces/cyfok3}"   # override for your account
: "${BIND_ROOT:=/research}"                         # filesystem root that contains $WS
PROJ=$WS/proj/csci5540
MODE=${MODE:-0}
LOG=$WS/containers/pp2-equiv-${MODE}.log
exec > "$LOG" 2>&1
set -x

SIF=$WS/containers/ngc-pytorch-25.03-py3.sif
PMP=$WS/third_party/BigMac/third_party/Pai-Megatron-Patch
BACKEND=$PMP/backends/megatron/Megatron-LM-250624
HF=$WS/models/Qwen2-VL-2B-Instruct
OUT=$PROJ/runs/track_b_n2_pilot_measured_v1/pp2_equivalence_mode${MODE}.json

export PYTHONPATH="$BACKEND:$PMP/examples:$PMP/examples/qwen2_vl:$PMP:$WS/third_party/BigMac:$WS/venvs/trackb_pydeps"
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_HF_HOME=$WS/caches/hf
export SINGULARITYENV_TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_USE_PP_EXECUTOR=$MODE
export SINGULARITYENV_PROJ_ROOT=$PROJ
export SINGULARITYENV_MANIFEST=$PROJ/runs/track_b_n2_pilot_measured_v1/pilot_manifest.json
export SINGULARITYENV_QWEN2VL_HF_DIR=$HF

singularity exec --nv \
  --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  -B "$BIND_ROOT:$BIND_ROOT" \
  "$SIF" \
  torchrun --standalone --tee 3 --nproc_per_node 2 --master_port 12666 \
  "$PROJ/track_b_cluster/bridge/pp2_equivalence.py" "$OUT"
echo "PP2_RC=$?"
