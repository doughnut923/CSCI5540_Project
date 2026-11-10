#!/bin/bash
#SBATCH --job-name=q2vl-2b-convert
#SBATCH --account=gpu
#SBATCH -p gpu_2h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu54
# Convert Qwen2-VL-2B-Instruct HF -> Megatron-Core checkpoint (TP=1, PP=1, bf16).
# Runs inside the NGC PyTorch 25.10 SIF, which provides torch + transformer_engine.
# Set MEGATRON_SRC to override the Megatron-LM/MCore checkout used.
set -euo pipefail
WS=/research/d7/fyp26/lkchau3/workspaces/cyfok3
LOG=$WS/containers/convert-q2vl-2b.log
exec > "$LOG" 2>&1
set -x

SIF=${SIF:-$WS/containers/ngc-pytorch-25.03-py3.sif}
PMP=$WS/third_party/BigMac/third_party/Pai-Megatron-Patch
MEGATRON_SRC=${MEGATRON_SRC:-$WS/third_party/Megatron-LM}
HF=$WS/models/Qwen2-VL-2B-Instruct
OUT=${OUT:-$WS/checkpoints/Qwen2-VL-2B-Instruct-mcore-tp1pp1}

export PYTHONPATH="$MEGATRON_SRC:$PMP:$WS/third_party/BigMac:$WS/venvs/trackb_pydeps"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export HF_HOME=$WS/caches/hf
# NGC image ships /sbin/ldconfig but drops /sbin from PATH; transformer_engine needs it.
export PATH=/sbin:/usr/sbin:$PATH

test -f "$SIF" || { echo "missing SIF $SIF"; exit 2; }
test -d "$HF" || { echo "missing HF model $HF"; exit 2; }
if [ -e "$OUT" ]; then echo "output exists: $OUT"; exit 2; fi

cd "$PMP/toolkits/model_checkpoints_convertor/qwen"
# Compute-node singularity has no `--env`; SINGULARITYENV_* is portable.
# Host driver is 565 (CUDA 12.7) so this must be a cu12.x image (NGC 25.03: torch 2.7+cu128, TE 2.1).
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
singularity exec --nv \
  --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  -B /research/d7/fyp26/lkchau3:/research/d7/fyp26/lkchau3 \
  "$SIF" \
  bash hf2mcore_qwen2_vl_convertor.sh 2B "$HF" "$OUT" 1 1 false bf16
echo "CONVERT_RC=$?"
du -sh "$OUT" 2>/dev/null || true
