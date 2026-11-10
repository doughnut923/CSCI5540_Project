#!/bin/bash
#SBATCH --job-name=q2vl-native-fwd
#SBATCH --account=gpu
#SBATCH -p gpu_2h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=gpu54
# Single-GPU parity: MCore Qwen2-VL forward vs HF reference on one N=2 Track B sample.
set -euo pipefail
WS=/research/d7/fyp26/lkchau3/workspaces/cyfok3
PROJ=$WS/proj/csci5540
LOG=$WS/containers/native-fwd.log
exec > "$LOG" 2>&1
set -x

SIF=$WS/containers/ngc-pytorch-25.03-py3.sif
PMP=$WS/third_party/BigMac/third_party/Pai-Megatron-Patch
BACKEND=$PMP/backends/megatron/Megatron-LM-250624
HF=$WS/models/Qwen2-VL-2B-Instruct
CKPT=$WS/checkpoints/Qwen2-VL-2B-Instruct-mcore-tp1pp1
OUT=$PROJ/runs/track_b_n2_pilot_measured_v1/native_forward_check.json

export PYTHONPATH="$BACKEND:$PMP:$WS/third_party/BigMac:$WS/venvs/trackb_pydeps"
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_HF_HOME=$WS/caches/hf
export SINGULARITYENV_TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false

singularity exec --nv \
  --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  -B /research/d7/fyp26/lkchau3:/research/d7/fyp26/lkchau3 \
  "$SIF" \
  torchrun --nproc_per_node 1 --master_port 12555 \
  "$PROJ/track_b_cluster/bridge/native_forward_check.py" \
  --hf-dir "$HF" --mcore-ckpt "$CKPT" \
  --manifest "$PROJ/runs/track_b_n2_pilot_measured_v1/pilot_manifest.json" \
  --repo-root "$PROJ" --pmp-root "$PMP" --output "$OUT"
echo "CHECK_RC=$?"
