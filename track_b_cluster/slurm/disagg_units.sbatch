#!/bin/bash
#SBATCH --job-name=q2vl-disagg
#SBATCH --account=gpu
#SBATCH -p gpu_2h
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --constraint=rtx3090
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu54,projgpu33
# Part B: disaggregated vision encoder (DP) + LLM pipeline (PP) as two worlds.
#   submit: sbatch --export=ALL,ENC_GPUS=4,LLM_PP=4,NUM_MICROBATCHES=4,TAG=v1 \
#             track_b_cluster/slurm/disagg_units.sbatch
set -euo pipefail
: "${WS:=/research/d7/fyp26/lkchau3/workspaces/cyfok3}"
: "${BIND_ROOT:=/research}"
PROJ=$WS/proj/csci5540
ENC_GPUS=${ENC_GPUS:-4}
LLM_PP=${LLM_PP:-4}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-4}
NUM_LAYERS=${NUM_LAYERS:-32}
BENCH_STEPS=${BENCH_STEPS:-3}
WARMUP_STEPS=${WARMUP_STEPS:-2}
TAG=${TAG:-v1}
if [ $((ENC_GPUS + LLM_PP)) -gt 8 ]; then
  echo "ENC_GPUS + LLM_PP must be <= 8 (got $ENC_GPUS + $LLM_PP)"; exit 1
fi
LOG=$WS/containers/disagg-${TAG}.log
exec > "$LOG" 2>&1
set -x

SIF=$WS/containers/ngc-pytorch-25.03-py3.sif
PMP=$WS/third_party/BigMac/third_party/Pai-Megatron-Patch
BACKEND=$PMP/backends/megatron/Megatron-LM-250624
export PYTHONPATH="$BACKEND:$PMP/examples:$PMP/examples/qwen2_vl:$PMP:$WS/third_party/BigMac:$WS/venvs/trackb_pydeps"
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_HF_HOME=$WS/caches/hf
export SINGULARITYENV_TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_PROJ_ROOT=$PROJ
export SINGULARITYENV_MANIFEST=$PROJ/runs/track_b_n2_pilot_measured_v1/pilot_manifest.json
export SINGULARITYENV_QWEN2VL_HF_DIR=$WS/models/Qwen2-VL-2B-Instruct
export SINGULARITYENV_NUM_MICROBATCHES=$NUM_MICROBATCHES
export SINGULARITYENV_LLM_PP=$LLM_PP
export SINGULARITYENV_NUM_LAYERS=$NUM_LAYERS
export SINGULARITYENV_BENCH_STEPS=$BENCH_STEPS
export SINGULARITYENV_WARMUP_STEPS=$WARMUP_STEPS
export SINGULARITYENV_TAG=$TAG

SHM=${TMPDIR:-/tmp}/disagg-${SLURM_JOB_ID:-$$}
rm -rf "$SHM"; mkdir -p "$SHM"
export SINGULARITYENV_DISAGG_SHM=$SHM

SING="singularity exec --nv --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 -B $BIND_ROOT:$BIND_ROOT -B /tmp:/tmp $SIF"

# Both units run at once, each in its own process group and GPU slice.
SINGULARITYENV_ROLE=enc SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ENC_GPUS - 1))) \
  $SING torchrun --standalone --nproc_per_node "$ENC_GPUS" --master_port 12690 \
  "$PROJ/track_b_cluster/bridge/disagg_units.py" > "$WS/containers/disagg-enc-${TAG}.log" 2>&1 &
ENC_PID=$!

set +e
SINGULARITYENV_ROLE=llm SINGULARITYENV_CUDA_VISIBLE_DEVICES=$(seq -s, "$ENC_GPUS" $((ENC_GPUS + LLM_PP - 1))) \
  $SING torchrun --standalone --nproc_per_node "$LLM_PP" --master_port 12691 \
  "$PROJ/track_b_cluster/bridge/disagg_units.py" > "$WS/containers/disagg-llm-${TAG}.log" 2>&1
LLM_RC=$?
wait $ENC_PID
ENC_RC=$?
set -e
echo "DISAGG_LLM_RC=$LLM_RC DISAGG_ENC_RC=$ENC_RC"
exit $(( LLM_RC != 0 || ENC_RC != 0 ))
