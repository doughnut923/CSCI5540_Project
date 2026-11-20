#!/usr/bin/env bash
#SBATCH --job-name=qwen-optimus-pp
#SBATCH --account=gpu
#SBATCH --partition=csci5540
#SBATCH --qos=csci5540
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --chdir=/tmp
# Prepared environment only: use the same pinned container/PMP/MCore as native.
set -euo pipefail
: "${WS:?Set your own GPU-visible workspace}"
: "${PROJ_ROOT:?Set the synchronized project checkout}"
: "${MANIFEST:?Set the common input manifest}"
: "${OUTPUT_DIR:?Set a NEW evidence directory}"
: "${BIND_ROOT:=$WS}"
MODE=${MODE:-optimus-pp}
PP=${PP:-${SLURM_GPUS_ON_NODE:-2}}
[[ "$PP" =~ ^[0-9]+$ ]] || { echo "Expected integer GPU count, got $PP" >&2; exit 2; }
test ! -e "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")" "$WS/container-home" "$WS/libcuda"
exec > "${OUTPUT_DIR}.log" 2>&1
PMP=$WS/third_party/BigMac/third_party/Pai-Megatron-Patch
BACKEND=$PMP/backends/megatron/Megatron-LM-250624
SIF=$WS/containers/ngc-pytorch-25.03-py3.sif
MODEL_DIR=${MODEL_DIR:-$WS/models/Qwen2-VL-2B-Instruct}
test -f "$SIF"
ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$WS/libcuda/libcuda.so"
export SINGULARITYENV_PYTHONPATH="$BACKEND:$PMP/examples:$PMP/examples/qwen2_vl:$PMP:$WS/third_party/BigMac:$WS/venvs/trackb_pydeps"
export SINGULARITYENV_LD_LIBRARY_PATH="$WS/libcuda:/.singularity.d/libs:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"
export SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin
export SINGULARITYENV_TRITON_LIBCUDA_PATH="$WS/libcuda"
export SINGULARITYENV_PROJ_ROOT="$PROJ_ROOT"
export SINGULARITYENV_SOURCE_COMMIT
SINGULARITYENV_SOURCE_COMMIT=$(git -C "$PROJ_ROOT" rev-parse HEAD)
export SINGULARITYENV_PMP_COMMIT
SINGULARITYENV_PMP_COMMIT=$(git -C "$PMP" rev-parse HEAD)
export SINGULARITYENV_MCORE_COMMIT
SINGULARITYENV_MCORE_COMMIT=$(git -C "$BACKEND" rev-parse HEAD)
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_TORCH_EXTENSIONS_DIR="$WS/cache/torch_extensions"
export SINGULARITYENV_NCCL_DEBUG=WARN
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv --home "$WS/container-home" \
  --bind "$BIND_ROOT:$BIND_ROOT" \
  --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  "$SIF" torchrun --standalone --nproc_per_node "$PP" \
  "$PROJ_ROOT/track_b_cluster/bridge/optimus_qwen.py" \
  --manifest "$MANIFEST" --model-dir "$MODEL_DIR" --output-dir "$OUTPUT_DIR" \
  --schedule "$MODE" --layers "${NUM_LAYERS:-28}" \
  --microbatches "${NUM_MICROBATCHES:-4}" \
  --warmup-steps "${WARMUP_STEPS:-1}" --steps "${BENCH_STEPS:-3}" \
  --validation-samples "${VALIDATION_SAMPLES:-0}" \
  --optimizer-device "${OPTIMIZER_DEVICE:-cpu}" \
  --batching "${MANIFEST_BATCHING:-manifest}"
