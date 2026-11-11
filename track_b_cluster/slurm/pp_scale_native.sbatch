#!/bin/bash
#SBATCH --job-name=q2vl-pp-native-scale
#SBATCH --account=gpu
#SBATCH -p gpu_2h
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --constraint=rtx3090
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --exclude=gpu54
# Native-schedule scaling run: submit with --gres=gpu:$PP --export=ALL,PP=$PP
# (PP=4 needs 4 GPUs on one node; PP=8 needs NUM_LAYERS=32 because 28 % 8 != 0).
# Benchmark mode: --export=ALL,PP=4,BENCH_STEPS=10,WARMUP_STEPS=2[,NUM_MICROBATCHES=8]
set -euo pipefail
WS=/research/d7/fyp26/lkchau3/workspaces/cyfok3
PROJ=$WS/proj/csci5540
PP=${PP:-4}
NUM_LAYERS=${NUM_LAYERS:-28}
BENCH_STEPS=${BENCH_STEPS:-0}
WARMUP_STEPS=${WARMUP_STEPS:-2}
NUM_MICROBATCHES=${NUM_MICROBATCHES:-2}
LOG=$WS/containers/pp-native-pp${PP}.log
exec > "$LOG" 2>&1
set -x

SIF=$WS/containers/ngc-pytorch-25.03-py3.sif
PMP=$WS/third_party/BigMac/third_party/Pai-Megatron-Patch
BACKEND=$PMP/backends/megatron/Megatron-LM-250624
HF=$WS/models/Qwen2-VL-2B-Instruct
OUT=$PROJ/runs/track_b_n2_pilot_measured_v1/pp_native_scale_pp${PP}.json
JSONL=${OUT%.json}.jsonl
SUMMARY=${OUT%.json}_summary.json

export PYTHONPATH="$BACKEND:$PMP/examples:$PMP/examples/qwen2_vl:$PMP:$WS/third_party/BigMac:$WS/venvs/trackb_pydeps"
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_HF_HOME=$WS/caches/hf
export SINGULARITYENV_TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_USE_PP_EXECUTOR=0
export SINGULARITYENV_PP_SIZE=$PP
export SINGULARITYENV_NUM_LAYERS=$NUM_LAYERS
export SINGULARITYENV_BENCH_STEPS=$BENCH_STEPS
export SINGULARITYENV_WARMUP_STEPS=$WARMUP_STEPS
export SINGULARITYENV_NUM_MICROBATCHES=$NUM_MICROBATCHES
export SINGULARITYENV_PROJ_ROOT=$PROJ
export SINGULARITYENV_MANIFEST=$PROJ/runs/track_b_n2_pilot_measured_v1/pilot_manifest.json
export SINGULARITYENV_QWEN2VL_HF_DIR=$HF

singularity exec --nv \
  --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  -B /research/d7/fyp26/lkchau3:/research/d7/fyp26/lkchau3 \
  "$SIF" \
  torchrun --standalone --tee 3 --nproc_per_node "$PP" --master_port 12677 \
  "$PROJ/track_b_cluster/bridge/pp2_equivalence.py" "$OUT"
echo "PP_NATIVE_SCALE_RC=$?"

if [ "$BENCH_STEPS" -gt 0 ]; then
  singularity exec \
    -B /research/d7/fyp26/lkchau3:/research/d7/fyp26/lkchau3 \
    "$SIF" \
    python3 "$PROJ/track_b_cluster/scripts/collect_step_metrics.py" \
      --input "$JSONL" --output "$SUMMARY" \
      --warmup-steps 0 --world-size "$PP" --batch-size 1 \
      --grad-accum "$NUM_MICROBATCHES" --run-label "pp${PP}-native"
fi
