#!/bin/bash
#SBATCH --job-name=q2vl-fused-bc
#SBATCH --account=gpu
#SBATCH -p gpu_2h
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --constraint=rtx3090
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu54,projgpu33
# Fused BigMac pp_executor, training-valid, on the B/C suites at 8 GPUs.
#   full:  Part B M=4, full coverage (94/186/129 steps) + Part C M=4, 8 steps
#   SMOKE=1: resolution_full_low and image_count_high, 2 steps, to check the
#            uneven PP=8 split, the optimizer step and the 576-token memory.
# Set PC_MB=2 if a Part C M=4 step does not fit on rank 0.
set -euo pipefail
: "${WS:=/research/d7/fyp26/lkchau3/workspaces/cyfok3}"
: "${BIND_ROOT:=/research}"
PROJ=$WS/proj/csci5540
SUITE_B=${SUITE_B:-$PROJ/runs/partb_suite_v1}
SUITE_C=${SUITE_C:-$PROJ/runs/partc_suite_v1}
OUT_DIR=${OUT_DIR:-$PROJ/runs/bigmac_fused_bc${SMOKE:+_smoke}_v1}
B_MB=${B_MB:-4}
PC_MB=${PC_MB:-4}
LOG=$WS/containers/fused-bc${SMOKE:+-smoke}.log
exec > "$LOG" 2>&1
set -x

mkdir "$OUT_DIR"   # refuses to overwrite earlier evidence

if [ "${SMOKE:-0}" = "1" ]; then
  B_CASES="resolution_full_low"; B_STEPS=2; B_FULL=0
  PC_CASES="image_count_high"; PC_STEPS=2
else
  B_CASES="resolution_full_low resolution_full_medium resolution_full_high"
  B_STEPS=0; B_FULL=1
  PC_CASES="image_count_low image_count_medium image_count_high"; PC_STEPS=8
fi

run() {  # manifest tag microbatches steps full_coverage
  MANIFEST=$1 TAG=$2 NUM_MICROBATCHES=$3 BENCH_STEPS=$4 FULL_COVERAGE=$5 \
  MODE=1 PP=8 NUM_LAYERS=28 PP_FIRST_LAYERS=2 PP_LAST_LAYERS=2 \
  TRAIN_STEP=1 WARMUP_STEPS=1 SEQ_LENGTH=1024 OUT_DIR=$OUT_DIR \
  bash "$PROJ/track_b_cluster/slurm/pp2_equivalence.sbatch"
}

for case in $B_CASES; do
  run "$SUITE_B/$case/manifest.json" "fused_b_${case#resolution_full_}" "$B_MB" "$B_STEPS" "$B_FULL"
done
for case in $PC_CASES; do
  run "$SUITE_C/$case/manifest.json" "fused_c_${case#image_count_}" "$PC_MB" "$PC_STEPS" 0
done
