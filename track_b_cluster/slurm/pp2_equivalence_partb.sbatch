#!/bin/bash
#SBATCH --job-name=q2vl-fused-partb
#SBATCH --account=gpu
#SBATCH -p gpu_2h
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --constraint=rtx3090
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu54,projgpu33
# Fused BigMac pp_executor on the three full Part B datasets, training-valid:
# PP=8 uneven 2|4|4|4|4|4|4|2, M=4, optimizer stepped, full sample coverage.
# Compare against the DistTrain arm's 8-GPU rows (same manifests, layers, seed).
set -euo pipefail
: "${WS:=/research/d7/fyp26/lkchau3/workspaces/cyfok3}"
: "${BIND_ROOT:=/research}"
PROJ=$WS/proj/csci5540
SUITE=${SUITE:-$PROJ/runs/partb_suite_v1}
OUT_DIR=${OUT_DIR:-$PROJ/runs/bigmac_fused_partb_v1}
LOG=$WS/containers/fused-partb.log
exec > "$LOG" 2>&1
set -x

mkdir "$OUT_DIR"   # refuses to mix with earlier evidence; pick a new --export=OUT_DIR=

for CASE in resolution_full_low resolution_full_medium resolution_full_high; do
  MANIFEST=$SUITE/$CASE/manifest.json \
  OUT_DIR=$OUT_DIR TAG=fused_${CASE#resolution_full_} MODE=1 PP=8 \
  NUM_LAYERS=28 NUM_MICROBATCHES=4 PP_FIRST_LAYERS=2 PP_LAST_LAYERS=2 \
  TRAIN_STEP=1 FULL_COVERAGE=1 WARMUP_STEPS=2 \
  bash "$PROJ/track_b_cluster/slurm/pp2_equivalence.sbatch"
done
