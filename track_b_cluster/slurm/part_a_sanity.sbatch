#!/bin/bash
#SBATCH --job-name=parta-qwen2vl-sanity
#SBATCH --account=gpu
#SBATCH --qos=csci5540
#SBATCH -p csci5540
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# CONFIRM account/qos/partition/gres/mem/time/container on cluster.

set -euo pipefail
export SLURM_CONF="${SLURM_CONF:-/opt1/slurm/gpu-slurm.conf}"
source "${PROJ_ROOT:-$CYFOK3_WORKSPACE/proj/csci5540}/track_b_cluster/scripts/common_env.sh"

SIF="${SIF:-}"  # set to approved .sif path
BIND="${BIND:-$CYFOK3_WORKSPACE:$CYFOK3_WORKSPACE}"  # CONFIRM

run_in_env() {
  if [[ -n "${SIF}" ]]; then
    singularity exec --nv -B "$BIND" "$SIF" bash -lc "$*"
  else
    bash -lc "$*"
  fi
}

cd "$PROJ_ROOT/part_a_model_check"
run_in_env "cd '$PROJ_ROOT/part_a_model_check' && python run_check.py --device cuda --dtype bfloat16 --model-dir '$MODEL_DIR' --output-dir '$RUN_ROOT/part_a_cuda_sanity_${SLURM_JOB_ID:-local}'"
run_in_env "cd '$PROJ_ROOT/part_a_model_check' && python run_check.py --device cuda --dtype bfloat16 --visual-token-budget 576 --model-dir '$MODEL_DIR' --output-dir '$RUN_ROOT/part_a_cuda_sanity_576_${SLURM_JOB_ID:-local}'"
