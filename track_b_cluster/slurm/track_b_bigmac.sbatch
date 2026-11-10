#!/bin/bash
#SBATCH --job-name=trackb-bigmac
#SBATCH --account=gpu
#SBATCH --qos=csci5540
#SBATCH -p csci5540
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# CONFIRM gres/count/limits/container. Same workload/env as native except USE_PP_EXECUTOR=1.

set -euo pipefail
export SLURM_CONF="${SLURM_CONF:-/opt1/slurm/gpu-slurm.conf}"
export USE_PP_EXECUTOR=1
source "${PROJ_ROOT:-$CYFOK3_WORKSPACE/proj/csci5540}/track_b_cluster/scripts/common_env.sh"

: "${MANIFEST:?set MANIFEST to Track B N=2 pilot_manifest.json}"
: "${BRIDGE_RUNNER:?set BRIDGE_RUNNER to the cluster BigMac/MCore Qwen2-VL runner entrypoint}"
SIF="${SIF:-}"
BIND="${BIND:-$CYFOK3_WORKSPACE:$CYFOK3_WORKSPACE}"

OUT="$RUN_ROOT/bigmac_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
CMD="cd '$PROJ_ROOT' && USE_PP_EXECUTOR=1 python '$BRIDGE_RUNNER' --manifest '$MANIFEST' --model-dir '$MODEL_DIR' --output-dir '$OUT' --metrics-jsonl '$OUT/step_metrics.jsonl' ${RUNNER_ARGS:-}"
if [[ -n "${SIF}" ]]; then singularity exec --nv -B "$BIND" "$SIF" bash -lc "$CMD"; else bash -lc "$CMD"; fi
python "$PROJ_ROOT/track_b_cluster/scripts/collect_step_metrics.py" --input "$OUT/step_metrics.jsonl" --output "$OUT/metrics_summary.json" --warmup-steps "${WARMUP_STEPS:-2}" --run-label track_b_bigmac
