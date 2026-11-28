#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ( "$1" != "image-count" && "$1" != "grouping" && "$1" != "matched" ) ]]; then
    echo "Usage: bash run.sh {image-count|grouping|matched}" >&2
    exit 2
fi
: "${BIGMAC_DIR:?Set BIGMAC_DIR to the pinned BigMac checkout}"
: "${PARTC_MODEL_DIR:?Set PARTC_MODEL_DIR to the pinned local Qwen2-VL checkpoint}"
root=$(cd "$(dirname "$0")" && pwd)
export PARTC_PRIMARY="$root/primary" PARTC_PREVIOUS_CODE="$root/adapter"
export PYTHONPATH="$BIGMAC_DIR:$PARTC_PREVIOUS_CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
py=${PYTHON:-python3}
output=${OUTPUT_DIR:-"$root/runs"}
mkdir -p "$output"
PARTC_RUN_DIR=$(mktemp -d "$output/$1-XXXXXXXX")
export PARTC_RUN_DIR
echo "Outputs: $PARTC_RUN_DIR"
mkdir -p "$PARTC_RUN_DIR/primary_results" "$PARTC_RUN_DIR/results"
"$py" "$root/matched/preflight.py"
launch=("$py" -m torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0)
case "$1" in
    image-count)
        "${launch[@]}" "$PARTC_PRIMARY/code/bigmac_qwen2vl.py"
        "$py" "$root/matched/summarize_primary.py"
        ;;
    grouping)
        "${launch[@]}" "$root/grouping/run.py"
        "$py" "$root/grouping/summarize.py" "$PARTC_RUN_DIR/results"
        "$py" "$root/grouping/verify_results.py" "$PARTC_RUN_DIR/results"
        ;;
    matched)
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        "${launch[@]}" "$root/matched/bigmac_qwen2vl.py"
        "$py" "$root/matched/summarize.py"
        ;;
esac
