#!/usr/bin/env bash
# Run inside ONE two-GPU allocation with the prepared native environment.
set -euo pipefail
: "${SUITE_ROOT:?Directory containing generated suite.json}"
: "${OUTPUT_ROOT:?New directory for all run results}"
: "${PROJ_ROOT:?Project checkout}"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
export MANIFEST_BATCHING=manifest
export NUM_MICROBATCHES=${NUM_MICROBATCHES:-2}
export WARMUP_STEPS=${WARMUP_STEPS:-1}
export BENCH_STEPS=${BENCH_STEPS:-2}
export VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-16}
export OPTIMIZER_DEVICE=${OPTIMIZER_DEVICE:-cpu}
REPEATS=${REPEATS:-1}
python3 - "$SUITE_ROOT" > "$OUTPUT_ROOT/cases.tsv" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for case in json.loads((root/'suite.json').read_text())['cases']:
    print(case['name']+'\t'+str(root/case['manifest']))
PY
for ((repeat=0; repeat<REPEATS; repeat++)); do
  while IFS=$'\t' read -r case_name manifest; do
    export MANIFEST=$manifest
    for MODE in serial-reference optimus-pp; do
      export MODE OUTPUT_DIR="$OUTPUT_ROOT/$case_name/repeat-$repeat/$MODE"
      bash "$PROJ_ROOT/track_b_cluster/slurm/optimus_pp.sbatch"
    done
  done < "$OUTPUT_ROOT/cases.tsv"
done
