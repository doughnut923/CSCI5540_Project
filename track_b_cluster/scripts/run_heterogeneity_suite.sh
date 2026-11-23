#!/usr/bin/env bash
# Run both arms inside the same allocation with the prepared native environment.
set -euo pipefail
: "${SUITE_ROOT:?Directory containing generated suite.json}"
: "${OUTPUT_ROOT:?New directory for all run results}"
: "${PROJ_ROOT:?Project checkout}"
: "${INPUT_PLANS_ROOT:?Run prepare_optimus_suite.py before reserving GPUs}"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
export MANIFEST_BATCHING=${MANIFEST_BATCHING:-bucket}
export NUM_MICROBATCHES=${NUM_MICROBATCHES:-2}
export WARMUP_STEPS=${WARMUP_STEPS:-1}
export BENCH_STEPS=${BENCH_STEPS:-8}
export VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-16}
export OPTIMIZER_DEVICE=${OPTIMIZER_DEVICE:-cpu}
export ENCODER_PLACEMENT=${ENCODER_PLACEMENT:-round-robin}
export DETERMINISTIC=${DETERMINISTIC:-1}
REPEATS=${REPEATS:-1}
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be positive" >&2; exit 2; }
python3 - "$SUITE_ROOT" > "$OUTPUT_ROOT/cases.tsv" <<'PY'
import json,sys,os
from pathlib import Path
root=Path(sys.argv[1])
cases=json.loads((root/'suite.json').read_text())['cases']
selected=set(filter(None,os.environ.get('SUITE_CASES','').split(',')))
if selected and not selected <= {c['name'] for c in cases}:raise ValueError('unknown SUITE_CASES')
for case in cases:
    if selected and case['name'] not in selected:continue
    print(case['name']+'\t'+str(root/case['manifest']))
PY
for ((repeat=0; repeat<REPEATS; repeat++)); do
  while IFS=$'\t' read -r case_name manifest; do
    export MANIFEST=$manifest
    export INPUT_PLAN="$INPUT_PLANS_ROOT/$case_name.json"
    test -f "$INPUT_PLAN"
    for MODE in serial-reference optimus-pp; do
      export MODE OUTPUT_DIR="$OUTPUT_ROOT/$case_name/repeat-$repeat/$MODE"
      bash "$PROJ_ROOT/track_b_cluster/slurm/optimus_pp.sbatch"
    done
    python3 "$PROJ_ROOT/track_b_cluster/scripts/compare_optimus_validation.py" \
      "$OUTPUT_ROOT/$case_name/repeat-$repeat" \
      --output "$OUTPUT_ROOT/$case_name/repeat-$repeat/comparison.json"
  done < "$OUTPUT_ROOT/cases.tsv"
done
