# Optimus-inspired PP on eight RTX 3090s

Slurm job 31748 completed on one node on 2026-11-27. It used source
`afa751384bbf753e045b387443301dd8bf83ef72`, seeded Qwen2-VL-2B initialization
and GPU-resident AdamW. The 28 LLM layers were partitioned as
`2 / 4 / 4 / 4 / 4 / 4 / 4 / 2`, with a full vision encoder replica on each
rank. The image-count High workload mixed one, two, four or eight images at
576 visual tokens per sample.

## Correctness and memory

Both checks paired Optimus with the internal serial reference using identical
initialization, encoder ownership and sample order. At `rtol=1e-3` and
`atol=1e-5`, the maximum absolute differences in loss, global gradient norm,
sampled parameter values and sampled used gradients were all zero.

| Check | Microbatches per step | Updates per arm | Sampled parameter values | Sampled gradient values |
| --- | ---: | ---: | ---: | ---: |
| Replay in manifest order | 8 | 2 | 106,464 | 106,464 |
| Timing configuration with length buckets | 4 | 9 | 479,088 | 479,088 |

The replay exercised all eight encoder replicas and checked 60 finite,
nonzero image gradients per arm. Each timing repeat measured eight steps after
one warm-up step, visiting all 12 distinct samples over 32 sample visits.

## Timing

| Repeat | Samples/s | Mean step time (s) | Maximum allocated peak (GiB) |
| --- | ---: | ---: | ---: |
| 1 | 1.7351 | 2.3054 | 20.723 |
| 2 | 1.7300 | 2.3121 | 20.723 |
| 3 | 1.7008 | 2.3518 | 20.723 |
| Pooled | 1.7218 | 2.3231 | 20.723 |

The pooled rate uses 96 sample visits divided by total time over 24 measured
steps. Each step includes model forward and backward, gradient synchronization,
and the optimizer update. Input preparation and calibration occur before timing.
Timing disabled parameter sampling and synchronized phase diagnostics.

| Rank | Optimus replay peak (GiB) | Highest Optimus timing peak (GiB) |
| ---: | ---: | ---: |
| 0 | 21.122 | 20.679 |
| 1 | 19.187 | 18.518 |
| 2 | 18.970 | 18.518 |
| 3 | 18.758 | 18.518 |
| 4 | 18.541 | 17.888 |
| 5 | 18.328 | 17.888 |
| 6 | 18.119 | 17.890 |
| 7 | 20.711 | 20.723 |

Memory is PyTorch peak allocated memory. The timing column takes the maximum
over all three repeats, including warm-up steps.

## Reproduce

Use the pinned environment and audited input suite in the
[Optimus protocol](../../OPTIMUS_EXPERIMENTS.md#single-node-pp8-correctness-check).
The same section contains the two-step replay. For the timing configuration,
generate a length-bucket plan from that suite before entering the allocation.

```bash
python track_b_cluster/scripts/prepare_optimus_suite.py \
  --suite-root "$SUITE_ROOT" --model-dir "$MODEL_DIR" \
  --output-dir "$INPUT_PLANS_ROOT" --cases image_count_high \
  --batching bucket --microbatches 4 --warmup-steps 1 --measured-steps 8
```

The archived plan SHA-256 is
`6f7443cf6023794243615d8c8cbad904808681e89cb61576807e003289dd4362`.
Inside the prepared eight-GPU allocation, set `OUTPUT_ROOT` to a new result
directory and rerun the pair at the timing configuration.

```bash
(
set -eu
export PP=8 NUM_LAYERS=28 NUM_MICROBATCHES=4 SEQ_LENGTH=1024
export MANIFEST_BATCHING=bucket WARMUP_STEPS=1 BENCH_STEPS=8 REPEATS=1
export OPTIMIZER_DEVICE=cuda ENCODER_PLACEMENT=round-robin DETERMINISTIC=1
export VALIDATION_SAMPLES=16 PROFILE_STEP_PHASES=0 TRACE_FIRST_ENCODER_GRADIENTS=0
export SUITE_CASES=image_count_high

bash "$PROJ_ROOT/track_b_cluster/scripts/run_heterogeneity_suite.sh"
python3 "$PROJ_ROOT/track_b_cluster/scripts/report_heterogeneity_suite.py" \
  --suite-root "$SUITE_ROOT" --runs-root "$OUTPUT_ROOT" \
  --cases image_count_high --output "$OUTPUT_ROOT/summary.json"

export MANIFEST="$SUITE_ROOT/image_count_high/manifest.json"
export INPUT_PLAN="$INPUT_PLANS_ROOT/image_count_high.json"
export VALIDATION_SAMPLES=0
for repeat in 1 2 3; do
  run="$OUTPUT_ROOT/timed-$repeat"
  MODE=optimus-pp OUTPUT_DIR="$run" \
    bash "$PROJ_ROOT/track_b_cluster/slurm/optimus_pp.sbatch"
  python3 "$PROJ_ROOT/track_b_cluster/scripts/collect_step_metrics.py" \
    --input "$run/metrics.jsonl" --output "$run/summary.json" \
    --batch-size 4 --grad-accum 4 --world-size 8 --run-label "optimus-c-high-pp8-$repeat"
done
)
```

The [validation record](validation.json) contains the sampled comparisons,
runtime configuration and archive checksum. The
[step records](timing-steps.jsonl) contain the fields needed to reproduce the
throughput and memory calculations. This result covers one short-window
Optimus configuration. The remaining workload levels and a shared protocol
for a cross-system comparison are pending.
