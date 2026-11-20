# Six-case input and training pilot

This extends the Optimus PP prototype with the team's existing workload ideas.
It does not change the experiment into a new dataset or a new model. Eight real
photo previews from Part C are reused across all cases; repeats are not independent
images. This limited pool is suitable for integration checks, not final quality claims.

## Reused workload definitions

| Case | Images/sample and preparation | Planned mean visual tokens/sample |
| --- | --- | --- |
| resolution_low | Two images, both 256 square | 162 |
| resolution_medium | Repeating pairs (128,384), (128,256), (256,256) | 163 |
| resolution_high | Repeating pairs (128,384), (128,384), (128,128) | 164 |
| image_count_low | N=1, Part C controlled padding | 576 |
| image_count_medium | Repeating N=1,2,3, Part C controlled padding | 576 |
| image_count_high | Repeating N=1,2,4,8, Part C controlled padding | 576 |

Resolution mixtures retain Part B's image-level ratios. Part C uses its existing
`prepare()` function, shape table, photo hashes, and per-image pixel cap. All
cases use the fixed response `These are photographs.` to avoid a response that
incorrectly asserts that every sample contains two images. This is a synthetic
training target for system checks, not a caption dataset.

The resolution means are close in **tokens**; that does not establish equal
encoder runtime. The final experiment needs calibration against measured encoder
forward/backward costs before attributing a difference to variance alone.

## Preserve heterogeneity within a batch

The legacy runner groups by visual-token total. Besides hiding resolution
variation, that assumption fails for mixed image counts: the start/end image
markers add tokens even when visual-token totals are equal.

`ManifestBatches(..., batching="manifest")` instead keeps the manifest order and
allows each microbatch to have its own sequence/grid shape. The Optimus executor
already exchanges tensors using per-microbatch metadata. The legacy default is
still `bucket`; other backends must explicitly support mixed lengths or agree
on a shared padding policy before a performance comparison.

The input audit verifies actual processor grids, token slots, image counts and
source hashes. During GPU checks, `--validation-samples` additionally records a
nonzero gradient norm for each image's output-feature slice. These checks connect
input retention to the real training path.

## Reproduce

Generate into a new directory outside versioned data paths:

```bash
python track_b_cluster/scripts/build_heterogeneity_suite.py \
  --output-dir <new-suite-directory> --samples-per-case 12 \
  --model-dir <pinned-Qwen2-VL-config-and-tokenizer>
```

Within one prepared two-GPU allocation, set `WS`, `BIND_ROOT`, and `PROJ_ROOT` as
in OPTIMUS.md. The suite launcher creates a new output root and runs each case
under both the serial correctness reference and Optimus PP:

```bash
export SUITE_ROOT=<suite-directory> OUTPUT_ROOT=<new-results-directory>
export PP=2 NUM_MICROBATCHES=2 OPTIMIZER_DEVICE=cpu
export WARMUP_STEPS=1 BENCH_STEPS=6 REPEATS=1 VALIDATION_SAMPLES=16
export ENCODER_PLACEMENT=round-robin DETERMINISTIC=1
bash track_b_cluster/scripts/run_heterogeneity_suite.sh
python track_b_cluster/scripts/report_heterogeneity_suite.py \
  --suite-root "$SUITE_ROOT" --runs-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json"
```

Six measured steps × two microbatches cover all 12 manifest samples once,
regardless of the warmup offset. One repeat is an integration pilot. For later
statistics, agree on repeated runs, warmup, step budget and memory policy across
the actual backends. Keep validation sampling off during a dedicated performance
run; it introduces synchronization outside and inside the measured step.

The paired suite fixes encoder ownership in both arms and enables deterministic
execution. It isolates scheduling with identical placement and arithmetic, rather
than changing placement at the same time. Broader BF16 comparisons between the
single-encoder reference and profiled ownership exceeded strict multi-step
parameter/loss tolerances on real photos, despite the earlier repeated-photo
smoke passing. Those results are not promoted to validated equivalence. The
profiled placement mode remains separately selectable, and needs its own
numerical/performance evaluation before broader claims.

## Verified controlled pilot (2026-11-20)

The [GPU validation record](reproduction/heterogeneity/validation.json) contains
all six completed case pairs. On two RTX 3090s, each arm performed one warmup
and six measured optimizer steps, using the 28-layer random-initialized model,
M=2 and CPU-offloaded AdamW. Both arms used round-robin encoder ownership and
deterministic execution. Executed source commit: `0a79f175775c12eb43027cba245a3b3937db6509`.

- All six pairs matched exactly in loss, global gradient norm, **659,232 sampled
  parameter values and 659,232 sampled used-gradient values**, including warmup.
- All expected image counts, sequence lengths and mean token budgets were
  observed during training. All **306 measured per-image gradient checks** were
  finite and positive, including the eight-image samples.
- The [processor audit](reproduction/heterogeneity/processor-summary.json) covers
  72 manifest entries built from eight repeated photographs. CPU input generation
  and GPU training have separate recorded software versions and were checked
  against the same observed token/sequence budgets.
- Resolution-series encoder forward means in the serial arm were **71.39, 123.43
  and 93.85 ms**. Their max/min spread was **72.9%**, so the report marks the mean
  workload as unmatched. These brief, instrumented measurements guide later
  calibration; they do not establish a speedup or a variance-only effect.

Earlier cross-layout validation failed after the FP32 accumulation fix, with
maximum absolute differences of 0.07148 in loss and 0.0003281 in sampled
parameters. The validation record retains this negative result separately.
The fixed-layout success above does not establish cross-layout equivalence.

## What remains for the proposal's final comparison

- Review and agree whether this PP-only implementation is an acceptable explicitly
  named Optimus-derived baseline, or whether additional original-paper mechanisms
  must be implemented.
- Run the same manifests and ordering on a complete BigMac and the chosen
  Megatron/DistTrain backend, with matched optimizer/offload/precision settings.
- Match resolution-series **measured mean encoder work**, then repeat measurements
  and report variability. The pilot report flags a greater-than-5% mean runtime
  spread rather than silently calling it controlled.
- Add comparable kernel-level bubble and FLOPs instrumentation. High-level
  operation intervals alone do not establish those metrics.

The serial reference used here is not relabelled as Megatron-DistTrain or BigMac.
