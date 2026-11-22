# Optimus bottleneck diagnostics

`PROFILE_STEP_PHASES=1` enables synchronized wall-time measurements in the
Optimus runner only. It adds CUDA synchronization at phase boundaries and at
the optimizer subphase boundaries. This changes potential overlap: use these
numbers to locate costs, not to claim a scheduler speedup.

## Measurement boundaries

The complete step starts after input preparation, calibration, gradient clearing
and the initial barrier. It ends after optimizer completion and the final CUDA
synchronization. Outer result gathering, file writes and validation sampling are
outside this clock, as in the existing runner. Profiling runs disable validation
sampling so per-image gradient diagnostics inside the executor are also absent.

| Top-level phase | Included work |
| --- | --- |
| `model_schedule` | Encoder/LLM forward and backward, P2P and waits, executor event processing and its result gathering |
| `ordinary_grad_collect` | Final collection of ordinary autograd gradients |
| `encoder_grad_sync` | Encoder replica gradient all-reduce and completion |
| `tied_grad_sync` | MCore finalization, including tied embeddings |
| `global_grad_norm` | Whole-model norm computation and communication |
| `optimizer` | Gradient preparation, clipping, AdamW, copyback and optimizer diagnostics |
| `unattributed` | Remaining Python/control overhead inside the step |

Optimizer children are `gradient_prepare`, `lr_and_clip`, `adamw_update`,
`parameter_copyback` and `diagnostics`. With CPU offload, gradient preparation
includes GPU-to-CPU copies; copyback moves updated parameters to the GPU.
Children are already included in `optimizer`; adding parent and children would
double-count them. The schedule phase is not pure compute, and its complement
is not a measured pipeline bubble or pure communication time.

For every step, select the rank with the longest complete-step time and report
that same rank's phase breakdown. Do not independently maximize each phase over
ranks and add those maxima. All rank timings remain in the raw JSONL.

## Run

Use the same pinned environment and input plans as `OPTIMUS_EXPERIMENTS.md`.
Choose a new output directory and run the existing Optimus launcher:

```bash
export MODE=optimus-pp PP=2 NUM_LAYERS=28 NUM_MICROBATCHES=2 SEQ_LENGTH=1024
export OPTIMIZER_DEVICE=cpu ENCODER_PLACEMENT=round-robin DETERMINISTIC=1
export PROFILE_STEP_PHASES=1 VALIDATION_SAMPLES=0
export WARMUP_STEPS=1 BENCH_STEPS=8 MANIFEST_BATCHING=bucket
# Set MANIFEST, INPUT_PLAN, OUTPUT_DIR, WS, BIND_ROOT and PROJ_ROOT.
bash track_b_cluster/slurm/optimus_pp.sbatch
python track_b_cluster/scripts/report_optimus_phases.py "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/phase-summary.json"
```

The reporter checks completed steps, finite training values and additive timing
closure. It excludes warmup and reports both top-level and optimizer-only shares.
The first diagnostic uses `image_count_high` and `resolution_high`, at M=2.
It is a small attribution check; independent repeats and uninstrumented runs are
required before drawing a throughput conclusion. GPU count, optimizer residence,
clipping and initialization differences across backends remain unresolved.

## Recorded diagnosis (2026-11-22)

[Validation record](reproduction/optimus_profiling_20261122/validation.json),
source `bcb82b8128be9ba30e6e046efba85a5e106fc55c`, two RTX 3090s, CPU optimizer,
28 language layers, M=2, one warmup and eight measured steps per case:

| Case | Complete step | Optimizer | Optimizer share | CPU AdamW update | Gradient preparation | Parameter copyback |
| --- | --- | --- | --- | --- | --- | --- |
| image_count_high | 13.39 s | 12.24 s | 91.4% | 8.71 s | 1.75 s | 1.26 s |
| resolution_high | 12.87 s | 12.01 s | 93.3% | 8.47 s | 1.73 s | 1.28 s |

The model-schedule phase was 0.61 s and 0.33 s respectively; it includes P2P,
waits and executor bookkeeping. Each optimizer child in the table is already
included in the optimizer parent. All top-level components close to the full
step time. The nine image-count-high losses and global gradient norms match the
previous unprofiled run exactly; profiling did not sample full GPU parameters.
CPU tests separately check unchanged optimizer parameter updates.

This diagnosis identifies the CPU-offloaded optimizer path as the dominant
cost in this configuration. The logs report torchrun's default of one OpenMP
thread per process. A next experiment can measure one/two/three threads per rank
within the six training CPU cores, before changing optimizer algorithms or
expanding the schedule benchmark. Neither threading speedup nor cross-backend
speedup has been measured here. One run per workload is a diagnosis, not a
statistical performance study.
