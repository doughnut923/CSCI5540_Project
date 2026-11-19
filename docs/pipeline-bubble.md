# Pipeline bubble measurement (Track B)

How the pipeline bubble is computed for the native MCore schedule, and what the number
does and does not include. Implemented in `track_b_cluster/bridge/pp2_equivalence.py`
(benchmark mode) and aggregated by `track_b_cluster/scripts/collect_step_metrics.py`.

## Definition

For one optimizer step with pipeline size `P`:

```
T           = globally synced wall time of the step
busy_i      = compute time on rank i during the step (forward + backward intervals)
bubble_time = P * T - sum(busy_i)
bubble_ratio = bubble_time / (P * T)
```

`P * T` is the total stage-time the pipeline could have used. Every slot not spent
computing is pipeline fill/drain (the bubble). The result is **measured**, not an
estimate from a formula: both terms come from CUDA events on the actual run.

## How the terms are measured

- `T`: `torch.distributed.barrier()` → CUDA event → `forward_backward_func(...)` →
  event → barrier. `T` is the max over ranks, so it is one global window.
- `busy_i`: CUDA events around each `model(...)` call in `forward_step` (forward), split
  at the vision-encoder / language-model boundary by a `forward_pre_hook` on
  `language_model`, so `enc` and `llm` are measured separately. Backward is timed by
  wrapping `torch.autograd.backward`, which is what MCore's `backward_step` calls per
  microbatch (`deallocate_pipeline_outputs` must stay off or MCore takes the
  `custom_backward` path instead). Module backward hooks cannot be used here: the stage
  activation enters the model through `set_input_tensor()`, not as a forward argument, so
  no forward input requires grad and the full-backward post-hook fires immediately after
  the pre-hook, giving zero-width intervals. In 1F1B (non-interleaved) forward and
  backward do not overlap on a rank, so the intervals are serial and are summed.
- `busy_i` is all-reduced (SUM); `T` is all-reduced (MAX). Both are computed per step,
  after `torch.cuda.synchronize()`.

Every step record is one JSONL line:

```
step, warmup, step_time_s, samples,
max_memory_allocated_bytes_by_rank, busy_time_s_total,
bubble_time_s, pipeline_bubble_ratio, backward_busy_timed,
loss, skipped, nan_or_inf, pp_size, num_layers, num_microbatches, mode
```

`busy_intervals` adds every busy interval as `[pp_rank, kind, start_s, end_s]`
relative to the step start, with `kind` = `enc` (vision encoder), `llm` (language
model), `bwd` (backward), `sync` (encoder gradient all-reduce) or `opt` (optimizer
step); `scripts/plot_pipeline_gantt.py` draws the utilization diagram
(time on x, stage on y) from it.

`samples` per step is `micro_batch_size * num_microbatches * data_parallel` = `num_microbatches`
here (micro batch 1, DP 1), which is what the throughput calculation divides by.

## Limits of the number

- **Compute-idle only.** P2P sends/recvs and NCCL run on separate streams and overlap
  compute. Their wait time is not counted as compute-busy, so time a rank spends purely
  waiting on communication *is* counted as bubble. This matches the standard definition
  of pipeline bubble (idle compute slots), but it is not "GPU idle" as a profiler would
  report it.
- **Event resolution.** CUDA events are accurate to ~microseconds; event markers add a
  small overhead to `step_time_s` when benchmark mode is on.
- **`backward_busy_timed` only means the wrapper is installed.** It does not prove the
  intervals are non-zero; the Gantt script flags a record whose backward intervals are
  all empty.
- **Fill/drain is not the whole story at PP=8.** Measured 2026-11-18 (PP=8, 32 layers,
  M=4): bubble 0.804 against `(P-1)/(M+P-1)` = 0.64, because stage 0 runs the whole
  vision encoder for every microbatch (0.475 s of encoder, 0.754 s of backward in a
  1.084 s step) and the other stages wait for it. Compare stages with the same encoder
  placement before attributing a bubble to the schedule.
- **Shared nodes.** Absolute timings depend on co-tenants; compare runs made in the same
  session and report per-step spread, not only the mean.
- **No NVLink on rtx3090 nodes.** PP P2P goes over PCIe, which affects drain time.

## How to run

```sh
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
cd /research/d7/fyp26/lkchau3/workspaces/cyfok3/proj/csci5540

# PP=4 benchmark: 2 warmup + 10 measured steps, M=2 (same microbatches as the gate)
sbatch --gres=gpu:4 --export=ALL,PP=4,BENCH_STEPS=10,WARMUP_STEPS=2 \
  track_b_cluster/slurm/pp_scale_native.sbatch

# PP=8 steady-state variant (32 layers because 28 % 8 != 0; M=8 for a smaller bubble)
sbatch --gres=gpu:8 --export=ALL,PP=8,NUM_LAYERS=32,BENCH_STEPS=10,WARMUP_STEPS=2,NUM_MICROBATCHES=8 \
  track_b_cluster/slurm/pp_scale_native.sbatch
```

Outputs under `runs/track_b_n2_pilot_measured_v1/`:

| File | Content |
| --- | --- |
| `pp_native_scale_pp{P}.json` | run config + final loss/grad norm |
| `pp_native_scale_pp{P}.jsonl` | one line per measured step (raw metrics) |
| `pp_native_scale_pp{P}_summary.json` | aggregated throughput, step p50/p95, peak mem, bubble |

Plot one step's utilization diagram:

```sh
python3 track_b_cluster/scripts/plot_pipeline_gantt.py --input <jsonl> [--step 0] \
  --output pp8_gantt.png
```

`docs/pp8_gantt_split_v1.png` is that diagram for the PP=8 / M=4 run (job 28700).

`BENCH_STEPS=0` (default) keeps the original single-step equivalence gate behavior and
writes no benchmark records.
