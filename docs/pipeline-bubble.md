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
- `busy_i`: CUDA events around each `model(...)` call in `forward_step` (forward), and
  around each backward pass via `register_full_backward_pre_hook` /
  `register_full_backward_hook` on the pipeline chunk. In 1F1B (non-interleaved) forward
  and backward do not overlap on a rank, so the intervals are serial and are summed.
- `busy_i` is all-reduced (SUM); `T` is all-reduced (MAX). Both are computed per step,
  after `torch.cuda.synchronize()`.

Every step record is one JSONL line:

```
step, warmup, step_time_s, samples,
max_memory_allocated_bytes_by_rank, busy_time_s_total,
bubble_time_s, pipeline_bubble_ratio, backward_busy_timed,
loss, skipped, nan_or_inf, pp_size, num_layers, num_microbatches, mode
```

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
- **If backward timing fails** to register (platform/hook incompatibility), the script
  sets `backward_busy_timed: false` and the bubble is overestimated (busy underestimates).
  Check that field before trusting a number.
- **Fill/drain dominates at small `M`.** With `num_microbatches = 2` the bubble is mostly
  fill/drain and is not steady-state. Use `NUM_MICROBATCHES=8` (or more) for a meaningful
  steady-state figure; the cross-check estimate for 1F1B is `(P-1)/(M+P-1)`.
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

`BENCH_STEPS=0` (default) keeps the original single-step equivalence gate behavior and
writes no benchmark records.
