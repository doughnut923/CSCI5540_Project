# DistTrain arm — handoff

One page. What this is, how to run it, what it produces, and what you must keep fixed if you
compare against it.

## What it is

The vision encoder and the language model run as **two separate training units** on one node:
the encoder occupies `ENC_GPUS` GPUs (data-parallel), the LLM occupies `LLM_PP` GPUs (one MCore
pipeline). They hand embeddings and embedding-gradients through `/dev/shm` and lock-step by flag
files. Both units step AdamW over MCore's fp32 `main_grad`.

`ENC_GPUS=1` means no encoder gradient exchange at all — that is the configuration that is fast
and valid on this cluster.

## How to run it

```sh
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf

# 5 GPUs: 1 encoder + 4 LLM pipeline stages, 10 measured steps
sbatch --gres=gpu:5 \
  --export=ALL,ENC_GPUS=1,LLM_PP=4,NUM_MICROBATCHES=4,BENCH_STEPS=10,WARMUP_STEPS=2,TAG=demo \
  track_b_cluster/slurm/disagg_units.sbatch
```

Output: `runs/track_b_n2_pilot_measured_v1/disagg_m4_<TAG>.jsonl` (one line per measured step),
plus per-unit logs in `$WS/containers/disagg-*.log`.

`ENC_GPUS + LLM_PP` must be ≤ 8, and `--gres=gpu:N` must be N = `ENC_GPUS + LLM_PP`.

## The knobs (that's all of them)

| env | default | meaning |
| --- | --- | --- |
| `ENC_GPUS` | 4 | encoder unit world size (DP degree) |
| `LLM_PP` | 4 | LLM pipeline stages; must divide `NUM_LAYERS` |
| `NUM_MICROBATCHES` | 4 | microbatches per step (4 samples per step) |
| `NUM_LAYERS` | 32 | LLM depth. **32** if you need PP=8, **28** is the real model |
| `BENCH_STEPS` | 3 | measured steps (warmup steps are not recorded) |
| `WARMUP_STEPS` | 2 | discarded steps before measuring |
| `TAG` | v1 | output file suffix; use a new one, never reuse |
| `ENC_GRAD_SYNC` | 1 | 0 = skip the encoder all-reduce (then it is **not** training-valid) |
| `MANIFEST` | pilot manifest | set by the sbatch; change only if the dataset changes |

The seed is fixed at 1234 inside `mcore_argv` in both harnesses — do not change it per arm.

`PLAN_ONLY=1` prints the sample plan and per-step sequence length with no GPUs — use it to check
what a run will consume before burning an allocation.

## What you get per step

Plain-language version of the JSONL:

| field | meaning |
| --- | --- |
| `step`, `warmup` | index, always `false` (warmup is not written) |
| `step_time_s` | wall clock for the whole step, both units, measured with CUDA events |
| `samples` | samples in the step (= `NUM_MICROBATCHES`, one sample per microbatch) |
| `sample_ids` | which manifest samples this step consumed, in order |
| `loss`, `microbatch_losses` | LLM loss, mean and per microbatch |
| `lr`, `grad_norm`, `grad_norm_total` | schedule value; per-rank norm; whole-model norm |
| `weight_norm`, `enc_weight_norm` | weight fingerprint, both units — if these do not move, nothing trained |
| `param_norm0_by_rank`, `module_norm0_by_rank` | initial weight fingerprints, for comparing initialisations |
| `max_memory_allocated_bytes_by_rank` | peak GPU memory per rank, keyed `"0".."N-1"` |
| `busy_time_s_total`, `bubble_time_s`, `pipeline_bubble_ratio` | schedule efficiency, both units |
| `busy_intervals` | `[rank, kind, start_s, end_s]`, kind ∈ `enc, llm, bwd, sync, opt` |
| `rank_labels` | which rank is which unit/stage, for the Gantt |
| `skipped`, `nan_or_inf` | sanity flags; both must be false |

Summarise with:

```sh
python3 track_b_cluster/scripts/collect_step_metrics.py \
  --input runs/track_b_n2_pilot_measured_v1/disagg_m4_demo.jsonl \
  --output runs/track_b_n2_pilot_measured_v1/disagg_m4_demo_summary.json --world-size 5
```

Plot one step with `track_b_cluster/scripts/plot_pipeline_gantt.py --input <jsonl> --step N`.

## If you compare against it, keep these fixed

1. Same manifest, same `NUM_MICROBATCHES`, same `NUM_LAYERS`, same seed, same container image,
   same commit.
2. Same sample plan. The plan is `ManifestBatches` in `bridge/manifest_data.py` — a pure
   function of (step, microbatch), so your arm can just import it.
3. Sequence length changes per step (194/82/253/309/424/138 tokens) and step time follows it.
   Mixing plans makes the comparison noise.
4. Report wall clock around the step with warmup discarded, median of several steps.
5. A row is only "training-valid" if it steps an optimizer and its gradients are complete. Say so
   in the table, or label the row placement-only.

## Known traps

- **Seeded init depends on the layout.** Losses from different PP splits are not comparable; only
  timing is. Cross-config loss comparison needs a shared checkpoint.
- **Encoder DP costs more than it saves.** `ENC=2` adds 0.4-0.5 s per step (40 % of the step) and
  buys nothing on this hardware.
- **`PP=2` with 32 layers uses ~20.7 GiB per GPU.** Very little headroom; `PP=4` is safe.
- **Exclude `gpu54` and `projgpu33`.** The first silently kills jobs, the second is shared and OOMs
  the optimizer. Both are already in the sbatch `--exclude`.
- **No P2P between these 3090s.** Any exchange goes through host memory at ~5.3 GB/s.

## Where things are

| path | what |
| --- | --- |
| `track_b_cluster/bridge/disagg_units.py` | the two-unit harness |
| `track_b_cluster/bridge/manifest_data.py` | `ManifestBatches` — the shared sample plan and input builder |
| `track_b_cluster/bridge/pp2_equivalence.py` | fused baseline; re-exports the plan for old launchers |
| `track_b_cluster/slurm/disagg_units.sbatch` | launch both units in one allocation |
| `track_b_cluster/slurm/pp2_equivalence.sbatch` | launch the fused baseline |
| `track_b_cluster/scripts/collect_step_metrics.py` | JSONL → p50/p95 summary |
| `track_b_cluster/scripts/plot_pipeline_gantt.py` | JSONL → Gantt figure |
| `docs/distrain/distrain-results.md` | measured numbers, caveats, what is not measured |
| `docs/pipeline-bubble.md` | how the bubble number is defined |

## Still open

1. Fused baseline has no optimizer step — ~10 lines of `MainGradAdamW` in `pp2_equivalence.py`
   before the two arms can be quoted side by side.
2. Bucketed-bf16 encoder exchange (would cut ~0.45 s to ~0.1 s, makes `ENC>1` viable).
3. Checkpoint loading for both units — the only way to compare loss across layouts.
4. BigMac (`MODE=1`) and Optimus arms are not ours and are not measured here.
