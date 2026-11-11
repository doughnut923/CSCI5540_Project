# Track B cluster implementation (Megatron + BigMac)

Execution layer for Track B: run the same Qwen2-VL-2B N=2 pilot under Megatron-Core with
BigMac's `pp_executor` on and off, and compare.

**Start here**

| Document | Use it for |
| --- | --- |
| [`SETUP.md`](SETUP.md) | build the environment from scratch on the CSE cluster — clone, pinned third-party, patches, container, deps, data, conversion, first runs |
| [`HANDOFF.md`](HANDOFF.md) | current state, verified numbers, known bugs, open work |
| [`patches/README.md`](patches/README.md) | why each compat patch exists and how to re-apply it |

Nothing here is tied to one account: pick any workspace with ~40 GB free on a filesystem the
GPU nodes can see, and set `WS` / `BIND_ROOT` (see `SETUP.md` §0). The SLURM templates read both
from the environment.

## The comparison contract

Only `USE_PP_EXECUTOR` may differ between the two arms. Same manifest, same sample order, same
seed, same microbatches, same step count, same container, same commits.

- Native: `USE_PP_EXECUTOR=0` — MCore's own schedule, running on BigMac's MCore fork. **Not**
  "stock upstream Megatron-LM"; do not claim that without a separate parity check.
- BigMac: `USE_PP_EXECUTOR=1` — plus `MEGATRON_USE_PP_EXECUTOR=1` and, for encoder nesting,
  `MEGATRON_PP_EXECUTOR_WITH_VE=1`.

`pp_executor` only wraps when `pp_size > 1`, so any PP=1 comparison is a no-op.

## Files

| Path | Purpose |
| --- | --- |
| `bridge/native_forward_check.py` | 1-GPU parity gate: MCore vs HF loss + gradient on one N=2 sample |
| `bridge/pp2_equivalence.py` | PP equivalence harness; `PP_SIZE`/`NUM_LAYERS`/`BENCH_STEPS`/`NUM_MICROBATCHES` come from env |
| `bridge/runner_template.py` | contract for the eventual training-step runner (not implemented) |
| `bridge/mock_step_runner.py` | synthetic step records for wiring tests — never evidence |
| `scripts/collect_step_metrics.py` | aggregates step JSONL into throughput / step-time percentiles / peak memory / bubble |
| `scripts/common_env.sh` | shared env for SLURM jobs (`WS`, `BIND_ROOT`, `PROJ_ROOT`, `MODEL_DIR`, `HF_HOME`) |
| `slurm/*.sbatch` | conversion, parity gate, PP=2 equivalence, PP scaling, legacy templates |

## Parity ladder

1. **G1 forward**: MCore vs HF on one sample — `rel_diff` ~1e-5. Done (2.7e-05).
2. **G1b backward**: finite, non-zero gradient norm. Done (263.50).
3. **G2 schedule equivalence**: native vs `pp_executor` at PP=2, identical seed/weights/data,
   matching loss and grad norm. Native passes; `MODE=1` currently fails — see `HANDOFF.md` §9.
4. **G3**: 0 skipped / 0 NaN across all measured steps, both arms.
5. **G4**: provenance recorded — manifest hash, prompt hash, container, commits, PP/M/steps.

Timing is only meaningful after G2 and with enough microbatches per step: the bubble scales as
roughly `(PP−1)/M`, so `M=2` at PP=4 measures fill/drain, not scheduling quality.

## Scope limits (keep claims honest)

- The bridge exercises the **text-side** schedule. Qwen2-VL encoder nesting (`LLMPPExecutorWithVE`)
  is not wired yet.
- `part_b_image_res` token totals are derived; only the processor gate makes them measured.
- No training-convergence claim comes out of these runs — they are systems measurements.
