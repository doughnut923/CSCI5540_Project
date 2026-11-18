# Track B handoff — Qwen2-VL-2B on Megatron-Core + BigMac

Status as of 2026-11-11. Everything here is measured unless marked otherwise.
Read this before touching the cluster; several "obvious" shortcuts are known-broken and
cost hours to rediscover.

---

## TL;DR

| Piece | State |
| --- | --- |
| N=2 pilot manifest (24 samples, hashes) | **done** |
| Processor gate (real images, pinned model) | **pass, 24/24** |
| HF → Megatron-Core checkpoint conversion | **done**, 4.2 GB |
| Native MCore forward parity vs HF | **pass**, `rel_diff 2.7e-05` |
| Native MCore backward | **pass**, grad norm 263.50, finite |
| PP=2 (pipeline parallel) | **native works** (loss 6.5671, grad norm 867.04, 2 GPUs); `MODE=1` (`pp_executor`) fails early with no output — see §9 |
| `USE_PP_EXECUTOR=1` (BigMac schedule) | **not started** (needs PP=2 + encoder bridge) |
| BigMac encoder nesting (VE executor) | **not started** (needs a Qwen2-VL bridge) |
| Timings / bubble ratio / training loop | **not started** |

---

## 1. Objective

Controlled comparison of two pipeline schedules on the same model and data:

- **native**: `USE_PP_EXECUTOR=0` (MCore's own schedule, on BigMac's MCore fork)
- **BigMac**: `USE_PP_EXECUTOR=1` (BigMac `pp_executor`)

Fixed axis is **resolution heterogeneity with image count fixed at N=2**. Nothing else may
differ between the two runs: manifest, sample order, seed, microbatches, steps, container,
commits.

There is deliberately **no HF quality baseline** in the experiment (user decision). HF is only
used to validate that the converted checkpoint is numerically correct.

---

## 2. Cluster facts (non-negotiable)

| Item | Value |
| --- | --- |
| Workspace | any directory you own with ~40 GB free, exported as `$WS` (examples use `/research/<group>/<user>/workspaces/<name>`); set `$BIND_ROOT` to the filesystem root that contains it |
| Project | `$WS/proj/csci5540` |
| SLURM config | `SLURM_CONF=/opt1/slurm/gpu-slurm.conf` |
| Account / partition | `--account=gpu -p gpu_2h` (or `gpu_8h`) |
| GPU selection | `--gres=gpu:N --constraint=rtx3090` |
| **Broken node** | **`gpu54` — jobs die in seconds with no output. Always `--exclude=gpu54`.** |
| Driver | 565.57.01 = CUDA 12.7 → **CUDA-13 images do not work** |
| Container | `$WS/containers/ngc-pytorch-25.03-py3.sif` (11 GB; torch 2.7+cu128, TE 2.1.0) |
| Unusable | `ngc-pytorch-25.10-py3.sif` (CUDA 13 torch, rejects the driver) |
| Python deps | `$WS/venvs/trackb_pydeps` (transformers 4.49.0 etc.) |

### Container invocation rules

- Compute nodes have an **older Singularity without `--env`** → use `SINGULARITYENV_*`.
- `--nv` alone does **not** give torch a working `libcuda`; bind the host libs explicitly:

```bash
singularity exec --nv \
  --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  -B "$BIND_ROOT:$BIND_ROOT" \
  "$SIF" ...
```

- TE imports subprocess `ldconfig`, and the image drops `/sbin` from PATH → set
  `SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin`.
- Triton needs `SINGULARITYENV_TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu`.
- `--mem=32G` / `16G` are rejected on rtx3090 nodes; omit the `#SBATCH --mem` line.
- **Vision attention needs sm80+.** On an sm75 node (RTX 2080 / Titan) TE disables flash and
  fused and raises `No dot product attention support for the provided inputs!`. That is the
  GPU, not the code. `--constraint=rtx3090` is mandatory.

---

## 3. Pinned checkouts (`$WS/third_party/`)

| Repo | Commit | Note |
| --- | --- | --- |
| `BigMac` | `2fa6da2fd3979448e89a56e734cbc3c490990d81` | `pp_executor`, `pp_scheduler` |
| `BigMac/third_party/Pai-Megatron-Patch` | `3413f297e9666aa75c681bb588a7a61d92a33e5b` | Qwen2-VL model + HF→MCore converter |
| `BigMac/third_party/big_mac_open_source` | `3a1cd24a465600e5721b66e7e5a46cbe702980ee` | required submodule |
| `.../backends/megatron/Megatron-LM-250624` | `2220aa0d…` (branch `pai_qwen3_vl_backend`, `Dots-Infra/BigMac-Megatron-Core`) | **the backend we actually run**: MCore 0.14 fork with BigMac hooks |
| `Megatron-LM` | `266f1c97…` (`core_v0.17.1`) | stock 0.17.1; imports fine, **cannot host the PMP Qwen2-VL model** |

Model: `$WS/models/Qwen2-VL-2B-Instruct`, HF revision `895c3a49bc3fa70a340399125c650a463535e71c`.

---

## 4. What exists

### Code (repo, `track_b_cluster/`)

| File | Purpose |
| --- | --- |
| `bridge/native_forward_check.py` | 1-GPU parity: MCore vs HF loss on one N=2 sample. Also runs a backward. This is the current correctness gate. |
| `bridge/pp2_equivalence.py` | PP=2 native vs `pp_executor`: identical seed/weights/data, compare loss + grad norm. **Unproven, just written.** |
| `bridge/runner_template.py` | Contract for the eventual real training-step runner. |
| `bridge/mock_step_runner.py` | Synthetic step records for wiring tests only. |
| `scripts/collect_step_metrics.py` | Plan §6 aggregation (throughput, step p50/p95, peak mem, optional bubble). |
| `slurm/native_forward_check.sbatch` | Gate G1/G1b runner. |
| `slurm/pp2_equivalence.sbatch` | PP=2 runner; set `MODE=0` or `MODE=1`. |
| `slurm/convert_qwen2vl_2b.sbatch` | HF→MCore conversion. |
| `patches/` | Three patches + `README.md` with the full rationale. |
| `BIGMAC_STAGING.md` | Older staging notes (partly superseded by this file). |

### Data / artifacts

| Path | What |
| --- | --- |
| `$PROJ/runs/track_b_n2_pilot_measured_v1/pilot_manifest.json` | 24 samples, 8 per Low/Medium/High, N=2, seed 55402026, 0 missing, real SHA256s, derived totals 50–392 |
| `.../processor_gate/n2_validation_report.json` | 24/24 pass, transformers 4.49.0 |
| `.../processor_gate/environment.json` | container + versions at gate time |
| `.../native_forward_check.json` | the parity + backward result |
| `$WS/checkpoints/Qwen2-VL-2B-Instruct-mcore-tp1pp1` | converted checkpoint (4.2 GB, `release/mp_rank_00`) |

### Environment bits that are not code

- `$WS/venvs/trackb_pydeps` — installed with `--no-deps` on top of the container's torch.
  Keep the container's **numpy 1.26** (container `cv2` is built against it) and
  **`pyarrow==17.0.0`** (18+ demands numpy 2).
- `$WS/venvs/trackb_pydeps/megatron/energon/` — a ~30-line stub (EPath, Sample, placeholder
  factories). The real energon needs `multistorageclient`, which is not installable here.

---

## 5. Verified numbers

Pilot: 24 samples · 8/mixture · N=2 · seed 55402026 · derived totals 50–392.

Processor gate: **24/24 pass**, per-image slice isolation holds, hashes verified.

Parity (`low_0000`, 2×256 px, seq 194, grid `[[1,18,18],[1,18,18]]`, 162 image tokens,
5 supervised tokens):

| metric | value |
| --- | --- |
| MCore loss | 3.9477005005 |
| HF loss | 3.9478073120 |
| relative diff | **2.7056e-05** |
| per-token loss correlation | 0.99926 |
| MCore grad norm | 263.497598 (finite, > 0) |
| HF grad norm | not measured (24 GB OOM — see traps) |

---

## 6. Bugs already found — do not reintroduce

1. **Megatron does not shift labels.** `post_process` applies cross-entropy to the labels it is
   handed. Passing the same unshifted labels to HF and MCore inflates loss ~4× (16.53 vs 3.95).
   MCore side needs `target[:, :-1] = labels[:, 1:]`, last position masked.
2. **sm75 nodes cannot run the model.** THD/varlen vision attention needs flash/fused, sm80+.
3. **TE 2.x wants unpacked 4-D qkv.** MCore 0.14 hands it packed 3-D `[s,b,h*d]`. Patched in
   `TEDotProductAttention` with a pack/unpack wrapper (see `patches/bigmac-mcore-250624-te2.patch`).
4. **MCore needs `param.main_grad`** for its fused grad path; a bare `model(...).backward()`
   raises `'Parameter' object has no attribute 'main_grad'`.
5. **24 GB cannot hold both stacks.** MCore weights + fp32 `main_grad` + HF weights + HF fp32
   grads > 23.5 GiB. Free the MCore model before loading HF; skip the HF backward.
6. `gpu54` silently kills jobs (no stdout). Always exclude it.
7. Compute-node Singularity has no `--env`; `pytorch-sam3.sif`-style hardcoded binds to
   `/usr/lib/...` do not match this image's layout.

---

## 7. How to reproduce

```bash
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
WS=${WS:?set WS to your workspace}   # e.g. /research/<group>/<user>/workspaces/<name>
export BIND_ROOT=${BIND_ROOT:-/research}
PROJ=$WS/proj/csci5540

# 1. parity gate (1 GPU)
cd "$PROJ" && sbatch track_b_cluster/slurm/native_forward_check.sbatch
cat runs/track_b_n2_pilot_measured_v1/native_forward_check.json

# 2. conversion (only if the checkpoint is missing)
sbatch track_b_cluster/slurm/convert_qwen2vl_2b.sbatch

# 3. PP=2 equivalence (2 GPUs) — native then pp_executor
MODE=0 sbatch --export=ALL,MODE=0 track_b_cluster/slurm/pp2_equivalence.sbatch
MODE=1 sbatch --export=ALL,MODE=1 track_b_cluster/slurm/pp2_equivalence.sbatch
cat runs/track_b_n2_pilot_measured_v1/pp2_equivalence_mode{0,1}.json
```

Logs: `$WS/containers/{native-fwd,convert-q2vl-2b,pp2-equiv-<mode>}.log`.
All sbatch scripts redirect inside the script (`exec > "$LOG"`), so logs appear even when
SLURM's `--output` does not.

---

## 8. Decisions already made by the user (do not relitigate)

| # | Decision |
| --- | --- |
| Q1 | Stay on PMP + TE 2.1 (shim), rather than switching MCore version or PAI image |
| Q3 | **Stratify** pairing so resolution heterogeneity is guaranteed per sample |
| Q4 | 24 samples for smoke tests |
| Q5 | Fixed prompt across all samples |
| Q6 | Loop for warmup + measured steps, fixed step count |
| Q7 | Cross-mixture content duplication: ignored for now |
| Q8 | Performance is the objective → `pp_executor` traces are in scope |
| Q9 | No HF baseline; native Megatron is the reference |
| Q10 | 2 GPUs now, up to 16 later |

**Q3 is not implemented yet.** The current manifest uses seeded random pairing, which leaves
3 of 8 "High" samples as `(128,128)` — locally zero variance. A stratified builder must replace
it and the processor gate must be re-run.

---

## 9. Remaining work, in order

1. **PP=2 equivalence (G2).** Script exists (`bridge/pp2_equivalence.py`). Needs: native run and
   `MODE=1` run to produce identical loss and grad norm. Uses seeded CPU init instead of the
   converted checkpoint, because weights are irrelevant to a schedule-equivalence test.
   - native (MODE=0): **passes** — loss 6.5671281814575195, grad norm 867.0419179601411,
     identical on both ranks (job 26665).
   - MODE=1: fails ~60 s in; log stops right after the torchrun banner with **no traceback**
     (job 26668). Likely an import/init failure in the patched path that kills the child before
     Python flushes (or a segfault). Next step: run the same script under `srun` from a wrapper
     file (not `bash -lc "..."`) so stderr survives, and bisect by importing
     `pp_executor.mcore_patch` on a GPU node before `initialize_megatron`.
2. **Encoder bridge for Qwen2-VL (`LLMPPExecutorWithVE`).** This is the real integration work.
   Reference implementation: `BigMac/examples/qwen/pretrain_qwen3_vl_bigmac.py` (~400 lines).
   Required hooks on the `forward_step` owner: `get_encoder_models`, `encoder_forward_func`,
   `get_encoder_output_meta`. Also needs a **vision replica on every PP rank** and a cross-PP
   parameter broadcast. Without `MEGATRON_PP_EXECUTOR_WITH_VE=1` the executor runs the LLM-only
   schedule — useful as an intermediate check, but it is not the BigMac claim.
3. **Stratified manifest** (Q3) + re-run the processor gate on it.
4. **Real training-step runner**: manifest → tokens/labels → optimizer step, emitting the §6
   JSONL (`step, warmup, step_time_s, samples, max_memory_allocated_bytes_by_rank, loss,
   skipped, nan_or_inf, use_pp_executor`, plus instrumented `pipeline_bubble_ratio`).
   Then `scripts/collect_step_metrics.py` aggregates it.
5. **Instrumentation** for Q8: wire `pp_executor` tracing (`MEGATRON_PP_EXECUTOR_TRACE=1`,
   `..._TRACE_DIR`) so bubble ratio and encoder latency are measured rather than claimed.
6. **Scale-out** to 4/8/16 GPUs; PP=4 changes the layer split and the encoder placement again.

---

## 10. Traps that will bite the next agent

- `USE_PP_EXECUTOR` is **our** name; BigMac's code reads `MEGATRON_USE_PP_EXECUTOR` (and
  `MEGATRON_PP_EXECUTOR_WITH_VE` for the encoder path). Set both consistently.
- `pp_executor` only wraps when `pp_size > 1`. At PP=1 it is a no-op, so any "BigMac vs native"
  comparison at PP=1 is meaningless.
- The converted checkpoint is TP1/PP1. PP=2 either needs a PP=2 conversion
  (`hf2mcore_qwen2_vl_convertor.sh 2B <hf> <out> 1 2 false bf16`) or manual layer-range sharding.
- PMP pins Qwen2-VL `mrope_section=[16,24,24]`; the HF config confirms it, so the hardcoded
  default is correct for 2B — do not "fix" it.
- `--mem` on rtx3090 nodes fails job submission; leave it out.
- Meshed verification: a single 5-token sample catches gross mapping errors only. A second
  sample with `(128,384)` is the cheapest extra check before trusting the full pilot.
