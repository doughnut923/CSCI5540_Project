# AGENTS (csci5540 proj)

MLLM input-heterogeneity experiments: three local parts, plus a cluster Track B (fused Megatron +
BigMac) and DistTrain (encoder and LLM as two units) arm. No root build/lint/test/CI config —
everything is scripts and READMEs. Locally you get data prep and the Part A reference check; the
Megatron/BigMac runs only work on the cluster.

## Start here

| Need | Read |
| --- | --- |
| Build the cluster environment | `track_b_cluster/SETUP.md` |
| Track B state, verified numbers, open bugs | `track_b_cluster/HANDOFF.md` |
| DistTrain contract (knobs, JSONL fields, traps) | `docs/distrain/distrain-handoff.md` |
| DistTrain measured numbers and caveats | `docs/distrain/distrain-results.md` |
| B/C workload suite on the arm | `docs/distrain/distrain-bc-suite.md` |
| How the pipeline bubble is defined | `docs/pipeline-bubble.md` |

## Layout

- `part_a_model_check/` — Qwen2-VL real-image forward/backward reference (MPS/CPU/CUDA).
- `part_b_image_res/` — resolution heterogeneity: `data_prep.py` → `image_sets/`, `distinct_variance.py`
  → the three 750-image schedules, `track_b_pilot.py` → the N=2 pilot manifest the fused and DistTrain
  runs consume.
- `part_c_image_count/` — image-count heterogeneity; `real_photo_pilot/` holds the 8-photo pool.
- `track_b_cluster/bridge/` — `native_forward_check.py` (parity gate), `pp2_equivalence.py` (fused
  baseline), `manifest_data.py` (the shared sample plan), `disagg_units.py` (DistTrain).
- `track_b_cluster/{slurm,scripts,patches}/` — launchers, metric/report tools, compat patches.

## Git

- Two long-lived branches off `main`: `track-b-megatron-bigmac` (fused baseline) and
  `distrain-integration` (DistTrain arm). Branch work, PRs into it, merge up — never straight to `main`.
- `runs/` is **untracked but not ignored** — never `git add -A`. Evidence lives on the cluster; only
  docs and small summaries get committed.
- `part_b_image_res/variance_datasets/dataset_{Low,Medium,High}.json` are committed image-level
  schedules. `image_sets/` and any `*_batches.json` are generated and must not be committed.
- `third_party/` is deliberately not committed; its pins are in `track_b_cluster/HANDOFF.md`.

## Cluster

- Pick a workspace the GPU nodes can see, export `WS` and `BIND_ROOT` (`/research`, or `$HOME`). The
  `/research/d7/fyp26/lkchau3/...` path baked into the scripts is the original author's — override it.
- The cluster clone has no GitHub credentials: push locally, then on the login node
  `git fetch origin <branch> && git reset --hard FETCH_HEAD`.
- `export SLURM_CONF=/opt1/slurm/gpu-slurm.conf` — without it `squeue`/`sacct`/`scancel` all say
  "Invalid job id".
- Submit with `--gres=gpu:N --constraint=rtx3090`; never pass `--mem` (rtx3090 nodes reject it). The
  vision encoder needs sm80+, hence the constraint.
- Exclude `gpu54` (silently kills batch jobs) and `projgpu33` (shared GPUs, OOMs the optimizer).
- An N-GPU job needs N free GPUs on **one** node, and these nodes are usually full of day-scale jobs:
  8-GPU runs can queue for many hours. The QOS also caps 4 jobs per user.
- Host driver is 565.57.01 / CUDA 12.7: CUDA-13 images do not run. Use
  `$WS/containers/ngc-pytorch-25.03-py3.sif` only.
- That image's `/usr/bin/python3` has torch but **not transformers** — transformers comes from
  `$WS/venvs/trackb_pydeps`, which must be on `SINGULARITYENV_PYTHONPATH` for any `python3` you run.
- Compute nodes run an old Singularity: no `--env`, use `SINGULARITYENV_<VAR>=...`. `--nv` alone is
  not enough; bind host `libcuda.so.1` and `libnvidia-ml.so.1`, and set
  `SINGULARITYENV_TRITON_LIBCUDA_PATH` plus `SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin`
  (TE shells out to `ldconfig`).
- Login node python is 3.6.8. Keep stdlib-only cluster scripts 3.6-safe (no `statistics.fmean`, no
  `X | None`); CPU-only prep there is fine, model work goes through SLURM.

## Third-party

- Pinned: BigMac `2fa6da2f`, Pai-Megatron-Patch `3413f297`, MCore backend `Megatron-LM-250624`
  `2220aa0d`, stock Megatron-LM `266f1c97`. Only the `250624` fork hosts PMP's Qwen2-VL model.
- Local edits under `third_party/` are required to run. Keep them as patches in
  `track_b_cluster/patches/`, re-apply with `git apply -p1`, and regenerate with `git -C <repo> diff`
  after any hand edit.
- `venvs/trackb_pydeps` shadows modules by design: install extras with `--no-deps`, keep the
  container's numpy 1.26 (its cv2 is built against it), pin `pyarrow==17`.

## Track B baseline

- `pp_executor` only wraps when `pp_size > 1`, so a PP=1 "BigMac vs native" comparison is a no-op.
  Set `USE_PP_EXECUTOR` and `MEGATRON_USE_PP_EXECUTOR`, plus `MEGATRON_PP_EXECUTOR_WITH_VE=1` for the
  encoder path.
- Megatron does not shift labels: the MCore side needs next-token targets, or loss inflates ~4×.
- `pp2_equivalence.py` never steps an optimizer, so its rows are timing-only and not like-for-like
  with DistTrain's training-valid rows. Say so in any table.
- MCore's fused grad path needs `param.main_grad` before `backward()`. A 24 GB card cannot hold HF and
  MCore with gradients at once — free one first.
- The converted checkpoint is TP1/PP1 (`checkpoints/Qwen2-VL-2B-Instruct-mcore-tp1pp1`); PP>1 needs a
  re-conversion or layer-range sharding.

## DistTrain arm

- Two torchrun worlds in one allocation: `ENC_GPUS + LLM_PP <= 8`, and `--gres=gpu:N` must equal that
  sum — the launcher assigns `CUDA_VISIBLE_DEVICES` by index, so a short allocation indexes GPUs that
  do not exist.
- One sequence length per step. MCore/PMP builds the rotary table for the step's first microbatch, so
  `MANIFEST_BATCHING=manifest` raises `size of tensor a (609) must match the size of tensor b (607)`.
  Samples are bucketed by `(visual tokens, image count)` in `bridge/manifest_data.py`; both harnesses
  import it, so a grouping change moves the fused baseline too.
- `--seq-length` must cover the longest sample: the Part C image-count cases are 607-621 tokens.
- The bucket plan advances one bucket per step, so the step budget decides coverage: 6 steps reached
  10/12 samples in two suite cases, 8 steps reached 12/12. Check `samples_consumed`, not the step count.
- Peak memory: PP=2 at 32 layers is ~20.7 GiB per GPU; PP=4 is safe.
- `PLAN_ONLY=1` (+ `PLAN_STEPS`) prints the plan and per-step length with no GPUs — use it before
  spending an allocation.

## Evidence rules

- Keep measured / derived / unmeasured apart. Derived token totals are not measurements; mock step
  records (`bridge/mock_step_runner.py`) are never performance data.
- Label placement-only rows: no optimizer step, or no encoder gradient exchange, means it is not a
  valid training configuration.
- Scripts refuse to overwrite output directories (`mkdir(exist_ok=False)`) — pick a new
  `--output-dir` rather than deleting old evidence.
- Every sbatch redirects its own log (`exec > "$LOG"`); SLURM `--output` files have been lost on some
  nodes.
- `scripts/report_bc_suite_distrain.py` is the gate for suite runs: it exits non-zero on a sanity or
  input-coverage regression.

## Key commands

```powershell
# local, Windows: part B image sets and the variance schedules
python -m pip install -r .\part_b_image_res\requirements.txt
python .\part_b_image_res\data_prep.py
python .\part_b_image_res\distinct_variance.py --output-dir .\part_b_image_res\variance_datasets
```

```sh
# cluster, from the repo root (export SLURM_CONF=/opt1/slurm/gpu-slurm.conf first)
python3 part_b_image_res/track_b_pilot.py --output-dir runs/track_b_n2_pilot_measured_v1 \
  --samples-per-mixture 8                              # the N=2 pilot manifest
sbatch track_b_cluster/slurm/convert_qwen2vl_2b.sbatch
sbatch track_b_cluster/slurm/native_forward_check.sbatch        # parity gate, expect rel_diff ~1e-5

# fused baseline and DistTrain arm on the same plan
sbatch --gres=gpu:4 --export=ALL,MODE=0,PP=4,NUM_LAYERS=32,NUM_MICROBATCHES=4,BENCH_STEPS=10 \
  track_b_cluster/slurm/pp2_equivalence.sbatch
sbatch --gres=gpu:3 --export=ALL,ENC_GPUS=1,LLM_PP=2,NUM_LAYERS=28,NUM_MICROBATCHES=2,\
BENCH_STEPS=10,WARMUP_STEPS=2,TAG=demo track_b_cluster/slurm/disagg_units.sbatch

# metrics / figures / suite gates (container + venv PYTHONPATH)
python3 track_b_cluster/scripts/collect_step_metrics.py --input <jsonl> --output <json> --world-size N
python3 track_b_cluster/scripts/plot_pipeline_gantt.py --input <jsonl> --step N
python3 track_b_cluster/scripts/build_part_b_suite.py --output-dir <new> --model-dir <QWEN2VL_HF_DIR>
python3 track_b_cluster/scripts/report_bc_suite_distrain.py \
  --suite-root <suite> --runs-root <runs> --output <summary.json>
```
