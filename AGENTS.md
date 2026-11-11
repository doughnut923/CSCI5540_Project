# AGENTS (csci5540 proj)

Compact, repo-specific guidance. Prefer executable sources (`docs/plan.md`, per-part `README.md`,
`requirements.txt`, scripts) over prose. For Track B, `track_b_cluster/SETUP.md` (how to build the
environment) and `track_b_cluster/HANDOFF.md` (current state, verified numbers, open bugs) are the
authoritative pair — read them before touching the cluster.

## Scope / layout
- MLLM input-heterogeneity experiments plus a cluster plan for Megatron-Core baseline vs BigMac vs Optimus.
- `part_a_model_check/` — Qwen2-VL real-image forward/backward reference (MPS/CPU/CUDA).
- `part_b_image_res/` — resolution heterogeneity (pilot builder, processor gate, variance datasets).
- `part_c_image_count/` — image-count heterogeneity + `real_photo_pilot/`.
- `track_b_cluster/` — Megatron+BigMac execution: bridge harnesses, SLURM templates, compat patches.
- `docs/plan.md` — authoritative experiment plan. No root build/test/CI config; everything is script/README-driven.

## Git
- Track B work belongs on branch `track-b-megatron-bigmac` (remote: `doughnut923/CSCI5540_Project`). Do not push to `part-a-*` or `main`.
- `.venv/`, `__pycache__/`, `runs/` are ignored/untracked. `third_party/` is deliberately **not** committed — its pins are recorded in `HANDOFF.md`.
- Do not commit generated data (`variance_datasets/*_batches.json`, `image_sets/`) or cluster paths into code.

## Cluster (facts that cost hours to rediscover)
- Keep everything under `/research/d7/fyp26/lkchau3/workspaces/cyfok3`. Never write elsewhere on the cluster.
- `export SLURM_CONF=/opt1/slurm/gpu-slurm.conf` — without it every `squeue`/`sacct`/`scancel` says "Invalid job id".
- Submit with `--account=gpu -p gpu_2h --gres=gpu:N --constraint=rtx3090 --exclude=gpu54`. Omit `--mem` (rtx3090 nodes reject it). Keep CPU:GPU ≤ 6:1 unless the node has `highcpucount` — the rtx3090 nodes do, so `-c 8` is fine there but fails without the constraint.
- `gpu54` silently kills batch jobs (no stdout). Always exclude it.
- Host driver is 565.57.01 (CUDA 12.7): **CUDA-13 images do not run**. Use only
  `$WS/containers/ngc-pytorch-25.03-py3.sif` (torch 2.7+cu128, TE 2.1.0).
- Compute nodes have an old Singularity: no `--env`. Use `SINGULARITYENV_<VAR>=...` instead.
- `--nv` is not sufficient; bind host libs explicitly:
  `--bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1`
  `--bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1`,
  and set `SINGULARITYENV_TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu`.
- Set `SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin` — TE shells out to `ldconfig`.
- The vision encoder runs THD/varlen attention and needs **sm80+**; on an sm75 node TE disables every backend and raises `No dot product attention support`. Hence `--constraint=rtx3090`.
- Login node has Python 3.6.8 (`/bin/python3`). Keep stdlib-only cluster scripts 3.6-safe (no `statistics.fmean`, no `X | None`, no `from __future__ import annotations`). CPU prep on the login node is fine; GPU/model work goes through SLURM.

## Third-party code
- Pinned: BigMac `2fa6da2f`, Pai-Megatron-Patch `3413f297`, MCore backend `Megatron-LM-250624` `2220aa0d` (BigMac fork, core 0.14), stock Megatron-LM `266f1c97` (`core_v0.17.1`).
- **Only** the `250624` fork hosts Pai-Megatron-Patch's Qwen2-VL model. Stock 0.17.1 passes `pg_collection` into attention modules PMP defines itself, so it cannot be used.
- Local edits to `third_party/` are required to run at all. Keep them as patches in `track_b_cluster/patches/` and re-apply with `git apply -p1`; never hand-edit without regenerating the patch (`git -C <repo> diff > ...`).
- `venvs/trackb_pydeps` shadows a few modules by design: install extras with `--no-deps`, keep the container's **numpy 1.26** (its `cv2` is built against it), pin **`pyarrow==17`** (18+ demands numpy 2), and replace `megatron/energon` with the local stub (real energon needs `multistorageclient`, unavailable).

## Track B gotchas
- `pp_executor` only wraps when `pp_size > 1`; a PP=1 "BigMac vs native" comparison is a no-op. Set both `USE_PP_EXECUTOR` (ours) and `MEGATRON_USE_PP_EXECUTOR` (BigMac's), plus `MEGATRON_PP_EXECUTOR_WITH_VE=1` for the encoder path.
- Megatron does **not** shift labels internally: the MCore side must receive next-token targets. Unshifted labels inflate loss ~4×.
- MCore's fused grad path requires `param.main_grad` before `backward()`.
- A 24 GB card cannot hold HF and MCore with gradients at once; free one before loading the other.
- The converted checkpoint is TP1/PP1 (`checkpoints/Qwen2-VL-2B-Instruct-mcore-tp1pp1`). PP>1 needs a re-conversion or layer-range sharding.

## Evidence rules
- **Measured vs derived vs unmeasured:** keep claims scoped, especially `part_c_image_count` and anything from `track_b_cluster`. Derived token totals are not measurements; mock step records (`mock_step_runner.py`) are not performance data.
- Scripts refuse to overwrite output directories (`mkdir(exist_ok=False)`). Pick a new `--output-dir` instead of deleting old evidence.
- Every sbatch redirects its own log (`exec > "$LOG"`) because SLURM `--output` files have been lost on some nodes.

## Key commands
```powershell
# part_b (local, Windows)
python -m pip install -r .\part_b_image_res\requirements.txt
python .\part_b_image_res\data_prep.py
python .\part_b_image_res\distinct_variance.py --output-dir .\part_b_image_res\variance_datasets
```
```sh
# part_c (from part_c_image_count/)
python -m pip install -r requirements.txt
python download_reference.py
python validate_pipeline.py --output-dir runs/pilot_v1
python real_photo_pilot/run_real_photo_pilot.py --output-dir real_photo_pilot/runs/reproduction_v1
```
```sh
# track B (cluster, from repo root)
python3 part_b_image_res/track_b_pilot.py --output-dir runs/track_b_n2_pilot_measured_v1 --samples-per-mixture 8
sbatch track_b_cluster/slurm/convert_qwen2vl_2b.sbatch
sbatch track_b_cluster/slurm/native_forward_check.sbatch          # parity gate, expect rel_diff ~1e-5
sbatch --gres=gpu:4 --export=ALL,PP=4,BENCH_STEPS=10,NUM_MICROBATCHES=8 track_b_cluster/slurm/pp_scale_native.sbatch
```
