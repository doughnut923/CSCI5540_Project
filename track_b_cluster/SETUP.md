# Track B setup guide — from clone to a Megatron run on the CSE GPUs

Target: run the Megatron-Core training pipeline for Qwen2-VL-2B with the Track B N=2
resolution-heterogeneity pilot, on the CSE cluster, with the option of toggling BigMac's
`pp_executor`.

**No special workspace is required.** Any account with ~40 GB of storage on a filesystem the
GPU nodes can see will do. The examples below use a placeholder `$WS`; point it at your own
directory and none of the commands change.

Every command below has been run on this cluster (with `$WS=/research/d7/fyp26/lkchau3/workspaces/cyfok3`).
Anything that looks odd is odd because the cluster requires it — see the troubleshooting table
before "fixing" something.

---

## 0. Prerequisites

| Need | Why |
| --- | --- |
| CSE cluster account | SLURM jobs, GPU nodes |
| **~40 GB** free on a shared filesystem visible from GPU nodes | container 11 GB (+~10 GB pull cache), model 4.5 GB, MCore checkpoint 4.2 GB, vendored repos ~2 GB, python extras 0.7 GB |
| Access to the `rtx3090` GPU nodes | the vision encoder needs sm80+; older cards cannot run it at all |
| GitHub access | `doughnut923/CSCI5540_Project`, `Dots-Infra/BigMac`, `NVIDIA/Megatron-LM`, `alibaba/Pai-Megatron-Patch` (submodule) |

Pick a workspace and put it in your shell profile:

```bash
export WS=<your workspace, e.g. /research/<group>/<user>/workspaces/<name>}
export PROJ=$WS/proj/csci5540
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf

# the filesystem root that must be bind-mounted into the container;
# /research for /research/... workspaces, $HOME if you keep the workspace in your home
export BIND_ROOT=/research
```

Check the space before you start:

```bash
df -h "$WS"        # want >= 40 GB available
```

> If your workspace is under `$HOME`, be careful: home quotas are small and Singularity will
> cache ~10 GB of OCI layers. Set `SINGULARITY_CACHEDIR`/`SINGULARITY_TMPDIR` on the same big
> filesystem (see §4) and `export BIND_ROOT=$HOME`.

Forgetting `SLURM_CONF` gives "Invalid job id" from every `squeue`/`sacct`/`scancel`.

---

## 1. Clone the project

```bash
mkdir -p "$PROJ"
git clone -b track-b-megatron-bigmac \
  https://github.com/doughnut923/CSCI5540_Project.git "$PROJ"
cd "$PROJ"
```

That branch contains the pilot builder, the processor gate, the parity harness, the PP=2/PP
scaling harness, the SLURM templates and `HANDOFF.md`.

Layout you will create next (name and location are yours to choose):

```
$WS/
├── proj/csci5540/          # this repo
├── third_party/            # BigMac, Pai-Megatron-Patch, Megatron-LM
├── models/                 # Qwen2-VL-2B-Instruct
├── checkpoints/            # converted MCore checkpoints
├── containers/             # .sif, logs
├── venvs/                  # python extras (no torch)
├── caches/hf/              # HF_HOME
└── runs/                   # optional, larger experiments
```

```bash
mkdir -p "$WS"/{third_party,models,checkpoints,containers,venvs,caches/hf,runs}
```

---

## 2. Third-party code (pinned)

Three repos, all pinned. Do not track `main` for any of them.

### 2.1 BigMac (includes Pai-Megatron-Patch as a submodule)

```bash
cd "$WS/third_party"
git clone https://github.com/Dots-Infra/BigMac.git
git -C BigMac checkout 2fa6da2fd3979448e89a56e734cbc3c490990d81
# Pai-Megatron-Patch + big_mac_open_source
git -C BigMac submodule update --init --recursive
```

### 2.2 The MCore backend we actually run

Pai-Megatron-Patch vendors several Megatron-LM checkouts; only one is used:

```bash
cd "$WS/third_party/BigMac/third_party/Pai-Megatron-Patch"
git submodule update --init backends/megatron/Megatron-LM-250624
git -C backends/megatron/Megatron-LM-250624 log -1 --format='%H %s'
# expect: 2220aa0d…  (branch pai_qwen3_vl_backend, Dots-Infra/BigMac-Megatron-Core)
```

`git submodule update --init --recursive` on BigMac may skip this one; if so run the command
above explicitly. It is the **only** MCore that can host Pai-Megatron-Patch's Qwen2-VL model —
stock 0.17.1 cannot (different process-group API).

### 2.3 Stock Megatron-LM 0.17.1 (optional, only for cross-checks)

```bash
cd "$WS/third_party"
git clone --depth 1 --branch core_v0.17.1 https://github.com/NVIDIA/Megatron-LM.git
```

### 2.4 Apply the compatibility patches

```bash
cd "$WS/third_party/BigMac/third_party/Pai-Megatron-Patch"
git apply -p1 "$PROJ/track_b_cluster/patches/pai-megatron-patch-mcore017.patch"

cd "$WS/third_party/BigMac/third_party/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624"
git apply -p1 "$PROJ/track_b_cluster/patches/bigmac-mcore-250624-te2.patch"

cd "$WS/third_party/Megatron-LM"
git apply -p1 "$PROJ/track_b_cluster/patches/megatron-lm-nvrx.patch"
```

What they do is documented in `track_b_cluster/patches/README.md`. Verify with
`git -C <repo> diff --stat` — you should see only the files listed there.

---

## 3. Model weights

Pinned revision, not `main`:

```bash
export HF_HOME=$WS/caches/hf
pip install --user "huggingface_hub[cli]"        # login node, small
hf download Qwen/Qwen2-VL-2B-Instruct \
  --revision 895c3a49bc3fa70a340399125c650a463535e71c \
  --local-dir "$WS/models/Qwen2-VL-2B-Instruct"
```

Check you have both weight shards plus the tokenizer/preprocessor files:

```bash
ls -lh "$WS/models/Qwen2-VL-2B-Instruct"
# model-00001-of-00002.safetensors  3.8G
# model-00002-of-00002.safetensors  410M
# config.json preprocessor_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt
```

If HF is unreachable, the ModelScope mirror works for the small files but its `resolve/<sha>`
URLs 404 for this revision, and `git clone` without `git-lfs` fails with
`RPC failed; result=18`. Use HTTP with `curl -L -o <file> <url>` per file instead.

---

## 4. Container

The host is CentOS 7 with GLIBC 2.17 and driver 565.57.01 (CUDA 12.7). Modern torch does not
run on the host, so everything runs inside a SIF.

```bash
# cache/tmp must live on the big filesystem, not the home quota
export SINGULARITY_CACHEDIR=$WS/singcache
export SINGULARITY_TMPDIR=$WS/tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
chmod 700 "$SINGULARITY_CACHEDIR"
```

Pull it **on a GPU node**: the login node has no `mksquashfs`.

```bash
cd "$WS/containers"
sbatch --account=gpu -p gpu_2h --gres=gpu:1 --constraint=rtx3090 --exclude=gpu54 \
  --time=01:00:00 --wrap '
    export WS='"$WS"'; export BIND_ROOT='"$BIND_ROOT"';
    export SINGULARITY_CACHEDIR=$WS/singcache; export SINGULARITY_TMPDIR=/tmp;
    cd "$WS/containers";
    singularity pull ngc-pytorch-25.03-py3.sif docker://nvcr.io/nvidia/pytorch:25.03-py3'
```

Expect `ngc-pytorch-25.03-py3.sif`, ~11 GB, torch 2.7.0a0+cu128 + Transformer Engine 2.1.0.

**Do not use NGC 25.10 or later** — those images carry CUDA-13 torch and refuse driver 565
(`driver too old (found 12080)`).

Verify (replace `$BIND_ROOT` with yours):

```bash
sbatch --account=gpu -p gpu_2h --gres=gpu:1 --constraint=rtx3090 --exclude=gpu54 \
  --time=00:10:00 --wrap '
  singularity exec --nv \
    --bind /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
    --bind /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    -B '"$BIND_ROOT:$BIND_ROOT"' '"$WS"'/containers/ngc-pytorch-25.03-py3.sif \
    python -c "import torch,transformer_engine as te;print(torch.__version__,te.__version__,torch.cuda.is_available())"'
```

`--nv` alone is not enough on these nodes; the two `--bind` lines are mandatory.

---

## 5. Python extras

The container already has torch, numpy, cv2, flash-attn. Only add what is missing, with
`--no-deps`, into a directory that is put on `PYTHONPATH` at run time (never into the image):

```bash
D=$WS/venvs/trackb_pydeps
mkdir -p "$D"

singularity exec --env PIP_NO_CACHE_DIR=1 --env TMPDIR=$WS/tmp \
  -B "$BIND_ROOT:$BIND_ROOT" \
  "$WS/containers/ngc-pytorch-25.03-py3.sif" \
  python3 -m pip install --target "$D" --no-deps \
    transformers==4.49.0 tokenizers==0.21.4 safetensors regex tqdm \
    huggingface-hub==0.26.2 httpcore2 h11 certifi qwen-vl-utils av \
    sentencepiece protobuf accelerate psutil einops matplotlib \
    pyparsing contourpy cycler fonttools kiwisolver python-dateutil six \
    datasets==3.2.0 pyarrow==17.0.0 megatron-energon webdataset braceexpand
```

Three constraints that are not optional:

- keep the container's **numpy 1.26** in front — `pip` will happily install numpy 2 and break
  the container's `cv2`. Do not add numpy to the list; if a package drags it in, delete
  `$D/numpy*` afterwards.
- **`pyarrow==17.0.0`** — 18+ requires numpy 2.
- `megatron.energon` needs `multistorageclient`, which is not installable here. Replace the
  installed package with the local stub:

```bash
rm -rf "$D/megatron/energon"
```

then create the stub described in `track_b_cluster/patches/README.md` §"Extra runtime pieces"
(~30 lines: `EPath`, `Sample`, placeholder factories). Without it, importing the Qwen2-VL model
fails on `multistorageclient`.

---

## 6. Data

`part_b_image_res/image_sets/{128,256,384,...}` and `variance_datasets/dataset_*.json` are
**not** in git (they are generated). Recreate them, or copy them in from a machine that has
them:

```bash
cd "$PROJ/part_b_image_res"
python -m pip install -r requirements.txt       # needs `datasets`; login node is fine
python data_prep.py                             # -> image_sets/{128,256,384,512,768}
python distinct_variance.py --output-dir variance_datasets
```

`distinct_variance.py` prints mean/std per mixture. Expect Low std 0, Medium ~57, High ~81.

Then build the pilot and run the processor gate (both CPU-only):

```bash
cd "$PROJ"
python3 part_b_image_res/track_b_pilot.py \
  --output-dir runs/track_b_n2_pilot_measured_v1 \
  --samples-per-mixture 8

singularity exec --env PYTHONPATH="$WS/venvs/trackb_pydeps" \
  -B "$BIND_ROOT:$BIND_ROOT" \
  "$WS/containers/ngc-pytorch-25.03-py3.sif" \
  python3 part_b_image_res/validate_n2_pilot.py \
    --manifest runs/track_b_n2_pilot_measured_v1/pilot_manifest.json \
    --output-dir runs/track_b_n2_pilot_measured_v1/processor_gate \
    --model "$WS/models/Qwen2-VL-2B-Instruct" \
    --local-files-only --check-hashes --repo-root .
```

Expected: `{"status": "pass", "samples_checked": 24, "passed": 24, "failed": 0}`.

---

## 7. Convert the checkpoint to Megatron-Core

Every `track_b_cluster/slurm/*.sbatch` reads `WS` (and `BIND_ROOT`) from the environment and
falls back to the original author's paths, so **always export them first** (or pass
`--export=ALL,WS=...,BIND_ROOT=...`):

```bash
export WS="$WS" BIND_ROOT="$BIND_ROOT"
cd "$PROJ"
sbatch track_b_cluster/slurm/convert_qwen2vl_2b.sbatch
tail -f "$WS/containers/convert-q2vl-2b.log"
```

Output: `$WS/checkpoints/Qwen2-VL-2B-Instruct-mcore-tp1pp1` (4.2 GB,
`release/mp_rank_00`), log line `CONVERT_RC=0`.

---

## 8. First run: single-GPU parity gate

This is the cheapest way to know the whole stack works.

```bash
cd "$PROJ"
sbatch track_b_cluster/slurm/native_forward_check.sbatch
cat runs/track_b_n2_pilot_measured_v1/native_forward_check.json
```

Healthy output:

```json
{"mcore_loss": 3.9477005005, "hf_loss": 3.9478073120, "rel_diff": 2.7e-05,
 "mcore_grad_norm": 263.497598, "mcore_grad_finite": true}
```

If `rel_diff` is ~4 (not 1e-5), the MCore side is getting unshifted labels — see
troubleshooting.

---

## 9. The Megatron pipeline at scale

### 9.1 Native schedule, PP scaling (works today)

```bash
sbatch --gres=gpu:4 \
  --export=ALL,WS="$WS",BIND_ROOT="$BIND_ROOT",PP=4,BENCH_STEPS=10,WARMUP_STEPS=2,NUM_MICROBATCHES=8 \
  track_b_cluster/slurm/pp_scale_native.sbatch
```

- `NUM_LAYERS` defaults to 28; 28/4 = 7 layers per stage. For **PP=8** you must pass
  `NUM_LAYERS=32` (28 % 8 ≠ 0).
- `NUM_MICROBATCHES=8` matters: the pipeline bubble is roughly `(PP−1)/M`, so `M=2` at PP=4 is
  mostly fill/drain and tells you nothing.
- Outputs: `runs/track_b_n2_pilot_measured_v1/pp_native_scale_pp4.{json,jsonl}` plus
  `..._summary.json` from the metrics collector.

Quick look:

```bash
python3 -c "
import json;d=json.load(open('runs/track_b_n2_pilot_measured_v1/pp_native_scale_pp4_summary.json'))
print(d['primary']['throughput_samples_per_s'], d['primary']['pipeline_bubble_ratio'])"
```

### 9.2 BigMac schedule

```bash
MODE=0 sbatch --export=ALL,WS="$WS",BIND_ROOT="$BIND_ROOT",MODE=0 track_b_cluster/slurm/pp2_equivalence.sbatch
MODE=1 sbatch --export=ALL,WS="$WS",BIND_ROOT="$BIND_ROOT",MODE=1 track_b_cluster/slurm/pp2_equivalence.sbatch
```

Compare `loss` and `grad_norm` in `pp2_equivalence_mode{0,1}.json` — they must match.

**Current status: `MODE=1` fails silently ~60 s in** (log stops after the torchrun banner, no
traceback). `MODE=1` at PP=2 is also only the LLM schedule; the encoder-nesting path
(`MEGATRON_PP_EXECUTOR_WITH_VE=1`) is not wired for Qwen2-VL yet. See `HANDOFF.md` §9.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Job fails in ~2 s, no output at all | landed on `gpu54` | add `--exclude=gpu54` |
| `Requested node configuration is not available` | rtx3090 nodes reject `--mem`; bare `--gres=gpu:rtx3090:N` also fails | use `--gres=gpu:N --constraint=rtx3090`, remove `--mem` |
| `Memory specification can not be satisfied` | same `--mem` issue | drop the `--mem` line |
| `No dot product attention support for the provided inputs!` | job landed on sm75 (RTX 2080/Titan); THD vision attention needs sm80+ | `--constraint=rtx3090` |
| `CUDA initialization: driver too old (found 12080)` | CUDA-13 image (NGC 25.10+) | use NGC 25.03 |
| `libcuda.so cannot found!` / triton driver error | `--nv` did not inject the host libs | bind `libcuda.so.1` + `libnvidia-ml.so.1` into `/usr/lib/x86_64-linux-gnu`, set `TRITON_LIBCUDA_PATH` |
| `unknown flag: --env` | compute-node Singularity is older | use `SINGULARITYENV_<VAR>=` |
| `ldconfig: command not found` | image drops `/sbin` from PATH | `SINGULARITYENV_PATH=/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin` |
| `'Parameter' object has no attribute 'main_grad'` | MCore fused grad path needs it | `param.main_grad = torch.zeros_like(param, dtype=torch.float32)` before backward |
| loss ~4× HF | labels not shifted for Megatron | `target[:, :-1] = labels[:, 1:]`, mask last position |
| `nvidia_resiliency_ext has no __version__` | NGC package quirk | `megatron-lm-nvrx.patch` |
| `No module named 'multistorageclient'` | energon | install the local stub, remove real energon |
| `numpy.core.multiarray failed to import` | numpy 2 shadowing the container | delete `$D/numpy*`, keep pyarrow 17 |
| two torch installs, root fs full | `pip install transformers` pulled torch | always `--no-deps`, `TMPDIR` on the big filesystem |
| file not found inside the container | `$WS` not covered by `-B` | bind `$BIND_ROOT` (or the exact directory) |

---

## 11. Where things are

| Thing | Path |
| --- | --- |
| Repo / this guide | `$PROJ`, `$PROJ/track_b_cluster/SETUP.md` |
| Handoff + open work | `$PROJ/track_b_cluster/HANDOFF.md` |
| Compat patches | `$PROJ/track_b_cluster/patches/` |
| Pilot + gate | `$PROJ/runs/track_b_n2_pilot_measured_v1/` |
| MCore checkpoint | `$WS/checkpoints/Qwen2-VL-2B-Instruct-mcore-tp1pp1` |
| Logs | `$WS/containers/*.log` |
| Env extras | `$WS/venvs/trackb_pydeps` |
