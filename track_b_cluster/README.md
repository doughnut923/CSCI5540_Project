# Track B cluster implementation (Megatron+BigMac plan)

Portable scaffolding for Phases 0–4 of `C:\Users\jonas\.opencode\plan\megatron-bigmac-integration.md`. **No heavy work on login nodes.** Run GPU jobs via SLURM under the confirmed `cyfok3` workspace.

## What's implemented here (local, commit to GitHub)
- `part_b_image_res/track_b_pilot.py` — deterministic Track B pilot builder: **N=2 images/sample**, seed, mixtures, manifest + SHA256 hashes, derived token totals (128→25,256→81,384→196).
- `part_b_image_res/validate_n2_pilot.py` — processor-only validation gate for N=2 manifests (grids/totals, per-image slice isolation). Requires Qwen2-VL processor assets; no model weights/forward.
- `track_b_cluster/scripts/collect_step_metrics.py` — post-warmup metric aggregation: throughput, step time mean/p50/p95, peak GPU mem/rank. Bubble/stall/encoder-variance recorded only when instrumentation provides them (else `unmeasured`).
- SLURM/script templates for Part A CUDA sanity and Track B native-vs-BigMac toggle runs.

## Must confirm on cluster before GPU runs (plan open items)
1. Canonical absolute workspace path (`CYFOK3_WORKSPACE`; plan notes a possible path-string mismatch).
2. Approved container `.sif` (NGC 25.10-like preferred per BigMac validation) and bind mounts; or approved venv/module stack.
3. GPU allocation/partition/GRES for 4-GPU smoke and 8-GPU runs; time/mem limits.
4. Qwen2-VL HF→MCore checkpoint/loader compatibility in deployed MCore/BigMac runtime.
5. BigMac checkout commit in `<WS>/third_party/BigMac` and exact Qwen2-VL bridge patch location.

## Phase 0 (staging, no login-node heavy work)
```sh
# On cluster (allocation or approved transfer path), under $CYFOK3_WORKSPACE
export CYFOK3_WORKSPACE=/research/d7/fyp26/lkchau3/workspaces/cyfok3  # CONFIRM
mkdir -p "$CYFOK3_WORKSPACE"/{proj,third_party,models,runs,caches}
export HF_HOME="$CYFOK3_WORKSPACE/caches/hf"
git clone <this-repo-url> "$CYFOK3_WORKSPACE/proj/csci5540"
git clone https://github.com/Dots-Infra/BigMac "$CYFOK3_WORKSPACE/third_party/BigMac" && git -C "$CYFOK3_WORKSPACE/third_party/BigMac" rev-parse HEAD
# Stage model (ModelScope or approved mirror) and verify revision/hash parity with 895c3a49bc3fa70a340399125c650a463535e71c.
# Stage/generate part_b image_sets + variance_datasets deterministically; record hashes.
```

## Build Track B N=2 pilot manifest (CPU-safe)
```sh
cd "$CYFOK3_WORKSPACE/proj/csci5540"
python part_b_image_res/track_b_pilot.py \
  --output-dir runs/track_b_n2_pilot_smoke \
  --mixtures Low Medium High --samples-per-mixture 24 --seed 55402026
```
Default pilot: 72 samples (24/mixture), N=2, seed 55402026. Adjust `--samples-per-mixture`/mixtures for the confirmed smoke spec; keep identical manifest for native and BigMac.

## Processor gate (Phase 3.2, CPU/job as appropriate)
```sh
python part_b_image_res/validate_n2_pilot.py \
  --manifest runs/track_b_n2_pilot_smoke/pilot_manifest.json \
  --output-dir runs/track_b_n2_pilot_smoke/processor_gate \
  --model "$CYFOK3_WORKSPACE/models/Qwen2-VL-2B-Instruct" \
  --local-files-only --check-hashes --repo-root .
```

### Verified cluster recipe (pytorch-sam3.sif + extra deps on /research)
Host GLIBC is too old for modern torch, so run inside the SIF. Keep pip TMPDIR on `/research`
(root fs is small) and install only the packages missing from the container with `--no-deps`:
```sh
export TMPDIR=$CYFOK3_WORKSPACE/tmp
D=$CYFOK3_WORKSPACE/venvs/trackb_pydeps
singularity exec --env PIP_NO_CACHE_DIR=1 --env TMPDIR=$TMPDIR \
  -B /research/d7/fyp26/lkchau3:/research/d7/fyp26/lkchau3 \
  /research/d7/fyp26/lkchau3/fyp/containers/pytorch-sam3.sif \
  python3 -m pip install --target $D --no-deps transformers==4.49.0 tokenizers==0.21.4 \
  safetensors regex tqdm huggingface-hub==0.26.2 httpcore2 h11 certifi \
  qwen-vl-utils av sentencepiece protobuf accelerate psutil
```
Then (CPU, no `--nv`):
```sh
singularity exec --env PYTHONPATH=$D -B /research/d7/fyp26/lkchau3:/research/d7/fyp26/lkchau3 \
  /research/d7/fyp26/lkchau3/fyp/containers/pytorch-sam3.sif \
  python part_b_image_res/validate_n2_pilot.py --manifest ... --local-files-only --check-hashes ...
```
Pitfall: a plain `pip install transformers` also pulls a second torch + CUDA 13 wheels (~2.5 GB)
and fills the container root fs; `--no-deps` + the container's torch avoids that.

## Controlled comparison (Phase 4)
Run the **same** pilot/hardware/env/config; only toggle `USE_PP_EXECUTOR`.
- Native: `USE_PP_EXECUTOR=0` (MCore native path in the adapted runtime; not claimed as stock upstream Megatron-LM without parity checks).
- BigMac: `USE_PP_EXECUTOR=1`.
Capture per-step JSON lines/records, then aggregate:
```sh
python track_b_cluster/scripts/collect_step_metrics.py \
  --input runs/track_b/native/step_metrics.jsonl \
  --output runs/track_b/native/metrics_summary.json \
  --warmup-steps 2 --world-size 4 --batch-size 1 --grad-accum 1
```

## Phase 2 bridge (critical path, not fully runnable here)
BigMac provides Qwen3-VL examples but not turnkey Qwen2-VL. `bridge/` contains a contract/spec scaffold. Actual work requires the cluster BigMac+MCore runtime:
1. Adapt `pp_executor/mcore_patch.py`/bridge for Qwen2-VL `visual` + merger, `image_grid_thw`/`pixel_values`, image token positions.
2. Preserve Part A loss masking (mask prompt/image slots/padding; ≥2 assistant targets).
3. Parity ladder: single-GPU native vs Part A → native PP N=2 → `USE_PP_EXECUTOR=1`. Debug shapes/token counts/grads first.
4. Resolve HF→MCore checkpoint/loader compatibility before full runs.
