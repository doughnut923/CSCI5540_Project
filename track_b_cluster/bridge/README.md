# Qwen2-VL bridge scaffold (Phase 2 — critical path)

BigMac has Qwen3-VL examples but no turnkey Qwen2-VL example. This directory provides the **contract** the cluster bridge/runner must satisfy so the rest of the plan (pilot, validation, native-vs-BigMac toggle, metrics) stays stable.

## Required runtime work on cluster (needs BigMac + MCore)
1. **Adapt bridge/patch** (pattern: BigMac `examples/qwen/pretrain_qwen3_vl_bigmac.py` + `pp_executor/mcore_patch.py`):
   - Qwen2-VL `visual` encoder + merger.
   - `pixel_values`, `image_grid_thw`, image token positions / placeholder mapping.
   - Stable cross-boundary tensor metadata (names, shapes, dtypes, seq dims, grad requirements) across PP boundaries.
   - Per-iteration state reset.
2. **Loss masking parity** with `part_a_model_check/run_check.py`: mask prompt/image slots/padding; require ≥2 assistant supervised targets.
3. **Toggle**: `USE_PP_EXECUTOR=0` = native path in adapted runtime; `USE_PP_EXECUTOR=1` = BigMac `pp_executor` executes scheduler plan. Keep all other config identical.
4. **HF→MCore checkpoint/loader**: resolve compatibility for Qwen2-VL in deployed MCore/BigMac version before full runs.

## Parity ladder (debug in this order)
1. Single-GPU native forward+backward vs Part A (tolerant fp tolerance), shapes/token counts/grads.
2. Native PP with N=2 pilot.
3. `USE_PP_EXECUTOR=1` with same pilot/config.

## Runner contract (what `BRIDGE_RUNNER` must do)
The SLURM templates call:
```sh
python "$BRIDGE_RUNNER" --manifest <pilot_manifest.json> --model-dir <path> --output-dir <dir> --metrics-jsonl <path> [runner args]
```
It must:
- Load N=2 samples from manifest (paths relative to repo root or absolute fields).
- Run the configured MCore/BigMac training/eval steps with identical data order for native/BigMac.
- Respect `USE_PP_EXECUTOR` from environment (or accept an explicit flag that records it).
- Append one JSON object per optimizer step to `--metrics-jsonl`:
  ```json
  {"step":1,"warmup":true,"step_time_s":1.23,"samples":4,"max_memory_allocated_bytes_by_rank":{"0":123,"1":122},"loss":2.3,"skipped":false,"nan_or_inf":false,"use_pp_executor":0}
  ```
- Optional instrumented fields (only if measured comparably): `pipeline_bubble_ratio`, `encoder_latency_s` (per bucket), `microbatch_stall_s`.
- Record full env/commits/config in `output-dir/run_config.json`.

## Mock runner
`mock_step_runner.py` emits synthetic step records for validating the metrics pipeline and SLURM wiring **without GPUs/models**. It is not evidence of training/system performance.
