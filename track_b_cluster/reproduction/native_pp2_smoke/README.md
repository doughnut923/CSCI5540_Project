# Native PP=2 real-photo smoke test

This records a successful forward/backward gate on 2026-11-18 with two RTX 3090
GPUs. It uses the existing image fixtures and the native MCore schedule on the
BigMac MCore fork. It does not use the original Track B pilot, load pretrained
weights, update an optimizer, or compare BigMac performance.

## Recorded result

| Setting or check | Value |
| --- | --- |
| PP / TP / DP | 2 / 1 / 1 |
| Layers / microbatches / seed | 28 / 2 / 1234 |
| Images per input / resized dimensions | 2 / 128 x 128 each |
| Actual sequence length | 82 |
| Loss | 6.259978771209717 |
| Global gradient norm | 852.1813920756542 |
| Rank reports | Both reported the same loss and global gradient norm |
| Exit / finite check | 0 / true |
| Optimizer updates | 0 |

The same loss/global gradient norm on both ranks is expected from the harness's
collectives; these are not independent experimental replicates. The values above
document this run, not a universal exact-value assertion for different hardware.

- `manifest.json`: source-image paths and SHA-256 hashes; the existing harness
  performs the resize. Images already live under `part_a_model_check/fixtures/`.
- `result.json`: the saved rank-0 result, without editing its numbers.
- `environment.json`: source pins, model-asset/container hashes, software and
  hardware versions, and both rank reports. Machine-specific paths and raw logs
  have been omitted from this public record.

## Changes required by this run

1. Select `Qwen2VLTokenizer`. The text-only `Qwen2Tokenizer` lacks the image,
   video and vision-start token IDs consumed by multimodal position encoding.
2. Apply `pai-megatron-patch-inference-context.patch` after the existing PMP
   compatibility patch. MCore passes `inference_context`; the pinned model's
   attention methods use the older `inference_params` name.
3. Include `chat_template.json` from the pinned model revision.
4. Install the extra dependencies into a separate target directory. Bypass the
   NGC global pip constraints for that installation: its PyArrow 19 constraint
   otherwise conflicts with the documented PyArrow 17 pin.
5. Create the documented import-only Energon shim with
   `scripts/prepare_energon_stub.py`. This does not implement an Energon loader.
6. Bind the host driver libraries and supply a private `libcuda.so` linker path
   to Triton. Use a writable container home when the login home is not mounted.

The tokenizer correction and attention alias were included in the measured run.
The updated `pp_scale_native.sbatch` was also submitted with this manifest and
completed with exit code 0 and the same loss/gradient values. Its source hash and
settings are in `launcher-validation.json`. The `pp2_equivalence.sbatch` wrapper
passed syntax and ShellCheck validation; it was not separately submitted.

## Rerun

First complete [SETUP.md](../../SETUP.md) using the pinned sources and patches.
`WS` must be a directory owned by you and visible on the allocated compute node;
`PROJ` must be `$WS/proj/csci5540`. Follow SETUP's filesystem-binding guidance for
`BIND_ROOT`. A login-home path that is absent on the GPU node will not work.

For a model-only gate, download these seven assets from model revision
`895c3a49bc3fa70a340399125c650a463535e71c` into
`$WS/models/Qwen2-VL-2B-Instruct`: `config.json`, `preprocessor_config.json`,
`tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`, and
`chat_template.json`. Pretrained weight shards are not needed by this harness.

From the login host, with `WS`, `PROJ` and `BIND_ROOT` already set:

```bash
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
export MANIFEST="$PROJ/track_b_cluster/reproduction/native_pp2_smoke/manifest.json"
export RUN_TAG="smoke-$(date -u +%Y%m%dT%H%M%SZ)"

sbatch --partition=csci5540 --qos=csci5540 --gres=gpu:2 \
  --export=ALL,PP=2,NUM_LAYERS=28,NUM_MICROBATCHES=2,BENCH_STEPS=0,WARMUP_STEPS=0 \
  "$PROJ/track_b_cluster/slurm/pp_scale_native.sbatch"
```

Logs are written to `$WS/containers/pp-native-pp2-$RUN_TAG.log`. The result is
`$PROJ/runs/track_b_n2_pilot_measured_v1/pp_native_scale_pp2-$RUN_TAG.json`.
Use a new tag for each attempt. Check successful job exit, `mode == "native"`,
`pp_size == 2`, `finite == true`, and finite positive gradient norm.

This is a systems-integration gate. Numerical parity with a pretrained reference,
optimizer-update correctness, heterogeneous training workloads and comparative
performance remain separate checks.
