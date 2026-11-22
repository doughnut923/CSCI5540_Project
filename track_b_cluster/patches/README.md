# Compat patches for Qwen2-VL checkpoint conversion

Applied on the cluster to the pinned checkouts (see `track_b_cluster/BIGMAC_STAGING.md`).
Regenerate with `git -C <repo> diff` after applying.

| File | Target | Why |
| --- | --- | --- |
| `pai-megatron-patch-mcore017.patch` | `third_party/BigMac/third_party/Pai-Megatron-Patch` | Pai-Megatron-Patch targets ~MCore 0.11-0.14; these make its Qwen2-VL path import/build on stock MCore 0.17.1 |
| `bigmac-mcore-250624-te2.patch` | `.../backends/megatron/Megatron-LM-250624` (BigMac's MCore fork) | MCore 0.14 hands TE 2.x packed 3-D qkv; TE 2.x wants unpacked 4-D |
| `megatron-lm-nvrx.patch` | `third_party/Megatron-LM` (core_v0.17.1) | NGC ships `nvidia_resiliency_ext` without `__version__`, which crashes MCore import |

Apply with `git apply -p1 < patch>` from each repo root.

## What each hunk does

1. `transformer_config.py` — `quick_gelu` moved from `megatron.training.activations` to `megatron.core.activations`.
2. `nvrx.py` — tolerate `nvidia_resiliency_ext` without `__version__`.
3. `pretrain_qwen.py`, `multimodal_dataset_helper.py` — make the `megatron.energon` import optional; energon needs `multistorageclient`, which is not distributable here (conversion and model-only paths never build energon datasets).
4. `multimodal_dataset.py` — `torchaudio` optional; Track B is image-only.
5. `arguments.py` — ignore duplicate argparse flags that MCore 0.17 already registers.
6. `tokenizer/__init__.py` — MCore 0.17 moved `MegatronTokenizer` and forbids direct instantiation; the Qwen2 wrapper only needs a no-op base.
7. `hf2mcore_qwen2_vl_convertor.sh` — drop `--no-async-tensor-model-parallel-allreduce` (removed in MCore 0.17).
8. `attention.py`, `attention_vision.py` — accept and forward newer process-group kwargs (`pg_collection` in 0.17, `model_comm_pgs` in the BigMac 0.14 fork).
9. `gpt_model.py` — default `mrope_section=[16,24,24]` (Qwen2-VL-2B/7B) when the converter does not pass it.

Note on the backend: conversion runs on BigMac's `Megatron-LM-250624` fork (core 0.14). Stock 0.17.1 cannot host the patched `qwen2_vl` model even with the patches above, because MCore 0.17 passes `pg_collection` into attention modules that Pai-Megatron-Patch defines itself.

## Extra runtime pieces (not patches)

- `venvs/trackb_pydeps/megatron/energon/` is a small local stub (EPath, Sample, placeholder factories) installed only so the import chain resolves.
- Install extras with `--no-deps`; keep the container's numpy 1.26 (container `cv2` is built against it) and use `pyarrow==17.0.0` (pyarrow >= 18 requires numpy 2).

## Hardware requirement (learned the hard way)

Qwen2-VL vision attention runs as **THD/varlen**, which only FlashAttention/FusedAttention support, and
those require **sm80+**. On an sm75 node TE disables all three backends and raises
`No dot product attention support for the provided inputs!`. Request a modern GPU explicitly:

```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx3090      # rtx3090 nodes are sm86; also need highcpucount-aware -c
```

## Verification status

`track_b_cluster/bridge/native_forward_check.py` (job 26605) on sample `low_0000`, 2x256px:

| metric | value |
| --- | --- |
| MCore loss | 3.9477005005 |
| HF loss | 3.9478073120 |
| relative difference | **2.7e-05** |
| per-token loss correlation | 0.9993 |

Both sides used the same processor output and the same assistant mask. The remaining ~3e-5 is
bf16 noise. This validates the conversion and the model path; it is **not** a training result.

Note: Megatron computes cross-entropy against the labels it is handed (no internal shift), so the
MCore side must receive next-token targets. Passing unshifted labels inflates the loss ~4x.
