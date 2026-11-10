# BigMac staging notes (cluster, under cyfok3)

Pinned checkouts (all under `$CYFOK3_WORKSPACE/third_party/`):

| Repo | Commit | Purpose |
| --- | --- | --- |
| `BigMac` | `2fa6da2fd3979448e89a56e734cbc3c490990d81` | `pp_executor` / `pp_scheduler` / Qwen3-VL examples |
| `BigMac/third_party/Pai-Megatron-Patch` | `3413f297e9666aa75c681bb588a7a61d92a33e5b` | **Qwen2-VL MCore model + HF→MCore converter** |
| `BigMac/third_party/big_mac_open_source` | `3a1cd24a465600e5721b66e7e5a46cbe702980ee` | submodule required by examples |
| `Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624` | `2220aa0d4a63dd6bb221d438152f05c2dd7e9a9e` (branch `pai_qwen3_vl_backend`, `Dots-Infra/BigMac-Megatron-Core`) | patched MCore with Qwen3-VL + BigMac hooks; **core 0.14.0rc0** |
| `Megatron-LM` | `266f1c97dca477ca8b92c16087506da2000b0b84` (`core_v0.17.1`) | stock MCore matching BigMac's validated version |

BigMac's tutorials were validated on NGC PyTorch 25.10 (MCore 0.17.0). That image also ships
`transformer_engine`, which Pai-Megatron-Patch's `qwen2_vl/layer_specs.py` needs unconditionally.
A plain `pip install transformer-engine[pytorch]` is not usable here: `transformer-engine-torch`
has no manylinux wheel (sdist only) and the container has no `nvcc`.

## Verified import compatibility (CPU, `pytorch-sam3.sif`)

```
PYTHONPATH=<250624 backend>:<Pai-Megatron-Patch>:<BigMac>:<venvs/trackb_pydeps>
import megatron_patch.model.qwen2_vl.model   # ok  (core 0.14.0rc0 backend)
from pp_executor.mcore_patch import install   # ok; schedules.get_forward_backward_func patched
```
Extra deps for that check (installed with `--no-deps` into `venvs/trackb_pydeps`):
`einops`, `matplotlib` (+ `pyparsing contourpy cycler fonttools kiwisolver python-dateutil six`).

## Qwen2-VL-2B config parity (already matches the converter's 2B branch)

`hidden_size=1536`, `num_layers=28`, `num_attention_heads=12`, `num_key_value_heads=2`,
`intermediate_size=8960`, `vocab_size=151936` (`EXTRA_VOCAB_SIZE=293`), `rms_norm_eps=1e-6`,
vision `depth=32 hidden=1536 heads=16 patch=14 merge=2 temporal_patch=2`.

Converter: `Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen/hf2mcore_qwen2_vl_convertor.sh`
(`2B <hf_dir> <out_dir> 1 1 false bf16`). Multi-image support is in
`examples/qwen2_vl/pretrain_qwen.py` (`image_grid_thw`, per-image `t,h,w` expansion) — that file is
the reference bridge for Track B.

## Open blocker

Need a runtime with `transformer_engine` + MCore 0.17.x: pull `nvcr.io/nvidia/pytorch:25.10-py3`
to a SIF on a GPU node (login node lacks `mksquashfs`). Pull job script:
`containers/pull_ngc.sbatch`, log `containers/pull-ngc-2510.log`.
`USE_PP_EXECUTOR=0` must still be parity-checked against Part A before being called stock Megatron-LM.
