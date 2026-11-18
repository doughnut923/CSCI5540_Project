# Qwen2-VL model forward/backward check

This package checks the complete `Qwen/Qwen2-VL-2B-Instruct` model with real image and text inputs. The same Python entry point runs on CPU, Apple MPS and NVIDIA CUDA. It is intended as a model-correctness reference before the group integrates its distributed training systems.

## Shared model and inputs

- Model and processor revision: `895c3a49bc3fa70a340399125c650a463535e71c`.
- Transformers: `4.49.0`, matching Part C. NumPy, Pillow and Jinja versions also match Part C's requirements.
- `requirements-lock.txt` records the complete package versions resolved in the isolated Mac environment. Install the platform's PyTorch wheel first, then use this lock file if exact Python dependency matching is required.
- Group source commit: `047be401a7dfc37c8051660f9a883ea1de61d720`.
- Real-photo fixtures and their SHA-256 hashes come from `part_c_image_count/real_photo_pilot/`. Their original dataset source, rows and dimensions are preserved in `fixtures/photo_manifest.json`.
- The shared processor/tokenizer files are checked against the six hashes in Part C's `assets_manifest.json`, copied here as `fixtures/reference_assets.json`.

The default cases use one image and two images. The two cases run in a fixed order on one model instance. Updates from the first case are retained for the second case. Use `--cases single_image` when an isolated checkpoint-starting reference is needed.

The diagnostic default is at most 128 visual tokens per input. It applies Part C's pixel-budget rule `max_pixels = (budget // image_count) * 28**2` to the original photo aspect ratios. The actual token total is recorded. `--visual-token-budget 576` exercises the group's larger budget. Part C's padded-canvas intervention remains in Part C's pilot.

## Checks performed

1. Verify fixture hashes and build a real image/text conversation using the official processor.
2. Mask the prompt, image slots and padding out of the supervised text loss. Require at least two assistant target tokens.
3. Compare image count, processor grids, image placeholders and actual visual-embedding shape.
4. Run a complete model forward pass and require finite loss.
5. Run backward and require finite, nonzero gradients in the vision encoder, vision merger and language model. Record the gradient that reaches the visual output.
6. Run AdamW and verify sampled parameters change in each of those three groups.
7. Repeat for two steps per case by default. Each step must pass independently. Loss does not have to decrease on every step.

All parameters are trainable. The checker uses eager attention, disables the KV cache and enables non-reentrant gradient checkpointing in both vision and language blocks. The diagnostic AdamW settings are explicit in `config.json`: learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, and `foreach=False`. These settings should be reviewed together when the group selects its final training recipe.

Parameters and gradients use the selected dtype. Standard PyTorch AdamW creates its moment tensors in the parameter dtype; their actual dtypes are saved in each step record. The BF16 check uses BF16 moments without FP32 master parameters. A future mixed-precision optimizer setup should be validated as its own CUDA configuration.

## Mac installation and run

Use Python 3.12 and a dedicated environment:

```sh
cd part_a_model_check
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python prepare.py
python run_check.py --device mps --dtype bfloat16 --output-dir runs/macos-check
```

Preparation downloads roughly 4.4 GB of model weights plus processor files into `.cache/huggingface/`, and verifies the small shared photos. Existing valid files are reused. Set `HF_HOME` before preparation to select a different cache location.

MPS fallback is disabled by default. To deliberately permit PyTorch's CPU fallback for an unsupported operation, add `--mps-fallback`; this is recorded in the result. Use `--device cpu --dtype float32` for a CPU reference where memory permits. `--mode backward` checks forward and gradients without creating AdamW state through an update.

The local PyTorch 2.6 MPS `masked_scatter` path was observed to flatten a two-dimensional source tensor's metadata, causing a backward shape error. `mps_compat.py` passes an explicit flattened view to that operation on MPS. This preserves the original tensor shape and the autograd view relationship. The workaround is scoped to the model forward call, is inactive on CPU/CUDA, and its use is counted in the results. The small regression check is `python -m unittest discover -s tests -v`.

## Linux / NVIDIA installation and run

The following setup targets the course's RTX 3090 class of GPU and the PyTorch 2.6 CUDA 12.4 wheel. The host must have a compatible NVIDIA driver.

On the course cluster, obtain an allocated GPU from a CSE/VPN-connected session first, following the [course GPU instructions](https://github.com/henryhxu/CSCI5540#gpu-resources):

```sh
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
srun --account gpu --qos csci5540 -p csci5540 --gres=gpu:1 --pty /bin/bash
```

```sh
cd part_a_model_check
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python prepare.py
CUDA_VISIBLE_DEVICES=0 python run_check.py --device cuda --dtype bfloat16 --output-dir runs/cuda-check
```

For the group's 576-token input budget:

```sh
CUDA_VISIBLE_DEVICES=0 python run_check.py --device cuda --dtype bfloat16 --visual-token-budget 576 --output-dir runs/cuda-budget576
```

For a machine with a predownloaded shared snapshot:

```sh
python prepare.py --model-dir /path/to/pinned/qwen2-vl-2b
python run_check.py --model-dir /path/to/pinned/qwen2-vl-2b --device cuda --dtype bfloat16 --output-dir runs/cuda-offline
```

`prepare.py --photos-only` installs/verifies just the small photo fixtures. Once preparation finishes, model-check runs use local files only.

## Results and interpretation

Every run creates `run.log` and `result.json`. The output directory must be new, so a rerun cannot replace an earlier result. `result.json` begins with status `running`, then records `passed`, `failed` or `interrupted` on completion.

The JSON records the checkpoint, code hashes, package versions, device, precision, input grids and shapes, loss, module gradients, parameter-update deltas, timings and memory observations. MPS memory is sampled every 0.1 seconds and at step boundaries. CUDA also records its allocator high-water value. Timing and memory fields are diagnostic observations for that host and configuration.

The [recorded Mac checks](results/README.md) passed six full-model AdamW steps across single-image and two-image inputs, using 128- and 576-token limits. Linux/CUDA commands are provided for teammates to execute in their environment. CUDA and distributed-system acceptance require their own result files.

Before a cross-device comparison, use the same model revision, case order, token budget, trainable modules, optimizer settings and step count. Compare loss/gradient behavior with precision-appropriate tolerance. BF16 and FP32 runs can differ numerically. Hardware speed comparisons require a separate benchmark protocol.

## Handoff to BigMac / Megatron

This entry point is a single-device full-model reference. Teammates can use the same inputs and supervised-loss construction when they split vision and language components. The next integration checks should compare loss and gradients, verify per-stage tensor shapes and token counts, and confirm that gradients return through inter-stage communication.

The upstream BigMac example currently targets Qwen3-VL. Its bridge needs an explicit Qwen2-VL adaptation. Multi-GPU launching, pipeline partitioning, NCCL communication and schedule correctness belong to that integration stage.

## Sources

- [Group Part C real-photo pilot](https://github.com/doughnut923/CSCI5540_Project/tree/047be401a7dfc37c8051660f9a883ea1de61d720/part_c_image_count/real_photo_pilot)
- [Pinned Qwen2-VL-2B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/895c3a49bc3fa70a340399125c650a463535e71c)
- [Transformers 4.49 Qwen2-VL model implementation](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)
