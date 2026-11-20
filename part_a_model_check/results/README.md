# Recorded Mac checks

Recorded on 2026-11-05 on Apple M5 Max with 48 GiB unified memory, macOS 27.0.1, PyTorch 2.6.0 and Transformers 4.49.0. Both runs load the pinned Qwen2-VL-2B-Instruct checkpoint. All 2,208,985,600 parameters are trainable.

| Run | Input case | Actual visual tokens | Completed AdamW steps | Loss values | Result |
| --- | --- | ---: | ---: | --- | --- |
| 128-token limit | Single image | 117 | 2 | 2.977974, 2.197868 | Passed |
| 128-token limit | Two images | 117 | 2 | 7.336908, 3.164746 | Passed |
| 576-token limit | Single image | 234 | 1 | 3.025573 | Passed |
| 576-token limit | Two images | 486 | 1 | 3.957309 | Passed |

Both runs passed all gradient checks for the vision encoder, vision merger and language model. The visual output received a finite nonzero gradient, and sampled trainable weights changed in all three groups after each update. Actual visual output shapes matched the processor grids and image-token placeholders.

Each run starts from the original checkpoint. Within a run, the two-image case follows the single-image case on the updated model. The different loss values belong to different input cases and model states.

## Runtime configuration

- Device and parameter precision: MPS, bfloat16.
- Optimizer: AdamW with BF16 moment tensors. Hyperparameters are recorded in each JSON.
- Eager attention and non-reentrant gradient checkpointing in both vision and language blocks.
- PyTorch automatic MPS-to-CPU fallback: disabled.
- MPS scatter compatibility handling: enabled, with one application per forward pass.

The 128-token run sampled a maximum of 17.056 GiB of MPS tensor allocation and 23.852 GiB of MPS driver allocation. The 576-token run sampled 17.441 GiB and 25.105 GiB respectively. Sampling occurred every 0.1 seconds. These are overlapping views of memory use, and the driver measure includes allocator/cache overhead.

## Evidence files

- [128-token run JSON](macos-bf16-128/result.json) and [log](macos-bf16-128/run.log).
- [576-token run JSON](macos-bf16-576/result.json) and [log](macos-bf16-576/run.log).
- [Scatter regression tests](regression-tests.txt) passed on CPU and MPS, including MPS float32 and bfloat16 gradients.
- [Dependency check](dependency-check.txt) reports no broken requirements.

Each JSON includes the executed source hashes, model revision, dependency versions, input hashes, loss, module gradient norms, sampled parameter updates, timings and memory observations.

Linux/NVIDIA and distributed BigMac/Megatron execution are the next validation stages. The root README gives the matching single-GPU CUDA commands and explains how to use these inputs and gradient checks during integration.
