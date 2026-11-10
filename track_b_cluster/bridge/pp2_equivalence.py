"""PP=2 equivalence check: native MCore pipeline schedule vs BigMac pp_executor.

Both runs build the same model (identical seed, CPU init), consume the same sample, and
execute the same two microbatches. Only the forward-backward function differs, so loss and
gradient norm must match. This is the G2 gate before any timing is collected.

Encoder nesting (LLMPPExecutorWithVE) is NOT exercised here; this is the text-side schedule.
"""

import json
import math
import os
import sys
from pathlib import Path

import torch


def mcore_argv(seq_length):
    return [
        "--num-layers", "28",
        "--hidden-size", "1536",
        "--ffn-hidden-size", "8960",
        "--num-attention-heads", "12",
        "--num-query-groups", "2",
        "--group-query-attention",
        "--max-position-embeddings", "32768",
        "--seq-length", str(seq_length),
        "--micro-batch-size", "1",
        "--global-batch-size", "2",
        "--pipeline-model-parallel-size", "2",
        "--tensor-model-parallel-size", "1",
        "--use-mcore-models",
        "--swiglu",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000",
        "--use-rotary-position-embeddings",
        "--no-rope-fusion",
        "--no-bias-swiglu-fusion",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--bf16",
        "--use-cpu-initialization",
        "--seed", "1234",
        "--train-iters", "1",
        "--eval-interval", "1000000",
        "--eval-iters", "0",
        "--save-interval", "1000000",
        "--lr", "1e-4",
        "--min-lr", "1e-5",
        "--lr-decay-iters", "1",
        "--lr-decay-style", "cosine",
        "--weight-decay", "0.01",
        "--clip-grad", "1.0",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        "--transformer-impl", "transformer_engine",
        "--patch-tokenizer-type", "Qwen2Tokenizer",
        "--extra-vocab-size", "293",
        "--spatial-merge-size", "2",
        "--load", os.environ["QWEN2VL_HF_DIR"],
        "--save", "/tmp/pp2_unused",
    ]


def build_batch(hf_dir, manifest_path, repo_root, device):
    import json as _json
    from PIL import Image
    from transformers import AutoProcessor

    manifest = _json.loads(Path(manifest_path).read_text())
    samples = manifest["samples"]
    sample = next((s for s in samples if s["images"][0]["resolution"] == 128), samples[0])
    images = []
    for img in sample["images"]:
        path = Path(img["path_relative_to_repo_root"])
        if not path.is_absolute():
            path = Path(repo_root) / path
        with Image.open(path) as src:
            res = int(img["resolution"])
            images.append(src.convert("RGB").resize((res, res)))

    processor = AutoProcessor.from_pretrained(hf_dir, local_files_only=True, use_fast=False)
    message = {"role": "user", "content": [{"type": "image"} for _ in images]
               + [{"type": "text", "text": "Describe these images."}]}
    prefix = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(
        [message, {"role": "assistant", "content": [{"type": "text", "text": "Two images."}]}],
        tokenize=False, add_generation_prompt=False)
    batch = dict(processor(text=[full], images=images, padding=True, return_tensors="pt"))
    prefix_len = processor(text=[prefix], images=images, padding=True,
                           return_tensors="pt")["input_ids"].shape[1]

    labels = batch["input_ids"].clone()
    labels[:, :prefix_len] = -100
    labels[batch["attention_mask"] == 0] = -100

    tokens = batch["input_ids"].to(device)
    grid = batch["image_grid_thw"].to(device)
    pixels = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_mask = tokens == image_token_id
    return tokens, labels.to(device), grid, pixels, image_mask, sample["sample_id"]


def main():
    out_path = Path(sys.argv[1])
    mode = os.environ.get("USE_PP_EXECUTOR", "0")
    repo_root = Path(os.environ["PROJ_ROOT"])
    hf_dir = os.environ["QWEN2VL_HF_DIR"]

    # Must happen before megatron initializes, so the patch is in place on import.
    if mode in ("1", "true", "yes", "on"):
        from pp_executor.mcore_patch import install
        install()

    sys.argv = ["pp2_equiv"] + mcore_argv(512)
    from megatron.training.initialize import initialize_megatron
    from megatron_patch.arguments import get_patch_args
    initialize_megatron(extra_args_provider=get_patch_args)

    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.training import get_args
    from qwen2_vl.pretrain_qwen import get_ltor_masks_and_position_ids, model_provider

    args = get_args()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    model = model_provider(pre_process=pp_rank == 0, post_process=pp_rank == pp_size - 1)
    model = model.to(device)
    model.train()
    for param in model.parameters():
        if param.requires_grad and getattr(param, "main_grad", None) is None:
            param.main_grad = torch.zeros_like(param, dtype=torch.float32)

    tokens, labels, grid, pixels, image_mask, sample_id = build_batch(
        hf_dir, os.environ["MANIFEST"], repo_root, device)

    target_mc = labels.clone()
    target_mc[:, :-1] = labels[:, 1:]
    target_mc[:, -1] = -100
    empty_video = torch.zeros((0, 3), dtype=torch.long, device=device)
    _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, grid, empty_video, target_mc, 0, ignore_index=-100)

    def forward_step(data_iterator, model):
        losses = model(
            input_ids=tokens,
            position_ids=position_ids,
            vision_data=pixels,
            vision_grid_thw=grid,
            video_start_index=int(image_mask.sum().item()),
            image_input_mask=image_mask,
            video_input_mask=None,
            attention_mask=None,
            labels=target_mc,
        )

        def loss_func(output_tensor):
            mask = loss_mask.view(-1).float()
            loss = (output_tensor.float().view(-1) * mask).sum() / mask.sum()
            return loss, {"loss": loss.detach()}

        return losses, loss_func

    data_iterator = iter([{}])  # unused by forward_step; data is already on device
    fwd_bwd = get_forward_backward_func()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = fwd_bwd(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=[model],
            num_microbatches=2,
            seq_length=tokens.shape[1],
            micro_batch_size=1,
            forward_only=False,
        )

    losses = [float(x[0]) for x in out] if out else []
    local_loss = sum(losses) / len(losses) if losses else None

    total = torch.zeros(1, device=device)
    for param in model.parameters():
        grad = getattr(param, "main_grad", None)
        if grad is not None:
            total += grad.detach().float().pow(2).sum()
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
    grad_norm = math.sqrt(float(total))

    result = {
        "mode": "pp_executor" if mode in ("1", "true", "yes", "on") else "native",
        "use_pp_executor_env": mode,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "sample_id": sample_id,
        "num_microbatches": 2,
        "seq_length": int(tokens.shape[1]),
        "loss": local_loss,
        "grad_norm": grad_norm,
        "finite": bool(local_loss is not None and math.isfinite(local_loss)
                       and math.isfinite(grad_norm)),
        "measured": ["PP=2 forward+backward loss and grad norm"],
        "unmeasured": ["encoder nesting (VE executor)", "timing", "training convergence"],
    }
    if pp_rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rank": pp_rank, "mode": result["mode"],
                      "loss": local_loss, "grad_norm": grad_norm}), flush=True)


if __name__ == "__main__":
    main()
