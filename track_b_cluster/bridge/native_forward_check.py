"""Single-GPU parity check: MCore Qwen2-VL native forward vs the HF reference.

Runs one Track B manifest sample (N=2 images) through:
  1. the HF `Qwen2VLForConditionalGeneration` checkpoint, and
  2. the converted Megatron-Core checkpoint built by Pai-Megatron-Patch,
and compares the assistant-token loss.

Both sides use the same processor output and the same loss mask, so any difference is
weight-mapping / model-code, not data. Forward only: this is parity step 1 of the ladder
(native vs Part A), not training evidence.

Requires: NGC 25.03 SIF (torch 2.7 + transformer_engine), the BigMac 250624 MCore backend,
and PYTHONPATH set by the caller (see track_b_cluster/slurm/native_forward_check.sbatch).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--mcore-ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--pmp-root", required=True)
    return p.parse_args()


def build_mcore_argv(hf_dir: str, seq_length: int) -> list:
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
        "--transformer-impl", "transformer_engine",
        "--patch-tokenizer-type", "Qwen2Tokenizer",
        "--extra-vocab-size", "293",
        "--spatial-merge-size", "2",
        "--load", hf_dir,
        "--save", "/tmp/mcore_check_unused",
    ]


def load_mcore_weights(model, ckpt_dir: str) -> None:
    from megatron.training.checkpointing import (
        get_checkpoint_name,
        get_checkpoint_tracker_filename,
        read_metadata,
    )

    iteration, release = read_metadata(get_checkpoint_tracker_filename(ckpt_dir))
    name = get_checkpoint_name(ckpt_dir, iteration, release, None, None, None, None, None)
    state = torch.load(name, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Vision/language/merger weights must all land; anything else is a mapping bug.
    if missing or unexpected:
        raise AssertionError(f"state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def build_sample(manifest: dict, sample_id, repo_root: Path):
    from PIL import Image

    samples = manifest["samples"]
    sample = next((s for s in samples if s["sample_id"] == sample_id), samples[0])
    images = []
    for img in sample["images"]:
        path = Path(img["path_relative_to_repo_root"])
        if not path.is_absolute():
            path = repo_root / path
        with Image.open(path) as src:
            images.append(src.convert("RGB").resize((int(img["resolution"]),) * 2))
    return sample, images


def main() -> None:
    cli = parse_args()
    sys.path.insert(0, str(Path(cli.pmp_root) / "examples"))
    sys.path.insert(0, str(Path(cli.pmp_root) / "examples/qwen2_vl"))

    manifest = json.loads(Path(cli.manifest).read_text())
    sample, images = build_sample(manifest, cli.sample_id, Path(cli.repo_root))

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cli.hf_dir, local_files_only=True, use_fast=False)
    message = {
        "role": "user",
        "content": [{"type": "image"} for _ in images]
        + [{"type": "text", "text": "Describe these images."}],
    }
    prefix = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(
        [message, {"role": "assistant", "content": [{"type": "text", "text": "Two images."}]}],
        tokenize=False,
        add_generation_prompt=False,
    )
    batch = dict(processor(text=[full], images=images, padding=True, return_tensors="pt"))
    prefix_batch = processor(text=[prefix], images=images, padding=True, return_tensors="pt")
    prefix_len = prefix_batch["input_ids"].shape[1]
    if not torch.equal(batch["input_ids"][:, :prefix_len], prefix_batch["input_ids"]):
        raise AssertionError("assistant boundary mismatch")

    labels = batch["input_ids"].clone()
    labels[:, :prefix_len] = -100
    labels[batch["attention_mask"] == 0] = -100
    supervised = int((labels != -100).sum())
    if supervised < 2:
        raise AssertionError("no supervised assistant tokens")

    seq_length = int(batch["input_ids"].shape[1])
    sys.argv = ["native_forward_check"] + build_mcore_argv(cli.hf_dir, seq_length)

    from megatron.training.initialize import initialize_megatron
    from megatron_patch.arguments import get_patch_args

    initialize_megatron(extra_args_provider=get_patch_args)
    from megatron.training import get_args
    from qwen2_vl.pretrain_qwen import get_ltor_masks_and_position_ids, model_provider

    args = get_args()
    model = model_provider()
    load_mcore_weights(model, cli.mcore_ckpt)
    model.eval()

    device = torch.device("cuda")
    tokens = batch["input_ids"].to(device)
    target = labels.to(device)
    grid = batch["image_grid_thw"].to(device)
    pixels = batch["pixel_values"].to(device)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_mask = tokens == image_token_id
    empty_video_grid = torch.zeros((0, 3), dtype=torch.long, device=device)

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, grid, empty_video_grid, target, processor.tokenizer.pad_token_id, ignore_index=-100
    )

    with torch.no_grad():
        mcore_losses = model(
            input_ids=tokens,
            position_ids=position_ids,
            vision_data=pixels,
            vision_grid_thw=grid,
            video_start_index=int(image_mask.sum().item()),
            image_input_mask=image_mask,
            video_input_mask=None,
            attention_mask=attention_mask,
            labels=target,
        )
    mask = loss_mask.view(-1)
    mcore_loss = float((mcore_losses.view(-1) * mask).sum() / mask.sum())

    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        cli.hf_dir, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device)
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(
            input_ids=tokens,
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=pixels,
            image_grid_thw=grid,
            labels=target,
        )
    hf_loss = float(hf_out.loss)

    result = {
        "check": "native_mcore_vs_hf_forward",
        "sample_id": sample["sample_id"],
        "images_per_sample": len(images),
        "resolutions": [i["resolution"] for i in sample["images"]],
        "seq_length": seq_length,
        "image_grid_thw": grid.tolist(),
        "derived_visual_tokens": int((grid.prod(dim=1) // 4).sum().item()),
        "image_placeholders": int(image_mask.sum().item()),
        "supervised_tokens": supervised,
        "mcore_loss": mcore_loss,
        "hf_loss": hf_loss,
        "abs_diff": abs(mcore_loss - hf_loss),
        "rel_diff": abs(mcore_loss - hf_loss) / max(abs(hf_loss), 1e-12),
        "finite": math.isfinite(mcore_loss) and math.isfinite(hf_loss),
        "mcore_head": "BigMac Megatron-LM-250624 (core 0.14 fork)",
        "measured": ["single-GPU forward loss on one N=2 sample"],
        "unmeasured": ["backward/gradients", "training", "pipeline parallelism", "throughput"],
    }
    Path(cli.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("mcore_loss", "hf_loss", "rel_diff", "finite")}))


if __name__ == "__main__":
    main()
