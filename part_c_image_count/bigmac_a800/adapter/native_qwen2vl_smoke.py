"""Train the actual pretrained Qwen2-VL vision projector for six small smoke steps."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen2VLImageProcessor

RUN = Path(__file__).resolve().parents[1]
SHAPE = {1: 672, 4: 336}
PROMPT = "Describe the visual input briefly."
ANSWER = "The sample contains natural images."


def checked_canvas(path, digest, count):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    side = 2 * SHAPE[count]
    with Image.open(path) as image:
        fitted = ImageOps.contain(image.convert("RGB"), (side, side), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(fitted, ((side - fitted.width) // 2, (side - fitted.height) // 2))
    return canvas


def assemble(sample, image_processor, tokenizer):
    count = sample["image_count"]
    images = [checked_canvas(RUN / "data" / img["file"], img["sha256"], count) for img in sample["images"]]
    batch = image_processor(images=images, return_tensors="pt")
    grids = batch["image_grid_thw"]
    assert grids.tolist() == sample["grid_thw"], sample["sample_id"]
    patch_counts = grids.prod(dim=1).tolist()
    for image, batch_slice in zip(images, batch["pixel_values"].split(patch_counts), strict=True):
        single = image_processor(images=[image], return_tensors="pt")
        assert torch.equal(single["pixel_values"], batch_slice), sample["sample_id"]

    messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": PROMPT}]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = prefix.split("<|image_pad|>")
    assert len(parts) == count + 1
    token_counts = sample["tokens_per_image"]
    prefix = parts[0] + "".join("<|image_pad|>" * n + suffix for n, suffix in zip(token_counts, parts[1:]))
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(ANSWER, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = torch.tensor([prefix_ids + answer_ids])
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_mask = ids == image_id
    assert image_mask.sum().item() == sum(token_counts) == 576
    labels = ids.clone()
    labels[:, : len(prefix_ids)] = -100
    return {
        **batch, "input_ids": ids, "attention_mask": torch.ones_like(ids),
        "labels": labels, "mm_token_type_ids": image_mask.long(), "use_cache": False,
    }, image_mask


def train_one(model, parameter, optimizer, batch, image_mask, sample):
    evidence = {}

    def observe_vision(_module, _args, out):
        evidence["vision"] = out.pooler_output
        evidence["vision"].retain_grad()

    def observe_llm(_module, _args, kwargs):
        evidence["llm_vision"] = kwargs["inputs_embeds"][image_mask].detach()

    vision_hook = model.model.visual.register_forward_hook(observe_vision)
    llm_hook = model.model.language_model.register_forward_pre_hook(observe_llm, with_kwargs=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    before = parameter.detach().clone()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        vision = evidence["vision"]
        llm_vision = evidence["llm_vision"]
        assert tuple(vision.shape) == tuple(llm_vision.shape) == (576, model.config.text_config.hidden_size)
        assert torch.equal(vision.detach().to(llm_vision.dtype), llm_vision)
        assert torch.isfinite(loss).item()
        loss.backward()
        norms = [segment.float().norm().item() for segment in vision.grad.split(sample["tokens_per_image"])]
        assert torch.isfinite(vision.grad).all().item() and all(norm > 0 for norm in norms)
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        grad_norm = parameter.grad.norm().item()
        assert grad_norm > 0
        optimizer.step()
        delta = parameter.detach() - before
        delta_l2 = delta.norm().item()
        assert torch.isfinite(delta).all().item() and delta_l2 > 0
        return {
            "sample_id": sample["sample_id"], "step": sample["step"], "image_count": sample["image_count"],
            "source_rows": [image["source_row"] for image in sample["images"]],
            "vision_grid_thw": sample["grid_thw"], "encoder_output_tokens_per_image": sample["tokens_per_image"],
            "llm_visual_embeddings": list(llm_vision.shape), "llm_image_slots_match_encoder": True,
            "per_image_embedding_grad_norm": norms, "loss": loss.detach().item(),
            "projector_grad_l2": grad_norm, "projector_update_l2": delta_l2,
            "projector_update_max_abs": delta.abs().max().item(),
            "observed_step_seconds_with_auditing": time.monotonic() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "status": "PASS",
        }
    finally:
        vision_hook.remove()
        llm_hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.manual_seed(55402026)
    manifest = json.loads((RUN / "data/inputs.json").read_text())
    assert args.model_dir.name == manifest["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    print(json.dumps({"stage": "loading", "framework": "Transformers native", "bigmac_executor": False,
                      "torch": torch.__version__, "transformers": transformers.__version__,
                      "device": torch.cuda.get_device_name(), "revision": manifest["model_revision"]}), flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.train()
    parameter = model.model.visual.merger.mlp[-1].weight
    parameter.data = parameter.detach().float()
    parameter.requires_grad_(True)
    original = parameter.detach().clone()
    with args.output.open("x", buffering=1) as results:
        for count in SHAPE:
            with torch.no_grad():
                parameter.copy_(original)
            optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.0)
            processor = Qwen2VLImageProcessor.from_pretrained(
                args.model_dir, local_files_only=True,
                min_pixels=3136, max_pixels=576 // count * 784,
            )
            for sample in (item for item in manifest["samples"] if item["image_count"] == count):
                batch, mask = assemble(sample, processor, tokenizer)
                batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
                record = train_one(model, parameter, optimizer, batch, mask.cuda(), sample)
                results.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            optimizer.zero_grad(set_to_none=True)
    print("All six audited native optimizer steps passed.", flush=True)


if __name__ == "__main__":
    main()
