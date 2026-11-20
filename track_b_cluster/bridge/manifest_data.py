"""Shared deterministic Qwen2-VL manifest input, extracted from the team runner."""
import json
import hashlib
import os
from copy import copy, deepcopy
from pathlib import Path

import torch


def load_processor(hf_dir):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(hf_dir, local_files_only=True, use_fast=False)


def build_sample(processor, sample, repo_root):
    """Processor tensors for one manifest sample. CPU, so it can be cached."""
    from PIL import Image

    images = []
    for img in sample["images"]:
        path = Path(img["path_relative_to_repo_root"])
        if not path.is_absolute():
            path = Path(repo_root) / path
        if img.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != img["sha256"]:
            raise ValueError("source image hash mismatch: " + str(path))
        with Image.open(path) as src:
            preparation = img.get("preparation", {})
            if preparation.get("type") == "part_c_padded_control":
                # Reuse the team's exact fit/pad intervention, not a new crop.
                import sys
                repo = str(Path(__file__).resolve().parents[2])
                if repo not in sys.path:
                    sys.path.insert(0, repo)
                from part_c_image_count.real_photo_pilot.run_real_photo_pilot import prepare
                image, _ = prepare(src.convert("RGB"), preparation["image_count"], "padded_control")
                images.append(image)
            else:
                res = int(img["resolution"])
                images.append(src.convert("RGB").resize((res, res)))

    if sample.get("processor_bounds"):
        # A private image processor prevents one sample's pixel cap from leaking
        # into later samples or the baseline. Share the read-only tokenizer.
        processor = copy(processor)
        processor.image_processor = deepcopy(processor.image_processor)
        for name, value in sample["processor_bounds"].items():
            if name not in ("min_pixels", "max_pixels") or not isinstance(value, int) or value <= 0:
                raise ValueError("invalid per-sample processor bound")
            setattr(processor.image_processor, name, value)

    message = {"role": "user", "content": [{"type": "image"} for _ in images]
               + [{"type": "text", "text": "Describe these images."}]}
    prefix = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(
        [message, {"role": "assistant", "content": [{"type": "text", "text": sample.get("target_text", "Two images.")}]}],
        tokenize=False, add_generation_prompt=False)
    batch = dict(processor(text=[full], images=images, padding=True, return_tensors="pt"))
    prefix_len = processor(text=[prefix], images=images, padding=True,
                           return_tensors="pt")["input_ids"].shape[1]

    labels = batch["input_ids"].clone()
    labels[:, :prefix_len] = -100
    labels[batch["attention_mask"] == 0] = -100

    tokens = batch["input_ids"]
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    visual_tokens = (batch["image_grid_thw"].prod(dim=1) // processor.image_processor.merge_size**2).tolist()
    return {
        "sample_id": sample["sample_id"],
        "tokens": tokens,
        "labels": labels,
        "grid": batch["image_grid_thw"],
        "pixels": batch["pixel_values"].to(torch.bfloat16),
        "image_mask": tokens == image_token_id,
        "input_observation": {"image_count": len(images), "visual_tokens_per_image": visual_tokens,
                              "visual_tokens_total": sum(visual_tokens),
                              "image_grid_thw": batch["image_grid_thw"].tolist(),
                              "sequence_length": tokens.shape[1],
                              "supervised_tokens": int((labels != -100).sum())},
    }


class ManifestBatches:
    """Deterministic (step, microbatch) -> sample plan over a manifest.

    Legacy bucket mode groups by derived visual-token total and requires equal
    actual sequence lengths (suitable for fixed image counts). Manifest mode keeps
    mixed-length samples in their original order for a runtime with per-microbatch
    tensor shapes. Both modes are deterministic functions of (step, microbatch).

    Samples are built once and cached; builds happen before a step is timed.
    """

    def __init__(self, hf_dir, manifest_path, repo_root, batching=None):
        self.hf_dir = hf_dir
        self.repo_root = repo_root
        self.samples = json.loads(Path(manifest_path).read_text())["samples"]
        if not self.samples:
            raise ValueError("manifest has no samples")
        ids = [s["sample_id"] for s in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest sample_id values must be unique")
        self.processor = None
        self.cache = {}
        self.batching = batching or os.environ.get("MANIFEST_BATCHING", "bucket")
        if self.batching not in ("bucket", "manifest"):
            raise ValueError("batching must be bucket or manifest")
        if self.batching == "manifest":
            self.groups = [self.samples]
            return
        groups = {}
        for s in self.samples:
            total = s.get("derived_visual_tokens_total")
            if total is None and all("derived_visual_tokens" in i for i in s["images"]):
                total = sum(i["derived_visual_tokens"] for i in s["images"])
            if total is None:
                # Smoke fixtures do not carry derived token metadata. Obtain
                # their group from the actual processor; never invent a count.
                if self.processor is None:
                    self.processor = load_processor(self.hf_dir)
                b = build_sample(self.processor, s, self.repo_root)
                self.cache[s["sample_id"]] = b
                total = int(b["image_mask"].sum())
            groups.setdefault((total, len(s["images"])), []).append(s)
        # Largest group first, so steps draw distinct samples before any group repeats.
        # Equal visual tokens do not imply equal length: image markers add tokens, so
        # samples with a different image count must never share a bucket.
        self.groups = [g for _, g in sorted(
            groups.items(), key=lambda kv: (-len(kv[1]), -kv[0][0], -kv[0][1]))]

    def plan(self, step, num_mb):
        group = self.groups[step % len(self.groups)]
        base = (step // len(self.groups)) * num_mb
        return [group[(base + mb) % len(group)] for mb in range(num_mb)]

    def build(self, step, num_mb):
        if self.processor is None:
            self.processor = load_processor(self.hf_dir)
        batches = []
        for sample in self.plan(step, num_mb):
            if sample["sample_id"] not in self.cache:
                self.cache[sample["sample_id"]] = build_sample(
                    self.processor, sample, self.repo_root)
            batches.append(self.cache[sample["sample_id"]])
        lengths = {b["tokens"].shape[1] for b in batches}
        assert self.batching == "manifest" or len(lengths) == 1, (
            "step %d microbatches have mixed sequence lengths %s: the derived-token "
            "grouping does not match this manifest" % (step, sorted(lengths)))
        return batches

    def step_tensors(self, step, num_mb, device, masks_fn, empty_video):
        """One step's microbatches on device, with next-token targets for MCore.

        Built before the step's start time is published, so host preprocessing and the
        host-to-device copies stay out of the measured step time.
        """
        prepared = []
        for b in self.build(step, num_mb):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            target = b["labels"].clone()
            target[:, :-1] = b["labels"][:, 1:]
            target[:, -1] = -100
            _, b["loss_mask"], b["position_ids"] = masks_fn(
                b["tokens"], b["grid"], empty_video, target, 0, ignore_index=-100)
            b["target"] = target
            prepared.append(b)
        return prepared
