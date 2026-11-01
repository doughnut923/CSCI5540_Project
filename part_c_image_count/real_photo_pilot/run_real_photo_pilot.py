"""CPU-only real-photo input check; no model weights, forward, or training."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import sys
import traceback

import numpy as np
from PIL import Image, ImageOps
from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor, Qwen2TokenizerFast


HERE = Path(__file__).resolve().parent
SEED = 55402026
COUNTS = (1, 2, 3, 4, 8)
SHAPES_HW = {1: (672, 672), 2: (448, 504), 3: (336, 448), 4: (336, 336), 8: (224, 252)}
TARGET = 576
MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(image, n, condition):
    if condition == "native_aspect":
        return image.copy(), {"method": "unchanged real preview", "content_box_xyxy": [0, 0, *image.size], "spatial_crop": False}
    h, w = SHAPES_HW[n]
    canvas_wh = (2 * w, 2 * h)
    # Keep the complete source field of view. Padding is part of this controlled
    # input intervention and consumes model-input patch positions like content.
    fitted = ImageOps.contain(image, canvas_wh, method=Image.Resampling.LANCZOS)
    x = (canvas_wh[0] - fitted.width) // 2
    y = (canvas_wh[1] - fitted.height) // 2
    canvas = Image.new("RGB", canvas_wh, (128, 128, 128))
    canvas.paste(fitted, (x, y))
    return canvas, {"method": "aspect-preserving fit plus gray padding", "content_box_xyxy": [x, y, x + fitted.width, y + fitted.height], "spatial_crop": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=HERE / "runs" / "pilot_v1")
    ap.add_argument("--samples-per-count", type=int, default=16)
    ap.add_argument("--assets-dir", type=Path, default=HERE.parent / "assets",
                    help="Pinned tokenizer/config assets prepared by ../download_reference.py")
    ap.add_argument("--photo-manifest", type=Path, default=HERE / "photo_manifest.json",
                    help="Photo paths are relative to this manifest")
    args = ap.parse_args()
    if args.samples_per_count < 1:
        raise ValueError("samples-per-count must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log")])
    try:
        run(out, args.samples_per_count, args.photo_manifest.resolve(), args.assets_dir.resolve())
    except BaseException:
        failure = traceback.format_exc()
        (out / "failure.log").write_text(failure)
        logging.exception("PILOT FAILED")
        raise


def run(out, count, source_manifest, assets):
    source_entries = json.loads(source_manifest.read_text())
    assert len(source_entries) == 8, "Expected the eight diverse, already-local previews"
    pool = []
    provenance = []
    for item in source_entries:
        path = source_manifest.parent / item["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["sha256"], f"Source hash mismatch: {path.name}"
        image = Image.open(path).convert("RGB")
        pool.append(image)
        prov = dict(item)
        prov.update(usage="real-photo processor pilot; repeatedly sampled small preview pool", original_wh=list(image.size))
        provenance.append(prov)
    write_json(out / "photo_provenance.json", provenance)
    config = {
        "seed": SEED, "samples_per_count_per_condition": count, "counts": COUNTS,
        "conditions": ["native_aspect", "padded_control"], "target_visual_tokens_per_sample": TARGET,
        "source_pool_size": len(pool), "source_type": "already-local Dataset Viewer JPEG previews, not full-resolution original dataset",
        "source_pool_reuse": "sources repeat across samples; distinct ordered samples do not imply independent unique photos",
        "selection": "random.Random(seed + N*100000 + sample_index).sample(range(8), N); same assignments in both conditions",
        "padded_control": {"processed_target_hw": SHAPES_HW, "prepared_canvas": "2x target H and W", "fit": "aspect ratio preserved, complete field of view retained, centered gray RGB128 padding, no crop", "caution": "padding is a controlled intervention; this does not establish model quality or a native solution for arbitrary image shapes"},
        "processor": {"model_id": "Qwen/Qwen2-VL-2B-Instruct", "revision": MODEL_REVISION, "transformers": importlib.metadata.version("transformers"), "min_pixels": 3136, "max_pixels": "(576 // N) * 784", "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2},
        "scope": "official processor plus tokenizer, NumPy, CPU; no weights or model forward or training", "performance_measurement": False,
    }
    write_json(out / "config.json", config)
    write_json(out / "environment.json", {"python": sys.version, "platform": platform.platform(), "device": "CPU", "packages": {k: importlib.metadata.version(k) for k in ["numpy", "Pillow", "transformers", "tokenizers"]}, "encoder_forward": False, "llm_forward": False, "training_step": False, "model_weights_loaded": False})
    tokenizer = Qwen2TokenizerFast.from_pretrained(assets, local_files_only=True)
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    samples = []
    manifest = []
    rows_path = out / "samples.jsonl"
    representative = out / "representative_N4_00"
    representative.mkdir()
    for condition in config["conditions"]:
        for n in COUNTS:
            ip = Qwen2VLImageProcessor.from_pretrained(assets, local_files_only=True, min_pixels=3136, max_pixels=(TARGET // n) * 784)
            proc = Qwen2VLProcessor(image_processor=ip, tokenizer=tokenizer, chat_template=tokenizer.chat_template)
            for idx in range(count):
                sid = f"{condition}_N{n}_{idx:02d}"
                selection = random.Random(SEED + n * 100000 + idx).sample(range(len(pool)), n)
                pairs = [prepare(pool[k], n, condition) for k in selection]
                images = [p[0] for p in pairs]
                prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": "Describe these images."}]}], tokenize=False, add_generation_prompt=True)
                batch = proc(text=[prompt], images=images, return_tensors="np", padding=False, truncation=False)
                grids = batch["image_grid_thw"]
                pixels = batch["pixel_values"]
                patch_counts = np.prod(grids, axis=1).astype(int)
                offsets = np.r_[0, np.cumsum(patch_counts)]
                merge_divisor = ip.merge_size ** 2
                assert all(patch_counts % merge_divisor == 0)
                derived = (patch_counts // merge_divisor).astype(int)
                slice_matches, perturb_matches = [], []
                for j, image in enumerate(images):
                    single = ip(images=[image], return_tensors="np")
                    slice_matches.append(bool(np.array_equal(single["pixel_values"], pixels[offsets[j]:offsets[j + 1]])))
                    changed = list(images)
                    changed[j] = Image.fromarray(255 - np.asarray(image))
                    alternative = ip(images=changed, return_tensors="np")["pixel_values"]
                    changes = [not np.array_equal(alternative[offsets[k]:offsets[k + 1]], pixels[offsets[k]:offsets[k + 1]]) for k in range(n)]
                    perturb_matches.append(changes == [k == j for k in range(n)])
                ids = batch["input_ids"][0]
                visual_slots = int(np.sum(ids == image_pad))
                specials = int(np.sum(ids == vision_start) + np.sum(ids == vision_end))
                padding = int(np.sum(batch["attention_mask"] == 0))
                retained = bool(len(grids) == n and all(slice_matches) and all(perturb_matches))
                consistent = retained and visual_slots == int(sum(derived)) and specials == 2 * n and padding == 0
                assert consistent, sid
                prepared_hw = [[im.height, im.width] for im in images]
                original_hw = [[pool[k].height, pool[k].width] for k in selection]
                processed_hw = [[int(g[1]) * 14, int(g[2]) * 14] for g in grids]
                sample = {
                    "sample_id": sid, "condition": condition, "sample_index": idx, "image_count": n,
                    "source": "real Dataset Viewer JPEG previews", "source_rows": [provenance[k]["row_idx"] for k in selection],
                    "source_sha256": [provenance[k]["sha256"] for k in selection], "original_hw": original_hw,
                    "prepared_hw": prepared_hw, "processed_hw": processed_hw,
                    "padding_intervention": condition == "padded_control", "spatial_crops": 0,
                    "image_grid_thw": grids.tolist(), "image_blocks": len(grids), "pixel_values_shape": list(pixels.shape),
                    "encoder_input_patch_rows_per_image": patch_counts.tolist(), "encoder_input_patch_rows_total": int(sum(patch_counts)),
                    "llm_visual_tokens_derived_per_image": derived.tolist(), "llm_visual_tokens_derived_total": int(sum(derived)),
                    "derivation": "sum(actual image_grid_thw products) / merge_size^2; spatial merge divisor=4",
                    "target_visual_tokens_per_sample": TARGET, "budget_delta": int(sum(derived)) - TARGET,
                    "exact_budget_met": bool(int(sum(derived)) == TARGET),
                    "expanded_image_token_slots_observed": visual_slots, "vision_special_markers": specials,
                    "padding_tokens": padding, "text_and_other_special_tokens": int(len(ids) - visual_slots - specials), "sequence_length": int(len(ids)),
                    "image_tensor_slices_match": all(slice_matches), "each_image_perturbation_pass": all(perturb_matches),
                    "all_images_retained_in_processor": retained, "token_slot_consistency": visual_slots == int(sum(derived)), "status": "PASS",
                    "original_pixels_total": sum(h * w for h, w in original_hw), "prepared_canvas_pixels_total": sum(h * w for h, w in prepared_hw),
                    "processed_pixels_total": sum(h * w for h, w in processed_hw),
                    "encoder_forward_tokens": "NOT MEASURED", "llm_embeddings_measured": "NOT MEASURED", "actual_llm_consumption": "NOT MEASURED",
                    "encoder_runtime": "NOT MEASURED", "training_step_runtime": "NOT MEASURED", "model_quality": "NOT MEASURED",
                }
                samples.append(sample)
                manifest.append({"sample_id": sid, "selection_seed": SEED + n * 100000 + idx, "images": [{"position": j, "pool_index": k, "source_row": provenance[k]["row_idx"], "source_path": provenance[k]["file"], "sha256": provenance[k]["sha256"], "original_hw": original_hw[j], "prepared_hw": prepared_hw[j], "preparation": pairs[j][1]} for j, k in enumerate(selection)]})
                if condition == "padded_control" and n == 4 and idx == 0:
                    for j, image in enumerate(images):
                        image.save(representative / f"image_{j:02d}.png")
                with rows_path.open("a") as f:
                    f.write(json.dumps(sample) + "\n")
                logging.info(json.dumps({k: sample[k] for k in ["sample_id", "image_count", "llm_visual_tokens_derived_total", "exact_budget_met", "all_images_retained_in_processor", "status"]}))
    write_json(out / "samples.json", samples)
    write_json(out / "manifest.json", manifest)
    with (out / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in samples)
    summaries = []
    for condition in config["conditions"]:
        for n in COUNTS:
            rows = [r for r in samples if r["condition"] == condition and r["image_count"] == n]
            totals = [r["llm_visual_tokens_derived_total"] for r in rows]
            summaries.append({"condition": condition, "image_count": n, "sample_count": len(rows), "source_image_instances": len(rows) * n, "all_images_retained_count": sum(r["all_images_retained_in_processor"] for r in rows), "min_derived_total": min(totals), "max_derived_total": max(totals), "unique_derived_totals": sorted(set(totals)), "exact_576_count": sum(r["exact_budget_met"] for r in rows), "processed_hw_values": sorted({str(hw) for r in rows for hw in r["processed_hw"]}), "patch_row_totals": sorted({r["encoder_input_patch_rows_total"] for r in rows})})
    summary = {"samples_total": len(samples), "source_image_instances_total": sum(r["image_count"] for r in samples), "source_pool_unique_images": len(pool), "all_samples_retention_pass": all(r["all_images_retained_in_processor"] for r in samples), "groups": summaries, "scope": config["scope"], "inference_or_training_performed": False}
    write_json(out / "summary.json", summary)
    logging.info("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
