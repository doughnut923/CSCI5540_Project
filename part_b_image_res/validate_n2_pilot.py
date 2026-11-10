"""Processor validation gate for Track B N=2 pilot manifests (Phase 3.2).

Checks each 2-image sample with the Qwen2-VL processor (no model weights/forward):
- images load and match manifest hashes (optional recheck),
- per-image grid/derived tokens match the fixed rule (128->25,256->81,384->196),
- per-image slice isolation: processing each image alone matches its slice in the
  2-image batch (guards against dropped/silently merged images),
- derived sample total matches manifest.

Outputs a machine-readable report; exits nonzero on failure.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
TOKEN_COST = {128: 25, 256: 81, 384: 196}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokens_from_grid(grid_thw: Any) -> int:
    t, h, w = (int(x) for x in grid_thw)
    return (t * h * w) // 4


def load_image(path: Path, resolution: int) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").resize((resolution, resolution))


def validate_sample(processor: Any, sample: dict, check_hashes: bool, repo_root: Optional[Path]) -> Dict:
    errors = []
    loaded = []
    for img in sample["images"]:
        # Support both new relative and old absolute manifest formats.
        raw = img.get("path_relative_to_repo_root") or img.get("image_id") or img.get("absolute_path_recorded")
        path = Path(raw)
        if not path.is_absolute():
            path = (repo_root or Path.cwd()) / path
        if not path.is_file() and img.get("absolute_path_recorded"):
            path = Path(img["absolute_path_recorded"])
        if not path.is_file():
            errors.append(f"missing {raw}")
            continue
        if check_hashes and sha256_file(path) != img["sha256"]:
            errors.append(f"hash mismatch {path}")
        loaded.append(load_image(path, int(img["resolution"])))

    result = {"sample_id": sample["sample_id"], "status": "fail", "errors": errors}
    if errors or len(loaded) != 2:
        return result

    batch = processor(images=loaded, text=["test", "test"], padding=True, return_tensors="np")
    grids = batch["image_grid_thw"].tolist()
    pixels = batch["pixel_values"]
    patch_counts = np.prod(batch["image_grid_thw"], axis=1).astype(int)
    offsets = np.r_[0, np.cumsum(patch_counts)]
    per_image = []
    for j, img in enumerate(sample["images"]):
        res = int(img["resolution"])
        expected = TOKEN_COST[res]
        actual = tokens_from_grid(grids[j])
        single = processor(images=[loaded[j]], text="test", return_tensors="np")
        slice_ok = bool(np.array_equal(single["pixel_values"], pixels[offsets[j] : offsets[j + 1]]))
        if actual != expected:
            errors.append(f"slot {j} res {res}: got {actual} tokens, expected {expected}")
        if not slice_ok:
            errors.append(f"slot {j}: single-image tensor != batch slice")
        per_image.append({"slot": j, "resolution": res, "grid_thw": grids[j], "derived_tokens": actual, "slice_matches_single": slice_ok})

    total = sum(x["derived_tokens"] for x in per_image)
    if total != int(sample["derived_visual_tokens_total"]):
        errors.append(f"manifest total {sample['derived_visual_tokens_total']} != processor total {total}")
    result.update(
        {
            "per_image": per_image,
            "derived_visual_tokens_total": total,
            "pixel_values_shape": list(pixels.shape),
            "status": "pass" if not errors else "fail",
            "errors": errors,
        }
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default=MODEL, help="HF model id or local path for AutoProcessor")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--check-hashes", action="store_true")
    p.add_argument("--repo-root", type=Path, default=None, help="Root for resolving manifest-relative image paths")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    repo_root = args.repo_root.resolve() if args.repo_root else args.manifest.resolve().parent.parent.parent

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)

    results = [validate_sample(processor, s, args.check_hashes, repo_root) for s in samples]
    failed = [r for r in results if r["status"] != "pass"]
    report = {
        "gate": "track_b_n2_processor",
        "model": args.model,
        "manifest": str(args.manifest.resolve()),
        "samples_checked": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "status": "pass" if not failed else "fail",
        "measured": ["processor grids", "pixel_values slices vs single-image", "derived token totals"],
        "unmeasured": ["model forward", "embeddings", "training", "cluster runtime behavior"],
        "results": results,
    }
    (out / "n2_validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "samples_checked", "passed", "failed")}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
