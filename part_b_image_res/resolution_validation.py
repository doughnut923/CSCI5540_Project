"""Validate dataset resolution metadata against the Qwen2-VL processor."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
RESOLUTIONS = (128, 256, 384)
DATASET_NAMES = ("Low", "Medium", "High")
DEFAULT_BATCH_TOLERANCE = 0.35


def processor_token_count(
    processor: Any, image: Image.Image
) -> tuple[tuple[int, int, int], int]:
    """Return Qwen2-VL's image grid and merged visual-token count."""
    inputs = processor(images=image, text="test", return_tensors="pt")
    t, height, width = inputs["image_grid_thw"][0].tolist()
    return (t, height, width), (t * height * width) // 4


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate a dataset JSON list."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def resolve_image_path(image_id: str, repository_root: Path) -> Path | None:
    """Resolve image IDs written relative to the Part B or repository root."""
    candidates = (
        repository_root / image_id,
        Path(__file__).resolve().parent / image_id,
    )
    return next((path for path in candidates if path.is_file()), None)


def load_resized_image(
    record: dict[str, Any], repository_root: Path
) -> Image.Image | None:
    """Load one record's source image and resize it to its target resolution."""
    resolution = record.get("resolution")
    image_id = record.get("image_id")
    if not isinstance(resolution, int) or resolution not in RESOLUTIONS:
        return None
    if not isinstance(image_id, str) or not image_id:
        return None
    image_path = resolve_image_path(image_id, repository_root)
    if image_path is None:
        return None
    with Image.open(image_path) as source_image:
        return source_image.convert("RGB").resize((resolution, resolution))


def processor_batch_token_count(
    processor: Any, images: list[Image.Image]
) -> tuple[list[int], int]:
    """Return per-image visual tokens and text padding for one processor batch."""
    inputs = processor(
        images=images,
        text=["test"] * len(images),
        padding=True,
        return_tensors="pt",
    )
    visual_tokens = [
        (int(t) * int(height) * int(width)) // 4
        for t, height, width in inputs["image_grid_thw"].tolist()
    ]
    padding_tokens = int((inputs["attention_mask"] == 0).sum().item())
    return visual_tokens, padding_tokens


def validate_dataset(
    path: Path,
    records: list[dict[str, Any]],
    repository_root: Path,
    processor: Any,
    token_cache: dict[tuple[str, int], tuple[tuple[int, int, int], int]],
) -> int:
    """Validate every sample's image, resolution, and processor token count."""
    errors = 0
    actual_tokens: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        resolution = record.get("resolution")
        image_id = record.get("image_id")
        tokens = record.get("token_cost_estimate")

        if not isinstance(resolution, int) or resolution not in RESOLUTIONS:
            print(f"ERROR {path}:{index}: unsupported resolution {resolution!r}")
            errors += 1
            continue

        if not isinstance(image_id, str) or not image_id:
            print(f"ERROR {path}:{index}: invalid image_id")
            errors += 1
            continue
        image = load_resized_image(record, repository_root)
        if image is None:
            print(f"ERROR {path}:{index}: image not found: {image_id}")
            errors += 1
            continue

        cache_key = (image_id, resolution)
        cached_result = token_cache.get(cache_key)
        if cached_result is None:
            grid, expected_tokens = processor_token_count(processor, image)
            token_cache[cache_key] = (grid, expected_tokens)
        else:
            grid, expected_tokens = cached_result
        actual_tokens.append((resolution, expected_tokens))

        if tokens != expected_tokens:
            print(
                f"ERROR {path}:{index}: token_cost_estimate={tokens}, "
                f"expected={expected_tokens} for {resolution}x{resolution} "
                f"(grid t={grid[0]}, h={grid[1]}, w={grid[2]})"
            )
            errors += 1
    # Per-resolution stats: all images resized to R x R must share one token
    # count; min != max means the processor broke the resolution guarantee.
    for resolution in RESOLUTIONS:
        values = [tokens for res, tokens in actual_tokens if res == resolution]
        if values:
            print(
                f"{path.name}: {resolution}x{resolution} tokens: "
                f"n={len(values)}, mean={statistics.fmean(values):.1f}, "
                f"variance={statistics.pvariance(values):.2f}, "
                f"resolution_preserved={'yes' if min(values) == max(values) else 'NO'}"
            )
    if actual_tokens:
        overall = [tokens for _, tokens in actual_tokens]
        print(
            f"{path.name}: overall tokens: mean={statistics.fmean(overall):.1f}, "
            f"variance={statistics.pvariance(overall):.2f}"
        )
    return errors


def sample_processor_tokens(
    record: dict[str, Any],
    repository_root: Path,
    processor: Any,
    token_cache: dict[tuple[str, int], tuple[tuple[int, int, int], int]],
) -> int | None:
    """Return processor-derived tokens for one dataset record."""
    resolution = record.get("resolution")
    image_id = record.get("image_id")
    if not isinstance(resolution, int) or resolution not in RESOLUTIONS:
        return None
    if not isinstance(image_id, str) or not image_id:
        return None

    image = load_resized_image(record, repository_root)
    if image is None:
        return None
    cache_key = (image_id, resolution)
    cached_result = token_cache.get(cache_key)
    if cached_result is not None:
        return cached_result[1]
    _, tokens = processor_token_count(processor, image)
    token_cache[cache_key] = ((0, 0, 0), tokens)
    return tokens


def validate_batch_file(
    path: Path,
    repository_root: Path,
    processor: Any,
    tolerance: float,
    token_cache: dict[tuple[str, int], tuple[tuple[int, int, int], int]],
) -> int:
    """Validate processor-derived totals and similarity for one batch file."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("batches"), list):
        print(f"ERROR {path}: expected an object containing a batches list")
        return 1

    budget = data.get("batch_token_budget")
    if not isinstance(budget, int) or budget <= 0:
        print(f"ERROR {path}: invalid batch_token_budget: {budget!r}")
        return 1

    errors = 0
    totals: list[int] = []
    batches = data["batches"]
    for batch_index, batch in enumerate(batches):
        samples = batch.get("samples") if isinstance(batch, dict) else None
        if not isinstance(samples, list):
            print(f"ERROR {path}: batch {batch_index} has no samples list")
            errors += 1
            continue

        actual_total = 0
        batch_images: list[Image.Image] = []
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                print(
                    f"ERROR {path}: batch {batch_index}, sample "
                    f"{sample_index} is not an object"
                )
                errors += 1
                continue
            actual_tokens = sample_processor_tokens(
                sample, repository_root, processor, token_cache
            )
            if actual_tokens is None:
                print(
                    f"ERROR {path}: batch {batch_index}, sample "
                    f"{sample_index} could not be processed"
                )
                errors += 1
                continue
            image = load_resized_image(sample, repository_root)
            if image is None:
                print(
                    f"ERROR {path}: batch {batch_index}, sample "
                    f"{sample_index} image could not be loaded"
                )
                errors += 1
                continue
            batch_images.append(image)
            stored_tokens = sample.get("token_cost_estimate")
            if stored_tokens != actual_tokens:
                print(
                    f"ERROR {path}: batch {batch_index}, sample "
                    f"{sample_index} stores {stored_tokens} tokens; "
                    f"processor returned {actual_tokens}"
                )
                errors += 1
            actual_total += actual_tokens

        if batch_images:
            batch_processor_tokens, padding_tokens = processor_batch_token_count(
                processor, batch_images
            )
            batch_processor_total = sum(batch_processor_tokens)
            if batch_processor_total != actual_total:
                print(
                    f"ERROR {path}: batch {batch_index} processor total is "
                    f"{batch_processor_total}; per-sample total is "
                    f"{actual_total}"
                )
                errors += 1
            print(
                f"{path.name}: batch {batch_index} visual_tokens="
                f"{batch_processor_total}, padding_tokens={padding_tokens}"
            )

        declared_total = batch.get("total_tokens")
        if declared_total != actual_total:
            print(
                f"ERROR {path}: batch {batch_index} declares "
                f"{declared_total} tokens; processor total is {actual_total}"
            )
            errors += 1
        if actual_total > budget:
            print(
                f"ERROR {path}: batch {batch_index} has {actual_total} "
                f"tokens, over budget {budget}"
            )
            errors += 1
        totals.append(actual_total)

    if totals:
        complete_totals = totals[:-1] if len(totals) > 1 else totals
        lower_bound = budget * (1 - tolerance)
        outliers = [
            (index, total)
            for index, total in enumerate(complete_totals)
            if total < lower_bound
        ]
        if outliers:
            print(
                f"ERROR {path}: {len(outliers)} non-final batches are below "
                f"{lower_bound:.1f} tokens ({tolerance:.0%} tolerance): "
                f"{outliers[:5]}"
            )
            errors += len(outliers)
        print(
            f"{path.name}: {len(totals)} batches, "
            f"actual range={min(totals)}-{max(totals)}, "
            f"mean={sum(totals) / len(totals):.1f}, "
            f"batch_total_variance={statistics.pvariance(totals):.2f}"
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated datasets against Qwen2-VL token counts."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("variance_datasets"),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--batch-json",
        type=Path,
        nargs="+",
        help="Batch JSON files to validate against the Qwen2-VL processor.",
    )
    parser.add_argument(
        "--batch-tolerance",
        type=float,
        default=DEFAULT_BATCH_TOLERANCE,
        help=(
            "Allowed underfill fraction for non-final batches "
            f"(default: {DEFAULT_BATCH_TOLERANCE * 100:.0f}%%)."
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.batch_tolerance < 1:
        parser.error("--batch-tolerance must be in the range [0, 1).")

    from transformers import AutoProcessor

    args.repository_root = args.repository_root.resolve()
    args.dataset_dir = (
        args.dataset_dir
        if args.dataset_dir.is_absolute()
        else args.repository_root / args.dataset_dir
    )
    args.batch_json = [
        path if path.is_absolute() else args.repository_root / path
        for path in (args.batch_json or [])
    ]
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    for resolution in RESOLUTIONS:
        image = Image.fromarray(
            np.zeros((resolution, resolution, 3), dtype=np.uint8)
        )
        grid, tokens = processor_token_count(processor, image)
        print(
            f"Input {resolution}x{resolution} -> "
            f"Grid t={grid[0]}, h={grid[1]}, w={grid[2]} -> Tokens: {tokens}"
        )

    errors = 0
    token_cache: dict[tuple[str, int], tuple[tuple[int, int, int], int]] = {}
    for name in DATASET_NAMES:
        path = args.dataset_dir / f"dataset_{name}.json"
        if not path.is_file():
            print(f"ERROR: dataset not found: {path}")
            errors += 1
            continue
        records = load_dataset(path)
        dataset_errors = validate_dataset(
            path, records, args.repository_root, processor, token_cache
        )
        errors += dataset_errors
        print(
            f"{name}: {len(records)} samples, "
            f"{'valid' if dataset_errors == 0 else f'{dataset_errors} errors'}"
        )

    if args.batch_json:
        for path in args.batch_json:
            if not path.is_file():
                print(f"ERROR: batch file not found: {path}")
                errors += 1
                continue
            errors += validate_batch_file(
                path,
                args.repository_root,
                processor,
                args.batch_tolerance,
                token_cache,
            )

    if errors:
        raise SystemExit(f"Validation failed with {errors} error(s).")
    print("Resolution validation passed.")


if __name__ == "__main__":
    main()
