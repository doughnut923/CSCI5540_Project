"""Validate dataset resolution metadata against the Qwen2-VL processor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoProcessor


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
RESOLUTIONS = (128, 256, 384)
DATASET_NAMES = ("Low", "Medium", "High")


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


def validate_dataset(
    path: Path,
    records: list[dict[str, Any]],
    repository_root: Path,
    processor: Any,
) -> int:
    """Validate every sample's image, resolution, and processor token count."""
    errors = 0
    for index, record in enumerate(records):
        resolution = record.get("resolution")
        image_id = record.get("image_id")
        tokens = record.get("tokens")

        if not isinstance(resolution, int) or resolution not in RESOLUTIONS:
            print(f"ERROR {path}:{index}: unsupported resolution {resolution!r}")
            errors += 1
            continue

        image_path = repository_root / image_id if isinstance(image_id, str) else None
        if not isinstance(image_id, str) or not image_id:
            print(f"ERROR {path}:{index}: invalid image_id")
            errors += 1
            continue
        if image_path is None or not image_path.is_file():
            print(f"ERROR {path}:{index}: image not found: {image_id}")
            errors += 1
            continue

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB").resize((resolution, resolution))
            grid, expected_tokens = processor_token_count(processor, image)

        if tokens != expected_tokens:
            print(
                f"ERROR {path}:{index}: tokens={tokens}, "
                f"expected={expected_tokens} for {resolution}x{resolution} "
                f"(grid t={grid[0]}, h={grid[1]}, w={grid[2]})"
            )
            errors += 1
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
    args = parser.parse_args()

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
    for name in DATASET_NAMES:
        path = args.dataset_dir / f"dataset_{name}.json"
        if not path.is_file():
            print(f"ERROR: dataset not found: {path}")
            errors += 1
            continue
        records = load_dataset(path)
        dataset_errors = validate_dataset(
            path, records, args.repository_root, processor
        )
        errors += dataset_errors
        print(
            f"{name}: {len(records)} samples, "
            f"{'valid' if dataset_errors == 0 else f'{dataset_errors} errors'}"
        )

    if errors:
        raise SystemExit(f"Validation failed with {errors} error(s).")
    print("Resolution validation passed.")


if __name__ == "__main__":
    main()
