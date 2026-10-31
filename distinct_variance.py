"""Generate resolution-mixture datasets with a target average token cost."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

RESOLUTIONS = (128, 256, 384)
TOKEN_COST = {128: 25, 256: 81, 384: 196}
DISTRIBUTIONS = {
    "Low": None,
    "Medium": None,
    "High": None,
}

ImageRecord = dict[str, Any]
DatasetMetadata = dict[str, list[ImageRecord]]


def _validate_image_ids(image_ids: Sequence[str]) -> list[str]:
    """Return image IDs after rejecting empty IDs and an empty input pool."""
    normalized_ids = [str(image_id).strip() for image_id in image_ids]
    if not normalized_ids:
        raise ValueError("At least one base image ID is required.")
    if any(not image_id for image_id in normalized_ids):
        raise ValueError("Image IDs must not be empty.")
    return normalized_ids


def generate_datasets(
    base_image_ids: Sequence[str],
    *,
    seed: int = 42,
) -> DatasetMetadata:
    """Generate exact-count resolution datasets for each variance level."""
    image_ids = _validate_image_ids(base_image_ids)
    dataset_metadata: DatasetMetadata = {}

    for group_name in DISTRIBUTIONS:
        if group_name == "Low":
            resolutions = [256] * len(image_ids)
        elif group_name == "Medium":
            resolutions = (
                [128, 128, 256, 256, 256, 384]
                * (len(image_ids) // 6)
            )
        else:
            resolutions = [128, 128, 384] * (len(image_ids) // 3)

        resolutions.extend([256] * (len(image_ids) - len(resolutions)))
        random.Random(seed).shuffle(resolutions)
        group_dataset = [
            {
                "image_id": image_id,
                "resolution": resolution,
                "tokens": TOKEN_COST[resolution],
            }
            for image_id, resolution in zip(image_ids, resolutions)
        ]
        dataset_metadata[group_name] = group_dataset

        tokens = np.array([record["tokens"] for record in group_dataset])
        print(
            f"{group_name}: mean={tokens.mean():.1f}, "
            f"std={tokens.std():.1f}"
        )

    return dataset_metadata


def verify_datasets(
    dataset_metadata: DatasetMetadata,
) -> None:
    """Print token mean and population standard deviation for each group."""
    for group_name, dataset in dataset_metadata.items():
        tokens = [record["tokens"] for record in dataset]
        average = statistics.mean(tokens)
        std_dev = statistics.pstdev(tokens) if len(tokens) > 1 else 0.0
        print(f"--- {group_name} Verification ---")
        print(f"Average Tokens: {average}")
        print(f"Standard Deviation (Heterogeneity): {std_dev}")


def save_datasets(
    dataset_metadata: DatasetMetadata, output_dir: Path
) -> None:
    """Save each generated group as a formatted JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for group_name, dataset in dataset_metadata.items():
        output_path = output_dir / f"dataset_{group_name}.json"
        output_path.write_text(
            json.dumps(dataset, indent=2) + "\n",
            encoding="utf-8",
        )


def load_image_ids(path: Path) -> list[str]:
    """Load image IDs from a JSON list or a CSV with an optional image_id column."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("JSON image ID input must be a list.")
        return [str(image_id) for image_id in data]

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if not rows:
        return []
    if rows[0] and rows[0][0].strip().lower() == "image_id":
        return [row[0] for row in rows[1:] if row]
    return [row[0] for row in rows if row]


def discover_image_ids(image_dir: Path = Path("image_sets")) -> list[str]:
    """Discover image paths from generated image-set folders."""
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return [path.as_posix() for path in image_paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate distinct-variance resolution datasets."
    )
    parser.add_argument(
        "image_ids",
        type=Path,
        nargs="?",
        help=(
            "JSON list or CSV file containing one image ID per row. "
            "If omitted, image files are discovered under image_sets."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for dataset_<group>.json files (default: current directory)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.image_ids is None:
        image_ids = discover_image_ids()
        if not image_ids:
            parser.error(
                "No image IDs were found. Provide a JSON/CSV ID file or run "
                "data_prep.py first to create image_sets."
            )
    else:
        if not args.image_ids.is_file():
            parser.error(
                f"Input file not found: {args.image_ids}. Provide an existing "
                "JSON/CSV file or omit the argument to discover image_sets."
            )
        image_ids = load_image_ids(args.image_ids)

    datasets = generate_datasets(image_ids, seed=args.seed)
    verify_datasets(datasets)
    save_datasets(datasets, args.output_dir)


if __name__ == "__main__":
    main()
