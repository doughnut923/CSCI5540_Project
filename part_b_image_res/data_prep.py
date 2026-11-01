"""Group ImageNet validation images into pools and create cropped image sets."""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


POOL_SIZES = (128, 256, 384, 512, 768, 1024)
IMAGE_SET_SIZES = (128, 256, 384, 512, 768)
IMAGES_PER_SET = 250
OUTPUT_DIR = Path("image_sets")
DATASET_NAME = "mrm8488/ImageNet1K-val"


def _short_side(image: Any) -> int:
    """Return the shortest image dimension."""
    width, height = image.size
    return min(width, height)


def _pool_for_short_side(short_side: int) -> int:
    """Return the first pool that can contain ``short_side``.

    Images larger than the largest configured pool are kept in the 1024 pool.
    """
    for pool_size in POOL_SIZES:
        if short_side <= pool_size:
            return pool_size
    return POOL_SIZES[-1]


def _images(dataset: Dataset | DatasetDict) -> Iterable[Any]:
    """Yield images from either a single split or the first dataset split."""
    if isinstance(dataset, DatasetDict):
        if not dataset:
            return
        dataset = next(iter(dataset.values()))

    image_column = next(
        (column for column in dataset.column_names if column.lower() in {"image", "img"}),
        None,
    )
    if image_column is None:
        raise ValueError(
            f"Could not find an image column. Available columns: {dataset.column_names}"
        )

    for row in dataset:
        image = row[image_column]
        if image is None:
            raise ValueError("The dataset contains a missing image.")
        yield image


def count_image_pools(dataset: Dataset | DatasetDict) -> Counter[int]:
    """Count dataset images by the configured short-side pools."""
    counts = Counter({pool_size: 0 for pool_size in POOL_SIZES})
    for image in _images(dataset):
        counts[_pool_for_short_side(_short_side(image))] += 1
    return counts


def _center_crop(image: Any, size: int) -> Any:
    """Crop the center square from an image."""
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def create_image_sets(
    dataset: Dataset | DatasetDict,
    output_dir: Path = OUTPUT_DIR,
    images_per_set: int = IMAGES_PER_SET,
) -> None:
    """Create fixed-size center-cropped PNG sets from the dataset.

    Each set prioritizes its matching short-side pool, then uses larger pools
    when that pool has fewer than ``images_per_set`` images.
    """
    selected_images = {size: [] for size in IMAGE_SET_SIZES}
    for image in _images(dataset):
        short_side = _short_side(image)
        for size in IMAGE_SET_SIZES:
            if len(selected_images[size]) < images_per_set and short_side >= size:
                selected_images[size].append(image)

    for size in IMAGE_SET_SIZES:
        candidates = selected_images[size]
        if len(candidates) < images_per_set:
            raise ValueError(
                f"Only {len(candidates)} images can be cropped to {size}x{size}; "
                f"{images_per_set} are required."
            )

        size_dir = output_dir / str(size)
        size_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(candidates[:images_per_set]):
            cropped = _center_crop(image, size).convert("RGB")
            cropped.save(size_dir / f"{index:04d}.png")


def main() -> None:
    dataset = load_dataset(DATASET_NAME)
    counts = count_image_pools(dataset)

    print("Pool counts (classified by image short side):")
    for pool_size in POOL_SIZES:
        print(f"{pool_size:>4}: {counts[pool_size]}")
    print(f"Total: {sum(counts.values())}")

    create_image_sets(dataset)
    print(
        f"Created {len(IMAGE_SET_SIZES)} sets of {IMAGES_PER_SET} images "
        f"under {OUTPUT_DIR}/."
    )


if __name__ == "__main__":
    main()