"""Batch dataset samples under a maximum visual-token budget."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


Sample = dict[str, Any]


def create_batches(
    samples: list[Sample],
    batch_token_budget: int = 800,
    *,
    shuffle: bool = True,
    seed: int = 42,
) -> list[list[Sample]]:
    """Group samples without exceeding the token budget when possible."""
    if batch_token_budget <= 0:
        raise ValueError("batch_token_budget must be greater than zero.")

    indices = list(range(len(samples)))
    if shuffle:
        random.Random(seed).shuffle(indices)

    batches: list[list[Sample]] = []
    current_batch: list[Sample] = []
    current_tokens = 0

    for index in indices:
        sample = samples[index]
        tokens = sample.get("tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ValueError(
                f"Sample at index {index} must have a non-negative integer 'tokens'."
            )

        if current_batch and current_tokens + tokens > batch_token_budget:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(sample)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def save_batches(
    batches: list[list[Sample]],
    output_path: Path,
    batch_token_budget: int,
) -> None:
    """Save batches and their token totals as JSON."""
    output = {
        "batch_token_budget": batch_token_budget,
        "num_batches": len(batches),
        "batches": [
            {
                "batch_index": index,
                "total_tokens": sum(sample["tokens"] for sample in batch),
                "samples": batch,
            }
            for index, batch in enumerate(batches)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create token-budget batches from a dataset JSON file."
    )
    parser.add_argument("dataset_json", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("token_budget_batches.json"),
    )
    parser.add_argument("--batch-token-budget", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep samples in their input order.",
    )
    args = parser.parse_args()

    if not args.dataset_json.is_file():
        parser.error(f"Input dataset not found: {args.dataset_json}")

    data = json.loads(args.dataset_json.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        parser.error("The input JSON must contain a list of samples.")

    batches = create_batches(
        data,
        batch_token_budget=args.batch_token_budget,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    save_batches(batches, args.output, args.batch_token_budget)
    print(
        f"Created {len(batches)} batches from {len(data)} samples: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
