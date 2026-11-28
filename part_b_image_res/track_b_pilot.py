"""Build deterministic Track B N=2 resolution-heterogeneity pilots.

Phase 0.3/3.1 of the Megatron+BigMac integration plan:
- Fixed image count = 2 images/sample, resolution held as the heterogeneity axis.
- Deterministic pairing from existing part_b variance datasets, with seed.
- Manifest records per-sample resolutions, image paths/hashes, and token totals.
- Token rule (Qwen2-VL square, measured in part_b): 128->25, 256->81, 384->196.

Output directory must not already exist (never overwrite evidence).
"""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

TOKEN_COST = {128: 25, 256: 81, 384: 196}
MIXTURES = ("Low", "Medium", "High")
DEFAULT_SEED = 55402026
DEFAULT_SAMPLES_PER_MIXTURE = 24
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"  # shortcut: parity with part_a/b/c pins; verify ModelScope checkout hash before cluster runs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def resolve_image(image_id: str, part_b_root: Path, repo_root: Path) -> Path:
    for base in (repo_root, part_b_root):
        p = base / image_id
        if p.is_file():
            return p
    raise FileNotFoundError(image_id)


def pair_records(records: List[Dict[str, Any]], n: int, seed: int) -> List[Tuple[Dict, Dict]]:
    # Deterministic: shuffle a copy, then take adjacent pairs; cycle if needed.
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    if len(pool) < 2 * n:
        # Reuse pool deterministically to reach requested sample count.
        extended = []
        i = 0
        while len(extended) < 2 * n:
            extended.append(pool[i % len(pool)])
            i += 1
        pool = extended
    return [(pool[2 * i], pool[2 * i + 1]) for i in range(n)]


def build_pilot(
    part_b_root: Path,
    repo_root: Path,
    dataset_dir: Path,
    mixtures: List[str],
    samples_per_mixture: int,
    seed: int,
    allow_missing_images: bool = False,
) -> Dict[str, Any]:
    samples = []
    for mix_i, mixture in enumerate(mixtures):
        records = load_dataset(dataset_dir / f"dataset_{mixture}.json")
        pairs = pair_records(records, samples_per_mixture, seed + mix_i)
        for j, (a, b) in enumerate(pairs):
            images = []
            for slot, rec in enumerate((a, b)):
                res = int(rec["resolution"])
                est = int(rec["token_cost_estimate"])
                if TOKEN_COST[res] != est:
                    raise ValueError(f"token mismatch for {res}: {est}")
                try:
                    path = resolve_image(str(rec["image_id"]), part_b_root, repo_root)
                    rel = path.resolve().relative_to(repo_root.resolve()).as_posix()
                    sha = sha256_file(path)
                    missing = False
                except (FileNotFoundError, OSError):
                    if not allow_missing_images:
                        raise
                    rel = str(rec["image_id"]).replace("\\", "/")
                    sha = "unmeasured_image_missing_on_build_host"
                    missing = True
                images.append(
                    {
                        "slot": slot,
                        "image_id": rec["image_id"],
                        "path_relative_to_repo_root": rel,
                        "sha256": sha,
                        "image_missing_on_build_host": missing,
                        "resolution": res,
                        "derived_visual_tokens": TOKEN_COST[res],
                    }
                )
            samples.append(
                {
                    "sample_id": f"{mixture.lower()}_{j:04d}",
                    "mixture": mixture,
                    "images_per_sample": 2,
                    "images": images,
                    "derived_visual_tokens_total": sum(i["derived_visual_tokens"] for i in images),
                    "resolutions": [i["resolution"] for i in images],
                }
            )
    any_missing = any(img["image_missing_on_build_host"] for s in samples for img in s["images"])
    measured = ["manifest fields", "per-sample derived totals from fixed rule"] + ([] if any_missing else ["image sha256"])
    unmeasured = ["actual processor grids on target runtime", "embeddings", "forward", "training", "throughput", "bubble ratio"] + (["image sha256 (missing on build host)"] if any_missing else [])
    return {
        "track": "B_resolution_heterogeneity",
        "spec": {
            "images_per_sample": 2,
            "heterogeneity_axis": "image_resolution",
            "image_count_held_fixed": True,
            "mixtures": mixtures,
            "samples_per_mixture": samples_per_mixture,
            "total_samples": len(samples),
            "seed": seed,
            "allow_missing_images": allow_missing_images,
            "token_rule": {
                "model": MODEL,
                "model_revision_pinned": MODEL_REVISION,
                "square_tokens": TOKEN_COST,
                "sample_total": "sum(derived_visual_tokens per image)",
                "note": "Derived from processor grid rule validated in part_b; re-validate with processor gate on target runtime.",
            },
        },
        "measured_derived_unmeasured": {
            "measured_by_this_script": measured,
            "derived": ["visual token totals from 128->25,256->81,384->196"],
            "unmeasured": unmeasured,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--part-b-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mixtures", nargs="+", default=list(MIXTURES), choices=MIXTURES)
    p.add_argument("--samples-per-mixture", type=int, default=DEFAULT_SAMPLES_PER_MIXTURE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--allow-missing-images", action="store_true", help="Record missing image hashes as unmeasured (for staging where image_sets not yet synced)")
    args = p.parse_args()
    if args.samples_per_mixture < 1:
        p.error("--samples-per-mixture must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset_dir = args.dataset_dir or (args.part_b_root / "variance_datasets")
    pilot = build_pilot(
        args.part_b_root.resolve(),
        args.repo_root.resolve(),
        dataset_dir.resolve(),
        args.mixtures,
        args.samples_per_mixture,
        args.seed,
        allow_missing_images=args.allow_missing_images,
    )
    (out / "pilot_manifest.json").write_text(json.dumps(pilot, indent=2) + "\n", encoding="utf-8")
    summary = {
        "total_samples": pilot["spec"]["total_samples"],
        "mixtures": pilot["spec"]["mixtures"],
        "samples_per_mixture": pilot["spec"]["samples_per_mixture"],
        "images_per_sample": 2,
        "derived_total_range": [
            min(s["derived_visual_tokens_total"] for s in pilot["samples"]),
            max(s["derived_visual_tokens_total"] for s in pilot["samples"]),
        ],
        "seed": args.seed,
    }
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
