"""Build six pilot manifests from the team's B recipes and C real-photo pool.

Generated data stays outside Git. This is a small-pool engineering pilot, not
an independent-sample dataset or a claim of matched encoder execution time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "track_b_cluster/bridge"))
from manifest_data import build_sample, load_processor


def build_manifests(output, samples_per_case=24):
    if samples_per_case < 12 or samples_per_case % 12:
        raise ValueError("samples per case must be a positive multiple of 12")
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "part_c_image_count/real_photo_pilot/photo_manifest.json"
    pool = json.loads(source.read_text())
    for image in pool:
        path = source.parent / image["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
            raise ValueError("photo pool hash mismatch")
    # Exact image-level ratios from Part B, arranged to expose heterogeneity
    # in mixed microbatches rather than hiding it through length bucketing.
    resolutions = {
        "low": [(256, 256)],
        "medium": [(128, 384), (128, 256), (256, 256)],
        "high": [(128, 384), (128, 384), (128, 128)],
    }
    image_counts = {"low": [1], "medium": [1, 2, 3], "high": [1, 2, 4, 8]}
    suite = {"scope": "real-photo small-pool engineering pilot",
             "source_pool_size": len(pool), "sample_pool_repeats": True,
             "batching": "manifest", "cases": [],
             "limitations": ["not the unavailable original training pilot",
                             "B mean visual tokens are close, encoder runtime is not matched yet",
                             "C uses the team's controlled padding intervention"]}
    for axis in ("resolution", "image_count"):
        for level in ("low", "medium", "high"):
            case = axis + "_" + level
            counts_seen = {}
            samples = []
            for idx in range(samples_per_case):
                if axis == "resolution":
                    sizes = resolutions[level][idx % len(resolutions[level])]
                    n = 2
                else:
                    n = image_counts[level][idx % len(image_counts[level])]
                    sizes = None
                count_idx = counts_seen.get(n, 0)
                counts_seen[n] = count_idx + 1
                # Same Part C selection rule; selection is retained in manifests.
                seed = 55402026 + n * 100000 + count_idx
                selected = random.Random(seed).sample(pool, n)
                images = []
                for position, record in enumerate(selected):
                    item = {"path_relative_to_repo_root": str((source.parent / record["file"]).relative_to(ROOT)),
                            "sha256": record["sha256"], "source_row": record["row_idx"]}
                    if sizes is not None:
                        item["resolution"] = sizes[position]
                    else:
                        item["preparation"] = {"type": "part_c_padded_control", "image_count": n}
                    images.append(item)
                sample = {"sample_id": "%s_%03d" % (case, idx), "images": images,
                          "target_text": "These are photographs.", "axis": axis, "level": level}
                if axis == "image_count":
                    sample["processor_bounds"] = {"min_pixels": 3136, "max_pixels": (576 // n) * 784}
                    sample["expected_visual_tokens"] = 576
                samples.append(sample)
            directory = output / case
            directory.mkdir()
            path = directory / "manifest.json"
            path.write_text(json.dumps({"scope": suite["scope"], "case": case,
                                        "batching": "manifest", "samples": samples}, indent=2) + "\n")
            suite["cases"].append({"name": case, "axis": axis, "level": level,
                                   "manifest": str(path.relative_to(output)),
                                   "samples": samples_per_case,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    return suite


def audit_inputs(output, model_dir):
    processor = load_processor(str(model_dir))
    suite = json.loads((output / "suite.json").read_text())
    for case in suite["cases"]:
        manifest = output / case["manifest"]
        data = json.loads(manifest.read_text())
        observations = []
        for sample in data["samples"]:
            built = build_sample(processor, sample, ROOT)
            observation = dict(built["input_observation"], sample_id=sample["sample_id"])
            slots = int(built["image_mask"].sum())
            if observation["image_count"] != len(sample["images"]) or slots != observation["visual_tokens_total"]:
                raise ValueError("image retention/token mismatch: " + sample["sample_id"])
            if sample.get("expected_visual_tokens") and slots != sample["expected_visual_tokens"]:
                raise ValueError("controlled C token budget failed: " + sample["sample_id"])
            observations.append(observation)
        totals = [r["visual_tokens_total"] for r in observations]
        audit = {"case": case["name"], "passed": True, "samples": observations,
                 "visual_tokens_mean": statistics.mean(totals),
                 "visual_tokens_population_variance": statistics.pvariance(totals),
                 "sequence_lengths": sorted({r["sequence_length"] for r in observations}),
                 "image_counts": sorted({r["image_count"] for r in observations}),
                 "encoder_runtime_measured": False}
        (manifest.parent / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps({k: audit[k] for k in ("case", "passed", "visual_tokens_mean", "sequence_lengths", "image_counts")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-case", type=int, default=24)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    build_manifests(args.output_dir, args.samples_per_case)
    audit_inputs(args.output_dir, args.model_dir)
