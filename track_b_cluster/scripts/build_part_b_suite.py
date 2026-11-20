"""Build the three Part B resolution datasets as manifests for the DistTrain arm.

`part_b_image_res/variance_datasets/dataset_{Low,Medium,High}.json` hold 750 images each: one
resolution per image, three equal-mean / different-variance schedules. This pairs consecutive
images into samples (2 by default, the convention the N=2 pilot uses), writes a manifest per
dataset in the same layout as the B/C suite, and audits every sample through the processor.

Output layout matches `build_heterogeneity_suite.py`, so `report_bc_suite_distrain.py` reads it
unchanged. Generated data stays outside Git.
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "track_b_cluster/bridge"))
from manifest_data import build_sample, load_processor  # noqa: E402


def build(dataset_dir, images_root, output, per_sample):
    if per_sample < 1:
        raise ValueError("images per sample must be positive")
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name in ("Low", "Medium", "High"):
        records = json.loads((dataset_dir / ("dataset_%s.json" % name)).read_text())
        if len(records) % per_sample:
            raise ValueError("%s: %d images do not split into %d per sample"
                             % (name, len(records), per_sample))
        case = "resolution_full_%s" % name.lower()
        samples, histogram = [], {}
        for index in range(0, len(records), per_sample):
            group = records[index:index + per_sample]
            images = []
            for record in group:
                path = images_root / record["image_id"]
                if not path.is_file():
                    raise ValueError("missing image: %s" % path)
                resolution = int(record["resolution"])
                histogram[resolution] = histogram.get(resolution, 0) + 1
                images.append({"path_relative_to_repo_root": str(path.relative_to(ROOT)),
                               "resolution": resolution,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if len(images) != per_sample:
                raise ValueError("%s: ragged last sample" % name)
            samples.append({
                "sample_id": "%s_%03d" % (case, len(samples)),
                "images": images,
                "target_text": "These are photographs.",
                "derived_visual_tokens_total": sum(int(r["token_cost_estimate"])
                                                   for r in group),
            })
        directory = output / case
        directory.mkdir()
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({
            "scope": "Part B variance dataset, %d-image samples" % per_sample,
            "case": case,
            "batching": "bucket",
            "source_dataset": "part_b_image_res/variance_datasets/dataset_%s.json" % name,
            "samples": samples}, indent=1) + "\n")
        source_counts = {}
        for record in records:
            source_counts[int(record["resolution"])] = (
                source_counts.get(int(record["resolution"]), 0) + 1)
        # Pairing must not change the image-level mix, or "equal mean, different variance"
        # is not what the run measured.
        if histogram != source_counts:
            raise ValueError("%s: pairing changed the resolution mix %s vs %s"
                             % (name, histogram, source_counts))
        cases.append({"name": case, "axis": "resolution", "level": name.lower(),
                      "manifest": str(manifest.relative_to(output)),
                      "samples": len(samples),
                      "images_per_sample": per_sample,
                      "resolution_counts": source_counts,
                      "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})
        print(json.dumps({"case": case, "samples": len(samples),
                          "resolution_counts": source_counts}))
    (output / "suite.json").write_text(json.dumps({
        "scope": "Part B resolution datasets as %d-image samples" % per_sample,
        "batching": "bucket", "cases": cases}, indent=1) + "\n")


def audit(output, model_dir):
    processor = load_processor(str(model_dir))
    suite = json.loads((output / "suite.json").read_text())
    for case in suite["cases"]:
        manifest_path = output / case["manifest"]
        rows, lengths, counts = [], set(), set()
        for sample in json.loads(manifest_path.read_text())["samples"]:
            built = build_sample(processor, sample, ROOT)
            observation = dict(built["input_observation"], sample_id=sample["sample_id"])
            slots = int(built["image_mask"].sum())
            if slots != sample["derived_visual_tokens_total"]:
                raise ValueError("%s: processor tokens %d != derived %d"
                                 % (sample["sample_id"], slots,
                                    sample["derived_visual_tokens_total"]))
            lengths.add(observation["sequence_length"])
            counts.add(observation["image_count"])
            rows.append(observation)
        totals = [r["visual_tokens_total"] for r in rows]
        result = {"case": case["name"], "passed": True, "samples": rows,
                  "visual_tokens_mean": statistics.mean(totals),
                  "visual_tokens_population_variance": statistics.pvariance(totals),
                  "sequence_lengths": sorted(lengths),
                  "image_counts": sorted(counts),
                  "encoder_runtime_measured": False}
        (manifest_path.parent / "input_audit.json").write_text(json.dumps(result) + "\n")
        print(json.dumps({k: result[k] for k in ("case", "visual_tokens_mean",
                                                 "visual_tokens_population_variance",
                                                 "sequence_lengths", "image_counts")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "part_b_image_res/variance_datasets")
    parser.add_argument("--images-root", type=Path, default=ROOT / "part_b_image_res")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--images-per-sample", type=int, default=2)
    args = parser.parse_args()
    build(args.dataset_dir, args.images_root, args.output_dir, args.images_per_sample)
    audit(args.output_dir, args.model_dir)
