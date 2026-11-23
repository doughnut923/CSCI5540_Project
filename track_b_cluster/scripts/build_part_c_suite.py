"""Build the Part C image-count cases as manifests for the DistTrain arm.

Two families, same eight-photo pool and preparation as Part C's own pilot:

* five per-count cases, N in {1,2,3,4,8} with 16 samples each — Part C's grid;
* three mixes (N=1 / N=1,2,3 / N=1,2,4,8) with 12 samples, copied from the B/C suite so
  they stay byte-identical to the cases PR #5 ran.

Every sample carries N images at exactly 576 visual tokens (`padded_control`), with Part C's
selection seed, so the per-count cases reproduce its own input set. Output layout matches the
other suites, so `report_bc_suite_distrain.py` reads it unchanged.
"""
import argparse
import hashlib
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "track_b_cluster/bridge"))
from manifest_data import build_sample, load_processor  # noqa: E402

SEED = 55402026
TARGET = 576
COUNTS = (1, 2, 3, 4, 8)
MIXES = (("image_count_low", [1]), ("image_count_medium", [1, 2, 3]),
         ("image_count_high", [1, 2, 4, 8]))


def sample_from(pool, manifest_path, n, count_index, sample_id):
    seed = SEED + n * 100000 + count_index
    selected = random.Random(seed).sample(pool, n)
    images = []
    for record in selected:
        images.append({
            "path_relative_to_repo_root": str((manifest_path.parent / record["file"])
                                              .relative_to(ROOT)),
            "sha256": record["sha256"],
            "source_row": record["row_idx"],
            "preparation": {"type": "part_c_padded_control", "image_count": n},
        })
    return {
        "sample_id": sample_id,
        "images": images,
        "target_text": "These are photographs.",
        "axis": "image_count",
        "processor_bounds": {"min_pixels": 3136, "max_pixels": (TARGET // n) * 784},
        "expected_visual_tokens": TARGET,
    }


def build(output, photo_manifest, per_count, mixes_root):
    output.mkdir(parents=True, exist_ok=False)
    pool = json.loads(photo_manifest.read_text())
    for record in pool:
        path = photo_manifest.parent / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("photo pool hash mismatch: %s" % path)
    cases = []
    if mixes_root:
        for name, _ in MIXES:
            source = mixes_root / name
            shutil.copytree(source, output / name)
            cases.append({"name": name, "axis": "image_count", "level": name.split("_")[-1],
                          "manifest": name + "/manifest.json", "samples": 12,
                          "source": "copied from " + str(source)})
    for n in COUNTS:
        case = "image_count_n%d" % n
        samples = [sample_from(pool, photo_manifest, n, index, "%s_%03d" % (case, index))
                   for index in range(per_count)]
        directory = output / case
        directory.mkdir()
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({
            "scope": "Part C padded_control, %d image(s) per sample" % n,
            "case": case, "batching": "bucket", "samples": samples}, indent=1) + "\n")
        cases.append({"name": case, "axis": "image_count", "level": "n%d" % n,
                      "manifest": case + "/manifest.json", "samples": len(samples),
                      "images_per_sample": n,
                      "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})
        print(json.dumps({"case": case, "samples": len(samples), "images": n}))
    (output / "suite.json").write_text(json.dumps({
        "scope": "Part C image-count cases", "batching": "bucket", "cases": cases}, indent=1) + "\n")


def audit(output, model_dir):
    processor = load_processor(str(model_dir))
    suite = json.loads((output / "suite.json").read_text())
    for case in suite["cases"]:
        target = output / Path(case["manifest"]).parent / "input_audit.json"
        if target.exists():  # copied mix, already audited by the B/C suite
            continue
        rows, lengths, counts = [], set(), set()
        for sample in json.loads((output / case["manifest"]).read_text())["samples"]:
            built = build_sample(processor, sample, ROOT)
            observation = dict(built["input_observation"], sample_id=sample["sample_id"])
            slots = int(built["image_mask"].sum())
            if slots != TARGET:
                raise ValueError("%s: %d visual tokens, expected %d"
                                 % (sample["sample_id"], slots, TARGET))
            if observation["image_count"] != len(sample["images"]):
                raise ValueError("%s: images lost in preprocessing" % sample["sample_id"])
            lengths.add(observation["sequence_length"])
            counts.add(observation["image_count"])
            rows.append(observation)
        result = {"case": case["name"], "passed": True, "samples": rows,
                  "visual_tokens_mean": statistics.mean(r["visual_tokens_total"] for r in rows),
                  "sequence_lengths": sorted(lengths), "image_counts": sorted(counts),
                  "encoder_runtime_measured": False}
        target.write_text(json.dumps(result) + "\n")
        print(json.dumps({k: result[k] for k in ("case", "sequence_lengths", "image_counts")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--photo-manifest", type=Path,
                        default=ROOT / "part_c_image_count/real_photo_pilot/photo_manifest.json")
    parser.add_argument("--samples-per-count", type=int, default=16)
    parser.add_argument("--mixes-root", type=Path,
                        help="B/C suite root whose image_count_* cases are copied in")
    args = parser.parse_args()
    build(args.output_dir, args.photo_manifest, args.samples_per_count, args.mixes_root)
    audit(args.output_dir, args.model_dir)
