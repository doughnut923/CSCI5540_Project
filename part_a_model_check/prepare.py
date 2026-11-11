"""Download and verify the shared Qwen checkpoint and small real-photo fixtures."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_reference(model_dir, reference):
    for item in reference["files"]:
        path = model_dir / item["name"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Shared reference asset differs: {item['name']}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weights = []
    for name in sorted(set(index["weight_map"].values())):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        weights.append({"file": name, "bytes": path.stat().st_size})
    return weights


def prepare_photos(config):
    manifest = json.loads((ROOT / "fixtures/photo_manifest.json").read_text())
    base = (
        "https://raw.githubusercontent.com/doughnut923/CSCI5540_Project/"
        + config["group_commit"]
        + "/part_c_image_count/real_photo_pilot/"
    )
    for item in manifest:
        path = ROOT / "fixtures" / item["file"]
        if path.is_file() and sha256(path) == item["sha256"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        request = urllib.request.Request(base + item["file"], headers={"User-Agent": "Group8-model-check/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            temporary.write_bytes(response.read())
        if sha256(temporary) != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Photo checksum differs: {item['file']}")
        temporary.replace(path)
        print(f"Photo ready: {item['file']}", flush=True)
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Verify an existing local snapshot instead of downloading it")
    parser.add_argument("--photos-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    reference = json.loads((ROOT / "fixtures/reference_assets.json").read_text())
    assert config["model_id"] == reference["repo"]
    assert config["revision"] == reference["revision"]
    photo_count = prepare_photos(config)
    if args.photos_only:
        print(f"Verified {photo_count} shared photo fixtures.")
        return
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    from huggingface_hub import snapshot_download

    if args.model_dir:
        model_dir = args.model_dir.resolve()
    else:
        print(f"Downloading {config['model_id']} at {config['revision']}", flush=True)
        model_dir = Path(snapshot_download(
            repo_id=config["model_id"], revision=config["revision"],
            allow_patterns=["*.json", "*.txt", "*.safetensors"], max_workers=2,
        ))
    weights = verify_reference(model_dir, reference)
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    (cache / "prepared.json").write_text(json.dumps({
        "model_id": config["model_id"], "revision": config["revision"],
        "model_dir": str(model_dir), "verified_shared_files": len(reference["files"]),
        "verified_photos": photo_count, "weights": weights,
    }, indent=2) + "\n")
    print(json.dumps({"verified_photos": photo_count, "verified_shared_files": len(reference["files"]),
                      "weight_files": weights, "status": "ready"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
