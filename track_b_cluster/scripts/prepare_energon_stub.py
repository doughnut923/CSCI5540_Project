"""Reproduce the documented import-only Energon shim for the model-only gate.

No dataset loader is implemented. Calling the stub's loader functions raises.
The original training harness uses its own already-processed image tensors.
"""
import hashlib
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]) / "megatron" / "energon"
    files = {
        "epathlib/__init__.py": "from .epath import EPath\n",
        "epathlib/epath.py": "class EPath(str):\n    pass\n",
        "flavors/__init__.py": "",
        "flavors/base_dataset.py": "from dataclasses import dataclass\n@dataclass\nclass Sample:\n    pass\n",
        "flavors/webdataset/__init__.py": (
            "class DefaultDecoderWebdatasetFactory:\n"
            "    @classmethod\n"
            "    def __class_getitem__(cls, item):\n"
            "        return cls\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        raise RuntimeError('Import-only Energon stub: dataset loading is unavailable')\n"
        ),
        "__init__.py": (
            "from dataclasses import dataclass\n"
            "from .epathlib import EPath\n"
            "from .flavors.base_dataset import Sample\n"
            "@dataclass\n"
            "class Batch:\n"
            "    pass\n"
            "@dataclass\n"
            "class VQASample(Sample):\n"
            "    pass\n"
            "class DefaultTaskEncoder:\n"
            "    @classmethod\n"
            "    def __class_getitem__(cls, item):\n"
            "        return cls\n"
            "class WorkerConfig:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        raise RuntimeError('Import-only Energon stub: worker configuration is unavailable')\n"
            "def unavailable(*args, **kwargs):\n"
            "    raise RuntimeError('Import-only Energon stub: dataset loading is unavailable')\n"
            "LimitDataset = RepeatDataset = WorkerConfig\n"
            "get_loader = get_savable_loader = get_train_dataset = get_val_datasets = unavailable\n"
        ),
    }
    for relative, text in files.items():
        path = root / relative
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise SystemExit("Refusing to replace a different existing module: " + str(path))
    hashes = {}
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        hashes[relative] = hashlib.sha256(text.encode()).hexdigest()
    receipt = {"scope": "Import-only shim for native PP=2 smoke test; not an Energon dataset implementation", "files": hashes}
    (root / "stub-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"energon_stub": str(root), "file_count": len(files)}))


if __name__ == "__main__":
    main()
