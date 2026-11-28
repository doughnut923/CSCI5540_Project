"""Rebuild the 36 measured input archives from the team's photos and pinned loader."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

DATA = Path(__file__).resolve().parent / 'primary/data'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('team_checkout', type=Path, help='Team repository checkout at d309514')
    parser.add_argument('processor_dir', type=Path, help='Local Qwen2-VL tokenizer and processor assets')
    parser.add_argument('--output', type=Path, default=DATA / 'tensors')
    args = parser.parse_args()
    repo = args.team_checkout.resolve()
    assets = args.processor_dir.resolve()
    for name, expected in json.loads((DATA / 'processor_hashes.json').read_text()).items():
        if digest(assets / name) != expected:
            raise ValueError(f'Processor asset checksum mismatch: {name}')
    source = repo / 'track_b_cluster/bridge/manifest_data.py'
    spec = importlib.util.spec_from_file_location('team_manifest_data', source)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    processor = loader.load_processor(assets)
    verification = json.loads((DATA / 'verification.json').read_text())['cases']
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as tmp:
        for condition, evidence in verification.items():
            manifest = DATA / condition / 'manifest.json'
            if digest(manifest) != evidence['manifest_sha256']:
                raise ValueError(f'Manifest checksum mismatch: {condition}')
            for sample in json.loads(manifest.read_text())['samples']:
                key = sample['sample_id']
                expected = evidence['cases'][key]['sha256']
                output = args.output / f'{key}.npz'
                if output.exists():
                    if digest(output) != expected:
                        raise ValueError(f'Existing archive has a different checksum: {output}')
                    continue
                batch = loader.build_sample(processor, sample, repo)
                tokens = batch['tokens'].numpy()
                dest = Path(tmp) / output.name
                np.savez_compressed(dest, input_ids=tokens, labels=batch['labels'].numpy(),
                                    attention_mask=np.ones_like(tokens), image_grid_thw=batch['grid'].numpy(),
                                    pixel_values=batch['pixels'].float().numpy())
                if digest(dest) != expected:
                    raise ValueError(f'Encoded {key} differs from the measured archive. Check the preprocessing versions.')
                dest.replace(output)
    print('Verified all 36 tensor archives.')


if __name__ == '__main__':
    main()
