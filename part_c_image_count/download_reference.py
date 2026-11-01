"""Fetch six hash-pinned tokenizer/config files. Never download model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
FILES = {'config.json', 'preprocessor_config.json', 'tokenizer_config.json',
         'tokenizer.json', 'vocab.json', 'merges.txt'}
MAX_BYTES = 20_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, default=ROOT / 'assets')
    parser.add_argument('--verify-only', action='store_true', help='Check existing assets without network access')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'assets_manifest.json').read_text())
    if {entry['name'] for entry in manifest['files']} != FILES:
        raise ValueError('Unexpected asset list; refusing download')
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        name = entry['name']
        target = args.assets_dir / name
        if target.exists():
            content = target.read_bytes()
        elif args.verify_only:
            raise FileNotFoundError(f'Missing asset: {name}')
        else:
            url = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise ValueError(f'Asset exceeds safety limit: {name}')
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError(f'Asset differs from pinned reference: {name}')
        if not target.exists():
            with target.open('xb') as handle:
                handle.write(content)
        print(f'VERIFIED {name} ({len(content)} bytes)', flush=True)
    print(json.dumps({'model': manifest['repo'], 'revision': manifest['revision'],
                      'total_bytes': sum(entry['bytes'] for entry in manifest['files']),
                      'weights_downloaded': False}))


if __name__ == '__main__':
    main()
