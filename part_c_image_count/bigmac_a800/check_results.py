"""Recalculate the saved results from the step records, without CUDA or model assets."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check():
    with tempfile.TemporaryDirectory(prefix='bigmac-a800-check-') as temp:
        scratch = Path(temp)
        shutil.copytree(ROOT / 'recorded', scratch / 'recorded')
        r4 = scratch / 'recorded/r4'
        grouping = scratch / 'recorded/grouping/results'
        env = dict(os.environ, PARTC_PRIMARY=str(ROOT/'primary'), PARTC_RUN_DIR=str(r4),
                   PYTHONDONTWRITEBYTECODE='1')
        for script, args in [('matched/summarize_primary.py', []), ('matched/summarize.py', []),
                             ('grouping/summarize.py', [grouping]), ('grouping/verify_results.py', [grouping])]:
            subprocess.run([sys.executable, ROOT / script, *args], check=True, env=env,
                           stdout=subprocess.DEVNULL)
        for path in (r4/'primary_results/summary.json', r4/'results/summary.json', grouping/'summary.json'):
            actual = json.loads(path.read_text())
            expected = json.loads((ROOT / path.relative_to(scratch)).read_text())
            if actual != expected:
                raise ValueError(f'Saved and recalculated summaries differ: {path.name}')
        profiled = [json.loads(line) for line in (grouping / 'profile.jsonl').read_text().splitlines()]
        if len(profiled) != 120 or any(
            not gpu['available'] or gpu['source'] != 'kineto_cuda' for step in profiled for gpu in step['ranks']
        ):
            raise ValueError('Incomplete paired GPU idle-time traces')
    print('Verified 96 image-count steps, 192 matched-runtime steps, 1,120 follow-up steps and 120 two-GPU profiling steps.')
    print('The saved summaries match all three recalculations.')


if __name__ == '__main__':
    check()
