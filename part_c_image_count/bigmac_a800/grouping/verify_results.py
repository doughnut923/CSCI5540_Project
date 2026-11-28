"""Check a downloaded run against the predeclared paired plan and all timed blocks."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from plans import SEEDS


def verify(directory):
    read = lambda filename: json.loads((directory / filename).read_text())
    lines = lambda filename: [json.loads(row) for row in (directory / filename).read_text().splitlines()]
    groups = read('plans.json')
    gpus = read('allocation.json')['gpus']
    assert len(gpus) == 2 and read('allocation.json')['distinct_devices_verified'] and all('A800' in gpu['name'] for gpu in gpus)
    checks = lines('padding.jsonl')
    assert len(checks) == 4 and all(c['grad_cosine'] >= .98 for c in checks)
    for name in ('time', 'profile'):
        data = lines(name + '.jsonl')
        by_key = defaultdict(list)
        for row in data:
            key = row['design'], str(row['seed']), row['arm'], row['repeat']
            by_key[key].append(row)
            assert row['seconds'] > 0 and math.isfinite(row['seconds'])
            assert len(row['losses']) == 2 and all(math.isfinite(v) for v in row['losses'])
            assert len(row['peak_bytes']) == 2 and all(v > 0 for v in row['peak_bytes'])
        designs = ('single_multi',) if name == 'profile' else ('single_multi','original_high','low_high')
        repeats = (1,) if name == 'profile' else (1,2,3,4)
        exp = {(d,str(s),a,r) for d in designs for s in SEEDS for a in groups[d][str(s)] for r in repeats}
        assert exp == set(by_key), (name,exp-set(by_key),set(by_key)-exp)
        for (d,s,a,r), block in by_key.items():
            assert [row['step'] for row in block] == list(range(1,len(block)+1))
            assert [row['ids'] for row in block] == groups[d][s][a]
            for row in block:
                if d != 'low_high':
                    assert row['seq_len'] == 621
                    n1,n2 = row['images']
                    assert (n1 == n2) == (a=='grouped')
                    if d == 'single_multi' and a == 'mixed':
                        assert (n1 == 1) != (n2 == 1)
                if name == 'profile':
                    assert len(row['ranks']) == 2
                    for rank in row['ranks']:
                        if rank['available']:
                            assert rank['gpu_events'] > rank['nccl_events'] > 0
                            assert abs(rank['window_ms'] - rank['gpu_idle_ms'] - rank['gpu_busy_ms']) < .51
                            assert abs(rank['gpu_busy_ms'] - rank['noncomm_busy_ms'] - rank['nccl_only_ms']) < .01
        print(name, len(data), 'steps;', len(by_key), 'complete blocks')
        if name == 'profile':
            print('Wait metric sources',Counter(v['source'] for row in data for v in row['ranks']))
    prior = read('summary.json')
    for design in ('single_multi','original_high'):
        for s,arrangements in groups[design].items():
            for rank in (0,1):
                assert Counter(p[rank] for p in arrangements['mixed']) == Counter(p[rank] for p in arrangements['grouped'])
    assert set(prior)=={'single_multi','original_high','low_high'}
    print('Allocation',gpus)


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
