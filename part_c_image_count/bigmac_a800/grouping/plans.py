from collections import Counter, defaultdict
from pathlib import Path
import json
import random

SEEDS = (101, 211, 307, 401, 503)
ARMS = ('grouped', 'mixed')

def make_plan(path, design, seed):
    rng = random.Random(seed)
    source = json.loads((path/'plan.json').read_text())
    meta = json.loads((path/'verification.json').read_text())['cases']
    counts = {key: record['image_count'] for arm in meta.values() for key, record in arm['cases'].items()}
    pool = sum(source['image_count_high'][1:], [])
    if design == 'single_multi':
        pool += sum(source['image_count_low'][1:], [])[:8]
    else:
        assert design == 'original_high'
    bins = defaultdict(list)
    for key in pool:
        bins[counts[key]].append(key)
    ranks = [dict(), dict()]
    for n, ids in bins.items():
        rng.shuffle(ids)
        assert len(ids) % 2 == 0
        for rank in (0,1):
            ranks[rank][n] = ids[rank::2]
    grouped = [pair for n in bins for pair in zip(ranks[0][n], ranks[1][n])]
    if design == 'single_multi':
        multi = [sum((r[n] for n in (2,4,8)),[]) for r in ranks]
        for seq in multi: rng.shuffle(seq)
        mixed = list(zip(ranks[0][1],multi[1])) + list(zip(multi[0],ranks[1][1]))
        assert len(mixed) == 12
        assert all((counts[a]==1)!=(counts[b]==1) for a,b in mixed)
    else:
        levels = list(bins)
        rng.shuffle(levels)
        offset = rng.randint(1, len(levels)-1)
        mixed = [pair for i,n in enumerate(levels) for pair in zip(ranks[0][n],ranks[1][levels[(i+offset)%len(levels)]])]
        assert len(mixed)==8
    assert all(counts[a]==counts[b] for a,b in grouped)
    assert all(counts[a]!=counts[b] for a,b in mixed)
    for rank in (0,1):
        assert Counter(pair[rank] for pair in grouped) == Counter(pair[rank] for pair in mixed)
    assert Counter(k for pair in grouped for k in pair)==Counter(pool)
    rng.shuffle(grouped)
    rng.shuffle(mixed)
    return dict(grouped=grouped, mixed=mixed), counts
