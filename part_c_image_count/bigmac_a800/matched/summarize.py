"""Reject incomplete pairs; summarize only synchronized, matched post-warmup timings."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics

root=Path(os.environ['PARTC_RUN_DIR'])
primary=Path(os.environ['PARTC_PRIMARY'])
folder=root/'results'
allocation=json.loads((folder/'allocation.json').read_text())
assert allocation['distinct_devices_verified'] and len(allocation['gpus']) == 2 and all('A800' in gpu['name'] for gpu in allocation['gpus'])
assert json.loads((folder/'numeric_pass.json').read_text())['steps_compared_per_rank']==108
plan=json.loads((primary/'data/plan.json').read_text())
rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
assert len(rows)==216
summary={'cases':{}}
for case,expected in plan.items():
    data={}
    for system in ('BigMac','Optimus-PP'):
        blocks=[]
        condition=[row for row in rows if row['case']==case and row['system']==system]
        for b in range(1,5):
            group=[row for row in condition if row['block']==b]
            assert [row['ids'] for row in group]==expected
            assert [row['step'] for row in group]==list(range(9))
            assert [row['warmup'] for row in group]==[True]+[False]*8
            assert all(row['checked_against_other']==((system=='Optimus-PP')==(b%2==1)) for row in group)
            blocks.append(statistics.mean(row['seconds'] for row in group[1:]))
        measured=[row for row in condition if not row['warmup']]
        assert len(measured)==32 and all(math.isfinite(row['seconds']) and row['seconds']>0 for row in measured)
        assert all(len(row['peak_bytes'])==2 and min(row['peak_bytes'])>0 for row in measured)
        mean=statistics.mean(row['seconds'] for row in measured)
        data[system]={'mean_step_ms':1000*mean, 'samples_s':2/mean,
            'max_rank_peak_allocated_gib':max(max(row['peak_bytes']) for row in measured)/2**30,
            'paired_block_mean_ms':[1000*t for t in blocks], 'timed_steps':32}
    summary['cases'][case]=data
out=folder/'summary.json'
tmp=out.with_suffix('.tmp')
tmp.write_text(json.dumps(summary,indent=2)+'\n')
tmp.replace(out)
print(out.read_text())

lines = [
    '# Image count and training speed on two A800 GPUs',
    '',
    '| Images per sample | Runtime | Mean step (ms) | Throughput (samples/s) | Peak per GPU (GiB) |',
    '| --- | --- | ---: | ---: | ---: |',
]
labels = {'low': 'Low (1)', 'medium': 'Medium (1, 2, 3)', 'high': 'High (1, 2, 4, 8)'}
for level, label in labels.items():
    for runtime, row in summary['cases']['image_count_' + level].items():
        display = 'Optimus-inspired PP' if runtime == 'Optimus-PP' else runtime
        lines.append(f"| {label} | {display} | {row['mean_step_ms']:.1f} | {row['samples_s']:.2f} | {row['max_rank_peak_allocated_gib']:.2f} |")

low = summary['cases']['image_count_low']['BigMac']
high = summary['cases']['image_count_high']['BigMac']
change = 100 * (high['mean_step_ms'] / low['mean_step_ms'] - 1)
slower = sum(h > l for h, l in zip(high['paired_block_mean_ms'], low['paired_block_mean_ms']))
lines.extend(['', f'On BigMac, the High mix changed mean step time by {change:+.1f}% relative to Low. High was slower in {slower} of four paired repeats.',
              '', 'Both runtimes used the same two A800 GPUs and model weights, 576 visual tokens per input, and identical sample order, and the deterministic math SDPA backend. Only the final visual-projector matrix was trained. Values use 32 timed steps per workload and runtime, excluding warm-up.'])
slide = folder/'Part_C_final_table.md'
temp = slide.with_suffix('.pending')
temp.write_text('\n'.join(lines)+'\n')
temp.replace(slide)
