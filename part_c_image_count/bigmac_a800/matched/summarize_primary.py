"""Validate plan coverage and summarize per-step wall time, throughput and memory."""
import os
import collections
import json
import math
from pathlib import Path
import statistics

root = Path(os.environ['PARTC_PRIMARY'])
results = Path(os.environ['PARTC_RUN_DIR']) / 'primary_results'
plan = json.loads((root / 'data/plan.json').read_text())
rows = [json.loads(line) for line in (results / 'steps.jsonl').read_text().splitlines()]
assert len(json.loads((results.parent/'results/allocation.json').read_text())['gpus']) == 2
requested = results / 'workloads.json'
cases = json.loads(requested.read_text()) if requested.is_file() else ['image_count_low', 'image_count_high']
assert set(cases) in ({'image_count_low', 'image_count_high'},
                      {'image_count_low', 'image_count_medium', 'image_count_high'})
summary = {}
for case in cases:
    pairs = plan[case]
    case_rows = [row for row in rows if row['case'] == case]
    assert len(case_rows) == 36
    for block in range(1, 5):
        group = [row for row in case_rows if row['block'] == block]
        assert [row['ids'] for row in group] == pairs
        assert [row['step'] for row in group] == list(range(9))
    measured = [row for row in case_rows if not row['warmup']]
    assert len(measured) == 32 and all(row['warmup'] == (row['step'] == 0) for row in case_rows)
    durations = [row['seconds'] for row in measured]
    assert all(math.isfinite(t) and t > 0 for t in durations)
    assert all(math.isfinite(v) for row in measured for v in row['losses'])
    assert all(len(row['peak_bytes']) == 2 and min(row['peak_bytes']) > 0 for row in measured)
    summary[case] = {
        'blocks': 4, 'measured_steps':len(measured), 'measured_samples':2*len(measured),
        'mean_step_s': statistics.mean(durations), 'median_step_s':statistics.median(durations),
        'samples_per_s': 2*len(measured)/sum(durations),
        'max_rank_peak_allocated_gib':max(max(row['peak_bytes']) for row in measured)/2**30,
        'block_mean_step_s':[statistics.mean(row['seconds'] for row in measured if row['block']==b) for b in range(1,5)],
        'measured_step_sequence_lengths':dict(collections.Counter(row['seq_len'] for row in measured)),
    }
assert len(rows) == len(cases) * 36 and len(summary) == len(cases)
output = results / 'summary.json'
temporary = output.with_suffix('.pending')
temporary.write_text(json.dumps(summary, indent=2)+'\n')
temporary.replace(output)
print(output.read_text())

table = ['# BigMac on two A800 GPUs', '', '| Workload | Mean step (ms) | Throughput (samples/s) | Peak per GPU (GiB) |','| --- | ---: | ---: | ---: |']
for level, name in [('low','Low (1)'), ('medium','Medium (1, 2, 3)'), ('high','High (1, 2, 4, 8)')]:
    row = summary['image_count_'+level]
    table.append(f"| {name} | {1000*row['mean_step_s']:.1f} | {row['samples_per_s']:.2f} | {row['max_rank_peak_allocated_gib']:.2f} |")
low,high = (summary['image_count_'+level] for level in ['low','high'])
change = 100*(high['mean_step_s']/low['mean_step_s']-1)
slower = sum(h>l for h,l in zip(high['block_mean_step_s'],low['block_mean_step_s']))
table.extend(['',f'High changed mean step time by {change:+.1f}% relative to Low. High was slower in {slower} of four paired repeats.', '', 'Two A800-SXM4-80GB GPUs, 576 visual tokens per sample, 32 measured steps per condition, Qwen2-VL-2B with a trainable visual projector and default SDPA.'])
(results/'Part_C_BigMac_table.md').write_text('\n'.join(table)+'\n')
