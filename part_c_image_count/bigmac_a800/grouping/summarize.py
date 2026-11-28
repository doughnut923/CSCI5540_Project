"""Report mean step costs and paired order-seed differences without pooling seeds as independent steps."""
import json
import statistics as st
import sys
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write_summary(path):
    times = rows(path/'time.jsonl')
    traces = rows(path/'profile.jsonl')
    report = {}
    for design in dict.fromkeys(r['design'] for r in times):
        arm_names = ('image_count_low','image_count_high') if design=='low_high' else ('grouped','mixed')
        ds = [r for r in times if r['design']==design]
        arms = {}
        for arm in arm_names:
            data = [r for r in ds if r['arm']==arm]
            if not data:
                continue
            seconds = st.mean(r['seconds'] for r in data)
            seeds = {s:1000*st.mean(r['seconds'] for r in data if r['seed']==s) for s in sorted({r['seed'] for r in data})}
            arms[arm] = dict(steps=len(data),mean_step_ms=1000*seconds,mean_ms_by_seed=seeds,samples_per_sec=2/seconds,
                             images_per_sec=sum(sum(r['images']) for r in data)/sum(r['seconds'] for r in data),
                             peak_GiB=max(max(r['peak_bytes']) for r in data)/2**30)
            prof = [r for r in traces if r['design']==design and r['arm']==arm]
            if not prof:
                continue
            metrics = {}
            for rank in (0,1):
                valid = [r for r in prof if r['ranks'][rank]['available']]
                metrics[rank] = {'total':len(prof), 'available':len(valid)}
                if valid:
                    for name in ('window_ms','gpu_idle_ms','gpu_busy_ms','nccl_only_ms','noncomm_busy_ms'):
                        metrics[rank][name] = st.mean(r['ranks'][rank][name] for r in valid)
            validsteps = [r for r in prof if all(x['available'] for x in r['ranks'])]
            by_seed = {s:st.mean(st.mean(v['gpu_idle_ms'] for v in r['ranks']) for r in validsteps if r['seed']==s)
                       for s in sorted({r['seed'] for r in validsteps})}
            arms[arm]['profile'] = dict(ranks=metrics, gpu_idle_two_rank_mean_ms_by_seed=by_seed,
                                       step_ms=1000*st.mean(r['seconds'] for r in prof))
        entry = {'arms':arms}
        if all(a in arms for a in arm_names):
            a,b = (arms[k]['mean_ms_by_seed'] for k in arm_names)
            paired = {s:b[s]-a[s] for s in a.keys() & b.keys()}
            entry['second_minus_first_ms_by_seed'] = paired
            entry['second_minus_first_mean_ms'] = st.mean(paired.values())
            entry['second_minus_first_pct'] = 100*(arms[arm_names[1]]['mean_step_ms']/arms[arm_names[0]]['mean_step_ms']-1)
        report[design] = entry
    temp = path/'summary.partial'
    temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temp.replace(path/'summary.json')
    return report


if __name__ == '__main__':
    write_summary(Path(sys.argv[1]))
