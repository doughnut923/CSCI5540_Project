"""Aggregate controlled-run step records into plan §6 metrics.

Input: JSONL with one object per optimizer step (post-launch). Expected fields
(best-effort; missing values become unmeasured):
  step (int), warmup (bool, optional), step_time_s (float),
  samples (int, samples in this optimizer step across all ranks or as stated),
  max_memory_allocated_bytes_by_rank ({rank:int} or list), or
  max_memory_allocated_bytes (float/int),
  pipeline_bubble_ratio (float 0..1, optional, from profiler),
  encoder_latency_s (object/buckets, optional),
  microbatch_stall_s (object, optional), loss (float, optional), skipped (bool),
  nan_or_inf (bool).

Outputs summary with explicit measured/derived/unmeasured labels.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup-steps", type=int, default=0, help="Discard first N records from input (or use warmup=true)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--world-size", type=int, default=None)
    p.add_argument("--run-label", default="")
    args = p.parse_args()

    rows = load_jsonl(args.input)
    post = []
    discarded = 0
    for i, r in enumerate(rows):
        if r.get("warmup") is True or i < args.warmup_steps:
            discarded += 1
            continue
        post.append(r)
    if not post:
        raise SystemExit("no post-warmup steps")

    step_times = [float(r["step_time_s"]) for r in post if "step_time_s" in r and r["step_time_s"] is not None]
    def _mean(vals):
        return sum(vals) / len(vals) if vals else None
    step_times_sorted = sorted(step_times)
    samples = [float(r["samples"]) for r in post if "samples" in r and r["samples"] is not None]
    skipped = sum(1 for r in post if r.get("skipped"))
    bad = sum(1 for r in post if r.get("nan_or_inf") or (isinstance(r.get("loss"), float) and not math.isfinite(r["loss"])))

    peak_bytes = None
    peak_note = "unmeasured"
    for r in post:
        by_rank = r.get("max_memory_allocated_bytes_by_rank")
        if isinstance(by_rank, dict):
            vals = [float(v) for v in by_rank.values()]
            peak_bytes = max(vals) if peak_bytes is None else max(peak_bytes, max(vals))
            peak_note = "measured (max across ranks from records)"
        elif isinstance(by_rank, list):
            vals = [float(v) for v in by_rank]
            peak_bytes = max(vals) if peak_bytes is None else max(peak_bytes, max(vals))
            peak_note = "measured (max across ranks from records)"
        elif "max_memory_allocated_bytes" in r:
            v = float(r["max_memory_allocated_bytes"])
            peak_bytes = v if peak_bytes is None else max(peak_bytes, v)
            peak_note = "measured"

    bubbles = [float(r["pipeline_bubble_ratio"]) for r in post if r.get("pipeline_bubble_ratio") is not None]
    bubble_times = [float(r["bubble_time_s"]) for r in post if r.get("bubble_time_s") is not None]
    summary = {
        "run_label": args.run_label,
        "input": str(args.input),
        "records_total": len(rows),
        "warmup_discarded": discarded,
        "post_warmup_steps": len(post),
        "config_record": {"batch_size": args.batch_size, "grad_accum": args.grad_accum, "world_size": args.world_size},
        "primary": {
            "throughput_samples_per_s": {
                "value": (sum(samples) / sum(step_times)) if samples and step_times and sum(step_times) > 0 else None,
                "status": "measured" if samples and step_times else "unmeasured",
                "formula": "sum(samples over post-warmup steps) / sum(step_time_s)",
            },
            "step_time_s": {
                "mean": _mean(step_times),
                "p50": percentile(step_times_sorted, 50) if step_times else None,
                "p95": percentile(step_times_sorted, 95) if step_times else None,
                "status": "measured" if step_times else "unmeasured",
            },
            "peak_gpu_memory_allocated_bytes": {"value": peak_bytes, "status": peak_note},
            "pipeline_bubble_ratio": {
                "values": bubbles or None,
                "mean": _mean(bubbles),
                "status": "measured (instrumented)" if bubbles else "unmeasured (no comparable instrumentation)",
            },
            "pipeline_bubble_time_s": {
                "values": bubble_times or None,
                "mean": _mean(bubble_times),
                "total": sum(bubble_times) if bubble_times else None,
                "status": "measured (instrumented)" if bubble_times else "unmeasured",
            },
        },
        "correctness": {"skipped_steps": skipped, "nan_or_inf_steps": bad, "pass": skipped == 0 and bad == 0},
        "secondary": {
            "encoder_latency_variance": "unmeasured unless per-bucket encoder timers recorded in input",
            "microbatch_stall_time": "unmeasured unless stage/rank wait events recorded in input",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": summary["correctness"]["pass"], "post_warmup_steps": len(post), "throughput": summary["primary"]["throughput_samples_per_s"]["value"]}))


if __name__ == "__main__":
    main()
