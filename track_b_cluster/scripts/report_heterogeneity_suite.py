"""Audit paired six-case results and summarize a small CPU-offload pilot.

The serial arm is a correctness control, not the team's Megatron performance
baseline. This report never labels it BigMac or a full Optimus reproduction.
"""
import argparse
import json
from pathlib import Path
import statistics

from compare_optimus_validation import compare


def summarize_run(path):
    rows = [json.loads(x) for x in (path / "metrics.jsonl").read_text().splitlines()]
    rows = [r for r in rows if not r["warmup"]]
    times = [r["step_time_s"] for r in rows]
    encoder_times = [(iv["end_s"] - iv["start_s"]) * 1000
                     for r in rows for iv in r["compute_intervals"] if iv["kind"] == "EF"]
    observations = [x for r in rows for x in r["input_observations"]]
    return {"steps": len(rows), "step_time_s_mean": statistics.mean(times),
            "step_time_s_stdev": statistics.stdev(times) if len(times) > 1 else None,
            "throughput_samples_per_s": sum(r["samples"] for r in rows) / sum(times),
            "peak_allocated_bytes": max(v for r in rows for v in r["max_memory_allocated_bytes_by_rank"].values()),
            "encoder_forward_ms_mean": statistics.mean(encoder_times),
            "encoder_forward_ms_stdev": statistics.stdev(encoder_times),
            "visual_tokens_mean": statistics.mean(x["visual_tokens_total"] for x in observations),
            "image_counts_exercised": sorted({x["image_count"] for x in observations}),
            "sequence_lengths_exercised": sorted({x["sequence_length"] for x in observations}),
            "image_gradient_checks": sum(len(values) for r in rows
                                         for values in r["image_embedding_grad_norms"].values())}


def report(suite_root, runs_root):
    suite = json.loads((suite_root / "suite.json").read_text())
    results = []
    for case in suite["cases"]:
        repeats = sorted((runs_root / case["name"]).glob("repeat-*"))
        if not repeats:
            raise ValueError("missing case: " + case["name"])
        audit = json.loads((suite_root / Path(case["manifest"]).parent / "input_audit.json").read_text())
        for repeat in repeats:
            checked = compare(repeat)
            summaries = {mode: summarize_run(repeat / mode) for mode in ("serial-reference", "optimus-pp")}
            expected_counts = audit["image_counts"]
            coverage = all(s["image_counts_exercised"] == expected_counts for s in summaries.values())
            budget = all(abs(s["visual_tokens_mean"] - audit["visual_tokens_mean"]) < 1e-9
                         for s in summaries.values())
            sequences = all(s["sequence_lengths_exercised"] == audit["sequence_lengths"]
                            for s in summaries.values())
            results.append({"case": case["name"], "axis": case["axis"], "repeat": repeat.name,
                            "parity": checked, "input_coverage_pass": coverage,
                            "input_budget_pass": budget, "input_sequence_coverage_pass": sequences,
                            "runs": summaries})
    resolution_means = [r["runs"]["serial-reference"]["encoder_forward_ms_mean"]
                        for r in results if r["axis"] == "resolution"]
    spread = max(resolution_means) / min(resolution_means) - 1
    return {"pass": all(r["parity"]["pass"] and r["input_coverage_pass"] and r["input_budget_pass"]
                        and r["input_sequence_coverage_pass"] for r in results),
            "case_count": len(suite["cases"]), "paired_runs": len(results), "results": results,
            "resolution_encoder_mean_relative_spread": spread,
            "resolution_mean_workload_matched_within_5_percent": spread <= .05,
            "scope": "six-case engineering pilot using the team's eight-photo pool",
            "performance_claim": False,
            "remaining": ["independent repeats and more measured steps",
                          "calibrate resolution mixtures against encoder runtime",
                          "match training/memory settings with actual BigMac and DistTrain backends"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.suite_root, args.runs_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("pass", "case_count", "paired_runs", "resolution_encoder_mean_relative_spread")}))
    raise SystemExit(0 if result["pass"] else 1)
