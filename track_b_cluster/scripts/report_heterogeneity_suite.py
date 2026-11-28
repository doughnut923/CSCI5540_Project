"""Audit paired six-case results and summarize a small CPU-offload pilot.

The serial arm is a correctness control, not the team's Megatron performance
baseline. This report never labels it BigMac or a full Optimus reproduction.
"""
import argparse
import json
from pathlib import Path
import statistics
from collections import Counter

from compare_optimus_validation import compare


def summarize_run(path):
    rows = [json.loads(x) for x in (path / "metrics.jsonl").read_text().splitlines()]
    rows = [r for r in rows if not r["warmup"]]
    times = [r["step_time_s"] for r in rows]
    encoder_times = [(iv["end_s"] - iv["start_s"]) * 1000
                     for r in rows for iv in r["compute_intervals"] if iv["kind"] == "EF"]
    observations = [x for r in rows for x in r["input_observations"]]
    ids = [sample for r in rows for sample in r["sample_ids"]]
    return {"sample_multiplicity": dict(sorted(Counter(ids).items())),
            "distinct_samples": len(set(ids)), "steps": len(rows), "step_time_s_mean": statistics.mean(times),
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


def report(suite_root, runs_root, selected=None):
    suite = json.loads((suite_root / "suite.json").read_text())
    names = {case["name"] for case in suite["cases"]}
    if selected and not set(selected) <= names:
        raise ValueError("unknown selected case")
    results = []
    for case in suite["cases"]:
        if selected and case["name"] not in selected:
            continue
        repeats = sorted((runs_root / case["name"]).glob("repeat-*"))
        if not repeats:
            raise ValueError("missing case: " + case["name"])
        audit = json.loads((suite_root / Path(case["manifest"]).parent / "input_audit.json").read_text())
        for repeat in repeats:
            checked = compare(repeat)
            summaries = {mode: summarize_run(repeat / mode) for mode in ("serial-reference", "optimus-pp")}
            expected = {row["sample_id"]: row for row in audit["samples"]}
            coverage = all(set(summary["sample_multiplicity"]) == set(expected)
                           for summary in summaries.values())
            budget, sequences = True, True
            for mode in summaries:
                rows = [json.loads(line) for line in (repeat / mode / "metrics.jsonl").read_text().splitlines()]
                for row in rows:
                    if row["warmup"]:
                        continue
                    if len(row["sample_ids"]) != len(row["input_observations"]):
                        raise ValueError("input observation count mismatch")
                    for sample_id, observed in zip(row["sample_ids"], row["input_observations"]):
                        wanted = expected.get(sample_id)
                        if wanted is None:
                            budget, sequences = False, False
                            continue
                        budget &= all(observed[key] == wanted[key] for key in
                                      ("image_count", "visual_tokens_total", "visual_tokens_per_image"))
                        sequences &= observed["sequence_length"] == wanted["sequence_length"]
            results.append({"case": case["name"], "axis": case["axis"], "repeat": repeat.name,
                            "parity": checked, "input_coverage_pass": coverage,
                            "input_budget_pass": budget, "input_sequence_coverage_pass": sequences,
                            "runs": summaries})
    resolution_means = [r["runs"]["serial-reference"]["encoder_forward_ms_mean"]
                        for r in results if r["axis"] == "resolution"]
    spread = max(resolution_means) / min(resolution_means) - 1 if resolution_means else None
    return {"pass": all(r["parity"]["pass"] and r["input_coverage_pass"] and r["input_budget_pass"]
                        and r["input_sequence_coverage_pass"] for r in results),
            "case_count": len({r["case"] for r in results}), "paired_runs": len(results), "results": results,
            "resolution_encoder_mean_relative_spread": spread,
            "resolution_mean_workload_matched_within_5_percent": spread <= .05 if spread is not None else None,
            "scope": suite.get("scope", "Optimus paired engineering validation"),
            "performance_claim": False,
            "remaining": ["independent repeats and more measured steps",
                          "calibrate resolution mixtures against encoder runtime",
                          "match training/memory settings with actual BigMac and DistTrain backends"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="")
    args = parser.parse_args()
    result = report(args.suite_root, args.runs_root, list(filter(None, args.cases.split(","))))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("pass", "case_count", "paired_runs", "resolution_encoder_mean_relative_spread")}))
    raise SystemExit(0 if result["pass"] else 1)
