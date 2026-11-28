"""Summarise the six-case B/C suite as run on the disaggregated DistTrain arm.

One JSONL per case, written by `bridge/disagg_units.py` with TAG=bc_<case>. Prints one row
per case and an overall pass flag; exits 1 if any case fails sanity or input coverage.

Coverage is checked against the suite's own input audit: the image counts must match
exactly, and the sequence lengths the run saw must be a subset of the audited ones. The
token mean is reported, not checked: the arm buckets samples by length, so a six-step run
does not necessarily consume all twelve samples of a case.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_step_metrics import percentile  # noqa: E402  (same-directory helper)


def summarise(rows):
    times = [r["step_time_s"] for r in rows]
    ordered = sorted(times)
    observations = [o for r in rows for o in (r.get("input_observations") or [])]
    by_length = {}
    for row in rows:
        length = max(o["sequence_length"] for o in row["input_observations"])
        by_length.setdefault(length, []).append(row["step_time_s"])
    encoder_ms = []
    for row in rows:
        for rank, kind, start, end in row["busy_intervals"]:
            if kind == "enc":
                encoder_ms.append((end - start) * 1000.0)
    peaks = [max(r["max_memory_allocated_bytes_by_rank"].values()) for r in rows]
    losses = [r["loss"] for r in rows]
    weights = [r["weight_norm"] for r in rows]
    enc_weights = [r["enc_weight_norm"] for r in rows if r.get("enc_weight_norm") is not None]
    image_grads = [value for r in rows
                   for values in (r.get("image_embedding_grad_norms") or {}).values()
                   for value in values]
    return {
        "steps": len(rows),
        "step_time_s": {"mean": statistics.mean(times), "p50": percentile(ordered, 50),
                        "p95": percentile(ordered, 95)},
        "throughput_samples_per_s": sum(r["samples"] for r in rows) / sum(times),
        "pipeline_bubble_ratio_mean": statistics.mean(
            r["pipeline_bubble_ratio"] for r in rows),
        "peak_allocated_gib": max(peaks) / 2 ** 30,
        "encoder_forward_ms_mean": (statistics.mean(encoder_ms) if encoder_ms else None),
        # Median too: one slow forward (allocator or scheduling stall) moves the mean ~15 ms.
        "encoder_forward_ms_median": (statistics.median(encoder_ms) if encoder_ms else None),
        # A bucketed step has one length, so this is the cost per length the case exercised.
        "step_time_s_by_sequence_length": {
            str(length): {"steps": len(values), "p50": percentile(sorted(values), 50)}
            for length, values in sorted(by_length.items())},
        "inputs": {
            "image_counts": sorted({o["image_count"] for o in observations}),
            "sequence_lengths": sorted({o["sequence_length"] for o in observations}),
            "visual_tokens_mean": (statistics.mean(o["visual_tokens_total"]
                                                  for o in observations)
                                   if observations else None),
            "samples_consumed": len({s for r in rows for s in r["sample_ids"]}),
            "images_seen": sum(o["image_count"] for o in observations),
        },
        "correctness": {
            "skipped_steps": sum(1 for r in rows if r.get("skipped")),
            "nan_or_inf_steps": sum(1 for r in rows if r.get("nan_or_inf")),
            "loss_finite": all(math.isfinite(x) for x in losses),
            "loss_first_last": [losses[0], losses[-1]],
            "llm_weight_norm_moved": weights[0] != weights[-1],
            "enc_weight_norm_moved": (enc_weights[0] != enc_weights[-1]
                                      if enc_weights else None),
            # One value per image slot: every image reached the model, not just the processor.
            "image_gradient_checks": len(image_grads),
            "image_gradient_checks_positive": sum(1 for value in image_grads if value > 0),
        },
    }


def report(suite_root, runs_root):
    suite = json.loads((suite_root / "suite.json").read_text())
    results = []
    for case in suite["cases"]:
        audit_path = suite_root / Path(case["manifest"]).parent / "input_audit.json"
        audit = json.loads(audit_path.read_text())
        found = sorted(runs_root.glob("disagg_m*_*%s.jsonl" % case["name"]))
        if len(found) != 1:
            raise SystemExit("%s: expected one jsonl in %s, found %d"
                             % (case["name"], runs_root, len(found)))
        rows = [json.loads(x) for x in found[0].read_text().splitlines() if x.strip()]
        summary = summarise(rows)
        inputs = summary["inputs"]
        coverage = (inputs["image_counts"] == audit["image_counts"]
                    and set(inputs["sequence_lengths"]) <= set(audit["sequence_lengths"]))
        checks = summary["correctness"]["image_gradient_checks"]
        grads_ok = (checks == 0 or (checks == summary["correctness"]
                                    ["image_gradient_checks_positive"]
                                    and checks == summary["inputs"]["images_seen"]))
        correct = (not summary["correctness"]["skipped_steps"]
                   and not summary["correctness"]["nan_or_inf_steps"]
                   and summary["correctness"]["loss_finite"]
                   and summary["correctness"]["llm_weight_norm_moved"]
                   and summary["correctness"]["enc_weight_norm_moved"] is not False
                   and grads_ok)
        results.append({"case": case["name"], "axis": case["axis"], "jsonl": str(found[0]),
                        "input_coverage_pass": coverage, "correctness_pass": correct,
                        "audit": {"image_counts": audit["image_counts"],
                                  "sequence_lengths": audit["sequence_lengths"],
                                  "visual_tokens_mean": audit["visual_tokens_mean"]},
                        "summary": summary})
    return {
        "pass": all(r["input_coverage_pass"] and r["correctness_pass"] for r in results),
        "case_count": len(results),
        "config": {"units": "encoder DP=1 + LLM PP", "resolved_from": "records",
                   "note": "seeded init, no checkpoint; timings only, no cross-arm claim"},
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.suite_root, args.runs_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("%-20s %2s %7s %7s %7s %6s %6s %7s %s" % (
        "case", "st", "mean_s", "p50_s", "p95_s", "samp/s", "bubble", "enc_ms", "inputs"))
    for item in result["results"]:
        summary = item["summary"]
        print("%-20s %2d %7.3f %7.3f %7.3f %6.2f %5.1f%% %7.1f counts=%s len=%s samples=%d%s" % (
            item["case"], summary["steps"], summary["step_time_s"]["mean"],
            summary["step_time_s"]["p50"], summary["step_time_s"]["p95"],
            summary["throughput_samples_per_s"], 100 * summary["pipeline_bubble_ratio_mean"],
            summary["encoder_forward_ms_median"], summary["inputs"]["image_counts"],
            summary["inputs"]["sequence_lengths"], summary["inputs"]["samples_consumed"],
            "" if item["input_coverage_pass"] and item["correctness_pass"] else "  FAIL"))
    print(json.dumps({"pass": result["pass"], "cases": result["case_count"]}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
