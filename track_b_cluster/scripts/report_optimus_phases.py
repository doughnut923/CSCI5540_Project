"""Summarize intrusive Optimus phase timings without double-counting ranks."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bridge"))
from optimus_profiling import TOP_LEVEL_PHASES


def report(run):
    config = json.loads((run / "run_config.json").read_text())
    if not config.get("profile_step_phases") or config["validation_samples"] != 0:
        raise ValueError("requires phase profiling with validation sampling disabled")
    if json.loads((run / "status.json").read_text())["status"] != "completed":
        raise ValueError("run is incomplete")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    rows = [row for row in rows if not row["warmup"]]
    if len(rows) != config["steps"]:
        raise ValueError("measured step count differs from configuration")
    names = list(TOP_LEVEL_PHASES) + ["unattributed"]
    children = ("gradient_prepare", "lr_and_clip", "adamw_update", "parameter_copyback", "diagnostics")
    totals = {name: 0.0 for name in names}
    optimizer_parts = {name: 0.0 for name in children}
    critical_ranks = Counter()
    for row in rows:
        elapsed = row["step_time_s"]
        values = row["critical_rank_phase_breakdown_s"]
        if (row.get("skipped") or row.get("nan_or_inf")
                or not all(math.isfinite(row[key]) for key in ("loss", "grad_norm"))):
            raise ValueError("invalid training step")
        if (set(values) != set(names) or not math.isfinite(elapsed) or elapsed <= 0
                or any(not math.isfinite(value) or value < 0 for value in values.values())
                or not math.isclose(sum(values.values()), elapsed, rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("phase accounting does not close to complete step time")
        rank = str(row["critical_rank"])
        critical_ranks[rank] += 1
        details = row["phase_times_s_by_rank"][rank]
        for name in names:
            totals[name] += values[name]
        for name in children:
            optimizer_parts[name] += details["optimizer." + name]
        if sum(details["optimizer." + name] for name in children) > values["optimizer"] + 1e-6:
            raise ValueError("optimizer children exceed parent")
    elapsed = sum(row["step_time_s"] for row in rows)
    if totals["optimizer"] <= 0:
        raise ValueError("optimizer timing is missing")
    return {"pass": True, "source_commit": config["source_commit"],
            "case_manifest_sha256": config["manifest_sha256"],
            "diagnostic_only": True, "adds_cuda_synchronization": True,
            "performance_claim": False, "measured_steps": len(rows),
            "step_time_s_mean": elapsed / len(rows),
            "step_time_s_stdev": statistics.stdev(row["step_time_s"] for row in rows) if len(rows) > 1 else None,
            "critical_rank_counts": dict(critical_ranks),
            "phase_mean_s": {name: value / len(rows) for name, value in totals.items()},
            "phase_fraction_of_step": {name: value / elapsed for name, value in totals.items()},
            "optimizer_child_mean_s": {name: value / len(rows) for name, value in optimizer_parts.items()},
            "optimizer_child_fraction_of_optimizer": {
                name: value / totals["optimizer"] for name, value in optimizer_parts.items()},
            "dominant_phase": max(TOP_LEVEL_PHASES, key=lambda name: totals[name]),
            "optimizer_device": config["optimizer_device"],
            "notes": ["Each step uses phases from the rank with the maximum complete-step wall time.",
                      "Optimizer children are inclusive in the optimizer parent; do not add both.",
                      "Model schedule includes compute, P2P/waits, and runtime measurement/collection.",
                      "Phase synchronization changes overlap; these are bottleneck diagnostics, not scheduling speedups."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("pass", "measured_steps", "step_time_s_mean", "dominant_phase", "phase_fraction_of_step")}))
