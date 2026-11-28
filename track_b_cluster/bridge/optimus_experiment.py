"""Input-plan checks owned by Optimus; the team's loader stays unchanged."""
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from manifest_data import ManifestBatches


class OptimusBatches(ManifestBatches):
    """Reuse the shared builder, optionally replaying an audited Optimus plan."""

    def __init__(self, *args, input_plan=None, **kwargs):
        self.input_plan = input_plan
        super().__init__(*args, **kwargs)
        self.by_id = {sample["sample_id"]: sample for sample in self.samples}

    def plan(self, step, num_mb):
        if self.input_plan is None:
            return super().plan(step, num_mb)
        ids = self.input_plan["steps"][step]["sample_ids"]
        if len(ids) != num_mb:
            raise ValueError("input plan microbatch count mismatch")
        return [self.by_id[sample_id] for sample_id in ids]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_input_plan(source, manifest_path, audit, microbatches, warmup_steps,
                    measured_steps, sequence_limit=1024, require_coverage=True):
    if min(microbatches, measured_steps, sequence_limit) < 1 or warmup_steps < 0:
        raise ValueError("invalid experiment step or sequence budget")
    observed = {row["sample_id"]: row for row in audit["samples"]}
    expected_ids = {sample["sample_id"] for sample in source.samples}
    if not audit.get("passed") or set(observed) != expected_ids:
        raise ValueError("input audit does not cover this manifest")
    if max(row["sequence_length"] for row in observed.values()) > sequence_limit:
        raise ValueError("audited sequence exceeds configured sequence limit")
    steps, counts = [], Counter()
    for step in range(warmup_steps + measured_steps):
        ids = [sample["sample_id"] for sample in source.plan(step, microbatches)]
        rows = [observed[sample_id] for sample_id in ids]
        if source.batching == "bucket" and len({r["sequence_length"] for r in rows}) != 1:
            raise ValueError("bucket contains mixed sequence lengths")
        is_warmup = step < warmup_steps
        if not is_warmup:
            counts.update(ids)
        steps.append({"step": step, "warmup": is_warmup, "sample_ids": ids,
                      "input_observations": rows})
    missing = sorted(expected_ids - set(counts))
    if require_coverage and missing:
        raise ValueError("measured step budget misses %d samples: %s" % (len(missing), missing))
    order = [{key: row[key] for key in ("step", "warmup", "sample_ids")} for row in steps]
    return {"schema_version": 1, "manifest_sha256": file_sha256(manifest_path),
            "batching": source.batching, "microbatches": microbatches,
            "warmup_steps": warmup_steps, "measured_steps": measured_steps,
            "sequence_limit": sequence_limit, "steps": steps,
            "sample_order_sha256": hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest(),
            "sample_multiset_sha256": hashlib.sha256(json.dumps(dict(counts), sort_keys=True).encode()).hexdigest(),
            "measured_samples": sum(counts.values()), "distinct_measured_samples": len(counts),
            "expected_distinct_samples": len(expected_ids), "missing_sample_ids": missing,
            "sample_multiplicity": dict(sorted(counts.items())), "full_coverage": not missing}


def check_plan_config(plan, manifest_path, batching, microbatches, warmup_steps,
                      measured_steps, sequence_limit):
    expected = {"manifest_sha256": file_sha256(manifest_path), "batching": batching,
                "microbatches": microbatches, "warmup_steps": warmup_steps,
                "measured_steps": measured_steps, "sequence_limit": sequence_limit}
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError("input plan configuration mismatch: " + key)
    if not plan.get("full_coverage"):
        raise ValueError("input plan lacks complete measured coverage")


def interleave_input_plan(plan):
    """Change measured ordering while retaining every sample occurrence."""
    result = deepcopy(plan)
    groups = {}
    for step in plan["steps"]:
        if not step["warmup"]:
            for sample_id, observed in zip(step["sample_ids"], step["input_observations"]):
                groups.setdefault(observed["sequence_length"], deque()).append((sample_id, observed))
    mixed = []
    while any(groups.values()):
        for length in sorted(groups):
            if groups[length]:
                mixed.append(groups[length].popleft())
    offset = 0
    for step in result["steps"]:
        if step["warmup"]:
            continue
        items = mixed[offset:offset + result["microbatches"]]
        offset += result["microbatches"]
        step["sample_ids"] = [item[0] for item in items]
        step["input_observations"] = [item[1] for item in items]
    result["batching"] = "manifest"
    result["order_policy"] = "interleave lengths from the exact source-plan sample multiset"
    result["source_sample_order_sha256"] = plan["sample_order_sha256"]
    order = [{key: row[key] for key in ("step", "warmup", "sample_ids")} for row in result["steps"]]
    result["sample_order_sha256"] = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
    return result


def check_step_inputs(plan, step, data):
    expected = plan["steps"][step]
    if [row["sample_id"] for row in data] != expected["sample_ids"]:
        raise ValueError("runtime sample order differs from input plan")
    for actual, wanted in zip(data, expected["input_observations"]):
        for key in ("image_count", "visual_tokens_per_image", "visual_tokens_total",
                    "image_grid_thw", "sequence_length", "supervised_tokens"):
            if actual["input_observation"][key] != wanted[key]:
                raise ValueError("runtime input differs from audit: " + key)
