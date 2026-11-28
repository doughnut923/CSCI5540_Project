"""Preflight Optimus runs against Yin's unchanged loader and input audits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "track_b_cluster/bridge"))
from manifest_data import ManifestBatches
from optimus_experiment import (check_plan_config, file_sha256, interleave_input_plan,
                                make_input_plan)


def prepare(suite_root, output_dir, model_dir, batching, microbatches,
            warmup_steps, measured_steps, sequence_limit=1024, selected=None,
            match_plans=None):
    suite = json.loads((suite_root / "suite.json").read_text())
    names = {case["name"] for case in suite["cases"]}
    if selected and not set(selected) <= names:
        raise ValueError("unknown selected case")
    plans = []
    for case in suite["cases"]:
        if selected and case["name"] not in selected:
            continue
        manifest = suite_root / case["manifest"]
        if file_sha256(manifest) != case["sha256"]:
            raise ValueError("suite manifest hash mismatch: " + case["name"])
        audit = json.loads((manifest.parent / "input_audit.json").read_text())
        if match_plans:
            if batching != "manifest":
                raise ValueError("matched-plan interleaving requires manifest batching")
            original = json.loads((match_plans / (case["name"] + ".json")).read_text())
            check_plan_config(original, manifest, original["batching"], microbatches,
                              warmup_steps, measured_steps, sequence_limit)
            plan = interleave_input_plan(original)
        else:
            source = ManifestBatches(str(model_dir), manifest, ROOT, batching=batching)
            plan = make_input_plan(source, manifest, audit, microbatches, warmup_steps,
                                   measured_steps, sequence_limit)
        plans.append((case["name"], plan))
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, plan in plans:
        (output_dir / (name + ".json")).write_text(json.dumps(plan, indent=2) + "\n")
    report = {"pass": True, "batching": batching,
              "cases": [{"case": name, "sample_order_sha256": plan["sample_order_sha256"],
                         "sample_multiset_sha256": plan["sample_multiset_sha256"],
                         "distinct_measured_samples": plan["distinct_measured_samples"],
                         "expected_distinct_samples": plan["expected_distinct_samples"]}
                        for name, plan in plans]}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--batching", choices=("bucket", "manifest"), default="bucket")
    p.add_argument("--microbatches", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--measured-steps", type=int, default=8)
    p.add_argument("--sequence-limit", type=int, default=1024)
    p.add_argument("--cases", default="")
    p.add_argument("--match-plans", type=Path,
                   help="Interleave the exact sample occurrences from these source plans")
    args = p.parse_args()
    print(json.dumps(prepare(args.suite_root, args.output_dir, args.model_dir,
                             args.batching, args.microbatches, args.warmup_steps,
                             args.measured_steps, args.sequence_limit,
                             list(filter(None, args.cases.split(","))), args.match_plans)))
