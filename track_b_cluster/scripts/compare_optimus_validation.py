"""Compare paired GPU smoke runs; checks sampled values, not full tensors."""
import argparse
import json
import math
from pathlib import Path


def compare(root, rtol=1e-3, atol=1e-5):
    reference, candidate = root / "serial-reference", root / "optimus-pp"
    configs = [json.loads((p / "run_config.json").read_text()) for p in (reference, candidate)]
    errors = []
    for key in ("manifest_sha256", "model_config_sha256", "code_sha256", "optimizer",
                "optimizer_device", "llm_parallelism", "microbatches", "layers", "steps"):
        if configs[0][key] != configs[1][key]:
            errors.append("configuration mismatch: " + key)
    rows = [[json.loads(s) for s in (p / "metrics.jsonl").read_text().splitlines()]
            for p in (reference, candidate)]
    if len(rows[0]) != len(rows[1]) or not rows[0]:
        raise ValueError("different or empty step counts")
    maxima = {"loss": 0.0, "grad_norm": 0.0, "parameter": 0.0, "used_gradient": 0.0}
    values_checked = {"parameter": 0, "used_gradient": 0}
    def check(a, b, field, where):
        maxima[field] = max(maxima[field], abs(a - b))
        if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            if len(errors) < 30:
                errors.append("%s: %s differs (%g vs %g)" % (where, field, a, b))
    for a, b in zip(*rows):
        step = a["step"]
        if a["sample_ids"] != b["sample_ids"] or step != b["step"]:
            errors.append("sample/step alignment differs")
        for field in ("loss", "grad_norm"):
            check(a[field], b[field], field, "step%d" % step)
        for rank in range(configs[0]["llm_parallelism"]["pp"]):
            name = "validation-step%03d-rank%d.json" % (step, rank)
            av, bv = [json.loads((p / name).read_text()) for p in (reference, candidate)]
            if av.keys() != bv.keys():
                errors.append("parameter names differ: " + name)
                continue
            for parameter in av:
                if av[parameter]["shape"] != bv[parameter]["shape"]:
                    errors.append("parameter shapes differ: " + parameter)
                for field in ("parameter", "used_gradient"):
                    aa, bb = av[parameter][field], bv[parameter][field]
                    if len(aa) != len(bb):
                        errors.append("sample count differs: " + parameter)
                    for x, y in zip(aa, bb):
                        values_checked[field] += 1
                        check(x, y, field, "%s/%s" % (name, parameter))
    for p in (reference, candidate):
        if json.loads((p / "status.json").read_text()).get("status") != "completed":
            errors.append("incomplete run: " + str(p))
    plans = [json.loads(p.read_text()) for p in sorted(candidate.glob("plan-*.json"))]
    return {"pass": not errors, "rtol": rtol, "atol": atol,
            "comparison_scope": "loss, global gradient norm and sampled parameters/used gradients",
            "max_abs_difference": maxima, "values_checked": values_checked,
            "steps": len(rows[0]), "encoder_owners": [p["owners"] for p in plans],
            "optimizer_device": configs[0]["optimizer_device"],
            "full_optimus_reproduction": False, "errors": errors}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.root, args.rtol, args.atol)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
