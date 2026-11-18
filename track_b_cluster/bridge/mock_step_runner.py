"""Mock Track B runner for metrics/SLURM pipeline testing (NOT performance evidence).

Emits step_metrics.jsonl in the contract expected by collect_step_metrics.py and
SLURM templates. Use only to validate wiring, aggregation, and file layout.
"""
import argparse
import json
import os
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metrics-jsonl", type=Path, required=True)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--samples-per-step", type=int, default=4)
    p.add_argument("--seed", type=int, default=55402026)
    args = p.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    use_pp = int(os.environ.get("USE_PP_EXECUTOR", "0"))
    rng = random.Random(args.seed + use_pp)

    config = {
        "mock": True,
        "warning": "Synthetic metrics; not training/performance evidence.",
        "manifest": str(args.manifest.resolve()),
        "manifest_samples": manifest.get("spec", {}).get("total_samples"),
        "use_pp_executor": use_pp,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_jsonl.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            # Synthetic timing: BigMac mock slightly lower mean to prove pipeline works; DO NOT interpret.
            base = 0.50 if use_pp else 0.55
            rec = {
                "step": step,
                "warmup": step <= args.warmup_steps,
                "step_time_s": round(base + rng.uniform(-0.03, 0.05), 6),
                "samples": args.samples_per_step,
                "max_memory_allocated_bytes_by_rank": {"0": 8_000_000_000 + rng.randint(0, 10_000_000)},
                "loss": round(2.0 / (step**0.3) + rng.uniform(-0.01, 0.01), 6),
                "skipped": False,
                "nan_or_inf": False,
                "use_pp_executor": use_pp,
                "mock": True,
            }
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({"mock": True, "use_pp_executor": use_pp, "steps": args.steps, "metrics_jsonl": str(args.metrics_jsonl)}))


if __name__ == "__main__":
    main()
