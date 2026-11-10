"""Template for the real Track B Qwen2-VL MCore/BigMac runner (Phase 2).

Copy/adapt this on the cluster where BigMac+Megatron-Core are installed. Fill in
the `# CLUSTER:` sections. Keep the CLI contract and metrics JSONL schema stable.
"""

import argparse
import json
import os
import time
from pathlib import Path


def append_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--metrics-jsonl", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=2)
    args = p.parse_args()

    use_pp = int(os.environ.get("USE_PP_EXECUTOR", "0"))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "status": "template_only",
                "use_pp_executor": use_pp,
                "manifest": str(args.manifest.resolve()),
                "model_dir": str(args.model_dir.resolve()),
                "note": "Fill CLUSTER sections: MCore init, Qwen2-VL bridge, data loader, loss mask parity with part_a, PP toggle, checkpoint loader.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # CLUSTER: initialize distributed/MCore/BigMac runtime here.
    # CLUSTER: load Qwen2-VL HF checkpoint into MCore model (verify compatibility).
    # CLUSTER: build data iterator from manifest samples (N=2, fixed order both modes).
    # CLUSTER: build visual inputs: pixel_values, image_grid_thw per sample/batch.
    # CLUSTER: apply Part A loss masking (mask prompt/image slots/padding; >=2 assistant targets).
    # CLUSTER: if use_pp==1 enable BigMac pp_executor plan execution; if 0 use native path.

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        # CLUSTER: run one optimizer step; capture loss, skipped/nan, per-rank peak memory.
        # torch.cuda.reset_peak_memory_stats(); ...; max_mem = {rank: torch.cuda.max_memory_allocated(rank)}
        # CLUSTER: if PP profiler available, record pipeline_bubble_ratio and stall/encoder timers; else omit.
        step_time = time.perf_counter() - t0
        append_metric(
            args.metrics_jsonl,
            {
                "step": step,
                "warmup": step <= args.warmup_steps,
                "step_time_s": step_time,
                "samples": 0,  # CLUSTER: global samples in optimizer step
                "max_memory_allocated_bytes_by_rank": {},  # CLUSTER
                "loss": None,  # CLUSTER
                "skipped": False,
                "nan_or_inf": False,
                "use_pp_executor": use_pp,
                # pipeline_bubble_ratio / encoder_latency_s / microbatch_stall_s: only when instrumented
            },
        )
    raise SystemExit(
        "runner_template.py is a scaffold: fill CLUSTER sections on the BigMac+MCore cluster runtime."
    )


if __name__ == "__main__":
    main()
