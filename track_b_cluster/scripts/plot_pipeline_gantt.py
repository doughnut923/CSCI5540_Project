"""Draw the pipeline utilization diagram for one benchmark step.

x = time since the step started, y = pipeline stage (PP rank). Each recorded busy
interval is a filled bar (vision encoder, language model, backward), so fill, steady
state and drain are visible as the staircase of bars. Requires the `busy_intervals`
field that pp2_equivalence.py writes in benchmark mode.

    python3 plot_pipeline_gantt.py --input runs/.../pp_native_scale_pp8-bench_m8.jsonl

Writes <input>_step<N>.png next to the JSONL.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

COLORS = {"enc": "tab:red", "llm": "tab:blue", "bwd": "tab:orange",
          "sync": "tab:gray", "fwd": "tab:gray"}
LABELS = {"enc": "vision encoder", "llm": "language model", "bwd": "backward",
          "sync": "grad all-reduce", "fwd": "forward"}


def load_step(path, step):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("step") == step:
                return row
    raise SystemExit("step %d not found in %s" % (step, path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="benchmark JSONL")
    p.add_argument("--step", type=int, default=0, help="measured step index")
    p.add_argument("--output", type=Path, default=None, help="PNG path")
    args = p.parse_args()

    row = load_step(args.input, args.step)
    intervals = row.get("busy_intervals")
    if not intervals:
        raise SystemExit(
            "%s has no busy_intervals; re-run the benchmark with the current "
            "pp2_equivalence.py" % args.input)

    step_time = float(row["step_time_s"])
    ranks = sorted({int(iv[0]) for iv in intervals})
    labels = row.get("rank_labels") or {}
    # A wrong offset basis (e.g. absolute CUDA timestamps) would put bars outside
    # the step window; fail loudly instead of drawing a misleading chart.
    bad = [iv for iv in intervals if iv[2] < -1e-6 or iv[3] > step_time * 1.01]
    if bad:
        raise SystemExit("intervals outside [0, step_time]; bad record: %r" % (bad[0],))

    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.42 * len(ranks)))
    for rank, kind, start, end in intervals:
        if end - start <= 0:  # zero-width interval: nothing to draw
            continue
        ax.barh(rank, end - start, left=start, height=0.62,
                color=COLORS.get(kind, "tab:gray"), edgecolor="none")

    busy = sum(iv[3] - iv[2] for iv in intervals)
    util = busy / (len(ranks) * step_time)
    bwd_total = sum(iv[3] - iv[2] for iv in intervals if iv[1] == "bwd")
    ax.set_yticks(ranks)
    ax.set_yticklabels([labels.get(str(r), "PP %d" % r) for r in ranks])
    ax.invert_yaxis()  # stage 0 at the top
    ax.set_xlim(0, step_time)
    ax.set_ylim(len(ranks) - 0.5, -0.5)
    ax.set_xlabel("time since step start (s)")
    ax.set_ylabel("pipeline stage")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(step_time, color="black", lw=0.8, ls="--")
    stage_desc = ("PP=%d + %d enc" % (row["pp_size"], row["enc_world"])
                  if row.get("mode") == "disaggregated" else "PP=%d" % len(ranks))
    ax.set_title(
        "step %d: %s, microbatches=%s, %.3f s  |  busy %.1f%%, bubble %.1f%%%s"
        % (row["step"], stage_desc, row.get("num_microbatches", "?"), step_time,
           100 * util, 100 * (1 - util),
           "" if bwd_total > 0.01 * busy else "  [backward not captured]"))
    ax.legend(handles=[mpatches.Patch(color=COLORS[k], label=LABELS[k])
                       for k in ("enc", "llm", "bwd", "sync", "fwd")
                       if any(iv[1] == k and iv[3] - iv[2] > 0 for iv in intervals)],
              loc="lower right", framealpha=0.9)

    out = args.output or args.input.with_name(
        "%s_step%d.png" % (args.input.stem, args.step))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(str(out))


if __name__ == "__main__":
    main()
