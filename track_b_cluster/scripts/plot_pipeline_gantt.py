"""Draw the pipeline utilization diagram for one benchmark step.

x = time since the step started, y = pipeline stage (PP rank). Each recorded busy
interval is a filled bar, labelled with its microbatch: F<k> forward, B<k> backward.
Encoder and LLM backwards get different colours when the record is a disaggregated
run (`enc_world` tells the encoder ranks apart). Requires the `busy_intervals` field
that pp2_equivalence.py / disagg_units.py write.

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

COLORS = {"enc": "tab:red", "llm": "tab:blue", "enc_bwd": "tab:brown",
          "llm_bwd": "tab:orange", "bwd": "tab:orange", "sync": "tab:gray",
          "fwd": "tab:gray"}
NAMES = {"enc": "encoder fwd", "llm": "LLM fwd", "enc_bwd": "encoder bwd",
         "llm_bwd": "LLM bwd", "bwd": "backward", "sync": "grad all-reduce",
         "fwd": "forward"}
ORDER = ("enc", "llm", "fwd", "enc_bwd", "llm_bwd", "bwd", "sync")
FORWARD = ("enc", "llm", "fwd")


def load_step(path, step):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("step") == step:
                return row
    raise SystemExit("step %d not found in %s" % (step, path))


def split_backwards(row, intervals):
    """Disaggregated runs: rank < enc_world is the encoder unit, the rest is the LLM."""
    base = int(row.get("enc_world") or 0)
    if not base:
        return intervals
    out = []
    for iv in intervals:
        rank, kind = int(iv[0]), iv[1]
        if kind == "bwd":
            kind = "enc_bwd" if rank < base else "llm_bwd"
        out.append([rank, kind] + list(iv[2:]))
    return out


def label_blocks(row, intervals):
    """F<k>/B<k> per bar. Uses the recorded microbatch id when present, else the
    position in that rank's stream (1F1B runs each rank's microbatches in order); the
    encoder unit's k-th bar on rank r is microbatch r + k*enc_world."""
    base = int(row.get("enc_world") or 0)
    counted = {}
    labelled = []
    for iv in sorted(intervals, key=lambda x: (int(x[0]), x[2])):
        rank, kind, start, end = int(iv[0]), iv[1], iv[2], iv[3]
        if kind == "sync":  # not a microbatch block, no F/B label
            labelled.append([rank, kind, start, end, ""])
            continue
        mb = iv[4] if len(iv) > 4 else None
        key = (rank, kind in FORWARD)
        idx = counted.get(key, 0)
        counted[key] = idx + 1
        if mb is None or mb < 0:
            mb = (rank + idx * base if (kind in ("enc", "enc_bwd") and base)
                  else idx)
        prefix = "F" if kind in FORWARD else "B"
        labelled.append([rank, kind, start, end, "%s%d" % (prefix, mb)])
    return labelled


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

    intervals = label_blocks(row, split_backwards(row, intervals))
    step_time = float(row["step_time_s"])
    ranks = sorted({iv[0] for iv in intervals})
    labels = row.get("rank_labels") or {}
    # A wrong offset basis (e.g. absolute CUDA timestamps) would put bars outside
    # the step window; fail loudly instead of drawing a misleading chart.
    bad = [iv for iv in intervals if iv[2] < -1e-6 or iv[3] > step_time * 1.01]
    if bad:
        raise SystemExit("intervals outside [0, step_time]; bad record: %r" % (bad[0],))

    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.42 * len(ranks)))
    for rank, kind, start, end, text in intervals:
        if end - start <= 0:  # zero-width interval: nothing to draw
            continue
        ax.barh(rank, end - start, left=start, height=0.62,
                color=COLORS.get(kind, "tab:gray"), edgecolor="none")
        if end - start > 0.012 * step_time and text:  # only label readable bars
            ax.text((start + end) / 2, rank, text, ha="center", va="center",
                    fontsize=6.5, color="black", clip_on=True)

    busy = sum(iv[3] - iv[2] for iv in intervals)
    util = busy / (len(ranks) * step_time)
    bwd_total = sum(iv[3] - iv[2] for iv in intervals if "bwd" in iv[1])
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
    ax.legend(handles=[mpatches.Patch(color=COLORS[k], label=NAMES[k])
                       for k in ORDER
                       if any(iv[1] == k and iv[3] - iv[2] > 0 for iv in intervals)],
              loc="lower right", framealpha=0.9)

    out = args.output or args.input.with_name(
        "%s_step%d.png" % (args.input.stem, args.step))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(str(out))


if __name__ == "__main__":
    main()
