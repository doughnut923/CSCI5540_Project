# B/C suite on the DistTrain arm — 8-GPU results

Eight RTX 3090s, 28 layers, M=4, one warmup, seeded init, AdamW over MCore's fp32 `main_grad`
(clip 1.0), sequences bucketed by `(visual tokens, image count)`. Every row is a real training
step: forward, backward, optimizer update. Runs on fewer GPUs are side reference only (§ Side
reference) and are not part of this comparison.

The manifests are the ones PR #5 ran, regenerated here and byte-identical (`manifest_sha256`
matches its recorded audit), so the inputs are shared. The numbers are this arm's.

## What ran

- **Part B, full size**: 750 images per case, 375 two-image samples, all images used once
  (`runs/partb_suite_v1/`). 94 / 186 / 129 measured steps for Low / Medium / High; 375/375 samples
  consumed in every row.
- **Part C, 12-sample pilot**: `image_count_low` (N=1), `image_count_medium` (1,2,3),
  `image_count_high` (1,2,4,8), eight photographs, 576 visual tokens per sample. 8 measured steps,
  12/12 samples consumed. Scaling repeats the same photos, so it adds samples, not information.
- `ENC=1 + PP=7`, `ENC=2 + PP=6` (uneven split 4|5|5|5|5|4) and `ENC=4 + PP=4` all run 8 GPUs.
- Part C at M=4 fits only with encoder DP: `ENC=1` OOMed at **23.54 of 23.56 GiB** on the 576-token
  samples, so its Part C rows are M=2.

## Part B, M=4

`p50 / mean step (s) · samples/s`

| dataset | `ENC=1 + PP=7` | `ENC=2 + PP=6` | `ENC=4 + PP=4` |
| --- | --- | --- | --- |
| resolution_full_low | **0.791 / 0.801 · 4.99** | 0.986 / 1.065 · 3.76 | 1.702 / 1.718 · 2.33 |
| resolution_full_medium | **0.824 / 0.859 · 4.66** | 1.001 / 1.022 · 3.91 | 1.733 / 1.760 · 2.27 |
| resolution_full_high | **0.821 / 0.884 · 4.52** | 1.012 / 1.027 · 3.90 | 1.735 / 1.747 · 2.29 |

| config | bubble | encoder fwd (median) | all-reduce (median) | peak | steps |
| --- | --- | --- | --- | --- | --- |
| `ENC=1 + PP=7` | 77.3–77.9 % | 79–88 ms | — | 14.8–19.8 GiB | 94 / 186 / 129 |
| `ENC=2 + PP=6` | 68.2–68.4 % | 83–89 ms | 416–431 ms | 14.0–15.6 GiB | 94 / 186 / 129 |
| `ENC=4 + PP=4` | 51.2–51.4 % | 83–87 ms | 1 208–1 224 ms | 14.0 GiB | 94 / 186 / 129 |

**Reading.**

- **Placement dominates.** `ENC=1 + PP=7` is **16–33 % faster** than giving the encoder two GPUs, and
  the entire deficit is the collective: 416–431 ms per step in the `ENC=2` rows, zero in `ENC=1`.
  DP=4 adds a ~1.2 s collective per step and loses another 30–40 %.
- **Variance costs +10.4 %** from Low to High in `ENC=1` (0.801 → 0.859 → 0.884 s, 95 % CI ±0.011 /
  ±0.023 / ±0.031). With encoder DP the fixed collective dominates and the mixture's effect drops to
  noise (within 4 %).
- **The cause is the length mixture, not the dataset.** Cost is flat to ~254 tokens, then
  ~1.2 ms/token (0.795 → 1.007 s over 171 tokens). At equal length the three datasets agree within
  2 % (195 tokens: 0.791 Low vs 0.795 Medium; 425: 0.998 vs 1.007), so the curves differ only in how
  much of each length they contain.
- **Bubble ranks the configs backwards** (77–78 % for the fastest, 51 % for the slowest) because a
  rank blocked in an all-reduce counts as busy. Quote step time or throughput, never bubble across
  `ENC_GPUS`.

Records: `runs/partb_suite_v1/runs_sweep/disagg_m4_enc{1pp7,2pp6}_res_<level>.jsonl`,
`runs/partb_suite_v1/runs_enc4/disagg_m4_8g4_res_<level>.jsonl`.

## Part C, M=4

| case | `ENC=2 + PP=6` p50 / mean · samples/s | `ENC=4 + PP=4` p50 / mean · samples/s |
| --- | --- | --- |
| image_count_low | **1.236 / 1.266 · 3.16** | — (not run) |
| image_count_medium | **1.225 / 1.392 · 2.87** | 1.916 / 2.026 · 1.97 |
| image_count_high | **1.184 / 1.374 · 2.91** | 1.868 / 1.980 · 2.02 |

Bubble 69.1–70.2 % and 53.1 %; all-reduce 423–427 ms and 1 234–1 261 ms; peak 17.5 and 14.5 GiB.
`ENC=1 + PP=7` has no M=4 row (OOM above).

**M=2 reference, where `ENC=1` fits** (8 steps, same manifests):

| case | `ENC=1 + PP=7` mean / p50 · samples/s | `ENC=2 + PP=6` mean / p50 · samples/s |
| --- | --- | --- |
| image_count_low | **0.912 / 0.869 · 2.19** | 1.416 / 1.411 · 1.41 |
| image_count_medium | **0.996 / 0.809 · 2.01** | 1.788 / 1.404 · 1.12 |
| image_count_high | **0.997 / 0.778 · 2.01** | 1.606 / 1.420 · 1.25 |

Bubble 82.0 / 82.0 / 81.8 % and 74.2 / 75.2 / 75.1 %. The collective measured **800 ms at M=2
against 423–431 ms at M=4** on the same gradients — unexplained, most likely host-staged reduction
sensitivity to node co-tenants. Compare the pair within a session, not the absolute sync cost across
them.

**Reading.**

- **At a fixed token budget the encoder cost tracks tokens, not images.** The forward stays
  99–106 ms across N = 1…8 (±3 %), so per-image cost falls as 1/N and any per-image overhead is below
  the ~10 ms noise floor of an eight-step run.
- **Mixing image counts moves the step by +9 %** at M=2 (0.912 → 0.997 s) against the 1.6–1.8× gap
  between configs: placement dominates, the data mix is a single-digit effect. The `ENC=2` M=2 row is
  noisy (one `image_count_medium` step stalled, CI ±0.34 s) — read it as "no trend resolved".
- **Every image reaches the model.** The encoder unit records one gradient norm per image slot:
  32 / 60 / 90 checks at M=4 and 16…128 across the per-N grid, all finite and positive.

Records: `runs/partc_suite_v1/runs_sweep/disagg_m4_enc2pp6_pc_<level>.jsonl`,
`runs/partc_suite_v1/runs_enc4/disagg_m4_8g4_pc_<level>.jsonl`,
`runs/partc_suite_v1/runs_sweep/disagg_m2_*_m2.jsonl`.

## Figures

- `variance_three_configs.png` — step time and bubble vs resolution variance, three 8-GPU configs.
- `enc1pp7_variance_trend.png` — `ENC=1 + PP=7` alone, zoomed, with the per-length panel.
- `variance_two_axes.png` — both heterogeneity axes in one layout (row 2 is M=2; bottom-right is the
  3-GPU per-count curve).
- `enc1pp7_schedules.png` — one representative `ENC=1 + PP=7` step per resolution dataset.
- `dp4_schedules.png` — one step per case for `ENC=4 + PP=4`; the collective is 67–72 % of work.
- `enc2pp6_schedule_step17.png` — one `ENC=2 + PP=6` step, uneven split, all-reduce to the end.

Machine-readable results for every row: `sweep_results.json` (20 records: mean/p50/sd/min/max,
throughput, bubble, encoder and all-reduce medians, peak memory, samples and gradient checks, source
record path).

## Why this arm buckets

The suite's convention keeps manifest order and lets a step mix lengths. This arm cannot: the probe —
`image_count_high`, manifest order, the 607- and 609-token samples in one step — dies in the rotary
table:

```
RuntimeError: The size of tensor a (609) must match the size of tensor b (607)
    # PMP qwen2_vl/rotary_pos_embedding.py:213, apply_rotary_pos_emb_bshd
```

MCore/PMP builds the rotary table for the step's first microbatch and reuses it, so one step means
one length. Consequence: the step budget decides coverage. Six steps reached 10/12 samples in two
cases; eight steps reaches 12/12 everywhere, and 94 / 186 / 129 covers 375/375 on Part B.

## Side reference: non-8-GPU runs

Kept for continuity with the earlier write-ups. **Not part of the 8-GPU comparison** — different
GPU count, microbatch count and, for the fused row, no optimizer step.

| config | GPUs | what it measured | headline | records |
| --- | --- | --- | --- | --- |
| `ENC=1 + PP=2`, M=2 | 3 | full Part B (375) | 0.511 / 0.544 / 0.550 s, 3.91 / 3.67 / 3.64 samples/s; variance +7.6 % | `runs/partb_suite_v1/runs/` |
| `ENC=1 + PP=2`, M=2 | 3 | Part C per-count grid, 16 samples/N | 0.782 / 0.728 / 0.713 / 0.711 / 0.729 s p50 for N=1,2,3,4,8; encoder 99–105 ms | `runs/partc_suite_v1/runs/` |
| `ENC=1 + PP=2`, M=2 | 3 | six-case pilot, 12 samples | 0.496–0.823 s; superseded by the rows above | `runs/bc_suite_v1/runs_s8/` |
| `ENC=2 + PP=2`, M=2 | 4 | DP=2 on both axes | res_high 0.946 s / 2.08 s⁻¹, ic_high 1.075 s / 1.74 s⁻¹; all-reduce 0.45–0.47 s | `runs/*/runs_dp2/` |
| fused PP=4, no optimizer | 4 | fused baseline, same plan | 0.770 s p50 / 0.792 mean, 5.05 samples/s | `runs/track_b_n2_pilot_measured_v1/` |

The 5-GPU training-valid pilot rows and the placement-only topology rows live in
`distrain-results.md`; they are not used here either.

## Caveats

1. **Seeded random init, no checkpoint.** Loss is not comparable across cases or with other arms;
   only timing, memory and input coverage are.
2. **Eight repeated photographs** behind the 12-sample Part C cases. Part B uses 750 ImageNet
   previews but reuses source files above 250 images per resolution; every target text is a fixed
   sentence. These are system checks, not a quality result.
3. **Bubble is not comparable across placements.** It is computed over all ranks in the run, and a
   rank blocked in a collective counts as busy. Use throughput.
4. **Part C budget is 8 steps on 12 samples.** One stalled step moves the mean 8–16 % above the
   median (`image_count_high`: 1.184 vs 1.374 s); quote both.
5. **The variance effect is a long-tail effect**, measured across steps because this arm cannot mix
   lengths inside one step. A runtime that batches the mixture differently would price it
   differently.

## Reproduce

```sh
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf

# Part B, M=4. Swap the three exports for ENC=2 (add PP_FIRST_LAYERS=4,PP_LAST_LAYERS=4)
# or ENC=4 (ENC_GPUS=4,LLM_PP=4,OUT_DIR=.../runs_enc4).
for level in low medium high; do
  case $level in low) STEPS=94;; medium) STEPS=186;; high) STEPS=129;; esac
  sbatch --gres=gpu:8 --export=ALL,ENC_GPUS=1,LLM_PP=7,NUM_LAYERS=28,NUM_MICROBATCHES=4,\
BENCH_STEPS=$STEPS,WARMUP_STEPS=1,TAG=m4_enc1pp7_res_$level,\
MANIFEST=$WS/runs/partb_suite_v1/resolution_full_$level/manifest.json,\
OUT_DIR=$WS/runs/partb_suite_v1/runs_sweep track_b_cluster/slurm/disagg_units.sbatch
done

# Part C, M=4 with encoder DP (the only M=4 layout that fits)
for level in low medium high; do
  sbatch --gres=gpu:8 --export=ALL,ENC_GPUS=2,LLM_PP=6,PP_FIRST_LAYERS=4,PP_LAST_LAYERS=4,\
NUM_LAYERS=28,NUM_MICROBATCHES=4,BENCH_STEPS=8,WARMUP_STEPS=1,TAG=m4_enc2pp6_pc_$level,\
MANIFEST=$WS/runs/partc_suite_v1/image_count_$level/manifest.json,\
OUT_DIR=$WS/runs/partc_suite_v1/runs_sweep track_b_cluster/slurm/disagg_units.sbatch
done

# per-run summary inside the container image
python3 track_b_cluster/scripts/collect_step_metrics.py \
  --input <run>.jsonl --output <run>_summary.json --world-size 8
```

`M` must be a multiple of `ENC_GPUS`; that is why DP=4 runs at M=4 and why Part C `ENC=1` is M=2.
