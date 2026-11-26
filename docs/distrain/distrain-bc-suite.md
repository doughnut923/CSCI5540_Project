# B/C suite on the DistTrain arm

Six Part B / Part C workloads, run on the disaggregated encoder + LLM arm. This closes the item
`track_b_cluster/HETEROGENEITY.md` lists as remaining: *"run the same manifests and ordering on …
the chosen Megatron/DistTrain backend, with matched optimizer/offload/precision settings"*.

The manifests are the ones PR #5 ran — regenerated here and byte-identical (`manifest_sha256`
matches its recorded audit) — so the inputs are shared. The numbers below are this arm's.

## What ran

- **Resolution axis: the full Part B datasets** — 750 images each, 375 two-image samples per case,
  all images used once (`$WS/runs/partb_suite_v1/`).
- **Image-count axis: a 12-sample pilot** from the eight-photo Part C pool. Scaling it repeats the
  same eight photographs, so it adds samples, not information.
- `ENC_GPUS=1 + LLM_PP=2` = **3 GPUs**, **28 layers**, **M=2**, 1 warmup, seeded init, AdamW on GPU.
  Depth, microbatch count and warmup are matched to the Optimus pilot; the placement, the optimizer
  device and the step budget are not (see caveats).
- Samples are bucketed by `(visual tokens, image count)`, so every step has one sequence length.
- **Every sample of every case is consumed**: 375/375 for the resolution datasets (375 steps) and
  12/12 for the image-count pilot (8 steps). At six steps two pilot cases reached only 10 of 12,
  because the plan advances one bucket per step and five of the six cases have three or four
  buckets.

## Results: six-case suite (12 samples per case)

The three `resolution_*` rows are ratio-matched pilots: 12 two-image samples that reproduce Part B's
image-level resolution ratios on 24 of its 750 images. The full datasets are measured separately
below, and those numbers supersede these three rows.

| case | images | seq lengths | step mean / p50 / p95 (s) | samples/s | bubble | peak | encoder fwd | samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resolution_low *(pilot)* | 2 | 195 | 0.496 / 0.498 / 0.510 | 4.03 | 41.2 % | 18.7 GiB | 81 ms | 12 / 12 |
| resolution_medium *(pilot)* | 2 | 139, 195, 254 | 0.664 / 0.543 / 1.136 | 3.01 | 44.6 % | 18.7 GiB | 114 ms | 12 / 12 |
| resolution_high *(pilot)* | 2 | 83, 254 | 0.626 / 0.513 / 1.067 | 3.19 | 43.4 % | 18.7 GiB | 109 ms | 12 / 12 |
| image_count_low | 1 | 607 | 0.783 / 0.753 / 0.928 | 2.55 | 46.7 % | 18.7 GiB | 119 ms | 12 / 12 |
| image_count_medium | 1, 2, 3 | 607, 609, 611 | 0.823 / 0.765 / 1.123 | 2.43 | 46.9 % | 18.7 GiB | 114 ms | 12 / 12 |
| image_count_high | 1, 2, 4, 8 | 607, 609, 613, 621 | 0.808 / 0.719 / 1.115 | 2.47 | 46.4 % | 18.7 GiB | 116 ms | 12 / 12 |

All six pass: no skipped or non-finite steps, loss finite, and both the LLM and the encoder weight
norms move. `summary_s8.json` next to the records holds the full breakdown; gate
`track_b_cluster/scripts/report_bc_suite_distrain.py` reproduces the table and exits non-zero if any
case regresses.

Records: `$WS/runs/bc_suite_v1/runs_s8/disagg_m2_bc_<case>.jsonl`, summary
`$WS/runs/bc_suite_v1/summary_s8.json`. The earlier six-step run is kept in `runs/` as the
superseded record; its p50s are within 3 % of these.

## Results: Part B resolution datasets, full size

The real experiment: `part_b_image_res/variance_datasets/dataset_{Low,Medium,High}.json` — **750
images each**, equal mean token cost, different variance. Consecutive images are paired into
two-image samples, so every case has **375 samples that use all 750 images exactly once**; the
builder refuses a pairing that would change the image-level resolution mix.

| dataset | resolutions | per-sample tokens mean / var | step mean / p50 / p95 (s) | tokens/s | samples/s | bubble | encoder fwd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Low | 750 × 256 | 162.0 / 0 | 0.511 / 0.510 / 0.534 | **634** | 3.91 | 40.3 % | 80 ms |
| Medium | 250 × 128, 375 × 256, 125 × 384 | 163.0 / 6 785 | 0.544 / 0.528 / 0.610 | 599 | 3.67 | 41.4 % | 83 ms |
| High | 500 × 128, 250 × 384 | 164.0 / 13 516 | 0.550 / 0.526 / 0.613 | 596 | 3.64 | 41.4 % | 84 ms |

375 measured steps per case — the bucket plan needs 188 (Low), 372 (Medium) and 255 (High) to reach
every sample, so one budget covers all three. All three pass: no skipped or non-finite step, both
weight norms move, 375 / 375 distinct samples consumed.

Cost per step length, pooled over the three cases:

| step length | 83 | 139 | 195 | 254 | 310 | 425 |
| --- | --- | --- | --- | --- | --- | --- |
| step p50 (s) | 0.516-0.521 | 0.520 | 0.510-0.515 | 0.517-0.523 | 0.548 | 0.604-0.606 |

**Reading.** At equal mean visual tokens (162 / 163 / 164 per sample), the high-variance dataset is
**7.6 % slower per step** than the uniform one (0.550 vs 0.511 s) and delivers **6 % fewer tokens per
second**. The per-length table says why: step cost is flat from 83 to 254 tokens — the fixed
per-step floor, about 0.51 s, dominates — and only rises at 310 (+7 %) and 425 (+18 %). So a mixture
pays full price for its long steps and saves almost nothing on its short ones. The same length costs
the same in every case (195 tokens: 0.510 s Low, 0.515 s Medium), which is the control that the
difference comes from the length mix, not from the dataset.

This is a *long-tail* penalty: the variance shows up across steps because the arm buckets. A batch
that mixes lengths inside one step — what PR #5's Optimus runtime does — would price the same
variance differently, and is the remaining measurement for that arm, not this one.

Records: `$WS/runs/partb_suite_v1/runs/disagg_m2_partb_resolution_full_<level>.jsonl`, summary
`$WS/runs/partb_suite_v1/summary.json`.

## Results: Part C image-count cases

Part C fixes the visual-token budget at **576 per sample** and varies how many images carry it:
each image is padded to 576/N tokens (`padded_control`). Sequence length moves only by the two
marker tokens per extra image — 607, 609, 611, 613, 621 for N = 1, 2, 3, 4, 8.

Per-count cases, 16 samples each, all consumed:

| N | seq len | step p50 | samples/s | encoder fwd (median) | per image | per-image gradient checks |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 607 | 0.782 s | 2.56 | 105.2 ms | 105.2 ms | 16 / 16 |
| 2 | 609 | 0.728 s | 2.65 | 99.3 ms | 49.7 ms | 32 / 32 |
| 3 | 611 | 0.713 s | 2.81 | 99.8 ms | 33.3 ms | 48 / 48 |
| 4 | 613 | 0.711 s | 2.70 | 100.3 ms | 25.1 ms | 64 / 64 |
| 8 | 621 | 0.729 s | 2.65 | 104.1 ms | 13.0 ms | 128 / 128 |

The three mixes PR #5 ran, byte-identical manifests, 12 samples each:

| case | N | step p50 | samples/s | encoder fwd (median) | gradient checks |
| --- | --- | --- | --- | --- | --- |
| image_count_low | 1 | 0.800 s | 2.40 | 112.1 ms | 16 / 16 |
| image_count_medium | 1, 2, 3 | 0.804 s | 2.07 | 105.3 ms | 30 / 30 |
| image_count_high | 1, 2, 4, 8 | 0.753 s | 2.34 | 105.3 ms | 60 / 60 |

**Reading.** At a fixed token budget the encoder cost follows **tokens, not images**: the forward
stays 99-105 ms across N = 1..8 (±3 %), so per-image cost falls as 1/N and any per-image overhead is
below the ~10 ms noise floor of an eight-step run. Step p50 lands in 0.711-0.782 s with no
resolvable trend — the 70 ms spread is the same run-to-run node variance seen in the resolution
datasets, not an image-count effect. What dominates a step is fixed: encoder backward 47-49 ms per
interval, LLM 44-56 ms, optimizer 108 ms in the encoder unit.

**Every image reaches the model.** The encoder unit records one gradient norm per image slot from the
embedding gradient it receives back: 16 checks at N=1 up to 128 at N=8, all finite and positive,
including the eight-image samples. Part C listed `gradients` as unmeasured; this closes it on this
arm, and matches the shape of the check the Optimus pilot ran (its 12 / 24 / 45).

One measurement artifact, stated rather than smoothed: in six of the eight cases a single encoder
forward of 0.32-0.37 s appears at the same place (second microbatch of the seventh measured step),
three times the median. Cause not identified — it is why the mean (121-126 ms in the mixes) sits
above the median and why p95 exceeds p50. Quote the median.

Records: `$WS/runs/partc_suite_v1/runs/disagg_m2_pc_<case>.jsonl`, summary
`$WS/runs/partc_suite_v1/summary.json`.

## Encoder DP=2 on both axes

Same cases, `ENC_GPUS=2 + LLM_PP=2` (4 GPUs) instead of `ENC_GPUS=1 + PP=2` (3 GPUs), 28 layers, M=2:

| case | DP=1 step p50 | DP=2 step p50 | DP=2 samples/s | all-reduce (median) | share of step | throughput change |
| --- | --- | --- | --- | --- | --- | --- |
| resolution_full_high | 0.526 s | 0.946 s | 2.08 | 468 ms | 49 % | **-43 %** |
| image_count_high | 0.753 s | 1.075 s | 1.74 | 451 ms | 42 % | **-26 %** |

The all-reduce costs ~0.45-0.47 s per step on **both** axes, so it is a fixed tax rather than a
function of the workload: half of a 0.95 s step on the resolution dataset, two-fifths of a 1.15 s
step on the image-count case. The encoder forward itself is not slower (86 ms / 101 ms median per
microbatch — each replica holds half of them), and nothing is lost: 1020 / 1020 and 60 / 60
per-image gradient checks pass. Encoder DP buys no throughput here and costs 26-43 % of it.

Records: `$WS/runs/partb_suite_v1/runs_dp2/`, `$WS/runs/partc_suite_v1/runs_dp2/`.

## Encoder DP=4 at 8 GPUs (`ENC=4 + PP=4`, M=4)

Five gated runs, 28 layers, M=4 (M must be a multiple of `ENC_GPUS`, so DP=4 needs M=4):

| case | steps | step p50 | samples/s | encoder fwd (median) | all-reduce (median) | bubble | peak | per-image checks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resolution_full_low | 94 | 1.702 s | 2.33 | 87 ms | 1 208 ms | 51.2 % | 14.0 GiB | 752 |
| resolution_full_medium | 186 | 1.733 s | 2.27 | 86 ms | 1 224 ms | 51.4 % | 14.0 GiB | 1 488 |
| resolution_full_high | 129 | 1.735 s | 2.29 | 83 ms | 1 218 ms | 51.4 % | 14.0 GiB | 1 032 |
| image_count_high | 8 | 1.870 s | 2.02 | 102 ms | 1 234 ms | 53.1 % | 14.5 GiB | 90 |
| image_count_medium | 8 | 1.919 s | 1.97 | 104 ms | 1 261 ms | 53.1 % | 14.5 GiB | 60 |

**The bubble is flat at 51-53 % across five very different workloads** (162 to 621 tokens, 1 to 8
images), which is the point: it is structural, not a property of the data. The collective is 67-72 %
of all recorded work, and it takes the same ~1.2 s whatever the case; the four pipeline stages are
finished by 0.5-0.6 s of a 1.7-1.9 s step and idle for the rest. Against DP=1 on the same cases the
bubble rises 8-10 points (resolution_low 41.2 -> 51.2, resolution_high 43.4 -> 51.4) and throughput
falls 26-43 % — with the standing caveat that a bubble over eight ranks is not the same quantity as
one over three, so compare within a rank count.

Figure: `dp4_schedules.png` — one representative step per case, stacked, each chosen as the step
closest to that run's median step time. The green optimizer blocks on the encoder rows sit inside
the grey collective: `all_reduce` is asynchronous on the host, so AdamW is *submitted* before the
collective completes, and the plot places intervals at host-submission time plus GPU duration. GPU
execution is stream-ordered, so the reduction still finishes before the gradients are used.

`image_count_low` is the one case with no DP=4 row; it ran at DP=1 and DP=2 instead.

## Depth at 8 GPUs: `ENC=1 + PP=7` vs `ENC=2 + PP=6` (M=4)

| dataset | config | step p50 | samples/s | encoder rank 0 busy | last stage busy | bubble |
| --- | --- | --- | --- | --- | --- | --- |
| resolution_low | `ENC=1 + PP=7` | **0.791 s** | **4.99** | 79 % | 19 % | 77.3 % |
| resolution_low | `ENC=2 + PP=6` | 0.986 s | 3.76 | 86 % | 16 % | 68.4 % |
| resolution_medium | `ENC=1 + PP=7` | **0.828 s** | **4.66** | 80 % | 18 % | 77.7 % |
| resolution_medium | `ENC=2 + PP=6` | 1.001 s | 3.91 | 86 % | 16 % | 68.2 % |
| resolution_high | `ENC=1 + PP=7` | **0.821 s** | **4.52** | 78 % | 19 % | 77.9 % |
| resolution_high | `ENC=2 + PP=6` | 1.012 s | 3.90 | 85 % | 16 % | 68.4 % |

`PP=6` uses the uneven split 4|5|5|5|5|4 (`PP_FIRST_LAYERS=4`, `PP_LAST_LAYERS=4`) — the only way to
run six stages on 28 layers.

- **`ENC=1 + PP=7` is 16-33 % faster**, and the whole deficit is the collective: 416-431 ms per step in
  the `ENC=2` rows, zero in the `ENC=1` rows.
- **The arm is encoder-bound in both.** The encoder rank is the busiest rank (78-86 %) while the last
  pipeline stage is only 16-19 % busy: four microbatches cannot fill six or seven stages, so the extra
  depth is over-provisioned.
- **The bubble ratio ranks these backwards** (77-78 % for the faster config, 68 % for the slower one),
  because a rank sitting in an all-reduce is counted as busy. Bubble must never be compared across
  different `ENC_GPUS`; quote step time or throughput.
- **Encoder DP buys memory, not speed.** At M=4 the image-count cases (576-token samples) exceed the
  single encoder's budget: `ENC=1` hit 23.54 GiB of 23.56 and OOMed, while `ENC=2` ran at 17.5 GiB
  because each replica holds half the microbatches.

### Image count at M=2, where both configs fit

| dataset | `ENC=1+PP=7` mean / p50 | samples/s | `ENC=2+PP=6` mean / p50 | samples/s | all-reduce |
| --- | --- | --- | --- | --- | --- |
| image_count_low | **0.912 / 0.869 s** | **2.19** | 1.416 / 1.411 s | 1.41 | 800 ms |
| image_count_medium | **0.996 / 0.809 s** | **2.01** | 1.788 / 1.404 s | 1.12 | 800 ms |
| image_count_high | **0.997 / 0.778 s** | **2.01** | 1.606 / 1.420 s | 1.25 | 797 ms |

`ENC=1 + PP=7` is **1.6-1.8x faster** on every image-count dataset, on both mean and p50. Bubble is
82 % for the winner and 74-75 % for the loser, again the wrong way round.

Two honest notes on this table. The steps are noisy (sd 0.5-1.1 s) because of the encoder stall
described below, so the mean sits 15-20 % above the p50; and the collective measured **800 ms at M=2
against 420-430 ms at M=4 on the same 2.6 GB of gradients**, which is unexplained — most likely
sensitivity of a host-staged reduction with no P2P to whichever co-tenants share the node (the same
exchange measured 0.41 s and 1.13 s on different days). Compare the pair within a session, not the
absolute sync cost across them.

Figure: `enc1pp7_schedules.png` — one representative step per resolution dataset, stacked.

## What the numbers say

- **Step time is set by sequence length, not by image count or resolution.** The C cases carry
  3.6x the visual tokens of the B cases (576 vs ~162) and are 1.5x slower per step, but the encoder
  forward grows only 81 -> 119 ms. The pipeline's LLM work and the sequence length do the rest. None
  of this is an argument for or against disaggregation; it is the shape of the load the suite
  produces.
- **Bubble sits at 40-47 %** for all six, close to the 30 % the pilot manifest gives at
  `ENC=1 + PP=2` and well under the 58-60 % of the 4-5 GPU rows. Three GPUs is the config where the
  encoder unit is still the pipeline's critical path.
- **The heterogeneity the suite exists to expose does not reach the step.** Bucketing gives one
  length per step, so what varies within a case is which length each step gets, not the mix inside a
  step. That is why p95 sits above p50 in four of the six rows: each row pools steps of different
  lengths.

## Why this arm buckets (measured, not assumed)

The suite's own convention keeps manifest order and lets a step mix lengths. This arm cannot. The
probe — `image_count_high`, manifest order, the 607-token and 609-token samples in one
`ENC=1 + PP=2` step — dies in the language model:

```
RuntimeError: The size of tensor a (609) must match the size of tensor b (607) at non-singleton
dimension 0        # PMP qwen2_vl/rotary_pos_embedding.py:213, apply_rotary_pos_emb_bshd
```

The rotary table is built for the step's first microbatch and then reused, so one step means one
length in this MCore/PMP path. PR #5's Optimus prototype sidesteps this with its own executor rather
than MCore's pipeline schedule — that is the difference between the arms, not a choice about how to
batch. Job 29389, cancelled after the traceback.

Consequence of bucketing: coverage depends on the step budget. Six measured steps (the Optimus
pilot's budget) reaches only 10 of 12 samples in `resolution_high` and `image_count_high`, because
the plan advances one bucket per step and those cases have four. Eight steps covers all twelve
everywhere, which is what the table reports; the six-step records stay in `runs/` as the superseded
run.

## Caveats

1. **Eight repeated photographs** behind the 12-sample pilot cases. The resolution datasets use 750
   ImageNet previews but reuse source files above 250 images per resolution (Part B's own
   convention), and every target text is a fixed sentence — these are system checks, not a
   caption dataset.
2. **Seeded random init, no checkpoint.** Loss is not comparable across cases, or with the Optimus
   pilot's losses. Only timing, memory and input coverage are.
3. **Not like-for-like with PR #5's Optimus rows.** Ours: 3 GPUs, GPU-resident AdamW, MCore pipeline,
   bucketed order, 375 steps. Theirs: 2 GPUs, CPU-offloaded AdamW, their own executor, manifest
   order, 6 steps on 12 samples. Depths (28), microbatch count (2) and the warmup are matched; the
   rest is not, so no cross-arm throughput claim is made here.
4. **The variance result is a long-tail result.** It is measured on steps, because this arm cannot
   mix lengths inside a step (see the probe below). A runtime that batches the mixture differently
   would price it differently; the Optimus arm is where that comparison belongs. Part B's own
   encoder-runtime spread (72.9 %) also means the three mixtures are not matched in encoder work,
   only in encoder tokens.
5. **Bubble here is defined over three ranks** (1 encoder + 2 stages). It is not the same quantity
   as the 4-5 GPU rows in `distrain-results.md`; do not put them in one column.
6. **Eight measured steps for the pilot, 375 for the datasets.** p50 is the number to quote among the
   pilot rows; p95 moves with which lengths the row pools. Repeating `image_count_high` on a second
   allocation gave p50 0.717 s against 0.725 s (-1.1 %) and a mean 0.8 % higher, with the same 12
   distinct samples, so the plan and the timing are stable.

## Reproduce

```sh
# once: the six-case pilot manifests, then the full Part B datasets (container CPU, no GPU)
singularity exec -B /research:/research -B /tmp:/tmp $WS/containers/ngc-pytorch-25.03-py3.sif \
  python3 track_b_cluster/scripts/build_heterogeneity_suite.py \
    --output-dir $WS/runs/bc_suite_v1 --samples-per-case 12 \
    --model-dir $WS/models/Qwen2-VL-2B-Instruct
singularity exec -B /research:/research -B /tmp:/tmp $WS/containers/ngc-pytorch-25.03-py3.sif \
  python3 track_b_cluster/scripts/build_part_b_suite.py \
    --output-dir $WS/runs/partb_suite_v1 --images-per-sample 2 \
    --model-dir $WS/models/Qwen2-VL-2B-Instruct

# one job per case (three GPUs each; the QOS allows four jobs at a time)
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
for case in resolution_low resolution_medium resolution_high \
            image_count_low image_count_medium image_count_high; do
  sbatch --gres=gpu:3 --export=ALL,ENC_GPUS=1,LLM_PP=2,NUM_LAYERS=28,NUM_MICROBATCHES=2,\
BENCH_STEPS=8,WARMUP_STEPS=1,TAG=bc_$case,MANIFEST_BATCHING=bucket,\
MANIFEST=$WS/runs/bc_suite_v1/$case/manifest.json,OUT_DIR=$WS/runs/bc_suite_v1/runs_s8 \
    track_b_cluster/slurm/disagg_units.sbatch
done
for level in low medium high; do
  sbatch --gres=gpu:3 --export=ALL,ENC_GPUS=1,LLM_PP=2,NUM_LAYERS=28,NUM_MICROBATCHES=2,\
BENCH_STEPS=375,WARMUP_STEPS=1,TAG=partb_resolution_full_$level,MANIFEST_BATCHING=bucket,\
MANIFEST=$WS/runs/partb_suite_v1/resolution_full_$level/manifest.json,\
OUT_DIR=$WS/runs/partb_suite_v1/runs track_b_cluster/slurm/disagg_units.sbatch
done

# summary + gates
singularity exec -B /research:/research -B /tmp:/tmp $WS/containers/ngc-pytorch-25.03-py3.sif \
  python3 track_b_cluster/scripts/report_bc_suite_distrain.py \
    --suite-root $WS/runs/partb_suite_v1 --runs-root $WS/runs/partb_suite_v1/runs \
    --output $WS/runs/partb_suite_v1/summary.json
```

Every job fits on one node (3 GPUs of the 8) and the launcher draws per-job rendezvous ports, so
several cases can share a node. 375 steps is about five minutes of measured time plus one minute of
initialisation per case.
