# B/C suite on the DistTrain arm

Six Part B / Part C workloads, run on the disaggregated encoder + LLM arm. This closes the item
`track_b_cluster/HETEROGENEITY.md` lists as remaining: *"run the same manifests and ordering on …
the chosen Megatron/DistTrain backend, with matched optimizer/offload/precision settings"*.

The manifests are the ones PR #5 ran — regenerated here and byte-identical (`manifest_sha256`
matches its recorded audit) — so the inputs are shared. The numbers below are this arm's.

## What ran

- Six cases, 12 samples each, from `$WS/runs/bc_suite_v1/` (untracked, together with the
  `input_audit.json` per case; PR #5 carries the same audit as a committed record).
- `ENC_GPUS=1 + LLM_PP=2` = **3 GPUs**, **28 layers**, **M=2**, 1 warmup + **6 measured steps**,
  seeded init, AdamW on GPU. Depth, microbatch count and step budget are matched to the Optimus
  pilot; the placement and optimizer device are not (see caveats).
- Samples are bucketed by `(visual tokens, image count)`, so every step has one sequence length.

## Results

| case | images | seq lengths | step mean / p50 / p95 (s) | samples/s | bubble | peak | encoder fwd | samples seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resolution_low | 2 | 195 | 0.506 / 0.504 / 0.517 | 3.96 | 40.4 % | 18.7 GiB | 78 ms | 12 / 12 |
| resolution_medium | 2 | 139, 195, 254 | 0.648 / 0.507 / 1.140 | 3.09 | 42.9 % | 18.7 GiB | 107 ms | 12 / 12 |
| resolution_high | 2 | 83, 254 | 0.681 / 0.520 / 1.279 | 2.94 | 43.8 % | 18.7 GiB | 107 ms | 10 / 12 |
| image_count_low | 1 | 607 | 0.765 / 0.764 / 0.767 | 2.62 | 46.1 % | 18.7 GiB | 104 ms | 12 / 12 |
| image_count_medium | 1, 2, 3 | 607, 609, 611 | 0.817 / 0.737 / 1.132 | 2.45 | 46.6 % | 18.7 GiB | 101 ms | 12 / 12 |
| image_count_high | 1, 2, 4, 8 | 607, 609, 613, 621 | 0.794 / 0.715 / 1.094 | 2.52 | 46.0 % | 18.7 GiB | 98 ms | 10 / 12 |

All six pass: no skipped or non-finite steps, loss finite, and both the LLM and the encoder weight
norms move. `summary.json` next to the records holds the full breakdown; gate
`track_b_cluster/scripts/report_bc_suite_distrain.py` reproduces the table and exits non-zero if any
case regresses.

Records: `$WS/runs/bc_suite_v1/runs/disagg_m2_bc_<case>.jsonl`, summary
`$WS/runs/bc_suite_v1/summary.json`.

## What the numbers say

- **Step time is set by sequence length, not by image count or resolution.** The C cases carry
  3.6x the visual tokens of the B cases (576 vs ~162) and are 1.5x slower per step, but the encoder
  forward grows only 78 -> 104 ms. The pipeline's LLM work and the padding-free sequence length do
  the rest. None of this is an argument for or against disaggregation; it is the shape of the load
  the suite produces.
- **Bubble sits at 40-47 %** for all six, close to the 30 % the pilot manifest gives at
  `ENC=1 + PP=2` and well under the 58-60 % of the 4-5 GPU rows. Three GPUs is the config where the
  encoder unit is still the pipeline's critical path.
- **The heterogeneity the suite exists to expose does not reach the step.** Bucketing gives one
  length per step, so what varies within a case is which length each step gets, not the mix inside a
  step. That is why p95 sits far above p50 in four of the six rows: six steps is also too few for a
  stable tail.

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

Consequence, recorded honestly: bucket order means a six-step run does not necessarily draw all
twelve samples. `resolution_high` and `image_count_high` saw 10 of 12, so their observed mean visual
tokens (135.5 for `resolution_high`) is below the case's 164. The audit column in `summary.json`
keeps both numbers apart.

## Caveats

1. **Eight repeated photographs** behind 12 samples per case. An integration pilot, not an
   independent dataset and not a caption dataset — the target text is a fixed sentence.
2. **Seeded random init, no checkpoint.** Loss is not comparable across cases, or with the Optimus
   pilot's losses. Only timing, memory and input coverage are.
3. **Not like-for-like with PR #5's Optimus rows.** Ours: 3 GPUs, GPU-resident AdamW, MCore pipeline,
   bucketed order. Theirs: 2 GPUs, CPU-offloaded AdamW, their own executor, manifest order. Depths
   (28), microbatch count (2) and the 1+6 step budget are matched; the rest is not, so no
   cross-arm throughput claim is made here.
4. **The resolution cases cannot support a variance-only conclusion.** Their own measurement found a
   72.9 % spread in mean encoder runtime across the three resolution mixtures, and the token means
   are close by construction. We report the measured encoder cost; we do not attribute the
   differences to variance.
5. **Bubble here is defined over three ranks** (1 encoder + 2 stages). It is not the same quantity
   as the 4-5 GPU rows in `distrain-results.md`; do not put them in one column.
6. **Six measured steps, one repeat.** p50 is the number to quote; p95 moves with which length the
   plan landed on. Repeating `image_count_high` under a second tag gave p50 0.724 s against 0.721 s
   (+0.4 %) with an identical sample order, so the plan and the timing are stable.

## Reproduce

```sh
# once: build the six manifests and audit them (container CPU, no GPU)
singularity exec -B /research:/research -B /tmp:/tmp $WS/containers/ngc-pytorch-25.03-py3.sif \
  python3 track_b_cluster/scripts/build_heterogeneity_suite.py \
    --output-dir $WS/runs/bc_suite_v1 --samples-per-case 12 \
    --model-dir $WS/models/Qwen2-VL-2B-Instruct

# one job per case (three GPUs each; the QOS allows four jobs at a time)
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
for case in resolution_low resolution_medium resolution_high \
            image_count_low image_count_medium image_count_high; do
  sbatch --gres=gpu:3 --export=ALL,ENC_GPUS=1,LLM_PP=2,NUM_LAYERS=28,NUM_MICROBATCHES=2,\
BENCH_STEPS=6,WARMUP_STEPS=1,TAG=bc_$case,MANIFEST_BATCHING=bucket,\
MANIFEST=$WS/runs/bc_suite_v1/$case/manifest.json,OUT_DIR=$WS/runs/bc_suite_v1/runs \
    track_b_cluster/slurm/disagg_units.sbatch
done

# summary + gates
singularity exec -B /research:/research -B /tmp:/tmp $WS/containers/ngc-pytorch-25.03-py3.sif \
  python3 track_b_cluster/scripts/report_bc_suite_distrain.py \
    --suite-root $WS/runs/bc_suite_v1 --runs-root $WS/runs/bc_suite_v1/runs \
    --output $WS/runs/bc_suite_v1/summary.json
```

Run pairs fit on one 8-GPU node (3 GPUs each) and the launcher now draws per-job rendezvous ports,
so several cases can share a node.
