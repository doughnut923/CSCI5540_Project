# DistTrain arm — measured results (Qwen2-VL-2B, 8x RTX 3090)

Every number here was produced by `track_b_cluster/bridge/disagg_units.py` (disaggregated
encoder unit + LLM unit) or `track_b_cluster/bridge/pp2_equivalence.py` (fused baseline) on the
24-sample pilot manifest, `runs/track_b_n2_pilot_measured_v1/pilot_manifest.json`.

Rows are split into **training-valid** (real optimizer step, gradients complete) and
**placement-only** (schedule/topology measurement). They are not comparable across the split, and
the second table must never be quoted as a training result.

Common config: 32 layers (32 is required for PP=8; the real model has 28), hidden 1536, 12 heads,
bf16, M=4 microbatches of one sample each, sequence length cycling 194 -> 82 -> 253 -> 309 -> 424
-> 138 by the manifest plan, LR cosine 1e-4 -> 1e-5, clip 1.0, AdamW, single node, rtx3090,
`ngc-pytorch-25.03-py3`.

## 1. Training-valid DistTrain configurations

Encoder unit DP=1 (one GPU, no gradient exchange needed), LLM unit is a real MCore pipeline.
Both units step AdamW over MCore's fp32 `main_grad`; see `bridge/disagg_units.py`.

| config | GPUs | measured steps | step time mean / p50 / p95 (s) | samples/s | bubble | peak GPU mem | skipped / non-finite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ENC=1 + LLM_PP=2` | 3 | 30 (3 runs x 10) | 0.926 / 0.932 / 1.022 | 4.32 | 30.0 % | 20.68 GiB | 0 / 0 |
| `ENC=1 + LLM_PP=4` | 5 | 10 | **0.837 / 0.830 / 0.960** | **4.78** | 57.9 % | 19.76 GiB | 0 / 0 |

- Source records: `runs/track_b_n2_pilot_measured_v1/disagg_m4_a3all.jsonl` (+ `_summary.json`),
  `disagg_m4_b1pp4.jsonl` (+ `_summary.json`). Aggregated by
  `track_b_cluster/scripts/collect_step_metrics.py`; step time is wall clock around the step with
  warmup (2 steps) discarded.
- `samples/s` = sum(samples) / sum(step time) over measured steps, 4 samples per step.
- Bubble is instrumented over both units: `(ranks * step_time - sum(busy intervals)) / (ranks *
  step_time)`, ranks = encoder ranks + LLM stages. Definition in `docs/pipeline-bubble.md`.
- Both rows train: inside a run the loss peaks at 1.56-1.59 when the plan reaches the 424-token
  group and ends at 0.26 (`PP=4`) / 0.18 (`PP=2`), and both `weight_norm` and `enc_weight_norm`
  move.

Figures: `docs/disagg_m4_a2run20_step10.png` (`ENC=1+PP=2`, step 10),
`docs/disagg_m4_b1pp4_step5.png` (`ENC=1+PP=4`, step 5).

**Reading the two rows together:** two extra stages cut step time 10 % but double the bubble
(30 % -> 58 %). With one encoder rank feeding the pipeline, every added stage adds a place to
wait for the encoder. That is the bubble DistTrain is meant to remove, reproduced here.

## 2. Fused baseline on the same plan (no optimizer step)

`pp2_equivalence.py` now consumes the same `ManifestBatches` plan, so this is the same data,
same M=4, same 32 layers as section 1 — but the harness never steps an optimizer, so the loss
sits at its initial ~3.0. Compare the timing columns only.

| config | GPUs | measured steps | step time mean / p50 / p95 (s) | samples/s | bubble |
| --- | --- | --- | --- | --- | --- |
| fused PP=4 (encoder in stage 0) | 4 | 10 | 0.792 / 0.770 / 0.922 | 5.05 | 60.1 % |
| DistTrain `ENC=1 + LLM_PP=4` (section 1) | 5 | 10 | 0.837 / 0.830 / 0.960 | 4.78 | 57.9 % |

Figure: `docs/fused_pp4_step5.png` (step 5, 0.756 s, busy 39.5 %, bubble 60.5 %).

The fused row is faster in wall clock and uses one GPU fewer, but it is **not** a like-for-like
training number: it has no optimizer step. The measured optimizer cost in the DistTrain arm is
83 ms on the encoder unit and 46-75 ms per LLM stage at PP=4, so a fused step that also stepped
AdamW would land near 0.92-0.95 s (**derived, not measured**). Giving the fused harness the same
`MainGradAdamW` is the smallest remaining change needed before the two rows can be quoted side by
side.

What *is* directly comparable and measured is the bubble: 60.1 % fused vs 57.9 % disaggregated.
The fused pipeline leaves more capacity idle because stage 0 must run the whole vision tower
before it can start the next microbatch, while the disaggregated run overlaps encoder backwards
with LLM work.

## 3. Earlier placement-only configurations (no training step)

Seeded CPU init, one fixed sample, forward+backward only, no optimizer, and (unless stated)
**no encoder gradient exchange**. Useful for topology comparison only.

| config | GPUs | step (s) | busy | note |
| --- | --- | --- | --- | --- |
| fused PP=8 | 8 | 1.088 | 19.6 % | encoder in stage 0 dominates the step |
| fused PP=4 | 4 | 0.968 | 37 % | |
| disagg 4 enc + PP=4, no sync | 8 | 0.560 | 34 % | placement only, encoder grads not exchanged |
| disagg 4 enc + PP=4, fp32 all-reduce | 8 | 2.313 | — | exchange alone 1.72 s |
| disagg 2 enc + PP=2, streamed | 4 | 0.695 | 54-59 % | placement only |
| disagg 2 enc + PP=2, fp32 all-reduce | 4 | 1.19 | — | exchange alone 1.13 s |

Figures: `docs/pp8_gantt_split_v1.png` (fused PP=8), `docs/disagg_pp4_enc4.png` and
`docs/disagg_pp4_enc4_gradsync.png` (4 encoder ranks, with and without the exchange),
`docs/disagg_overlap_4gpu_nosync.png` (2 encoder ranks, streamed).

Encoder DP does not pay on this hardware: 2.6 GB of fp32 vision gradients, ~300 small
collectives, no P2P between these 3090s (~5.3 GB/s host-staged), i.e. 1.13 s at N=2 and 1.72 s at
N=4. Sync-free disaggregation is 2x faster than the fused baseline; with the naive exchange it
is 2x slower. Bucketed-bf16 exchange is the cheapest untried fix (workpackage B2).

## 4. Not measured

| arm | status |
| --- | --- |
| BigMac VE executor (`MODE=1`, `pp_executor`) | fails with no output; no BigMac number exists |
| Optimus | no code in this repository (only a mention in `AGENTS.md`) |
| encoder-as-pipeline-stages | not implemented (needs the ViT layer split); est. 0.28-0.36 s |

## 5. Caveats that apply to any table built from this

1. **Placement-only rows are not training results.** They have no optimizer step and, where
   marked, no encoder gradient exchange, so they are not a valid training configuration.
2. **Loss and gradient norms are not comparable across configurations.** Seeded CPU init is
   layout-dependent: at PP=2 with identical data the language-model weights are bit-identical
   across arms (squared norms 340426.1 / 341956.9 on both stages) but the vision tower's are not
   (fused 242603.1 vs encoder unit 294916.5), and fused loss changes with the split (PP=2 3.2159,
   PP=4 3.0689). Only timing columns may be compared across configurations until both units load
   the same checkpoint.
3. **Step time tracks sequence length**, not just topology: 424-token steps are ~25 % slower than
   82-token steps. Every arm must use the same `ManifestBatches` plan or the comparison is noise.
4. **Bubble is instrumented for the fused Megatron baseline and DistTrain, not for BigMac or
   Optimus.** The two measured rows in section 2 are therefore comparable; a three-way bubble
   column is not available, and must not be invented.
5. **Peak memory is per rank, max over ranks.** At `PP=4` the encoder unit is the peak
   (16.0 -> 19.8 GiB across a run) and the LLM stages are 7.7-13.0 GiB. At `PP=2` the LLM unit
   reaches 20.7 GiB, which is within ~3 GiB of the card limit.
6. **32 layers is a synthetic depth** chosen so PP=8 divides evenly; the real checkpoint has 28.
7. **Single node, 24 GB cards, no P2P.** Any conclusion about exchange cost is a property of this
   interconnect, not of the design.
8. **24-sample dataset**, repeated within a run. These are efficiency measurements, not a
   convergence result.

## 6. Reproduce

```sh
# training-valid DistTrain, 5 GPUs
sbatch --gres=gpu:5 --export=ALL,ENC_GPUS=1,LLM_PP=4,NUM_MICROBATCHES=4,BENCH_STEPS=10,WARMUP_STEPS=2,TAG=b1pp4 \
  track_b_cluster/slurm/disagg_units.sbatch

# metrics summary (inside the container image the sbatch uses)
python3 track_b_cluster/scripts/collect_step_metrics.py \
  --input runs/track_b_n2_pilot_measured_v1/disagg_m4_b1pp4.jsonl \
  --output runs/track_b_n2_pilot_measured_v1/disagg_m4_b1pp4_summary.json --world-size 5

# Gantt for one step
python3 track_b_cluster/scripts/plot_pipeline_gantt.py \
  --input runs/track_b_n2_pilot_measured_v1/disagg_m4_b1pp4.jsonl --step 5
```
