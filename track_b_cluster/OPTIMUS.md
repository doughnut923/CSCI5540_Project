# Optimus PP integration

This branch incorporates the team's `distrain-integration` through
`a49368b8dad76700c573a550b1f2453b1bd73149`. Every file present in that upstream
commit is retained unchanged. Optimus-specific extensions are isolated in new
files; the native setup suggestions remain separately reviewable in PR #4.
The [experiment alignment protocol](OPTIMUS_EXPERIMENTS.md) describes the current
workload contract and the remaining differences between backends.

## Reuse and ownership

| Component | Source / status |
| --- | --- |
| Qwen2-VL model and processor | Same pinned PMP model_provider as native |
| Sample order, prompts, labels, image handling | Team ManifestBatches, extracted to bridge/manifest_data.py |
| BF16 parameters, FP32 master weights, AdamW, cosine LR | Optimus-owned extension of the team AdamW in bridge/optimus_optimizer.py |
| Metrics | Existing JSONL schema and scripts/collect_step_metrics.py |
| Colocation, dependency graph, placement selection and executor | New Optimus PP implementation |

The upstream pp2_equivalence.py and disagg_units.py remain unchanged. The
Optimus tokenizer override is applied locally by bridge/optimus_config.py.
Missing derived token metadata is handled by measuring processor output, allowing
the recorded two-photo smoke manifest to use the same input path.

## Fidelity boundary

**Name results `optimus-inspired-pp-whole-encoder`, not full Optimus.** This is a
first implementation of the paper's colocation and bubble-filling idea:

- Each GPU owns a language-model pipeline stage and a complete vision encoder.
- LLM PP=P, TP=DP=1; encoder DP=P, PP=TP=1. Parameters of encoder replicas are
  broadcast initially; gradients are summed after each global batch.
- LLM order is fixed non-interleaved 1F1B. Encoder forward/backward operations
  fill dependency-safe idle intervals; one GPU runs one compute task at a time.
- A no-update calibration pass measures encoder/stage forward/backward calls on
  actual processed shapes. Placement candidates are evaluated with these costs.
  Each encoder measurement is reused across the identical replicas. The
  round-robin path calibrates on the same owners as training.
- A separate P2P channel handles each direction of encoder and LLM transfers.
  Microbatch IDs are retained; no sample order is changed.
- Encoder forward precedes that microbatch's LLM forward; encoder backward waits
  for its LLM input gradient; all backward work precedes optimizer updates.

**Not implemented:** kernel-level encoder partitioning, TP/DP communication bubble
exploitation, encoder pipeline parallelism, interleaved LLM chunks, full 3D-plan
search, automatic checkpoint loading. Initialization is seeded random, as in the
team's schedule tests. No convergence or published-paper speedup is claimed.

Paper: [Optimus, ATC 2025, sections 3–4](https://arxiv.org/html/2408.03505v2).

## Training correctness

Every loss is normalized by the number of microbatches. Each encoder replica has
only a subset of the global batch, so its gradients must be **summed**, not divided
again by the number of encoder replicas. The DistTrain optimizer is unchanged; this runner passes one global clipping
norm to its own optimizer extension. This difference is recorded in the experiment
contract and prevents a strict cross-backend update-equivalence claim.

Ordinary autograd `.grad` values (embedding/norm/etc.) are combined with fused
`main_grad` values. The pinned MCore gradient finalizer synchronizes tied embeddings.
The global norm counts encoder replicas and tied output embeddings only once.

Two/three/four-rank schedule checks and two-process CPU training tests cover
unequal microbatch ownership, zero-work encoder replicas, gradient transfer,
three AdamW updates, and full gradient/parameter parity with an unpartitioned
small model. CPU tests are correctness evidence, not GPU performance evidence.
The recorded [CPU validation](reproduction/optimus_cpu/validation.json) covers
two- and four-process training on PyTorch 2.6.0. Eight tests also cover mixed
sequence lengths, FP32 accumulation of ordinary BF16 gradients after each
microbatch, and large parameter sampling where floating-point indices can
round out of bounds.

```bash
cd track_b_cluster/bridge
python -m unittest -v test_optimus
```

## GPU smoke test

For the 28-layer single-node PP=8 layout and its correctness check, use the
[eight-GPU protocol](OPTIMUS_EXPERIMENTS.md#single-node-pp8-correctness-check).
The recorded GPU evidence below uses two GPUs.

Use the existing pinned native environment (SETUP.md plus the PR #4 corrections).
The course queue verified for this account is `csci5540`, with a per-user limit
of **two GPUs** (`sacctmgr show qos csci5540`, 2026-11-20). Older `gpu_2h` examples
in the team documentation refer to another account/environment.

Full AdamW needs BF16 model parameters + FP32 gradients/masters/moments. A
replicated vision encoder makes full GPU AdamW infeasible on two 24GB GPUs:
the measured build has about 1.554 billion trainable parameters on each rank,
with 15.1GB free but 18.6GB additional optimizer allocation needed per rank.
The launcher therefore uses **2 GPUs with CPU optimizer offload**, retaining
28 unchanged language layers. Both comparison arms must use this same setting.
CPU offload is an Optimus-only option; the upstream DistTrain runner is unchanged.
This is a functional check, not an all-GPU Optimus throughput comparison. On a
larger GPU allocation use `OPTIMIZER_DEVICE=cuda` consistently across all arms.

```bash
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
export WS=<your-existing-GPU-visible-environment>
export PROJ_ROOT=<synchronized-checkout>
export MANIFEST=$PROJ_ROOT/track_b_cluster/reproduction/native_pp2_smoke/manifest.json
export OUTPUT_DIR=$WS/results/<new-run-id>
sbatch --export=ALL track_b_cluster/slurm/optimus_pp.sbatch
```

For the reference arm, use a different new output directory and `MODE=serial-reference`.
It is a correctness/control arm with the same colocated model, not the team's
native Megatron performance baseline. Both arms use the same input/optimizer.
The launcher now defaults to matched round-robin ownership and deterministic
execution. Use `ENCODER_PLACEMENT=profiled DETERMINISTIC=0` only as a separately
labelled placement experiment; multi-step cross-layout BF16 equivalence is not
claimed. See [the heterogeneous-input protocol](HETEROGENEITY.md).

Each run retains config, input hash, per-step measured profiles and estimated
plans, optimizer-step metrics, operation intervals and final completion status.
Preprocessing/calibration/planning are outside the measured optimizer step.
Step time includes forward/backward, transfers, gradient synchronization and
optimizer updates. The runtime deliberately does not emit `pipeline_bubble_ratio`
or MFU from high-level operation ranges. Precise bubble/MFU measurement needs
additional comparable kernel instrumentation across all backends.

```bash
python track_b_cluster/scripts/collect_step_metrics.py \
  --input <run>/metrics.jsonl --output <run>/summary.json \
  --run-label optimus-inspired-pp-whole-encoder
```

The proposal's two axes × three levels are implemented as an integration pilot
in [HETEROGENEITY.md](HETEROGENEITY.md). A final controlled performance matrix
requires workload calibration and agreement on the actual comparison backends.

## Verified GPU result (2026-11-20)

[Validation record](reproduction/optimus_gpu/validation.json): two RTX 3090 GPUs,
28-layer random-initialized Qwen2-VL-2B architecture, four microbatches per step,
two optimizer updates, with identical CPU-offloaded AdamW in both arms.

| Step | Reference loss | Optimus PP loss | Both global gradient norms |
| --- | --- | --- | --- |
| 0 | 12.519957542419434 | 12.519957542419434 | 854.281986329445 |
| 1 | 9.399199485778809 | 9.399199485778809 | 400.9212604800243 |

All **31,392 sampled parameter values and 31,392 sampled used-gradient values**
matched exactly across the two runs. This is sampled Qwen equivalence; the CPU
small-model tests separately compare every parameter and gradient.
The selected bubble plan assigned encoder microbatches to ranks `[1, 0, 1, 0]`,
so this check exercised actual cross-GPU encoder handoff and gradient return.

This is not a speedup result: two cold steps with CPU optimizer offload do not
establish steady-state GPU training performance. It is not pretrained-weight
reproduction, convergence validation, or a complete ATC Optimus implementation.

To compare an independent pair of evidence directories:

```bash
python track_b_cluster/scripts/compare_optimus_validation.py <parent-of-both-runs> \
  --output <parent-of-both-runs>/comparison.json
```

## Six-case heterogeneous-input follow-up

The [controlled B/C pilot](HETEROGENEITY.md#verified-controlled-pilot-2026-11-20)
passed all six case pairs, with exact sampled parameter/gradient agreement and
per-image gradient coverage. Its fixed-placement, deterministic scope is narrower
than general profiled-placement equivalence. The record also preserves the earlier
cross-layout failure and flags unmatched mean encoder work in the resolution axis.

The team's [DistTrain results](../docs/distrain/distrain-results.md) include both
the older 32-layer/M=4 configurations and the newer 28-layer/M=2 B/C runs.
The newer B/C runs align model depth and microbatch count, but GPU count, optimizer
placement, clipping and batching still differ from the historical Optimus pilot.
Their fused baseline omits optimizer updates. See [the alignment protocol](OPTIMUS_EXPERIMENTS.md)
before comparing results.

## Synchronization validation (2026-11-22)

The [local synchronization record](reproduction/optimus_sync_20261122/validation.json)
checks that all 113 upstream files are unchanged, ten CPU tests pass, and all six
bucketed and reordered plans retain identical measured sample multiplicities and
complete input coverage. The reordered plans are an Optimus-only adapter over the
shared builder. The [post-sync GPU check](reproduction/optimus_gpu_sync_20261122/validation.json)
now also passes for the image-count-high case in bucketed and interleaved order.
Each order matches its serial reference for nine optimizer steps, including
282,528 sampled parameter values and the same number of used-gradient values
across both pairs. All 240 measured per-image gradient checks pass. The actual
measured sample multisets match across orders. This is a targeted follow-up,
not a rerun of all six cases or a performance comparison. The earlier GPU
records remain tied to their original source commits.

## Bottleneck diagnosis

The [phase-timing follow-up](OPTIMUS_PROFILING.md#recorded-diagnosis-2026-11-22)
measured both the high image-count and high-resolution workloads with validation
sampling disabled. In this two-GPU CPU-offload configuration, the optimizer path
accounts for 91.4–93.3% of the synchronized diagnostic step. CPU AdamW and data
movement should be investigated before interpreting whole-step timing as a
pipeline-scheduling result. These intrusive phase timings are not speedup evidence.
