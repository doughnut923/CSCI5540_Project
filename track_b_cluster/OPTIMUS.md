# Optimus PP integration

This branch builds on the team's `distrain-integration` commit
`3760175772c5952b7cfca2203558edb778c11bc7`, plus the native smoke environment fixes
from PR #4. It does not change the team's remote branches.

## Reuse and ownership

| Component | Source / status |
| --- | --- |
| Qwen2-VL model and processor | Same pinned PMP model_provider as native |
| Sample order, prompts, labels, image handling | Team ManifestBatches, extracted to bridge/manifest_data.py |
| BF16 parameters, FP32 master weights, AdamW, cosine LR | Team MainGradAdamW, extracted to bridge/training_common.py |
| Metrics | Existing JSONL schema and scripts/collect_step_metrics.py |
| Colocation, dependency graph, placement selection and executor | New Optimus PP implementation |

The original imports remain available from pp2_equivalence.py and disagg_units.py.
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
  Encoder costs measured on rank 0 are assumed equal on identical replica GPUs.
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
again by the number of encoder replicas. The original DistTrain default optimizer
behavior is preserved; this runner instead passes one global clipping norm.

Ordinary autograd `.grad` values (embedding/norm/etc.) are combined with fused
`main_grad` values. The pinned MCore gradient finalizer synchronizes tied embeddings.
The global norm counts encoder replicas and tied output embeddings only once.

Two/three/four-rank schedule checks and two-process CPU training tests cover
unequal microbatch ownership, zero-work encoder replicas, gradient transfer,
three AdamW updates, and full gradient/parameter parity with an unpartitioned
small model. CPU tests are correctness evidence, not GPU performance evidence.
The recorded [CPU validation](reproduction/optimus_cpu/validation.json) passed
five tests, including two- and four-process training, on PyTorch 2.6.0.

```bash
cd track_b_cluster/bridge
python -m unittest -v test_optimus
```

## GPU smoke test

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
The shared optimizer also exposes this as `OPTIMIZER_DEVICE=cpu` to DistTrain.
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

The proposal's final two axes × three levels are a later experiment matrix,
after Qwen GPU parity and actual colocation/communication behavior are verified.
