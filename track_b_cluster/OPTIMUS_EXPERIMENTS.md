# Optimus experiment alignment

## Ownership and source

The Optimus branch incorporates Yin's `distrain-integration` through
`a49368b8dad76700c573a550b1f2453b1bd73149`. All files already present in that
commit remain byte-identical to upstream. In particular, the shared loader,
dataset builders, DistTrain/native runners, launchers and team reports are not
modified. Optimus reuses their interfaces. Its tokenizer adaptation and optimizer
extension live in `bridge/optimus_config.py` and `bridge/optimus_optimizer.py`.

The earlier six-pair GPU record was produced before this synchronization. It
remains historical evidence for its recorded commit, not validation of new code.

## Workload-aligned integration check

| Setting | Yin's B/C pilot | Optimus integration check |
| --- | --- | --- |
| Manifest and processing | Shared six-case builder and loader | Same files, unchanged |
| Language layers / microbatches | 28 / 2 | 28 / 2 |
| Sequence limit / initialization seed | 1024 / 1234 | 1024 / 1234 |
| Warmup / measured steps | 1 / 8 | 1 / 8 |
| Sample order | Bucket by visual tokens and image count | Same bucket planner |
| Physical GPUs | 3: encoder 1 + LLM PP2 | 2: colocated PP2 and two encoder replicas |
| Optimizer state | GPU | CPU offload under the account's two-GPU quota |
| Gradient clipping | Unit/rank-local | Whole-model global norm |
| Initial weights | Layout-dependent seeded initialization | Layout-dependent seeded initialization |

The shared rows align the workload; the last four rows prevent a controlled
cross-backend training-time or loss comparison. No DistTrain run is launched or
changed by the Optimus scripts. The serial-reference arm remains an internal
Optimus correctness control, not native Megatron.

Prepare the six manifests with the existing team builder, or reuse the audited
suite directory. Then preflight **before requesting GPUs**, in an environment
with the pinned processor dependencies:

```bash
python track_b_cluster/scripts/prepare_optimus_suite.py \
  --suite-root "$SUITE_ROOT" --model-dir "$MODEL_DIR" \
  --output-dir "$INPUT_PLANS_ROOT" --batching bucket \
  --microbatches 2 --warmup-steps 1 --measured-steps 8 --sequence-limit 1024
```

This uses the actual shared planner and checks all measured sample IDs against
the processor audit. It refuses an incomplete measured budget; in particular,
six steps miss two samples in the four-bucket cases. Every plan records ordered
sample IDs, multiplicities, shapes and hashes. The Optimus runtime checks these
again against the tensors it actually receives.

Inside a prepared allocation, with `WS`, `BIND_ROOT`, `PROJ_ROOT`, `SUITE_ROOT`,
`MODEL_DIR` and `INPUT_PLANS_ROOT` set:

```bash
export PP=2 NUM_LAYERS=28 NUM_MICROBATCHES=2 SEQ_LENGTH=1024
export WARMUP_STEPS=1 BENCH_STEPS=8 MANIFEST_BATCHING=bucket
export OPTIMIZER_DEVICE=cpu ENCODER_PLACEMENT=round-robin DETERMINISTIC=1
export VALIDATION_SAMPLES=16 REPEATS=1 OUTPUT_ROOT=<new-results-directory>
bash track_b_cluster/scripts/run_heterogeneity_suite.sh
python track_b_cluster/scripts/report_heterogeneity_suite.py \
  --suite-root "$SUITE_ROOT" --runs-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json"
```

## Within-Optimus batching experiment

Keep the model, optimizer, hardware, placement and measurement policy fixed.
First generate the bucket plans above, then interleave their **exact measured
sample occurrences** in an Optimus-only plan:

```bash
python track_b_cluster/scripts/prepare_optimus_suite.py \
  --suite-root "$SUITE_ROOT" --model-dir "$MODEL_DIR" \
  --output-dir "$MIXED_PLANS_ROOT" --batching manifest \
  --match-plans "$INPUT_PLANS_ROOT" \
  --microbatches 2 --warmup-steps 1 --measured-steps 8 --sequence-limit 1024
```

Use `MANIFEST_BATCHING=manifest INPUT_PLANS_ROOT="$MIXED_PLANS_ROOT"` for the
second run. The Optimus input adapter replays the audited order while using the
unchanged shared processor/build methods. Warmup order and measured sample
multiplicities are retained. Require equal `sample_multiset_sha256` across both
plans; `sample_order_sha256` differs when multiple lengths exist. Merely choosing
the same step count, or even more steps, is insufficient: bucket sizes differ,
so their long-run sampling weights can differ from uniform manifest cycling.

First perform gradient/sampled-parameter checks against the serial control
separately for each order. Dedicated timing runs must disable validation sampling
and use independent repeats. This phase tests batching within Optimus only; it
does not fix the resource, clipping or initialization differences above.

## Full Part B and remaining comparison work

Reuse Yin's unchanged `scripts/build_part_b_suite.py` and its audited manifests
for `resolution_full_low`, `resolution_full_medium`, and `resolution_full_high`.
The original image files are required; do not substitute the eight-photo pilot
while labelling it the full dataset. Match the 375-step budget when reproducing
his workload. Coverage checks use actual sample IDs rather than assuming a
particular step count guarantees coverage.

A fair cross-backend performance experiment still requires agreement on equal
GPU budgets, optimizer residence/clipping, initialization/checkpoint policy and
sample multiplicities. The formal resource target is eight RTX 3090s per run.
The single-node correctness check below verifies feasibility for Optimus before
the group freezes the final training settings.

Mean encoder work is also not yet calibrated across the resolution distributions.
Token equality alone is not a sufficient control. Do not quote a final speedup,
variance-only effect, or full-paper Optimus reproduction from these checks.

## Verified targeted follow-up (2026-11-22)

[GPU evidence](reproduction/optimus_gpu_sync_20261122/validation.json), executed
source `1c977ec91da5d09a1703bdd826cf6481ec53a34d`: the image-count-high case
(N=1/2/4/8) passed in bucketed and interleaved order on two RTX 3090s. Each
order was paired with the internal serial reference for one warmup plus eight
measured steps. Loss, global gradient norm, sampled parameters and used gradients
matched exactly within each pair. The runtime logs also confirm equal measured
sample multiplicities across the two orders and nonzero gradients for every image.

Complete raw samples/logs are retained in the user's server archive with its
SHA-256 recorded in the evidence file. Configs, step metrics and comparator
summaries were verified locally. The GPU allocation was released. This validates
the new adapter path for the targeted case; other cases and dedicated performance
runs remain separate experiments.

## Single-node PP=8 correctness check

With `PP=8 NUM_LAYERS=28`, the LLM stages use `2 / 4 / 4 / 4 / 4 / 4 / 4 / 2`
layers. The endpoint stages also hold the embedding or output projection. Each
rank builds the same full vision encoder. Startup verifies the actual layer
numbers and records them alongside per-rank AdamW memory estimates. The GPU
path initializes P2P communicators before training. Round-robin mode also uses
the same encoder owners during calibration to bound per-rank activation memory.

Use the existing, audited `image_count_high` pilot for two updates per arm.
Eight microbatches activate every encoder replica. Manifest order covers all
12 samples over two steps, including samples with eight images. This check
uses seeded initialization and GPU AdamW. The group can choose a different
microbatch count when freezing the formal comparison.

From the local checkout, prepare the replay plan using the pinned model config
and tokenizer directory and the existing six-case pilot `SUITE_ROOT`:

```bash
python track_b_cluster/scripts/prepare_optimus_suite.py \
  --suite-root "$SUITE_ROOT" --model-dir "$MODEL_DIR" \
  --output-dir "$INPUT_PLANS_ROOT" --cases image_count_high \
  --batching manifest --microbatches 8 --warmup-steps 0 --measured-steps 2
```

On the CSE login node, check the account/QoS before staging a run. On 2026-11-23,
the course QoS `csci5540` had a two-GPU per-user limit. The authorized department
QoS `gpu` allowed eight and passed the 3090 placement check on `gpu_2h` and
`gpu_8h`. Recheck eligibility and availability on the account that will run.

```bash
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
sbatch --test-only --account=gpu --partition=gpu_2h --qos=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:8 --time=00:30:00 \
  --constraint=rtx3090 --exclude=gpu54,projgpu33 --chdir=/tmp --wrap=true
```

Within one approved 30-minute, eight-RTX-3090 allocation, use the prepared
environment (`WS`, `BIND_ROOT`, `PROJ_ROOT`, `MODEL_DIR`) and point `SUITE_ROOT`
and `INPUT_PLANS_ROOT` to the staged pilot and replay plan on the compute node.
Set `OUTPUT_ROOT` to a fresh evidence directory. From the allocation shell:

```bash
export PP=8 NUM_LAYERS=28 NUM_MICROBATCHES=8 SEQ_LENGTH=1024
export MANIFEST_BATCHING=manifest WARMUP_STEPS=0 BENCH_STEPS=2 REPEATS=1
export OPTIMIZER_DEVICE=cuda ENCODER_PLACEMENT=round-robin DETERMINISTIC=1
export VALIDATION_SAMPLES=16 PROFILE_STEP_PHASES=0 SUITE_CASES=image_count_high
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:8 --chdir=/tmp \
  bash "$PROJ_ROOT/track_b_cluster/scripts/run_heterogeneity_suite.sh"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=none --chdir=/tmp \
  python3 "$PROJ_ROOT/track_b_cluster/scripts/report_heterogeneity_suite.py" \
  --suite-root "$SUITE_ROOT" --runs-root "$OUTPUT_ROOT" \
  --cases image_count_high --output "$OUTPUT_ROOT/summary.json"
```

Require two completed optimizer steps in each arm, all eight ranks in the
memory and placement records, 28 decoder layers with no gaps, and a full
32-layer encoder on each rank. The report checks workload coverage, image
gradients, loss, global gradient norm, and sampled gradients and parameters
against the internal serial reference. Preserve the first failure and its logs
if the memory, launch or correctness check fails. Treat the result as a
correctness record for this layout and input. Formal throughput measurements
use the group's common settings and a separate run.
