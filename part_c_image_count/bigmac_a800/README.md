# Part C · Image count on two A800 GPUs

BigMac trained the last weight matrix of the Qwen2-VL-2B-Instruct visual projector on two A800-SXM4-80GB GPUs. Each step consumed two samples. Every sample contributed 576 visual tokens. Input preparation was outside the timed step.

## Measured results

| Default-attention workload | Images per sample | Mean step (ms) | Samples/s | Peak per GPU (GiB) |
| --- | --- | ---: | ---: | ---: |
| Low | 1 | 140.5 | 14.23 | 4.20 |
| Medium | 1, 2, 3 | 138.8 | 14.41 | 4.21 |
| High | 1, 2, 4, 8 | 142.1 | 14.07 | 4.23 |

Under default attention, step time varied from 138.8 to 142.1 ms. Each step paired inputs with the same image count. Four timing blocks used model seed 1234. Each block included one warm-up and eight measured steps per workload. [Primary step records](recorded/r4/primary_results/steps.jsonl)

The follow-up ran both schedules on one pair of A800s. Each schedule visited the same 24 samples counting repeats (20 distinct inputs), with identical per-rank sample counts. The grouped schedule paired inputs with the same number of images. Each mixed step paired one single-image input with a 2-, 4- or 8-image input. Both schedules used the same padded sequence length of 621. The sample-order seeds were 101, 211, 307, 401 and 503, with four timing repetitions per seed and a fixed model seed of 1234.

| Mean per step (ms) | Grouped | Mixed | Mixed minus grouped |
| --- | ---: | ---: | ---: |
| Training time, profiler off | 141.92 | 143.00 | +1.07 |
| GPU 0 idle, profiler on | 54.46 | 54.59 | +0.14 |
| GPU 1 idle, profiler on | 59.80 | 59.97 | +0.16 |

Mixing added 0.75% to mean step time, with the same direction in all five orders. GPU idle time changed in both directions across orders. It measures time with no GPU operation recorded on that device. Profiling increased wall time by 27.5–28.6%. Each schedule had 240 ordinary timing steps and 60 separately profiled steps. Differences use unrounded measurements. [Sample plans](recorded/grouping/results/plans.json) · [Step records](recorded/grouping/results/time.jsonl) · [Per-rank profiling records](recorded/grouping/results/profile.jsonl)

Two checks are in the same follow-up summary. Mixing the original High sample pool added 1.16% per step. Reordering the original Low/High workloads gave a 0.55% longer mean step for High. A separate two-runtime benchmark applied deterministic math attention to both BigMac and the team's Optimus-inspired PP. Its numerical replay covered losses and projector updates. [Follow-up results](recorded/grouping/results/summary.json) · [Matched-runtime results](recorded/r4/results/Part_C_final_table.md) · [Numerical checks](recorded/r4/results/numeric_pass.json)

## Verify the published tables on a CPU

From this directory, run the following command with Python 3.9 or newer. It uses the standard library to check the saved schedules and recalculate all three summaries from step-level measurements in a temporary directory. The GPU traces also include the validity status for each rank.

```sh
python3 check_results.py
```

## Reproduce the workloads

1. Prepare CPU inputs with a local checkout of the team's dataset revision `d309514eb012c33b5b2c619e6f88b9fd43d43c2f`, including its eight source photos, and the tokenizer and processor files of `Qwen/Qwen2-VL-2B-Instruct` at `895c3a49bc3fa70a340399125c650a463535e71c`. The preprocessing environment used Python 3.12, Torch 2.5.1, Transformers 4.49.0, Pillow 13.0.0 and NumPy 2.3.5. [Team loader](https://github.com/doughnut923/CSCI5540_Project/blob/d309514eb012c33b5b2c619e6f88b9fd43d43c2f/track_b_cluster/bridge/manifest_data.py) · [Processor asset checksums](primary/data/processor_hashes.json)
2. Run the export below from this directory. The exporter validates the three manifests and all 36 compressed arrays against the checksums recorded for the original experiments. It writes to `primary/data/tensors/`. Existing verified files may be reused.
3. On a Linux host allocated exactly two A800s, use the full local checkpoint at the same revision and the upstream BigMac checkout at `2fa6da2fd3979448e89a56e734cbc3c490990d81`. The recorded training environment used PyTorch 2.11.0+cu128, Transformers 5.9.0 and Megatron-Core 0.16.1. The grouping experiment needs working CUDA activity profiling to measure idle time. [BigMac executor](https://github.com/Dots-Infra/BigMac/tree/2fa6da2fd3979448e89a56e734cbc3c490990d81/pp_executor)

```sh
# CPU preprocessing environment
python3 prepare_inputs.py "$TEAM_CHECKOUT" "$QWEN_PROCESSOR_DIR"

# GPU training environment. Set both values to local checkout directories.
export BIGMAC_DIR=/path/to/BigMac
export PARTC_MODEL_DIR=/path/to/Qwen2-VL-2B-Instruct
bash run.sh image-count
bash run.sh grouping
bash run.sh matched
```

Each invocation creates a new directory under `runs/` and prints its path. The group comparison and its five sample orders run within a single invocation. Set `OUTPUT_DIR` to choose a different output parent or `PYTHON` to select the training interpreter.

## Code and provenance

- `primary/code/` contains BigMac step timing, synchronized over two ranks. `primary/data/` contains the team's sample manifests, published order, and input checksums.
- `adapter/` contains the project's earlier Qwen2-VL bridge and its input helper. All three measurements use the same bridge.
- `grouping/` contains the paired sample schedules, timed and profiled steps, and result checks.
- `matched/` contains the paired numerical checks, the team's Optimus schedule, and the runtime adapted to disable internal trace collection during ordinary timing. [Team runtime revision](https://github.com/doughnut923/CSCI5540_Project/tree/1ca3e2a15ae83b6a86dba802b59153b45ce8865e/track_b_cluster/bridge)
- `recorded/` contains the measurements underlying each result. The two allocations are documented by device count, type, runtime versions, and a distinct-device check. Each experiment used a single pair for both sides of its comparison.

## Handoff verification

The 36 input archives were regenerated on CPU and matched every recorded SHA-256. All three summary files were independently recalculated from the included step records. The eight upstream executor modules and both adapter files matched their measured copies by SHA-256. The GPU numbers come from the completed A800 benchmarks. The portable shell launcher passed syntax checks and has not itself been rerun on GPUs.
