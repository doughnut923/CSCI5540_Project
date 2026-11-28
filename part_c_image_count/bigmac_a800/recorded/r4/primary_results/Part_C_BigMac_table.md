# BigMac on two A800 GPUs

| Workload | Mean step (ms) | Throughput (samples/s) | Peak per GPU (GiB) |
| --- | ---: | ---: | ---: |
| Low (1) | 140.5 | 14.23 | 4.20 |
| Medium (1, 2, 3) | 138.8 | 14.41 | 4.21 |
| High (1, 2, 4, 8) | 142.1 | 14.07 | 4.23 |

High changed mean step time by +1.1% relative to Low. High was slower in 3 of four paired repeats.

Two A800-SXM4-80GB GPUs, 576 visual tokens per sample, 32 measured steps per condition, Qwen2-VL-2B with a trainable visual projector and default SDPA.
