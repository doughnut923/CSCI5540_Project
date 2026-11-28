# Image count and training speed on two A800 GPUs

| Images per sample | Runtime | Mean step (ms) | Throughput (samples/s) | Peak per GPU (GiB) |
| --- | --- | ---: | ---: | ---: |
| Low (1) | BigMac | 342.8 | 5.83 | 4.89 |
| Low (1) | Optimus-inspired PP | 330.6 | 6.05 | 4.89 |
| Medium (1, 2, 3) | BigMac | 304.8 | 6.56 | 4.92 |
| Medium (1, 2, 3) | Optimus-inspired PP | 291.6 | 6.86 | 4.91 |
| High (1, 2, 4, 8) | BigMac | 300.3 | 6.66 | 4.94 |
| High (1, 2, 4, 8) | Optimus-inspired PP | 288.7 | 6.93 | 4.94 |

On BigMac, the High mix changed mean step time by -12.4% relative to Low. High was slower in 0 of four paired repeats.

Both runtimes used the same two A800 GPUs and model weights, 576 visual tokens per input, and identical sample order, and the deterministic math SDPA backend. Only the final visual-projector matrix was trained. Values use 32 timed steps per workload and runtime, excluding warm-up.
