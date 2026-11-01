# Real-photo image-count preprocessing pilot

This is the check reported on slides 6 and 7 of the Group 8 midterm presentation.
It runs locally on CPU with NumPy and the official Qwen2-VL processor/tokenizer.
It does not load model weights or run the vision encoder, language model,
backward pass, training step, or a BigMac performance comparison.

The group plans to use `Qwen/Qwen2-VL-2B-Instruct`. These preprocessing checks use
its pinned tokenizer/config revision `895c3a49bc3fa70a340399125c650a463535e71c`
with Transformers 4.49.0. Full-model and training-system integration remain future work.

## Reproduce

From the repository's `part_c_image_count` folder, use Python 3.12 in a personal environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python download_reference.py
python real_photo_pilot/run_real_photo_pilot.py --output-dir real_photo_pilot/runs/reproduction_v1
```

The existing downloader fetches only six hash-pinned tokenizer/config files,
about 11.5 MB, and never model weights. The pilot itself uses only local files.
The output directory must not exist. For an offline run with existing assets:

```sh
python download_reference.py --assets-dir /path/to/existing/assets --verify-only
python real_photo_pilot/run_real_photo_pilot.py --assets-dir /path/to/existing/assets --output-dir real_photo_pilot/runs/offline_v1
```

The included eight small JPEG previews make the input check reproducible without
downloading ImageNet. `photo_manifest.json` records the dataset, viewer rows,
dimensions and SHA-256 hashes. Paths are relative to the manifest; the script
does not depend on the original author's folders. These files are previews from
[`mrm8488/ImageNet1K-val`](https://huggingface.co/datasets/mrm8488/ImageNet1K-val),
config `default`, split `train`, rows 0, 2500, 7500, 12500, 17500, 22500, 27500 and 35000.
They are not full-resolution original dataset images or a new labeled training set.

## What 160, 80 and 576 mean

- Eight unique real photographs are repeatedly sampled with fixed seed `55402026`.
- Each input contains N = 1, 2, 3, 4 or 8 images, with 16 inputs for each N.
- Five counts times 16 inputs gives 80 inputs per preparation condition.
- The same ordered photo combinations are checked under two conditions: 160
  paired input checks, not 160 independent photographs.
- 576 is the target total visual-token budget shared by all images in one input.
  It is unrelated to the coincidental total of 576 image instances across both conditions.

1. `native_aspect`: give the original previews directly to the processor.
2. `padded_control`: fit each complete photo on a fixed canvas using gray padding,
   preserving aspect ratio and field of view. No stretching or cropping is used.

Both conditions use `min_pixels=3136` and `max_pixels=(576 // N) * 784`.
For padded inputs the processed H x W per image is 672 x 672, 448 x 504,
336 x 448, 336 x 336, or 224 x 252 for N = 1, 2, 3, 4, or 8 respectively.
The prepared canvas is twice the target H and W. Shape changes across N are
part of this feasibility intervention, not an isolated causal training experiment.

| Images/input | Inputs/condition | Native-aspect derived token range | Padded derived total/input |
| --- | --- | --- | --- |
| 1 | 16 | 216-252 | 576 |
| 2 | 16 | 432-486 | 576 |
| 3 | 16 | 536-576 | 576 |
| 4 | 16 | 512-520 | 576 |
| 8 | 16 | 498 | 576 |

All 160 input checks retained every image through preprocessing. All 80 padded
inputs met the derived shared budget. Only one native-aspect input happened to
have exactly 576 derived tokens. PASS means retention/consistency, not accuracy
or necessarily meeting the budget.

## Measured, derived and unmeasured

Measured: source/prepared/processed dimensions, grids, patch-tensor shapes,
per-image tensor-slice equality, independent image perturbations, and separately
counted image slots, visual markers and text padding. Each image's tensor slice
must equal processing it alone; inverting one image may change only its slice.

Derived: merged visual-token count is `sum(grid_t * grid_h * grid_w) / 4` using
the model's 2 x 2 spatial merge rule. Each padded input produced 2,304 actual
processor patch rows, shape `[2304, 1176]`, giving 2,304 / 4 = 576 derived tokens.
Image placeholders are a separate consistency check, not measured embeddings.

Unmeasured: actual encoder output embeddings, LLM consumption, encoder/LLM
forward, gradients, training speed, GPU memory, model quality and scheduler advantage.
Padding occupies token positions; equal token totals do not prove equal useful
photo content, equal encoding work or equal training time. Eight source photos
are enough for this reproducible input probe, not a representative model benchmark.

## Files

- `run_real_photo_pilot.py`: portable implementation; default is 16 inputs/count/condition.
- `photo_manifest.json`, `photos/`: source previews and provenance, checked by SHA-256.
- `reference_results/summary.json`, `config.json`, `samples.csv`: original 160-check records.
- `reference_results/verification.json`: fresh portable rerun compared with those records.

Each run writes the full per-input JSON/JSONL/CSV, deterministic image assignments,
configuration, environment, log and four representative padded inputs. Existing
outputs are never overwritten. No synthetic, encoder or training results are
substituted for these real-photo preprocessing records.

## Official references

- [Qwen2-VL documentation](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/qwen2_vl)
- [Pinned tokenizer/config assets](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/895c3a49bc3fa70a340399125c650a463535e71c)
- [Processor implementation](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
