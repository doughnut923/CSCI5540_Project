# Image count heterogeneity: processor feasibility pilot

This folder contains Group 8's image-count input-pipeline check. It is separate from the repository's image-resolution data-preparation script. It does not implement a BigMac integration or run training.

**Planned group model:** `Qwen/Qwen2-VL-2B-Instruct`. The checks use tokenizer/config revision `895c3a49bc3fa70a340399125c650a463535e71c`. Full-model training-system integration has not been run by this pilot. The proposal originally listed a provisional LLaVA-style model; the group now plans to use Qwen2-VL.

## Latest real-photo check

The midterm real-photo check is in [`real_photo_pilot/`](real_photo_pilot/).
It checks 160 paired inputs built from eight real photo previews, with N = 1, 2,
3, 4 and 8. All images survived preprocessing; all 80 padded inputs reached the
derived shared 576-token target. That folder includes a portable script, the
eight small previews, provenance, original records and reproduction instructions.
It is processor-only, not a model training run.

The sections below describe the earlier synthetic-input pilot and its original evidence.

## Reproduce

Python 3.12 was used. Create a personal environment, then run from this folder:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python download_reference.py
python validate_pipeline.py --output-dir runs/pilot_v1
```

The download is limited to six hash-verified official tokenizer/config files, about 11.5 MB total. No model weights or real dataset are downloaded. Assets, new runs, virtual environments and generated images are ignored by Git.

To reuse assets already available locally, without network access:

```sh
python download_reference.py --assets-dir /path/to/existing/assets --verify-only
python validate_pipeline.py --assets-dir /path/to/existing/assets --output-dir runs/offline_v1
```

A full run generates eight aligned synthetic samples for each image count N = 1, 2, 3, 4, 8 (40 total), plus five square stress samples and a separate padding check. Use `--samples-per-count 1` for a smaller smoke run. `--save-images` optionally retains generated PNGs; seeds, dimensions, PNG hashes and sample-to-image assignments are always recorded in `manifest.json`. A new output directory is required, so previous results are never overwritten.

## What was checked

Fixed seed: `55402026`. Each synthetic image has a distinct deterministic seed. Images are artificial input probes, not evidence of model quality or real training performance.

The official Transformers 4.49.0 processor uses `min_pixels=3136` and `max_pixels=(576/N)*28*28`, with patch size 14, spatial merge size 2 and temporal patch size 2. A static image's internal temporal duplication does not represent two input images. There are no spatial crops/tiles in these constructed cases.

| Images per sample | Processed H x W per image | Derived visual tokens/image | Derived total/sample |
| --- | --- | --- | --- |
| 1 | 672 x 672 | 576 | 576 |
| 2 | 448 x 504 | 288 | 576 |
| 3 | 336 x 448 | 192 | 576 |
| 4 | 336 x 336 | 144 | 576 |
| 8 | 224 x 252 | 72 | 576 |

Original images have twice the listed height and width. For each image, its processor tensor slice is compared with processing it alone. Each image is then inverted independently: only its corresponding tensor slice may change. These checks guard against missing images or silently using only the first image **through preprocessing**.

### Measured, derived and unmeasured are different

- **Measured:** processed grids, tensor shapes, 2,304 encoder-input patch rows per aligned sample, resized/source pixel totals, individual image tensor checks, image-token expansion, padding and special-marker counts.
- **Derived:** expected LLM visual-token count = sum of actual grid patch rows / `merge_size**2`. The aligned cases give 2,304 / 4 = 576. Observed expanded image-token slots are a consistency check; they are not measured model embeddings.
- **Unmeasured:** encoder output embeddings, actual LLM consumption, encoder/LLM forward, gradients, training steps, GPU throughput and any BigMac-versus-baseline result.

`pixel_values.shape = [2304, 1176]` for each aligned sample. The second dimension is `3 channels * 2 temporal positions * 14 * 14`. Vision start/end markers (`2*N`), text and other special tokens are counted separately from visual slots. The single-sample checks have no padding. A separate two-sample padding check uses 72 tokens **per image** solely to expose padding; it is not the shared-576 experiment.

## Limitations and next steps

A per-image pixel cap is not a guarantee of exactly 576 shared tokens for arbitrary shapes. With square source images under the same caps, derived totals for N = 1, 2, 3, 4, 8 are **576, 512, 507, 576, 512**. All images still pass the retention checks. Grid rounding and image shape affect the realized budget.

Equal token totals do not establish equal encoder compute. The constructed shapes/resolutions change with image count, so this is a feasibility check, not a causal isolation of image-count effects. `cpu_processor_ms_uncontrolled` is an untuned wall-clock observation with no warmup, not an encoder or training-speed benchmark.

A2-Low uses N=1; A2-Med lists N=1/2/3; A2-High lists N=1/2/4/8. Equal mixture weights recorded in `reference_config.json` are temporary pilot settings, not a requirement or a performed mixed training run.

The real-photo preprocessing follow-up is now available above. Next: integrate the planned Qwen2-VL model, verify all embeddings in an actual model forward pass, and only then measure matched encoder/runtime/training behavior on authorized hardware.

## Included evidence

Files named `reference_*.csv` and `reference_*.json` contain the original successful pilot's compact CSV/JSON records and sanitized environment, with `reference_manifest.json` as the sample manifest that omits private paths. No generated images, model assets or private course documents are included. PNG hashes allow regenerated probes to be compared when the pinned Pillow version is used.

`verification.json` records a fresh portable-code rerun and comparison against the reference results. Runtime values are deliberately excluded from equality checks.

## Sources

- [Qwen2-VL official model documentation](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/qwen2_vl)
- [Pinned candidate files](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/895c3a49bc3fa70a340399125c650a463535e71c)
- [Transformers 4.49.0 image processor](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
- [Transformers 4.49.0 multimodal processor](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/processing_qwen2_vl.py)

