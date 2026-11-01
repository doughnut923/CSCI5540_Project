# Image Dataset Preparation

## Install

```powershell
python -m pip install -r .\requirements.txt
```

## Run

```powershell
python .\data_prep.py
```

The script downloads the dataset and creates five folders under `image_sets\`:

```text
image_sets\128\
image_sets\256\
image_sets\384\
image_sets\512\
image_sets\768\
```

Each folder contains 250 cropped PNG images of the corresponding size.

## Generate distinct-variance datasets

`distinct_variance.py` creates one JSON metadata file per feasible resolution
group. The input may be a JSON list of image IDs or a CSV containing one ID
per row (optionally with an `image_id` header):

```powershell
python .\distinct_variance.py .\image_ids.json --output-dir .\variance_datasets
```

Alternatively, omit the ID file and use images already generated under
`image_sets\`:

```powershell
python .\distinct_variance.py --output-dir .\variance_datasets
```

Run `python .\data_prep.py` first if `image_sets\` does not exist.

The generator uses only source images from the `128`, `256`, and `384` folders.
Each source image is assigned its native folder resolution, so no image is
upscaled or downscaled. Since each source folder has 250 images, source paths
are reused when a mixture needs more than 250 samples at one resolution. The
assigned target resolutions are 128x128, 256x256, and 384x384 with measured
Qwen2-VL token costs of 25, 81, and 196 respectively. It uses exact
resolution mixtures rather than random probabilities:

```text
Low:    750 x 256
Medium: repeating [128, 128, 256, 256, 256, 384]
High:   repeating [128, 128, 384]
```

Any remainder is filled with 256 and the assignments are shuffled. Use
`--seed` to reproduce the shuffle.

## Valiadation
```powershell
python .\part_b_image_res\resolution_validation.py `
  --dataset-dir .\part_b_image_res\variance_datasets `
  --repository-root . `
  --batch-json .\variance_datasets\low_batches.json `
              .\variance_datasets\medium_batches.json `
              .\variance_datasets\high_batches.json
```

The validator recomputes every sample's visual tokens through the processor,
checks each batch total and budget, and requires non-final batches to be at least 65% full by default, allowing for
the discrete 25/81/196-token sample sizes. Adjust this with
`--batch-tolerance`.

For each dataset it also simulates the actual token size of every sample and
prints per-resolution statistics alongside an overall mean and variance. All
images are resized to a square `R x R` before processing, so every sample at
one resolution must produce the same token count; any spread is reported as
`resolution_preserved=NO` (per-sample mismatches against
`token_cost_estimate` are still reported as errors). Results from the
current datasets:

| Dataset | 128x128 | 256x256 | 384x384 | Overall mean | Overall variance |
|---|---|---|---|---|---|
| Low (750)   | —                 | n=750, mean=81.0, var=0.00 | —                          | 81.0 | 0.00 |
| Medium (750)| n=250, mean=25.0, var=0.00 | n=375, mean=81.0, var=0.00 | n=125, mean=196.0, var=0.00 | 81.5 | 3249.25 |
| High (750)  | n=500, mean=25.0, var=0.00 | —                          | n=250, mean=196.0, var=0.00 | 82.0 | 6498.00 |
