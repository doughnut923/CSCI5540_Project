# Image Dataset Preparation

## Install

```powershell
pip install datasets pillow
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

The generator uses only 128x128, 256x256, and 384x384 inputs with measured
Qwen2-VL token costs of 25, 81, and 196 respectively. It uses exact
resolution mixtures rather than random probabilities:

```text
Low:    1250 x 256
Medium: repeating [128, 128, 256, 256, 256, 384]
High:   repeating [128, 128, 384]
```

Any remainder is filled with 256 and the assignments are shuffled. Use
`--seed` to reproduce the shuffle.

## Create token-budget batches

Create batches from one generated dataset. Each batch is filled until adding
the next sample would exceed the token budget:

```powershell
python .\token_budget_sampler.py `
  .\variance_datasets\dataset_Medium.json `
  --batch-token-budget 800 `
  --output .\variance_datasets\medium_batches.json
```

Use `--no-shuffle` to preserve the input order. The output JSON includes each
batch's samples and `total_tokens`.
