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
