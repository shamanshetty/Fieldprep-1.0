# FieldPrep 1.0

A compact PyTorch pipeline for image-based plant disease classification. The repository contains code only: no datasets, checkpoints, test artifacts, or generated results.

## Included tools

- `preprocessing.py` cleans images, removes duplicates, augments training data, and creates train/validation/test splits.
- `train.py` trains a ResNet-34, ResNet-50, or MobileNetV2 classifier.
- `evaluate.py` produces classification metrics and visualizations for a labelled split.
- `predict.py` classifies a single image from a trained checkpoint.

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

Set `PreprocessingConfig.INPUT_DIR` in `config.py` to a directory arranged as one folder per class:

```text
raw_dataset/
  early_blight/
  late_blight/
  healthy/
```

## Workflow

Preprocess the source images. The processed output is written to `processed_dataset/` by default.

```bash
python preprocessing.py
```

Train a model. The processed directory must contain `train/` and `val/` folders.

```bash
python train.py --processed_dir processed_dataset --model_name mobilenet_v2 --epochs 20 --model_path models/mobilenet_v2.pth
```

Evaluate a labelled split (for example, the test split):

```bash
python evaluate.py --data_dir processed_dataset/test --model_name mobilenet_v2 --model_path models/mobilenet_v2.pth
```

Classify one image:

```bash
python predict.py --image_path sample.jpg --model_name mobilenet_v2 --model_path models/mobilenet_v2.pth
```

The training process saves a class-index mapping next to each checkpoint; keep that JSON file with the model when evaluating or predicting.
