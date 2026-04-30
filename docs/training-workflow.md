# Fluke Orca Identifier Training Workflow

This document is the working guide for training Fluke's dorsal-fin/orca identifier.

The first real model work uses public dataset releases, not scraped photos. The current source is the Happywhale Whale and Dolphin Identification dataset through Kaggle, filtered to killer whale/orca rows.

## What We Are Training

The model is an embedding model. It does not directly answer "this is Tahlequah" as a classification head. Instead, it converts each image into a vector. Images of the same individual should land close together; images of different individuals should land farther apart.

At inference time:

1. Embed the query photo.
2. Search a FAISS index of reference photo embeddings.
3. Aggregate matches by individual.
4. Return top-k candidates with similarity scores.

This matters because the catalog can grow. Adding a new whale should mean adding reference embeddings, not retraining the model from scratch.

## Data Policy

Allowed:

- Official public datasets whose terms allow research/training use.
- The Happywhale/Kaggle dataset after accepting Kaggle terms.
- Future licensed or permissioned reference photos.

Not allowed:

- Random Google Images.
- Social media photos.
- ID guide PDFs.
- Researcher catalog photos unless terms or permission explicitly allow ML training.
- Whale-watching photos copied from public webpages.

Public visibility is not permission.

## Local Machine First

Cale's current machine is suitable for the first training phase:

- Apple M3 Pro
- 18 GB RAM
- PyTorch MPS available

Use this machine for:

- Dataset download and verification.
- Orca manifest generation.
- Train/val/test split creation.
- Tiny overfit tests.
- One-to-three epoch local training runs.
- Debugging scripts and understanding the training loop.

Use BYU GPU Lab later for:

- Full dataset runs.
- Bigger backbones like ConvNeXt-Tiny.
- Larger image sizes.
- Longer training.
- Hyperparameter sweeps.

## End-to-End Commands

Run all commands from `fluke-model/`.

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Kaggle

Create a Kaggle API token from Kaggle account settings and place it at:

```text
~/.kaggle/kaggle.json
```

Then open the Happywhale dataset or competition page in Kaggle and accept the terms.

### 3. Download Happywhale

```bash
uv run python scripts/download_happywhale.py
```

Expected output:

- `data/happywhale/train.csv`
- `data/happywhale/train_images/`
- `data/happywhale/metadata_summary.json`

### 4. Build Orca Manifest

```bash
uv run python scripts/build_orca_manifest.py
```

Expected output:

- `data/manifests/happywhale_orca.jsonl`
- `data/manifests/happywhale_orca.summary.json`

This filters rows to killer whale/orca species labels only.

### 5. Split Data

```bash
uv run python scripts/split_manifest.py
```

Expected output:

- `data/manifests/happywhale_orca_splits/train.jsonl`
- `data/manifests/happywhale_orca_splits/val.jsonl`
- `data/manifests/happywhale_orca_splits/test.jsonl`
- `data/manifests/happywhale_orca_splits/split_summary.json`

Individuals with too few images are dropped from closed-set evaluation. No image path should appear in more than one split.

### 6. Tiny Overfit Test

```bash
uv run python scripts/train_embedder.py --overfit-tiny --epochs 3 --no-pretrained --device mps
```

This is the first neural-network sanity check. It asks: can the model memorize a tiny repeated-ID subset? If this fails, do not run a bigger training job yet.

### 7. First Local Training Run

```bash
uv run python scripts/train_embedder.py \
  --backbone resnet50 \
  --epochs 3 \
  --identities-per-batch 4 \
  --images-per-identity 2 \
  --device mps
```

Expected output:

- checkpoint under `artifacts/models/orca/<run-name>/model.pt`
- metrics under `results/orca/<run-name>.json`

### 8. Evaluate Against Baseline

```bash
uv run python scripts/evaluate_orca.py \
  --trained-checkpoint artifacts/models/orca/<run-name>/model.pt
```

This compares the trained model and MiewID on the same orca split.

For a checkpoint-only smoke test:

```bash
uv run python scripts/evaluate_orca.py \
  --skip-baseline \
  --trained-checkpoint artifacts/models/orca/<run-name>/model.pt
```

## Reading Results

Core metrics:

- `top_1`: true individual is the first result.
- `top_3`: true individual is in the first three results.
- `top_5`: true individual is in the first five results.
- `mrr`: mean reciprocal rank, a ranking-quality score where earlier correct answers count more.

For Fluke, top-3 matters most because the UI will show likely candidates, not one authoritative answer.

## Decision Rule

- If the trained model beats MiewID on the same public-orca test split, it becomes the candidate V1.
- If it does not, MiewID remains the fallback and the trained model remains the learning/research track.

Either outcome is useful.
