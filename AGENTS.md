# AGENTS.md

Guidance for agents working in `fluke-model`, the ML workspace for Fluke's dorsal-fin/orca identifier.

## Working Rules

- Use official public datasets only. Do not scrape arbitrary public webpages, social media, ID guides, researcher catalogs, or photos unless a license or written permission explicitly allows ML training.
- Keep raw images and model checkpoints out of git. Dataset files belong under `data/`; trained weights and checkpoints belong under `artifacts/`. Both are gitignored.
- Commit reproducible code, manifests that do not expose copied image data, metrics JSON, summaries, docs, and tests.
- Prefer small local smoke tests before expensive training. Every new model path should pass a tiny overfit run before a full run.
- Record dataset source, terms reviewed, split seed, image count, individual count, model version, and top-k metrics for every result.
- Do not wire a trained model into the Fluke API until it beats the agreed baseline or the user explicitly asks for a prototype integration.

## Standard Commands

Run from this directory:

```bash
uv sync
uv run pytest
uv run ruff check .
```

Public-orca data pipeline:

```bash
uv run python scripts/download_happywhale.py
uv run python scripts/build_orca_manifest.py
uv run python scripts/split_manifest.py
```

First local training checks:

```bash
uv run python scripts/train_embedder.py --overfit-tiny --epochs 3 --no-pretrained --device mps
uv run python scripts/train_embedder.py --backbone resnet50 --epochs 3 --device mps
```

Evaluation:

```bash
uv run python scripts/evaluate_orca.py --trained-checkpoint artifacts/models/orca/<run-name>/model.pt
```

Use `--skip-baseline` when validating only the trained-checkpoint path and avoiding MiewID downloads.

## Model Strategy

- MiewID-msv3 is the benchmark and fallback.
- The local training path starts with ResNet-50 or ConvNeXt-Tiny plus batch-hard triplet loss.
- The first goal is learning the neural-network workflow, not claiming production accuracy.
- If the trained model beats MiewID on the same public-orca split, it becomes the candidate V1.
- If it does not beat MiewID, keep MiewID as production fallback and continue the training track as research.

## Data Hygiene

- `data/happywhale/` stores the official Kaggle download.
- `data/manifests/happywhale_orca.jsonl` stores local image paths and individual IDs.
- `data/manifests/happywhale_orca_splits/` stores train/val/test JSONL splits.
- Do not copy images into docs, results, commits, issues, or PR descriptions.
- If Kaggle credentials or terms are missing, stop and report the exact setup steps. Do not find alternate scraped images.

## Reporting

When finishing work, include:

- Files changed.
- Commands run and whether they passed.
- Dataset status: not downloaded, downloaded, manifest built, splits built.
- Training status: not run, smoke run, local run, full run.
- Metrics if available: top-1, top-3, top-5, MRR.
