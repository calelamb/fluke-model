# First Local Training Pass

This is the immediate run plan for Cale's M3 Pro. The goal is to complete the full training loop locally once, not to maximize accuracy.

## Goal

Produce the first trained Fluke orca embedding checkpoint from public Happywhale orca data and compare it against the MiewID baseline.

## Hardware

- Machine: Apple M3 Pro
- RAM: 18 GB
- Device: `mps`

This is enough for a small ResNet-50 run. If memory becomes tight, reduce batch shape before using a GPU lab.

## Pass 0: Data Setup

```bash
uv sync
uv run python scripts/download_happywhale.py
uv run python scripts/build_orca_manifest.py
uv run python scripts/split_manifest.py
```

Stop if any of these fail. Do not substitute scraped images.

## Pass 1: Tiny Overfit

```bash
uv run python scripts/train_embedder.py \
  --overfit-tiny \
  --epochs 3 \
  --no-pretrained \
  --device mps
```

Expected outcome:

- The command completes.
- Loss decreases or at least remains finite.
- A checkpoint and results JSON are written.

If this fails, debug the training loop before running anything larger.

## Pass 2: First Real Local Run

```bash
uv run python scripts/train_embedder.py \
  --backbone resnet50 \
  --epochs 3 \
  --identities-per-batch 4 \
  --images-per-identity 2 \
  --device mps
```

This uses a small balanced-batch metric-learning setup:

- 4 individual IDs per batch.
- 2 photos per individual.
- Batch size = 8 images.

If MPS memory fails:

```bash
uv run python scripts/train_embedder.py \
  --backbone resnet18 \
  --epochs 3 \
  --identities-per-batch 3 \
  --images-per-identity 2 \
  --image-size 192 \
  --device mps
```

## Pass 3: Evaluation

Use the checkpoint path printed by the training command:

```bash
uv run python scripts/evaluate_orca.py \
  --trained-checkpoint artifacts/models/orca/<run-name>/model.pt
```

If MiewID loading blocks the first evaluation, run:

```bash
uv run python scripts/evaluate_orca.py \
  --skip-baseline \
  --trained-checkpoint artifacts/models/orca/<run-name>/model.pt
```

Then debug MiewID separately.

## What To Record

Save these from the output:

- manifest rows
- number of individuals
- split seed
- training run name
- checkpoint path
- top-1
- top-3
- top-5
- MRR
- whether MiewID was evaluated on the same split

## When To Use BYU GPU Lab

Do not start there. Use BYU GPU Lab when one of these is true:

- The local run completes and we want a longer full-data experiment.
- MPS memory prevents ResNet-50 or ConvNeXt-Tiny from training.
- We want image size above 224.
- We want more than 5-10 epochs.
- We want to run multiple experiments in parallel.

The first useful GPU-lab run should be based on a successful local run, not a guess.
