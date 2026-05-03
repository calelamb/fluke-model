# BYU GPU Lab Fine-Tuning Plan

This is the next step after the M3 Pro fine-tuning ceiling was hit
(see `docs/training-pass-003-and-miewid-finetune.md`). Frozen MiewID-msv3 hits
**82.6% Top-1 / MRR 0.875** on the combined orca test split, and no fine-tuning
variant tried on M3 Pro (linear head, MLP head, ResNet-50 from scratch,
ConvNeXt-Tiny from scratch) generalised better than the frozen baseline.

The realistic path to surpass MiewID is full backbone fine-tuning at MiewID's
native 440×440 resolution with a metric-learning head (ArcFace) and a larger
batch than M3 Pro can hold. Both the compute budget and the GPU memory needed
push this to BYU's GPU lab.

## Goals

| Run | Approach | Realistic outcome |
|---|---|---|
| **F1** | Last-block fine-tune + linear head | 83–86% Top-1 |
| **F2** | Full backbone fine-tune + ArcFace head | 86–92% Top-1 |
| **F3** | F2 + heavy augmentation + longer training | 88–94% Top-1 |

Stop when we beat the frozen baseline by ≥3 points on the held-out test split
two runs in a row, or when validation curves plateau across all three.

## Hardware target

- 1× NVIDIA GPU with **≥16 GB VRAM** (full fine-tune at batch 32, 440×440 needs
  about 12–14 GB peak; ArcFace adds a small classifier head proportional to the
  number of identities, so 16 GB is comfortable).
- Disk: ~5 GB for repo + datasets + checkpoints.

## Pre-flight (do this on M3 Pro before going to BYU)

1. Verify the combined orca pipeline still runs end-to-end:

   ```bash
   ./.venv/bin/python scripts/build_orca_manifest.py
   ./.venv/bin/python scripts/build_finid20_manifest.py
   ./.venv/bin/python scripts/build_combined_orca_manifest.py
   ./.venv/bin/python scripts/split_manifest.py \
       --manifest data/manifests/orca_all.jsonl \
       --out-dir data/manifests/orca_all_splits \
       --min-images-per-individual 5
   ```

2. Make sure these are committed and pushed:
   - `data/manifests/orca_all.jsonl` (manifest)
   - `data/manifests/orca_all_splits/` (splits, frozen by seed=42)
   - `results/orca/baseline-miewid-combined.json` (baseline numbers)

3. Bring the dataset locally on the BYU machine. Two options:
   - **Re-download** (preferred): run `download_happywhale.py` and
     `download_finid20.py` on the BYU machine (about 10 GB total).
   - **Rsync** the `data/` directory from the dev machine.

## On the BYU machine

### Setup

```bash
git clone <fluke-model remote>
cd fluke-model
uv sync                              # or pip install -e . with the pyproject
./.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 1 — re-cache MiewID features (sanity)

Confirm MiewID loads on CUDA and produces the same numbers as the M3 Pro cache.
The cache script falls back to CPU; before running, edit
`src/fluke_model/embedders.py`'s `_select_device()` for `_load_miewid` to return
`torch.device("cuda")` when available, then:

```bash
./.venv/bin/python scripts/cache_miewid_features.py \
    --manifest data/manifests/orca_all.jsonl \
    --out artifacts/miewid_features/orca_all.npz
```

This should run in seconds–minutes (vs. 4 minutes on CPU).

### Step 2 — build the fine-tune script (TODO before BYU)

The current M3 Pro fine-tune (`scripts/finetune_miewid_head.py`) trains a head
on cached features. The BYU script needs to:

1. Load MiewID with `requires_grad=True` (full or partial), not via the frozen
   `load_embedder()` path.
2. Train end-to-end through the backbone with ArcFace loss (not pure triplet —
   ArcFace is the metric-learning standard for re-ID and what MiewID itself
   was trained with).
3. Use the same `BalancedBatchSampler` from `src/fluke_model/trainable.py` and
   the same combined orca splits.
4. Save checkpoints in a format `serve_identifier.py` can load (need a small
   loader change so the runtime can pick up a fine-tuned MiewID instead of the
   frozen HuggingFace checkpoint).

Suggested file: `scripts/finetune_miewid_backbone.py` with this CLI:

```bash
./.venv/bin/python scripts/finetune_miewid_backbone.py \
    --splits-dir data/manifests/orca_all_splits \
    --image-size 440 \
    --epochs 60 \
    --warmup-epochs 5 \
    --identities-per-batch 8 \
    --images-per-identity 4 \
    --lr 1e-4 \
    --backbone-lr-multiplier 0.1 \
    --loss arcface \
    --arcface-margin 28.6 \
    --arcface-scale 64.0 \
    --freeze-until-block N    # optional: freeze early blocks for run F1
    --early-stop-patience 8 \
    --run-name miewid-arcface-full-001
```

ArcFace reference implementation: `pytorch-metric-learning` package
(`pip install pytorch-metric-learning`) provides `losses.ArcFaceLoss` and
plays well with the existing `BalancedBatchSampler`.

### Step 3 — three fine-tune runs

#### F1: Last-block fine-tune, linear head

Conservative warm-up. Freeze MiewID up through the EfficientNet-V2-M's
penultimate block; unfreeze the final block + add a 256-dim linear projection.
Triplet loss with margin 0.3 (a bit higher than M3 Pro to avoid the loss-to-zero
collapse seen on cached features). Backbone LR 1e-5, head LR 1e-4. 30 epochs.

Expected runtime on a single 16 GB GPU: ~30 min.

#### F2: Full fine-tune + ArcFace

Unfreeze entire backbone. ArcFace head with the standard margin (28.6°) and
scale (64). Backbone LR 1e-5, head LR 1e-3. 50 epochs with early stopping.
The number of classes for ArcFace is `len(unique_individuals_in_train)` (45 on
the current combined manifest).

Expected runtime: ~90 min.

#### F3: F2 + heavy augmentation

Add RandAugment, RandomErasing, ColorJitter heavy. Same ArcFace head. Train
80–100 epochs.

Expected runtime: ~3 hours.

### Step 4 — evaluation

For every run, evaluate on the **same combined test split**:

```bash
./.venv/bin/python scripts/evaluate_orca.py \
    --splits-dir data/manifests/orca_all_splits \
    --trained-checkpoint artifacts/miewid_finetuned/<run-name>/model.pt \
    --out results/orca/<run-name>-eval.json
```

The eval script should be extended to load a fine-tuned MiewID checkpoint
(currently it loads timm-backboned `EmbedderNet` checkpoints). One small change.

Compare each run head-to-head against the frozen baseline:
`results/orca/baseline-miewid-combined.json`.

### Step 5 — ship checkpoint back

Once a run beats the frozen baseline by ≥3 points on test:

1. Save the checkpoint + a small JSON metadata file (image size, normalization,
   embed_dim) to `artifacts/miewid_finetuned/<run-name>/`.
2. Push to GitHub if size permits (Git LFS likely needed for the 100+ MB model).
   Alternative: upload to a private S3 bucket and put the URL in a release note.
3. On the dev machine, update `serve_identifier.py` to load the fine-tuned
   checkpoint instead of the HuggingFace MiewID. Re-run the integration smoke
   test described in `docs/identifier-service.md`.

## What to bring back

A README addition with:

- The frozen baseline test number (already known: 82.6% Top-1 / MRR 0.875)
- The best fine-tune run's test number
- A short note on which run won and why
- The commit hash + checkpoint name so the dev machine can pin to the same
  artifact

## Out of scope for BYU

- New data sources. Stay on combined HappyWhale + FinID-20 for the first BYU
  pass. Adding more sources is a separate research thread (see Track A in the
  spec — Salish Sea licensing).
- Detector training (`yolov8n` for fin/saddle bbox). That's M-Model-1, separate
  effort.

## Stop conditions

- F1, F2, and F3 all fail to beat the frozen baseline by ≥3 points → write up,
  ship MiewID frozen as the V1 forever.
- A run beats the frozen baseline by ≥5 points → ship it as V1.5.
- A run beats by 90%+ Top-1 with stable val curves → start working on the
  citizen-science feedback loop and iterate from there.
