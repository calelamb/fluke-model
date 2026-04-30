# Repeated-Identity Local Orca Training Pass

Date: 2026-04-30

## Goal

This pass expanded the Happywhale/Kaggle orca subset from a small smoke-test dataset into a repeated-identity benchmark that is useful for metric learning. The target was 25 identities with 8-15 images per identity, trained locally on the M3 Pro using MPS, and evaluated against MiewID on the same split.

No scraped photos were used. Data came through the official Kaggle access path after accepting the Happywhale competition terms.

## Dataset

- Source: Happywhale Whale and Dolphin Identification, Kaggle official access path
- Species included: `killer_whale` and Kaggle typo `kiler_whale`
- Species excluded: `false_killer_whale`, `pygmy_killer_whale`, and all non-orca rows
- Repeated-identity plan: 25 identities, 15 planned images per identity
- Planned images: 375
- Already present before this pass: 102
- Newly downloaded: 273
- Failed downloads: 0
- Rate limited: false
- Local Happywhale train image count after expansion: 914

The download plan was written locally to `data/happywhale/orca_download_plan.json`. Raw images and local manifests stay out of git.

## Split

Split seed: 42

Closed-set filtering retained only identities with at least 5 local images.

| Split | Images | Identities | Min imgs/id | Median imgs/id | Max imgs/id |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 336 | 25 | 11 | 11 | 31 |
| Val | 64 | 25 | 2 | 2 | 7 |
| Test | 64 | 25 | 2 | 2 | 7 |

Dropped from closed-set evaluation: 447 low-count identities / 450 images. Those images remain useful later for open-set or pretraining experiments, but they do not support closed-set retrieval metrics yet.

## Training Runs

### ResNet-50

Command:

```bash
uv run python scripts/train_embedder.py \
  --backbone resnet50 \
  --epochs 5 \
  --identities-per-batch 4 \
  --images-per-identity 2 \
  --device mps \
  --run-name resnet50-happywhale-orca-local-002
```

The run completed in 106 seconds. Loss stayed finite and decreased from 0.298 to 0.014. Validation peaked in epoch 1, then declined, which suggests overfitting.

Best validation top-1: 32.8%

### ConvNeXt-Tiny

Command:

```bash
uv run python scripts/train_embedder.py \
  --backbone convnext_tiny \
  --epochs 3 \
  --identities-per-batch 4 \
  --images-per-identity 2 \
  --device mps \
  --run-name convnext-tiny-happywhale-orca-local-002
```

The run completed, but it underperformed ResNet-50 on this small local setup.

Best validation top-1: 20.3%

## Test Evaluation

All models were evaluated against the same 336-image train reference set and 64-image test query set.

| Model | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| MiewID-msv3 | 65.6% | 81.2% | 85.9% | 0.755 |
| Trained ResNet-50 | 35.9% | 53.1% | 60.9% | 0.495 |
| Trained ConvNeXt-Tiny | 15.6% | 32.8% | 45.3% | 0.303 |

## Interpretation

The repeated-identity pipeline works: we can select useful identities, download selectively from Kaggle, build leakage-free splits, train locally, write checkpoints, and evaluate against MiewID on the exact same split.

MiewID remains the production fallback. It is substantially stronger than both local trained models on this pass.

The ResNet result is still useful. It confirms the training loop learns signal above random chance, but the validation curve shows the current setup overfits quickly. The next meaningful improvement is not more epochs. It is better training data and stronger metric-learning setup:

- crop or localize fins/saddle patches instead of training on full uncropped images
- add stronger augmentations that preserve identity markings
- move from basic triplet loss toward ArcFace / supervised contrastive loss
- use a larger repeated-identity set if local download limits allow it
- evaluate open-set behavior, not just closed-set retrieval

## Artifacts

Tracked:

- `results/orca/resnet50-happywhale-orca-local-002.json`
- `results/orca/resnet50-vs-miewid-happywhale-orca-local-002.json`
- `results/orca/convnext-tiny-happywhale-orca-local-002.json`
- `results/orca/convnext-tiny-vs-miewid-happywhale-orca-local-002.json`
- this summary

Ignored locally:

- `data/happywhale/`
- `data/manifests/`
- `artifacts/models/orca/resnet50-happywhale-orca-local-002/model.pt`
- `artifacts/models/orca/convnext-tiny-happywhale-orca-local-002/model.pt`
