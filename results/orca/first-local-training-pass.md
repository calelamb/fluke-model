# First Local Training Pass

Completed: 2026-04-30

## Dataset

- Source: Happywhale Whale and Dolphin Identification via Kaggle.
- Terms: Kaggle competition rules accepted before download.
- Download strategy: selective official Kaggle file downloads, capped for a laptop first pass.
- Local images downloaded: 641 orca/killer-whale image files.
- Manifest rows used after missing-file filtering: 641.
- Split seed: 42.

The first broad 500-image pull covered too many one-photo identities. A second top-up pass changed selection toward repeated identities, then Kaggle rate-limited further image pulls. The resulting first-pass split is intentionally small but valid for learning the full training loop.

## Split

Closed-set split with `min_images_per_individual=3`:

| Split | Images | Individuals |
| --- | ---: | ---: |
| Train | 105 | 4 |
| Val | 22 | 4 |
| Test | 22 | 4 |

This is not a production benchmark. It is a first local sanity run.

## Training

Tiny overfit:

- Command: `uv run python scripts/train_embedder.py --overfit-tiny --epochs 3 --no-pretrained --device mps --identities-per-batch 3 --images-per-identity 2 --run-name tiny-overfit-mps`
- Loss: `0.7946 -> 0.1406`
- Result: training loop, optimizer, triplet loss, checkpoint writing, and MPS path all work.

First local model:

- Backbone: ResNet-50
- Embedding dim: 256
- Loss: batch-hard triplet loss
- Device: MPS
- Epochs: 3
- Train images: 105
- Validation top-1: 54.5%
- Validation top-3: 95.5%

## Test Evaluation

Same test split for MiewID and the trained ResNet-50:

| Model | Reference Images | Query Images | Individuals | Top-1 | Top-3 | Top-5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiewID-msv3 | 105 | 22 | 4 | 81.8% | 100.0% | 100.0% | 0.909 |
| trained ResNet-50 | 105 | 22 | 4 | 54.5% | 90.9% | 100.0% | 0.735 |

## Interpretation

The first trained model works, but MiewID is still stronger. That is the expected result for a first local training pass on a tiny split. The useful outcome is that the entire workflow now runs locally:

1. Authenticate with Kaggle.
2. Download official public orca images.
3. Build manifest.
4. Create leakage-safe splits.
5. Train with triplet loss.
6. Evaluate against MiewID on the same split.

## Next Step

Wait for Kaggle rate limiting to cool down, then continue downloading repeated-identity images. The next target should be at least 25-50 identities with 5+ images each before treating metrics as meaningful.
