# Training Pass 003 + MiewID Fine-Tune Findings (2026-05-03)

This session pivoted the local training track. We started by trying to close
the gap to MiewID with from-scratch ResNet-50 / ConvNeXt-Tiny on combined data,
hit a ceiling, then switched fully to MiewID fine-tuning. M3 Pro can't
meaningfully improve on frozen MiewID, so the next move is the BYU GPU lab
(see `docs/byu-gpu-finetuning-plan.md`).

## What changed

### Data

Added FinID-20 (Zenodo `10.5281/zenodo.16786268`, CC-BY-4.0):
- 500 cropped killer-whale dorsal-fin images
- 20 individuals × 25 photos each
- IDs prefixed `finid_<10-char-hex>` to avoid HappyWhale collisions
- Built-in train/val/test split is ignored — combined re-split via
  `split_manifest.py` keeps the no-leakage rule per individual

Combined manifest (`data/manifests/orca_all.jsonl`):
- 1414 rows total (914 HappyWhale + 500 FinID-20)
- 492 individuals (472 HW + 20 FinID-20)

After `split_manifest.py --min-images-per-individual 5`:

| Split | Images | Individuals |
|---|---|---|
| Train | 676 | 45 |
| Val | 144 | 45 |
| Test | 144 | 45 |
| Dropped | 450 | 447 (singletons + 3 with 2 photos) |

The HappyWhale orca distribution is **bimodal**: 444 individuals with 1 photo,
3 with 2 photos, then a gap up to 25 individuals with 10+ photos. Lowering
the threshold below 5 yields almost nothing (only 6 extra images at threshold
2). The only meaningful way to grow the training set was to add FinID-20.

### Code

- `scripts/download_finid20.py` — idempotent Zenodo download
- `scripts/build_finid20_manifest.py` — converts FinID-20 cropped images to
  `OrcaManifestRow` JSONL (same schema as HappyWhale)
- `scripts/build_combined_orca_manifest.py` — concatenates per-source manifests
- `src/fluke_model/miewid_finetune.py` — cached-feature dataset, learnable
  head architectures (Linear, MLP), checkpoint helpers
- `scripts/cache_miewid_features.py` — pre-compute MiewID embeddings once
- `scripts/finetune_miewid_head.py` — train a head on cached features
- `scripts/train_embedder.py` — added cosine LR schedule with linear warmup,
  early stopping on val Top-1, configurable scheduler/patience
- `tests/test_train_scheduler.py` — covers the new scheduler logic

### Results

**Frozen MiewID (the bar):**

| | Val Top-1 | Val Top-3 | Val MRR | Test Top-1 | Test Top-3 | Test MRR |
|---|---|---|---|---|---|---|
| MiewID-msv3 frozen (combined) | 79.9% | 91.0% | 0.863 | **82.6%** | 90.3% | **0.875** |

(The earlier 65.6% Top-1 number lived on the HappyWhale-only val split — not
comparable. Combined splits are easier because FinID-20 is well-balanced at
25 photos per individual.)

**From-scratch (training pass 003, combined data, cosine LR, early stop):**

| Run | Backbone | Best Val Top-1 | Stopped at epoch | MRR |
|---|---|---|---|---|
| `resnet50-orca-combined-003` | ResNet-50 | 36.8% (epoch 6) | 11/20 (early) | 0.512 |
| `convnext-tiny-orca-combined-003` | ConvNeXt-Tiny | 36.1% (epoch 1) | 6/20 (early) | 0.502 |

Improvement vs. pass 002 (32.8% on the easier HW-only split) is real, but the
gap to MiewID stayed enormous. From-scratch on M3 Pro is a dead end —
MiewID's pretraining on millions of cetacean images at 64 species cannot be
replicated.

**MiewID fine-tuning on cached features (M3 Pro):**

| Run | Head | Val Top-1 | Test Top-1 | Test Top-3 | Test MRR |
|---|---|---|---|---|---|
| `miewid-mlp-512-256-001` | Linear → BN → GELU → Dropout → Linear → BN | 82.6% (epoch 1) | 79.9% | 89.6% | 0.855 |
| `miewid-linear-512-margin05` | Single Linear, margin 0.5, weight-decay 1e-3 | 82.6% (epoch 2) | 81.9% | 88.9% | 0.868 |

Both heads peaked early and overfit (loss → 0 within 4–11 epochs). Val gain
of ~3 points didn't transfer to test. Net: **no reliable improvement over
frozen MiewID** on this scale of data.

## Why fine-tuning a head didn't help

1. Train set is small (676 images, 45 identities). A head with even 1 M
   parameters can memorize this perfectly.
2. MiewID's 2152-dim representation is already strong enough that a learnable
   linear projection has nothing useful to add — cosine distance over MiewID's
   raw features is already near-optimal for this data.
3. Triplet loss with margin 0.2 collapses to 0 once train pairs separate. We
   bumped to 0.5 and added weight-decay; same story.
4. The val/test gap of ~3 points is sample-size noise (144 queries each). To
   reliably detect a 3-point true improvement we'd need a much larger held-out
   set or proper bootstrapping.

## Decision

- **V1 stays MiewID frozen.** Already shipped via `serve_identifier.py`.
- **Local training track is reframed** as MiewID fine-tuning, not from-scratch.
- **From-scratch ResNet-50 / ConvNeXt artifacts are research-only.** They are
  not candidates for V1.
- **The actual fine-tune work moves to BYU GPU lab.** See
  `docs/byu-gpu-finetuning-plan.md` — full backbone fine-tune at 440×440 with
  ArcFace, larger batch, longer training.

## What stays useful from this session

- The combined manifest + split logic (used by all future runs).
- `cache_miewid_features.py` (sanity-checks the BYU MiewID load + provides a
  fast path for any future linear-probe experiments).
- `finetune_miewid_head.py` + the head module (template for the BYU full
  fine-tune script).
- The cosine LR + early-stop patches in `train_embedder.py` (also relevant for
  the BYU script).
- The new MiewID baseline number (82.6% Top-1 on combined test) — the bar to
  beat going forward.
