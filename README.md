# fluke-model

Model workspace for the [Fluke](../fluke) orca photo-identification project.

The current workspace has two tracks:

1. **Frozen embedder baseline (V1, shipped)** — MiewID-msv3 evaluated on the
   combined HappyWhale + FinID-20 orca split. Hits **82.6% Top-1 / MRR 0.875**
   on the held-out test split. Served behind `scripts/serve_identifier.py`.
2. **MiewID fine-tuning track (V1.5 candidate)** — fine-tune MiewID on combined
   public-orca data. M3 Pro hits a ceiling at the frozen baseline; full backbone
   fine-tune with ArcFace at 440×440 needs the BYU GPU lab. See
   [`docs/byu-gpu-finetuning-plan.md`](docs/byu-gpu-finetuning-plan.md).

> **Historical note:** an earlier track tried from-scratch ResNet-50 / ConvNeXt
> training. That topped out around 36% Top-1 — far below MiewID — and is
> retained as research artifacts only. See
> [`docs/training-pass-003-and-miewid-finetune.md`](docs/training-pass-003-and-miewid-finetune.md)
> for the full breakdown.

The important boundary: this project does **not** train on random scraped photos
from Google Images, social media, ID guide PDFs, researcher catalogs, or whale
watching sites unless an explicit license/permission is recorded. Public
visibility is not permission.

The full spec lives at
[`fluke/docs/specs/m-model-0-prototype.md`](../fluke/docs/specs/m-model-0-prototype.md).
The supporting model plan is at [`fluke/docs/model.md`](../fluke/docs/model.md).

## Layout

```
fluke-model/
  pyproject.toml         # uv-managed
  configs/embedders.yaml # embedder model ids and image sizes
  src/fluke_model/       # importable package
    embedders.py         # load_embedder(name) -> (model, preprocess, dim)
    index.py             # FAISS build / save / load / query
    metrics.py           # top-k accuracy, MRR
    io.py                # manifest + image helpers
    orca_data.py          # public-orca JSONL manifests and split rules
    trainable.py          # trainable embedding model + triplet loss
    retrieval_eval.py     # closed-set retrieval evaluation
  scripts/
    download_beluga.py             # idempotent dataset download
    download_happywhale.py         # official Kaggle download path
    download_finid20.py            # FinID-20 (Zenodo, CC-BY-4.0) download
    embed_catalog.py               # embed a manifest -> FAISS index
    identify.py                    # query the index with a single image
    evaluate.py                    # leave-one-out validation
    build_orca_manifest.py         # filter Happywhale to killer whale/orca rows
    build_finid20_manifest.py      # convert FinID-20 cropped images to OrcaManifestRow JSONL
    build_combined_orca_manifest.py # concatenate per-source orca manifests
    split_manifest.py              # train/val/test image splits by individual
    train_embedder.py              # local metric-learning training loop
    evaluate_orca.py               # compare trained model vs MiewID on same split
    cache_miewid_features.py       # pre-compute MiewID embeddings (one-time)
    finetune_miewid_head.py        # train a head on cached MiewID features (M3 Pro path)
    serve_identifier.py            # FastAPI service hosting the frozen MiewID identifier
    build_reference_index.py       # build FAISS reference index for the identifier service
  tests/                 # pytest unit tests (metrics + index round-trip)
  data/                  # gitignored datasets and indices
  artifacts/             # gitignored trained checkpoints
  results/               # eval outputs; tracked in git
```

## Tooling

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.
The first eval run was bootstrapped with `uv 0.11.8` on Apple Silicon (arm64).

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Running the frozen baseline evaluation

```bash
uv sync                                # install dependencies
uv run python scripts/download_beluga.py   # download Beluga ID 2022 to data/
uv run python scripts/evaluate.py --embedder dinov2-small --subset 100
uv run python scripts/evaluate.py --embedder dinov3-convnext-tiny --subset 100
uv run python scripts/evaluate.py --embedder miewid-msv3 --subset 100
```

Each `evaluate.py` run writes `results/<embedder>-eval.json` and appends to
`results/summary.md`.

## Combined public-orca pipeline

Two licensed sources are stitched into a single closed-set training pipeline:

- **HappyWhale** (Kaggle competition release, terms-restricted) — 914 orca rows
  out of the full Whale and Dolphin Identification dataset
- **FinID-20** (Zenodo `10.5281/zenodo.16786268`, CC-BY-4.0) — 500 cropped
  Bigg's killer whale dorsal-fin images, 20 individuals × 25 photos each

> **Source policy:** no random scraped photos. Every image must come from a
> dataset with explicit licensing terms permitting ML training. Public
> visibility is not permission.

### 1. Prepare Kaggle access (HappyWhale only)

1. Create a Kaggle API token from `https://www.kaggle.com/settings`.
2. Save it as `~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME` and
   `KAGGLE_KEY`.
3. Open the Happywhale dataset/competition page in Kaggle and accept the terms.

### 2. Download and build the combined manifest

```bash
uv sync
uv run python scripts/download_happywhale.py
uv run python scripts/download_finid20.py
uv run python scripts/build_orca_manifest.py
uv run python scripts/build_finid20_manifest.py
uv run python scripts/build_combined_orca_manifest.py
uv run python scripts/split_manifest.py \
    --manifest data/manifests/orca_all.jsonl \
    --out-dir data/manifests/orca_all_splits \
    --min-images-per-individual 5
```

Outputs (all gitignored):

- raw HappyWhale files: `data/happywhale/`
- raw FinID-20 files: `data/finid-20/raw/`
- per-source manifests: `data/manifests/happywhale_orca.jsonl`,
  `data/manifests/finid20_orca.jsonl`
- combined manifest: `data/manifests/orca_all.jsonl`
- closed-set splits: `data/manifests/orca_all_splits/` (676 train / 144 val /
  144 test, 45 individuals)

### 3. Frozen MiewID baseline (V1)

```bash
uv run python scripts/evaluate_orca.py \
    --splits-dir data/manifests/orca_all_splits \
    --out results/orca/baseline-miewid-combined.json
```

Latest numbers on combined test: **82.6% Top-1 / 90.3% Top-3 / MRR 0.875**.

### 4. M3 Pro fine-tune track (cached features)

Pre-cache MiewID's 2152-dim features once, then iterate quickly on a learnable
head:

```bash
# 4-minute one-time cache (CPU)
uv run python scripts/cache_miewid_features.py \
    --manifest data/manifests/orca_all.jsonl \
    --out artifacts/miewid_features/orca_all.npz

# Train an MLP head on cached features
uv run python scripts/finetune_miewid_head.py \
    --features artifacts/miewid_features/orca_all.npz \
    --splits-dir data/manifests/orca_all_splits \
    --head mlp --hidden-dim 512 --embed-dim 256 \
    --epochs 60 --warmup-epochs 5 --early-stop-patience 10 \
    --identities-per-batch 8 --images-per-identity 4 \
    --run-name miewid-mlp-head-001
```

Empirically, head fine-tuning on this ~676-image train set does **not**
generalise past frozen MiewID. The realistic upgrade path is the BYU GPU lab —
see [`docs/byu-gpu-finetuning-plan.md`](docs/byu-gpu-finetuning-plan.md) for
the full backbone + ArcFace fine-tune approach.

### 5. From-scratch training (research only)

`scripts/train_embedder.py` runs the original from-scratch ResNet-50 /
ConvNeXt-Tiny pipeline with the new cosine LR schedule and early stopping.
This top-out around 37% Top-1 on combined data — left in the repo as a
reference baseline for the metric-learning loop, not as a V1 candidate.

```bash
uv run python scripts/train_embedder.py \
    --splits-dir data/manifests/orca_all_splits \
    --backbone resnet50 --epochs 20 \
    --warmup-epochs 2 --early-stop-patience 5 \
    --identities-per-batch 6 --images-per-identity 4 \
    --run-name resnet50-orca-combined-XXX
```

## Data rules

- Raw images stay in `data/` and are never committed.
- Model weights stay in `artifacts/` or on Hugging Face Hub, never in the app repo.
- Every metric report should record dataset source, split seed, image count,
  individual count, model version, and top-k metrics.
- Do not claim “orca model” performance from non-orca-only metrics. Broader
  cetacean training is allowed as pretraining, but the reported Fluke benchmark
  is the killer-whale/orca split.

## Caveats

- Run on CPU by default for parity across embedders. MPS works for DINOv2/DINOv3 but
  has occasional gaps for some operators that affect MiewID's load path.
- MiewID is gated behind a HuggingFace login as of 2026-04. If `transformers`
  loading fails, `embedders.py` documents the fallback and the eval continues with
  DINOv2 + DINOv3 only.
- The full Beluga ID dataset is ~680 MB. The first pass uses a ~100-image subset;
  the full eval is opt-in.
- The Happywhale download requires Kaggle credentials and accepted terms. If those
  are missing, `scripts/download_happywhale.py` fails with setup instructions.

## License

MIT for the harness code. Beluga ID 2022 is CDLA-Permissive 2.0; MiewID and DINOv2/v3
weights carry their own per-model licenses (see `configs/embedders.yaml`).
