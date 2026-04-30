# fluke-model

Model workspace for the [Fluke](../fluke) orca photo-identification project.

The current workspace has two tracks:

1. **Frozen embedder baseline** — evaluate MiewID-msv3 and DINO embedders without
   training.
2. **Public orca training path** — train a beginner-friendly metric-learning
   model on official public dataset releases, starting with Happywhale/Kaggle
   filtered to killer whale/orca rows.

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
    download_beluga.py   # idempotent dataset download
    embed_catalog.py     # embed a manifest -> FAISS index
    identify.py          # query the index with a single image
    evaluate.py          # leave-one-out validation, writes results/<name>-eval.json
    download_happywhale.py    # official Kaggle download path
    build_orca_manifest.py    # filter Happywhale to killer whale/orca rows
    split_manifest.py         # train/val/test image splits by individual
    train_embedder.py         # local metric-learning training loop
    evaluate_orca.py          # compare trained model vs MiewID on same split
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

## Public orca training path

This is the first real training loop for Fluke. It uses the Happywhale Whale and
Dolphin Identification dataset through the official Kaggle access path, then
filters to killer whale/orca images.

### 1. Prepare Kaggle access

1. Create a Kaggle API token from `https://www.kaggle.com/settings`.
2. Save it as `~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME` and
   `KAGGLE_KEY`.
3. Open the Happywhale dataset/competition page in Kaggle and accept the terms.

### 2. Download and build manifests

```bash
uv sync
uv run python scripts/download_happywhale.py
uv run python scripts/build_orca_manifest.py
uv run python scripts/split_manifest.py
```

Outputs:

- raw Happywhale files: `data/happywhale/` (gitignored)
- orca-only manifest: `data/manifests/happywhale_orca.jsonl` (gitignored)
- closed-set splits: `data/manifests/happywhale_orca_splits/` (gitignored)

The split script drops individuals with too few images for closed-set eval and
guarantees no image path appears in more than one split.

### 3. First local training run

Start small. This is educational first; accuracy comes after the loop is solid.

```bash
# Tiny sanity run: can the model overfit a few repeated IDs?
uv run python scripts/train_embedder.py --overfit-tiny --epochs 3 --no-pretrained

# First real local run on the public orca split
uv run python scripts/train_embedder.py \
  --backbone resnet50 \
  --epochs 3 \
  --identities-per-batch 4 \
  --images-per-identity 2
```

The training script writes:

- checkpoint: `artifacts/models/orca/<run-name>/model.pt` (gitignored)
- metrics and training history: `results/orca/<run-name>.json`

### 4. Compare against MiewID

Evaluate MiewID and the trained checkpoint on the exact same orca split:

```bash
uv run python scripts/evaluate_orca.py \
  --trained-checkpoint artifacts/models/orca/<run-name>/model.pt
```

Results are written to `results/orca/evaluation.json` and appended to
`results/orca/summary.md`.

For a checkpoint-only smoke test that does not load MiewID:

```bash
uv run python scripts/evaluate_orca.py \
  --skip-baseline \
  --trained-checkpoint artifacts/models/orca/<run-name>/model.pt
```

Decision rule:

- If the trained model beats MiewID on the orca test split, it becomes the V1
  candidate.
- If it does not, MiewID remains the production fallback and the trained model
  remains the learning track.

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
