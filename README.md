# fluke-model

M-Model-0 prototype: a zero-shot photo-identification evaluation harness for the
[Fluke](../fluke) project. The goal is to answer one question with real numbers:

> Does any frozen embedder (MiewID-msv3, DINOv2-Small, DINOv3-ConvNeXt-Tiny) clear the
> 70% top-3 bar on Beluga ID 2022 zero-shot?

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
  scripts/
    download_beluga.py   # idempotent dataset download
    embed_catalog.py     # embed a manifest -> FAISS index
    identify.py          # query the index with a single image
    evaluate.py          # leave-one-out validation, writes results/<name>-eval.json
  tests/                 # pytest unit tests (metrics + index round-trip)
  data/                  # gitignored datasets and indices
  results/               # eval outputs; tracked in git
```

## Tooling

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.
The first eval run was bootstrapped with `uv 0.11.8` on Apple Silicon (arm64).

If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Running the evaluation

```bash
uv sync                                # install dependencies
uv run python scripts/download_beluga.py   # download Beluga ID 2022 to data/
uv run python scripts/evaluate.py --embedder dinov2-small --subset 100
uv run python scripts/evaluate.py --embedder dinov3-convnext-tiny --subset 100
uv run python scripts/evaluate.py --embedder miewid-msv3 --subset 100
```

Each `evaluate.py` run writes `results/<embedder>-eval.json` and appends to
`results/summary.md`.

## Caveats

- Run on CPU by default for parity across embedders. MPS works for DINOv2/DINOv3 but
  has occasional gaps for some operators that affect MiewID's load path.
- MiewID is gated behind a HuggingFace login as of 2026-04. If `transformers`
  loading fails, `embedders.py` documents the fallback and the eval continues with
  DINOv2 + DINOv3 only.
- The full Beluga ID dataset is ~680 MB. The first pass uses a ~100-image subset;
  the full eval is opt-in.

## License

MIT for the harness code. Beluga ID 2022 is CDLA-Permissive 2.0; MiewID and DINOv2/v3
weights carry their own per-model licenses (see `configs/embedders.yaml`).
