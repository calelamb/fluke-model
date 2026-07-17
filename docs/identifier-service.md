# Identifier service launch contract

The production service is an authenticated, CPU-compatible DINOv2 retrieval service. It is
deliberately **not ready** until a reference index with a written production-rights attestation has
been published.

## Required environment

```text
FLUKE_MODEL_API_KEY=<at least 32 random characters>
FLUKE_REFERENCE_ALLOWED_HOSTS=<comma-separated exact HTTPS object-storage hosts>
FLUKE_REFERENCE_INDEX_DIR=/persistent/reference-index
FLUKE_MODEL_ARTIFACT_DIR=/read-only/dinov2-small
PORT=4100
```

Generate the API key directly in the deployment secret store. Do not commit or paste it into logs.
The index directory must be persistent storage; immutable versions live under `versions/` and an
atomically replaced `current.json` pointer selects the live version.

The model directory must contain regular-file copies of `config.json`, `model.safetensors`, and
`preprocessor_config.json` matching the SHA256 allowlist in `model_artifact.py`. The production
container bakes those verified files at build time and forces Hugging Face and Transformers offline;
runtime startup never downloads weights.

## Endpoints

- `GET /health` is public liveness and returns `200` when the process is responsive.
- `GET /ready` returns `200` only when a rights-attested index and the pinned model can both load and
  a finite model probe matches the index dimension; otherwise it returns `503`.
- `POST /identify` accepts bounded JPEG, PNG, or WebP multipart uploads.
- `POST /identify-json` accepts the same formats as validated base64.
- `POST /rebuild-index` downloads reference images only from exact HTTPS allowlisted hosts, blocks
  local/file/private targets and redirects, pins TLS connections to the already validated public IP
  while preserving certificate SNI/Host checks, validates image bounds, and publishes atomically.

Rebuilds are serialized, cooperatively cancelled at their total deadline, enforce an aggregate pixel
budget, and embed one decoded reference at a time so a maximum-size catalog cannot become a single
unbounded tensor/image batch. A failed, partial, cancelled, or timed-out rebuild never changes the
live pointer.

CPU identification is single-flight. If an inference exceeds its response deadline, its worker keeps
the sole slot until it exits and closes its image; additional requests fail closed with `503` instead
of accumulating workers or decoded images.

All POST endpoints require `X-Fluke-Model-Key`. Inference and rebuild endpoints have independent
in-process rate limits. Index rebuild/publication requires one service replica because the generation
fence is process-local. Any future multi-replica deployment needs both a distributed publication
fence and a shared gateway rate limit before scaling out.

## Rights gate

Every rebuild request must include `rightsAttestation` covering the exact pinned model revision and
every reference's `rightsSourceId`. The attestation records an approver, approval time, SPDX model
license, commercial-use flags, and HTTPS evidence URLs. Missing, mismatched, non-commercial, or
uncovered rights fail before any image is fetched.

Production currently permits only:

- model: `facebook/dinov2-small`
- revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- license recorded by the publisher: Apache-2.0

MiewID and DINOv3 are excluded from executable production paths. Existing research results are not
a production-rights grant for their checkpoints or training images.

## Confidence semantics

Returned `score` values are cosine retrieval similarities, not probabilities or confirmed IDs.
Until a held-out, rights-cleared calibration set and versioned calibration artifact are approved,
the response always reports:

```json
{
  "confidenceBand": "unavailable",
  "confidenceSemantics": "uncalibrated_similarity_not_probability"
}
```

Human confirmation is required before presenting an identity as confirmed.

## Local verification

```bash
uv sync --locked
uv run ruff check .
uv run pytest --cov=src/fluke_model --cov-report=term-missing --cov-fail-under=80
uv run bandit -q -lll -r src scripts
uv run pip-audit --skip-editable
docker build -t fluke-model:local .
```
