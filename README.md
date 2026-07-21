# fluke-model

Rights-safe model and retrieval-service workspace for Fluke's orca photo-identification track.

## Launch status

The service foundation is deployable, authenticated, bounded, rate-limited, and fail-closed. Its
only production-enabled backbone is the exact pinned `facebook/dinov2-small` revision recorded in
[`docs/identifier-service.md`](docs/identifier-service.md). Reference images and indexes cannot be
used until a written commercial-production rights attestation covers every source.

The repository intentionally does not make a production accuracy claim. Existing MiewID,
HappyWhale, and FinID-20 experiments remain research evidence only; their presence on disk or in
historical result files does not establish production rights. MiewID remote code/checkpoints and
non-commercial DINOv3 weights are excluded from the executable registry.

The on-device Core ML package and fail-closed release verifier are implemented, but mobile release
readiness remains blocked until a rights-cleared production catalog and digest-bound production
evaluations exist. See the [mobile model card](docs/mobile-model-card.md) for intended use,
exclusions, score semantics, provenance requirements, fixed release layout, and binding thresholds.

## Service

Required environment:

```bash
export FLUKE_MODEL_API_KEY='<random secret with at least 32 characters>'
export FLUKE_REFERENCE_ALLOWED_HOSTS='rights-cleared-object-storage.example'
export FLUKE_REFERENCE_INDEX_DIR='artifacts/reference-index'
export FLUKE_MODEL_ARTIFACT_DIR='artifacts/models/dinov2-small'
uv run python scripts/serve_identifier.py
```

`GET /health` is process liveness. `GET /ready` returns `503` until an attested index is published.
All inference and rebuild routes require `X-Fluke-Model-Key`. See the
[service launch contract](docs/identifier-service.md) for request, rights, confidence, and storage
semantics.

Build a complete rights-attested request from the CLI:

```bash
uv run python scripts/build_reference_index.py \
  --request /secure/path/rebuild-request.json \
  --model-artifact-dir artifacts/models/dinov2-small \
  --allowed-host rights-cleared-object-storage.example
```

The request file and evidence may contain private operational information and must not be committed.

## Verification

```bash
uv sync --locked
uv run ruff check .
uv run pytest --cov=src/fluke_model --cov-report=term-missing --cov-fail-under=80
uv run bandit -q -lll -r src scripts
uv run pip-audit --skip-editable
docker build -t fluke-model:local .
```

Verify an assembled mobile candidate with:

```bash
uv run python scripts/verify_mobile_release.py \
  --release-dir artifacts/mobile-release/candidate
```

The command writes `mobile-release-report.json` in the candidate directory and exits nonzero unless
every package, catalog, digest, rights, parity, closed-set, open-set, cohort-presence, finiteness, and
sample-count gate passes. Synthetic test fixtures are never production release evidence.

Once a launch owner supplies a rights-cleared corpus and approval plan, build the complete release
on macOS with an executable Core ML runtime:

```bash
uv sync --locked --extra coreml-export
uv run python scripts/build_mobile_release.py \
  --corpus-manifest /secure/release/corpus.json \
  --corpus-root /secure/release/images \
  --evaluation-plan /secure/release/evaluation-plan.json \
  --rights /secure/release/rights-attestation.json \
  --model-artifact artifacts/dinov2-small \
  --model-package artifacts/mobile-release/candidate/FlukeEmbedder.mlpackage \
  --model-metadata artifacts/mobile-release/candidate/export-metadata.json \
  --manifest-version RELEASE_ID \
  --minimum-app-build APPROVED_MINIMUM \
  --maximum-app-build APPROVED_MAXIMUM \
  --score-threshold APPROVED_SCORE_THRESHOLD \
  --margin-threshold APPROVED_MARGIN_THRESHOLD \
  --output-dir artifacts/mobile-release/RELEASE_ID
```

Thresholds are required approval inputs; the builder never searches or tunes them. It validates and
hashes every manifest-named image, executes both pinned runtimes for parity, emits raw retrieval
decisions, recomputes all metrics, stages a fresh release, and publishes only after the authoritative
verifier returns `ready: true`. Core ML execution is mandatory and is never replaced with fabricated
output on an unsupported platform.

Export the exact three catalog resources for a compatible iOS build with:

```bash
uv run python scripts/export_verified_mobile_catalog.py \
  --release-dir artifacts/mobile-release/RELEASE_ID \
  --app-build APPROVED_APP_BUILD \
  --output-dir artifacts/ios/IdentifierCatalog
```

Production evidence is explicit: rights attestations use `purpose: production`, while the committed
fixture uses `purpose: test` and cannot satisfy the production verifier. Evaluation reports and
`evaluation/parity.json` use `evidencePurpose: production`, an absolute HTTPS `provenanceUrl`, and a
digest-bound `fixtureSetSha256`. The parity report also binds the package, catalog manifest, pinned
source model, preprocessing version, sample count, and both `.npy` SHA256 values. The verifier
reloads an isolated Core ML package and requires the exact Float32 `pixels (1,3,224,224)` to
`embedding (1,384)` interface; synthetic package bytes fail by default.

Catalog manifests expose `minimumAppBuild` and `maximumAppBuild` as an ordered positive-integer
compatibility range. The client-facing score literal is
`uncalibrated_similarity_not_probability`; cosine is only the internal ranking operation and must
never be surfaced as a probability.

The container downloads the exact revision during its image build, verifies every required file
against a committed SHA256 allowlist, and runs Transformers in offline/local-only mode. CI builds a
rights-attested project-generated fixture index, requires container readiness `200`, and exercises a
real DINOv2 identification before accepting the image.

## Research layout

```text
src/fluke_model/       importable model, evaluation, index, and service code
scripts/               research, dataset, index, and service entrypoints
configs/               model registry with explicit production enablement
tests/                 unit, integration, security, and evaluation tests
data/                  gitignored local datasets
artifacts/             gitignored features, weights, and reference indexes
results/               historical research metrics and reports
```

Research scripts retain prior metric-learning work, including ResNet/ConvNeXt experiments and
MiewID-derived historical metrics. They are not enabled service dependencies. Do not download or
train on arbitrary public images, social media, ID guides, researcher catalogs, or other material
without a license or written permission explicitly permitting the intended ML and production use.

## Data and result rules

- Keep raw images in `data/` and checkpoints/indexes in `artifacts/`; both are gitignored.
- Record source, terms reviewed, split seed, counts, model revision, and top-k/MRR for every result.
- Never treat public visibility, a downloadable checkpoint, or a dataset metadata label as a rights
  grant when the underlying terms conflict or are absent.
- Never report similarity as probability or a confirmed identity. Human review remains mandatory.

The harness code is MIT. Third-party model weights and datasets retain their own terms, and the
runtime rights gate remains authoritative for production use.

## Contributions

Fluke-model is part of the [Fluke](https://github.com/calelamb/fluke) project — a
non-commercial field guide for Pacific Northwest orcas — and is a single-author
personal project. It is **not accepting outside contributions**; pull requests and
feature issues aren't being reviewed. The source is public so the rights-safe
methodology can be read and learned from, and the MIT-licensed harness code reused
under its terms. Model weights and datasets keep their own licenses, and the
runtime rights gate stays authoritative for any production use.

The one message always welcome: if you hold rights to imagery the project could
train on, or want to discuss a licensed catalog, reach the maintainer through the
contact details on the live Fluke site.
