# OrcaWatch mobile embedder model card

## Release status

The mobile embedder and release tooling are implemented, but no production mobile release is
approved. This repository does not contain a rights-cleared production reference catalog or a
rights-cleared production evaluation set. Historical HappyWhale, FinID-20, MiewID, and local
research results are not substitutes for those inputs and must not be copied into a release.

`scripts/verify_mobile_release.py` is the authoritative machine gate. A candidate is releasable
only when it writes `mobile-release-report.json` with `ready: true` and exits zero. Missing inputs,
invalid schemas, unsafe paths, digest drift, absent rights, non-finite metrics, empty evaluations,
or any missed threshold produce `ready: false` and a nonzero exit.

## Model and intended use

The model is the exact pinned `facebook/dinov2-small` revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`, exported as a fixed-shape iOS 17 Core ML ML Program.
It accepts one preprocessed RGB image tensor with shape `(1, 3, 224, 224)` and returns one
384-dimensional L2-normalized embedding. The package is reloaded in an isolated directory and must
expose exact Float32 `pixels (1,3,224,224)` and `embedding (1,384)` features. The mobile catalog uses
cosine similarity internally to rank candidate matches for human review, while the API/catalog
handoff emits `uncalibrated_similarity_not_probability`.

Intended use is decision support for authorized OrcaWatch users comparing a suitable orca dorsal
fin photograph with a rights-cleared, redistributed reference catalog. The output may help a human
find plausible catalog candidates. It does not establish an identity.

## Excluded uses

- Do not present cosine similarity as a probability, confidence percentage, or confirmed identity.
- Do not use the model for autonomous identification, enforcement, tracking, or consequential
  decisions without qualified human review.
- Do not use people, other species, arbitrary web images, social-media images, researcher catalogs,
  ID guides, or any source lacking explicit permission for the intended ML and commercial use.
- Do not reuse the repository's synthetic test fixtures as production references or evaluation
  evidence.
- Do not extrapolate release results to devices, image conditions, populations, or catalog versions
  absent from the approved evaluation design.

## Data provenance and rights requirements

Every bundled reference row must name a stable `sourceId`. The release rights attestation must cover
the exact pinned model revision and exactly the reference source IDs in `catalog/metadata.json`.
Every source must explicitly allow commercial production use, redistribution in the mobile bundle,
and mobile ML use, with absolute HTTPS evidence and a named, timezone-aware approval. The verifier
checks that the attestation bytes match the catalog's recorded SHA256.
The attestation has an exact `purpose` field: `purpose: production` is mandatory for a release;
the committed synthetic fixture is `purpose: test` and intentionally fails that production gate.

The production evaluation owner must separately document the provenance, permission basis,
collection protocol, split policy, exclusions, and cohort definitions for all evaluation images.
Evaluation data need not be redistributed in the app, but its terms must explicitly permit the
commercial ML evaluation being reported. Approval records belong in controlled release evidence;
private image data and non-redistributable benchmark material must not be committed to this
repository. No historical result file establishes these rights.

## Score and threshold semantics

Catalog ranking uses cosine similarity between L2-normalized embeddings. Higher scores mean closer
vectors under the evaluated model; they are not calibrated probabilities. The catalog also records
a score threshold and margin threshold selected by the approved evaluation process. The release
verifier does not silently tune either threshold.

Every release must meet all binding gates:

| Gate | Requirement |
|---|---:|
| PyTorch/Core ML parity | minimum paired cosine similarity `>= 0.999` |
| Closed-set top-1 | `>= 0.65` |
| Closed-set top-3 | `>= 0.80` |
| Worst required open-set cohort false-accept rate | `<= 0.05` |

Each parity and evaluation cohort must have a positive integer sample count. Every metric must be
finite. The false-accept gate uses the worst rate across `openSet`, `nonOrca`, `poorQuality`,
`occlusion`, and `distributionShift`; passing an aggregate while a required cohort fails is not
allowed. The production owner remains responsible for demonstrating that the cohort design and
sample sizes support the claimed operating decision.

## Fixed release layout

The verifier accepts exactly this layout under `--release-dir`:

```text
FlukeEmbedder.mlpackage/
export-metadata.json
rights-attestation.json
catalog/
  manifest.json
  metadata.json
  references.f16
evaluation/
  parity-pytorch.npy
  parity-coreml.npy
  fixture-manifest.json
  decisions.json
  evaluation-plan.json
  parity.json
  closed-set.json
  open-set.json
  non-orca.json
  poor-quality.json
  occlusion.json
  distribution-shift.json
mobile-release-report.json  # optional prior output; never trusted as input evidence
```

No other root, catalog, or evaluation entries are allowed. A prior report may be present so the
verifier can safely replace it, but its contents are ignored. The newly generated report is bound
to current inputs with this exact top-level schema:

```text
schemaVersion, modelPackageSha256, catalogManifestSha256, ready, thresholds, gates
```

The two digest fields contain the actual verified Core ML package-tree and catalog-manifest SHA256
values. Either is literal JSON `null` when the corresponding input cannot be safely hashed; a stale
or copied report can therefore never supply release identity. Both fields are independent readiness
gates: each must contain a valid lowercase SHA256 even when all metric and boundary gates pass.

Within the release root, verifier output may be written only to the canonical
`mobile-release-report.json` path. A custom `--report` destination is allowed only outside the
release root, preventing verifier output from creating an extra entry that would invalidate the
next exact-layout check.

The two parity files are exact two-dimensional Float32 NumPy arrays with equal positive `(N, 384)`
shape, finite values, and unit-normalized rows. Before allocation, the verifier bounds file bytes,
parses and validates the `.npy` header, and enforces a maximum sample-row count. Parity is the
minimum per-row cosine similarity.

The evaluation directory also contains canonical `fixture-manifest.json` and `decisions.json`
files. The fixture manifest binds each stable fixture ID and approved role to a canonical relative
path and the SHA256 of the actual image bytes. The builder rejects absolute paths, traversal,
symbolic links, duplicate paths/IDs, missing files, and byte-digest mismatches. Every report's
fixture-set digest is recomputed from this canonical manifest.

Raw decisions record the fixed evaluation type, fixture ID, truth identity where applicable,
ranked identities, first and second scores, and the eligibility result. They also carry the exact
catalog score and margin thresholds. Verification requires exact fixture-role coverage, checks the
thresholds against the catalog, recomputes eligibility with the iOS Float32 rule, and recomputes
closed-set top-1/top-3 and every cohort false-accept rate. Hand-edited metric summaries cannot pass.

The production corpus manifest has exact top-level keys `schemaVersion`, `evidencePurpose`,
`provenanceUrl`, and `rows`. Each row has exactly `fixtureId`, `relativePath`, `imageSha256`,
`roles`, `referencePhotoId`, `whaleId`, `catalogId`, and `sourceId`. Reference rows require all four
identity/source values; closed-set rows require `whaleId`.

The evaluation plan has exactly `schemaVersion`, `evidencePurpose`, `approvedBy`, timezone-aware
`approvedAt`, HTTPS `provenanceUrl`, and `cohortDefinitions`; it defines parity, closed set, and all
five open/robustness cohorts. Production construction rejects any corpus, plan, or rights
attestation whose purpose is not `production`. The canonical plan is preserved as
`evaluation/evaluation-plan.json`, and every report provenance URL must match it.

`evaluation/parity.json` has this exact schema:

```text
schemaVersion, evaluationType, evidencePurpose, provenanceUrl,
modelPackageSha256, catalogManifestSha256, sourceModelSha256,
preprocessingVersion, fixtureSetSha256, sampleCount,
pytorchEmbeddingsSha256, coremlEmbeddingsSha256
```

`closed-set.json` has exactly these keys:

```text
schemaVersion, evaluationType, evidencePurpose, provenanceUrl, fixtureSetSha256,
modelPackageSha256, catalogManifestSha256, sampleCount, top1, top3
```

Each of the five open-set JSON files has exactly these keys:

```text
schemaVersion, evaluationType, evidencePurpose, provenanceUrl, fixtureSetSha256,
modelPackageSha256, catalogManifestSha256, sampleCount, falseAcceptRate
```

`schemaVersion` is `1`. `evaluationType` must match the fixed filename: `closedSetRetrieval`,
`openSet`, `nonOrca`, `poorQuality`, `occlusion`, or `distributionShift`. Every report must name the
actual Core ML package-tree SHA256 and the exact `catalog/manifest.json` file SHA256. Missing or
extra fields fail the release.

Every evaluation JSON uses `evidencePurpose: production`, an absolute HTTPS `provenanceUrl`, and a
lowercase SHA256 `fixtureSetSha256`. Catalog manifests also carry positive integer
`minimumAppBuild` and `maximumAppBuild` values with `minimumAppBuild <= maximumAppBuild`; clients
must reject catalogs outside that range. Export metadata binds all three pinned source artifacts
(`config.json`, `model.safetensors`, and `preprocessor_config.json`) and records Core ML Tools,
NumPy, Pillow, Python, PyTorch, Transformers, macOS, and Xcode versions.

## Known limitations

- Dorsal-fin appearance varies with pose, occlusion, water, lighting, image quality, injury, age,
  camera processing, and catalog coverage.
- Closed-set retrieval assumes the correct identity is represented in the reference catalog;
  real-world queries are open set.
- Threshold behavior can shift when the catalog, population, capture conditions, preprocessing,
  device runtime, or model package changes. Every changed package or catalog requires fresh,
  digest-bound evaluation.
- A positive sample count is a structural machine check, not proof of statistical power or broad
  ecological representativeness.
- Core ML/PyTorch parity demonstrates numerical agreement on the approved parity inputs, not model
  accuracy, calibration, fairness, or field robustness.

## Human review and release ownership

A launch owner must approve the catalog rights, evaluation rights, cohort design, sample adequacy,
and documented limitations before distribution. Reviewers should inspect every failed gate in
`mobile-release-report.json`; bypassing or deleting a failed report does not authorize release.
