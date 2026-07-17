# Model Release Verification

Verified on 2026-07-17 against `53553fa` (`feat: harden deployable model service`).

## Verdict

The model service foundation is deployable and fail-closed. It is not authorized or ready for
production identification yet. Production readiness correctly remains blocked until Fluke has both:

1. a written commercial-production rights attestation covering the exact pinned model revision and
   every reference source; and
2. a complete rights-cleared reference catalog published through the authenticated rebuild path.

Do not substitute the repository's historical HappyWhale, FinID-20, MiewID, or DINOv3 research
artifacts for either requirement. They are not executable production inputs.

## Fresh local evidence

```text
uv sync --locked
PASS

uv run pytest --cov=src/fluke_model --cov-report=term-missing --cov-fail-under=80
98 passed; total coverage 87.06%

uv run ruff check .
All checks passed

uv run pip-audit
No known vulnerabilities found (the local editable project itself is not on PyPI)

uv run pytest tests/test_container_contract.py -q
3 passed
```

`uv run bandit -q -r src scripts` reported no high-severity findings. Its medium findings were the
required public service bind and two conservative false positives at Transformers calls: the online
path passes the pinned revision, while the container path loads only SHA256-verified local files with
offline mode forced. The CI policy command is `uv run bandit -q -lll -r src scripts`, which passed on
the exact remote commit.

Docker is not installed on this workstation, so a fresh local container build could not run. The
exact commit's GitHub Actions run `29567752357` did run the container path successfully: it built
image `sha256:3efcff99acb23ad6133ace51a0869aab928fb35feb8c2cb7e920cce3499a6762`, built a
project-owned synthetic rights-attested index, returned `200` from `/health` and `/ready`, and
completed authenticated inference with the required uncalibrated-confidence semantics.

A direct local service boot with an empty reference-index directory returned:

```text
GET /health -> 200 {"status":"ok"}
GET /ready  -> 503 {"status":"not_ready","reason":"index_unavailable"}
POST /identify-json without the service key -> 401
POST /rebuild-index with a wrong service key -> 401
```

## Contracts verified

- The container downloads the exact pinned DINOv2 revision at build time, verifies all required
  files by SHA256, then runs with Hugging Face and Transformers offline.
- Readiness validates a complete atomic index, exact model identity and revision, written rights for
  every catalog source, non-empty consistent metadata, and a finite real model probe.
- Identify accepts only bounded JPEG, PNG, or WebP input, validates decoded pixels, runs single-flight,
  times out without releasing the occupied worker slot early, and requires a constant-time API key.
- Rebuild has schema and count bounds, exact HTTPS host allowlisting, DNS/IP and redirect SSRF
  defenses, TLS hostname verification, per-image and aggregate resource limits, a total deadline,
  serialization, and latest-generation atomic publication.
- Similarity scores are explicitly not probabilities or confirmed identities.
- MiewID and DINOv3 are absent from the executable model registry.

This verification adds a CI contract that boots the built container without a catalog and requires
the exact fail-closed `503 index_unavailable` response before the synthetic positive-readiness test.

## External launch blocker

The remaining blocker is not code: Fluke needs a launch owner to supply and approve written evidence
of commercial-production rights for every reference-photo source, assemble the corresponding catalog
with stable photo/catalog IDs on an exact HTTPS object-storage host, and submit the matching signed
attestation through `/rebuild-index`. Until then `/ready` must remain `503`, and the API/iOS app must
keep identification disabled.

The container can be deployed on a later free allowance if a provider offers sufficient CPU, memory,
image size, and persistent storage, but this verification neither selected a paid service nor claimed
that a zero-cost host can satisfy those resource requirements.
