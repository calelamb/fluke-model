# Production Mobile Release Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently verify a deterministic mobile release from an approved, rights-cleared image corpus without trusting hand-authored metric summaries.

**Architecture:** Add immutable corpus and decision contracts, a deterministic metric engine, real PyTorch/Core ML runtime adapters, and an atomic release orchestrator. Extend the authoritative verifier to bind a canonical fixture manifest to image bytes and recompute every reported retrieval metric from raw decisions. Add a separate verified-catalog exporter for the future iOS bundle boundary.

**Tech Stack:** Python 3.11, NumPy, Pillow, PyTorch, Transformers, Core ML Tools, pytest.

## Global Constraints

- Production CLI rejects test-purpose rights/evaluation plans and never fabricates model output.
- Thresholds are explicit launch-approved inputs and are never tuned.
- All corpus inputs are regular non-symlink files beneath one corpus root with unique canonical paths, IDs, and SHA256 digests.
- Core ML execution is mandatory for production and fails clearly when the runtime cannot load the package.
- Catalog reference count is at most 50,000 and app build compatibility is explicit.
- Publication uses a fresh sibling staging directory and atomic rename.
- Synthetic fixtures remain test-purpose and cannot pass production verification.

---

### Task 1: Canonical evidence and metric contracts

**Files:**
- Create: `src/fluke_model/mobile_release_evidence.py`
- Test: `tests/test_mobile_release_evidence.py`

- [ ] Write failing tests for exact schemas, traversal/symlink/duplicate rejection, canonical image hashing, score-and-margin decisions, top-1/top-3, and all five false-accept cohorts.
- [ ] Run the focused tests and confirm they fail because the module is absent.
- [ ] Implement immutable validated records, canonical hashing, decision serialization, and deterministic metric recomputation.
- [ ] Run the focused tests and retain green output.

### Task 2: Authoritative verifier recomputation

**Files:**
- Modify: `src/fluke_model/mobile_release_contracts.py`
- Modify: `tests/test_mobile_release.py`
- Modify: `tests/test_mobile_release_cli_and_reports.py`

- [ ] Write failing tamper tests proving changed report metrics, decisions, fixture rows, image bytes, and cohort identity are rejected.
- [ ] Run focused verifier tests and confirm expected failures.
- [ ] Require `fixture-manifest.json` and `decisions.json`, recompute the fixture digest and metrics, and compare exact report values/counts.
- [ ] Run focused verifier tests and retain green output.

### Task 3: Fail-closed production builder

**Files:**
- Create: `src/fluke_model/mobile_release_builder.py`
- Create: `scripts/build_mobile_release.py`
- Test: `tests/test_mobile_release_builder.py`

- [ ] Write failing tests for production-purpose enforcement, path overlap rejection, unavailable Core ML runtime, reference limits, deterministic evidence, and atomic publication.
- [ ] Run focused builder tests and confirm expected failures.
- [ ] Implement pinned PyTorch/Core ML adapters, normalized embeddings, catalog construction, decision generation, evidence reports, fresh staging, and final verifier invocation.
- [ ] Run focused builder tests and retain green output.

### Task 4: Verified catalog export contract

**Files:**
- Create: `src/fluke_model/mobile_catalog_export.py`
- Create: `scripts/export_verified_mobile_catalog.py`
- Test: `tests/test_mobile_catalog_export.py`

- [ ] Write failing tests for ready-report binding, exact three-file output, 50,000-row cap, app-build compatibility, symlink/overlap rejection, and atomic publication.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement verified release revalidation and exact catalog export.
- [ ] Run focused tests and retain green output.

### Task 5: Documentation and full verification

**Files:**
- Modify: `docs/mobile-model-card.md`
- Modify: `docs/superpowers/model-release-verification.md`
- Modify: `README.md`

- [ ] Document both input schemas, commands, non-tuning policy, Core ML platform requirement, raw-evidence contract, and iOS export handoff.
- [ ] Run the complete pytest suite with coverage and require at least 80% overall.
- [ ] Run Ruff, Bandit, and pip-audit; resolve applicable findings.
- [ ] Review the full diff for security and contract compatibility, then commit with a conventional message.
