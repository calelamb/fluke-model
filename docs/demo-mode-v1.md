# Demo-Mode V1 (2026-05-03)

The first deployable shape of the identifier feature, running against the 45
publicly-licensed orca individuals in the combined HappyWhale + FinID-20
catalog. Built so the project has a working thing to point at while
Salish-Sea catalog licensing is in flight (see
`fluke/docs/data-licensing.md`).

## What demo mode actually is

A FAISS index over MiewID-msv3 features for **5 reference photos per
individual** across **45 individuals** = **225 reference vectors**. Wired
through `serve_identifier.py` so a user upload returns a top-3 list of
licensed individuals.

```
Upload (any orca photo)
    │
    ▼
MiewID-msv3 (frozen, 440×440, 2152-dim)
    │
    ▼
FAISS k-NN against 225 reference vectors
    │
    ▼
Per-individual aggregation → top-3 catalogIds with scores
```

## Performance — runtime path on the val split

These are the deployed-shape numbers (5 reference photos per individual,
runtime aggregation), not the offline eval numbers (full train set as
reference, leave-one-out k-NN). The runtime numbers are what users actually
experience.

| Source | Top-1 | Top-3 |
|---|---|---|
| FinID-20 | **95.0%** | 97.5% |
| HappyWhale | 45.3% | 64.1% |
| **Combined** | **72.9%** | **82.6%** |

FinID-20 is essentially solved. HappyWhale is harder because the Kaggle
release is uncropped and noisier than FinID-20's pre-cropped dorsal-fin
crops. The combined Top-3 of 82.6% maps cleanly to Branch A of the
M-Model-0 spec ("ship as V1").

## What this is and isn't

**Is.** A real, working photo identifier serving against 45 publicly-licensed
orca individuals. Useful as a "try the matcher" feature, useful as a
demonstrable thing to send research orgs whose catalogs we want to license.

**Isn't.** A real V1 — the matched individuals are not Salish Sea residents
the user is likely to have just seen. The UI must say so plainly (see "UX
copy" below).

## How to run it

```bash
# Build the demo reference manifest (file:// URLs to local images)
uv run python scripts/build_demo_reference_manifest.py

# Build the FAISS index from the manifest
uv run python scripts/build_reference_index.py \
    --manifest data/manifests/demo_reference.jsonl \
    --out-dir artifacts/reference-index/demo

# Run the service
FLUKE_REFERENCE_INDEX_DIR=artifacts/reference-index/demo \
    uv run python scripts/serve_identifier.py
```

`/health` returns `{"status": "ok", ...}`. Upload a JPEG to `/identify` and
get back a top-3 list of catalogIds with scores. Wire the Fluke web app's
`/api/v1/identify` stub to this endpoint and the feature is live.

## File layout this adds

```
fluke-model/
  scripts/
    build_demo_reference_manifest.py    # NEW — orca_all → reference manifest
  data/manifests/
    demo_reference.jsonl                # NEW (gitignored — built locally)
  artifacts/reference-index/demo/       # NEW (gitignored — built locally)
  src/fluke_model/
    identify_runtime.py                 # MODIFIED — file:// URL support
```

## UX copy (what the Fluke web app should say on the identify page)

> Currently matching against **45 publicly-catalogued orcas** from
> [HappyWhale](https://happywhale.com) and the [FinID-20](https://zenodo.org/records/16786268)
> Bigg's killer whale dataset. **Salish Sea residents will be added as
> licensing closes** — until then, this is for trying the feature, not
> identifying whales you've actually seen.
>
> The match is useful, not authoritative. For real ID, see the
> [Center for Whale Research](https://www.whaleresearch.com) (Southern
> Residents) and [Bay Cetology](https://baycetology.org) (Bigg's).

## Retirement plan

When the first Salish Sea organization grants catalog access:

1. Build a separate `salish-sea` reference index from the licensed
   catalog photos.
2. Have the runtime serve that index by default.
3. Move demo-mode catalog into a "public examples" toggle (or remove
   entirely once real-catalog coverage exceeds 30+ individuals).
4. Update the UI copy. Drop "Currently matching against 45 publicly-
   catalogued orcas" and replace with the licensed catalog's framing.

## Why this is policy-clean

Per `fluke/docs/data-licensing.md`, every photograph in the demo index has
an explicit ML-training-permitting license:

- HappyWhale: Kaggle competition terms permit ML training.
- FinID-20: CC-BY-4.0, with attribution to Bergler et al., 2025.

No scraping, no implicit-permission assumptions. The thumbnail-display
guarantees from `data-licensing.md` (attribution, link-back, take-down
within 48 hours, no commercial use) apply identically to the demo catalog.
