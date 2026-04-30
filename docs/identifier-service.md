# Identifier Service

This package runs the local MiewID-powered identifier service for Fluke.

## Start the Service

```bash
cd fluke-model
uv run python scripts/serve_identifier.py
```

Default URL: `http://localhost:4100`

Health check:

```bash
curl http://localhost:4100/health
```

## Build the Reference Index

The normal path is through the Fluke admin UI:

1. Start the API and web app.
2. Start this service.
3. Upload reference photos at `/admin/reference-photos`.
4. Click **Rebuild index**.

The API sends reference-photo metadata to `POST /rebuild-index`, and this service:

- downloads each reference image from the API's local `/uploads/` URL
- optionally applies a crop if one is recorded
- embeds the image with `conservationxlabs/miewid-msv3`
- writes `artifacts/reference-index/index.faiss`
- writes `artifacts/reference-index/metadata.json`
- writes `artifacts/reference-index/index_info.json`

`artifacts/` is gitignored.

## CLI Index Builder

You can also build an index from a manifest:

```bash
uv run python scripts/build_reference_index.py --manifest references.json
```

Manifest shape:

```json
{
  "references": [
    {
      "referencePhotoId": "ref_1",
      "catalogId": "J35",
      "name": "Tahlequah",
      "url": "http://localhost:4000/uploads/reference-photos/...",
      "side": "LEFT",
      "quality": "USABLE",
      "crop": null
    }
  ]
}
```

## Identify

The Fluke API calls:

```http
POST /identify-json
```

with a base64 encoded image. The service returns:

```json
{
  "matches": [
    {
      "catalogId": "J35",
      "name": "Tahlequah",
      "score": 0.82,
      "rank": 1,
      "matchedReferencePhotoIds": ["..."],
      "explanation": "Closest visual match across 3 reference photos."
    }
  ],
  "confidenceBand": "medium",
  "model": "miewid-msv3",
  "indexVersion": "20260430T090000Z"
}
```

Scores are cosine-similarity retrieval scores, not confirmed IDs.
