"""Closed-set retrieval evaluation helpers."""

from __future__ import annotations

import numpy as np

from fluke_model.index import aggregate_per_individual, build_index, search
from fluke_model.metrics import mean_reciprocal_rank, top_k_accuracy
from fluke_model.orca_data import OrcaManifestRow


def evaluate_retrieval(
    reference_embeddings: np.ndarray,
    reference_rows: list[OrcaManifestRow],
    query_embeddings: np.ndarray,
    query_rows: list[OrcaManifestRow],
    *,
    embedder_name: str,
    neighbors: int = 50,
) -> dict:
    """Evaluate query images against a reference image bank."""
    metadata = [
        {"path": row.path, "individual_id": row.individual_id, "species": row.species}
        for row in reference_rows
    ]
    bundle = build_index(reference_embeddings.astype(np.float32), metadata, embedder_name=embedder_name)
    predictions: list[list[str]] = []
    truths: list[str] = []
    per_query: list[dict] = []

    for i, row in enumerate(query_rows):
        hits = search(bundle, query_embeddings[i : i + 1].astype(np.float32), k=neighbors)
        aggregated = aggregate_per_individual(hits, top_n=3)
        pred_ids = [individual_id for individual_id, _score in aggregated]
        predictions.append(pred_ids)
        truths.append(row.individual_id)
        per_query.append(
            {
                "path": row.path,
                "truth": row.individual_id,
                "top5": [
                    {"individual_id": individual_id, "score": score}
                    for individual_id, score in aggregated[:5]
                ],
            }
        )

    return {
        "n_reference_images": len(reference_rows),
        "n_query_images": len(query_rows),
        "n_reference_individuals": len({r.individual_id for r in reference_rows}),
        "n_query_individuals": len({r.individual_id for r in query_rows}),
        "metrics": {
            "top_1": top_k_accuracy(predictions, truths, k=1),
            "top_3": top_k_accuracy(predictions, truths, k=3),
            "top_5": top_k_accuracy(predictions, truths, k=5),
            "mrr": mean_reciprocal_rank(predictions, truths),
        },
        "per_query": per_query,
    }
