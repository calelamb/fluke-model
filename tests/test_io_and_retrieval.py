"""I/O round trips and deterministic retrieval evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fluke_model.io import (
    ManifestRow,
    load_image,
    read_json,
    read_manifest,
    write_json,
    write_manifest,
)
from fluke_model.orca_data import OrcaManifestRow
from fluke_model.retrieval_eval import evaluate_retrieval


def test_manifest_json_and_image_io_round_trip(tmp_path: Path) -> None:
    rows = [ManifestRow(path="/catalog/a.jpg", individual_id="J35")]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    assert read_manifest(manifest) == rows

    payload_path = tmp_path / "nested" / "result.json"
    write_json(payload_path, {"top_1": 0.75})
    assert read_json(payload_path) == {"top_1": 0.75}

    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 5), color=(1, 2, 3)).save(image_path)
    assert load_image(image_path).size == (4, 5)


def test_manifest_reader_rejects_missing_file_and_columns(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "missing.csv")
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("wrong,column\na,b\n")
    with pytest.raises(ValueError, match="must have columns"):
        read_manifest(malformed)


def test_retrieval_eval_reports_exact_closed_set_metrics() -> None:
    reference_rows = [
        OrcaManifestRow("a-1.jpg", "A", "killer_whale"),
        OrcaManifestRow("b-1.jpg", "B", "killer_whale"),
    ]
    query_rows = [
        OrcaManifestRow("a-2.jpg", "A", "killer_whale"),
        OrcaManifestRow("b-2.jpg", "B", "killer_whale"),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    report = evaluate_retrieval(
        embeddings,
        reference_rows,
        embeddings,
        query_rows,
        embedder_name="deterministic-eval",
        neighbors=2,
    )

    assert report["metrics"] == {"top_1": 1.0, "top_3": 1.0, "top_5": 1.0, "mrr": 1.0}
    assert report["n_reference_individuals"] == 2
    assert report["per_query"][0]["top5"][0]["individual_id"] == "A"
