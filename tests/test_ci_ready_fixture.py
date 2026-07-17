"""CI fixture proves a container can become genuinely ready."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fluke_model.embedders import LoadedEmbedder
from fluke_model.ci_fixture import build_ci_reference_index
from fluke_model.identify_runtime import IdentifierRuntime
from fluke_model.index_store import AtomicIndexStore


def test_ci_fixture_builds_a_consistent_rights_attested_ready_index(tmp_path: Path) -> None:
    def embed(images: list[Image.Image]) -> np.ndarray:
        values = np.asarray(
            [[float(image.getpixel((0, 0))[0]) + 1.0, 1.0] for image in images],
            dtype=np.float32,
        )
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    embedder = LoadedEmbedder(embed_fn=embed, embed_dim=2, name="dinov2-small")

    build_ci_reference_index(tmp_path, embedder=embedder)

    runtime = IdentifierRuntime(index_store=AtomicIndexStore(tmp_path), embedder=embedder)
    assert runtime.readiness() == (True, "ready")
    assert (tmp_path / "ci-query.jpg").is_file()
