"""Frozen embedder loaders for the M-Model-0 prototype.

Each `load_embedder(name)` returns `(embed_fn, embed_dim)` where:
  - embed_fn: callable taking a list[PIL.Image] and returning an L2-normalized
    np.ndarray of shape (N, embed_dim) on CPU.
  - embed_dim: integer dimensionality of the output vector.

We support one production executable embedder by name:
  * 'dinov2-small'           -> facebook/dinov2-small via transformers AutoModel

MiewID is intentionally not executable: its checkpoint has no documented
commercial-production license and requires third-party remote code.
DINOv3 is intentionally not executable because its weights are non-commercial.

If a model fails to load (license gate, missing weights, dependency mismatch),
`load_embedder` raises `EmbedderUnavailable` with a human-readable explanation.
The evaluation script catches this and skips the failing embedder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fluke_model.model_artifact import verify_dinov2_artifact


class EmbedderUnavailable(RuntimeError):
    """Raised when an embedder cannot be loaded (license, network, deps)."""


@dataclass(frozen=True)
class LoadedEmbedder:
    embed_fn: Callable[[list[Image.Image]], np.ndarray]
    embed_dim: int
    name: str


DINO_V2_MODEL_ID = "facebook/dinov2-small"
DINO_V2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


# ----- Internal helpers -------------------------------------------------------


def _select_device() -> torch.device:
    # CPU only for first pass; MPS coverage is still patchy for some operators.
    return torch.device("cpu")


def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1)


# ----- DINOv2 / DINOv3 (HF AutoModel + AutoImageProcessor) -------------------


def _load_hf_vision_embedder(
    repo: str,
    expected_dim: int,
    name: str,
    *,
    revision: str | None = None,
    artifact_dir: Path | None = None,
) -> LoadedEmbedder:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as e:  # pragma: no cover
        raise EmbedderUnavailable(f"transformers unavailable: {e}") from e

    source = str(artifact_dir) if artifact_dir is not None else repo
    processor_options = (
        {"local_files_only": True} if artifact_dir is not None else {"revision": revision}
    )
    model_options = {**processor_options, "use_safetensors": True}
    try:
        processor = AutoImageProcessor.from_pretrained(source, **processor_options)
        model = AutoModel.from_pretrained(source, **model_options)
    except Exception as e:
        raise EmbedderUnavailable(f"could not load {repo}: {e}") from e

    device = _select_device()
    model = model.to(device).eval()

    @torch.no_grad()
    def embed_fn(images: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        # Use the CLS / pooled output. Most HF vision models expose `pooler_output`;
        # if not, fall back to mean-pooling the last hidden state.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state.mean(dim=1)
        else:
            raise EmbedderUnavailable(f"{repo}: unexpected output structure")
        feats = _l2_normalize(feats)
        return feats.cpu().numpy().astype(np.float32)

    return LoadedEmbedder(embed_fn=embed_fn, embed_dim=expected_dim, name=name)


# ----- Public API -------------------------------------------------------------

_REGISTRY = frozenset({"dinov2-small"})


def load_embedder(name: str, *, artifact_dir: Path | None = None) -> LoadedEmbedder:
    """Load an embedder by name.

    Raises:
        EmbedderUnavailable: if the model cannot be loaded.
        ValueError: if `name` is not a registered embedder.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown embedder '{name}'. Choices: {sorted(_REGISTRY)}")
    if artifact_dir is not None:
        verify_dinov2_artifact(artifact_dir)
    return _load_hf_vision_embedder(
        DINO_V2_MODEL_ID,
        384,
        "dinov2-small",
        revision=DINO_V2_REVISION,
        artifact_dir=artifact_dir,
    )


def available_embedders() -> list[str]:
    return sorted(_REGISTRY)
