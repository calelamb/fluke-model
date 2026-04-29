"""Frozen embedder loaders for the M-Model-0 prototype.

Each `load_embedder(name)` returns `(embed_fn, embed_dim)` where:
  - embed_fn: callable taking a list[PIL.Image] and returning an L2-normalized
    np.ndarray of shape (N, embed_dim) on CPU.
  - embed_dim: integer dimensionality of the output vector.

We support three embedders by name:
  * 'dinov2-small'           -> facebook/dinov2-small via transformers AutoModel
  * 'dinov3-convnext-tiny'   -> facebook/dinov3-convnext-tiny-pretrain-lvd1689m
  * 'miewid-msv3'            -> conservationxlabs/miewid-msv3 (HF transformers)

If a model fails to load (license gate, missing weights, dependency mismatch),
`load_embedder` raises `EmbedderUnavailable` with a human-readable explanation.
The evaluation script catches this and skips the failing embedder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class EmbedderUnavailable(RuntimeError):
    """Raised when an embedder cannot be loaded (license, network, deps)."""


@dataclass(frozen=True)
class LoadedEmbedder:
    embed_fn: Callable[[list[Image.Image]], np.ndarray]
    embed_dim: int
    name: str


# ----- Internal helpers -------------------------------------------------------

def _select_device() -> torch.device:
    # CPU only for first pass; MPS coverage is still patchy for some operators.
    return torch.device("cpu")


def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1)


# ----- DINOv2 / DINOv3 (HF AutoModel + AutoImageProcessor) -------------------

def _load_hf_vision_embedder(repo: str, expected_dim: int, name: str) -> LoadedEmbedder:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except Exception as e:  # pragma: no cover
        raise EmbedderUnavailable(f"transformers unavailable: {e}") from e

    try:
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo)
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


# ----- MiewID (HF transformers; trust_remote_code path) ----------------------

def _load_miewid() -> LoadedEmbedder:
    """Load conservationxlabs/miewid-msv3.

    Per the model card, the recommended loading API is:
        AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
    The model exposes an `extract_features` method that returns a 2048-dim L2-normalized
    embedding. We try that path first; if it fails for any reason we surface the error
    via EmbedderUnavailable so the eval can skip and continue.
    """
    try:
        from transformers import AutoModel
    except Exception as e:  # pragma: no cover
        raise EmbedderUnavailable(f"transformers unavailable: {e}") from e

    repo = "conservationxlabs/miewid-msv3"
    try:
        model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    except Exception as e:
        raise EmbedderUnavailable(
            f"miewid-msv3 not loadable via AutoModel: {e}. "
            "May require Wildbook-specific loader or a license gate."
        ) from e

    device = _select_device()
    model = model.to(device).eval()

    # MiewID expects 440x440 input per the model card; we replicate the canonical
    # ImageNet normalization since the model card does not surface a separate processor.
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize((440, 440)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    @torch.no_grad()
    def embed_fn(images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([preprocess(img) for img in images]).to(device)
        # MiewIdNet's forward returns a (B, 2152) feature tensor directly.
        if hasattr(model, "extract_features"):
            feats = model.extract_features(batch)
        else:
            feats = model(batch)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
        if isinstance(feats, dict):
            feats = feats.get("embeddings", next(iter(feats.values())))
        feats = _l2_normalize(feats)
        return feats.cpu().numpy().astype(np.float32)

    # MiewIdNet over EfficientNetV2-M outputs 2152-dim features (2048 backbone +
    # auxiliary heads, depending on the checkpoint). We probe at load time.
    probe = torch.zeros(1, 3, 440, 440, device=device)
    with torch.no_grad():
        probe_out = model(probe)
        if isinstance(probe_out, (list, tuple)):
            probe_out = probe_out[0]
        if isinstance(probe_out, dict):
            probe_out = next(iter(probe_out.values()))
    embed_dim = int(probe_out.shape[-1])
    return LoadedEmbedder(embed_fn=embed_fn, embed_dim=embed_dim, name="miewid-msv3")


# ----- Public API -------------------------------------------------------------

_REGISTRY = {
    "dinov2-small": lambda: _load_hf_vision_embedder("facebook/dinov2-small", 384, "dinov2-small"),
    "dinov3-convnext-tiny": lambda: _load_hf_vision_embedder(
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m", 768, "dinov3-convnext-tiny"
    ),
    "miewid-msv3": _load_miewid,
}


def load_embedder(name: str) -> LoadedEmbedder:
    """Load an embedder by name.

    Raises:
        EmbedderUnavailable: if the model cannot be loaded.
        ValueError: if `name` is not a registered embedder.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown embedder '{name}'. Choices: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available_embedders() -> list[str]:
    return sorted(_REGISTRY)
