"""Stable model interface and metadata for on-device DINOv2 export."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from fluke_model.embedders import DINO_V2_MODEL_ID, DINO_V2_REVISION


@dataclass(frozen=True)
class MobileModelContract:
    """Fixed model and preprocessing metadata consumed by mobile exports."""

    model_id: str
    revision: str
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int]
    preprocessing_version: str


class MobileDINOv2Wrapper(torch.nn.Module):
    """Expose the normalized DINOv2 CLS embedding as a single tensor."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = deepcopy(model).eval()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        output = self._model(pixel_values=pixels, return_dict=False)
        hidden = output[0]
        embedding = hidden[:, 0, :]
        return torch.nn.functional.normalize(embedding, dim=-1)


def mobile_model_contract() -> MobileModelContract:
    """Return the immutable contract for the production mobile embedder."""
    return MobileModelContract(
        model_id=DINO_V2_MODEL_ID,
        revision=DINO_V2_REVISION,
        input_shape=(1, 3, 224, 224),
        output_shape=(1, 384),
        preprocessing_version="dinov2-imagenet-v1",
    )
