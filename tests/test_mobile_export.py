"""Contract tests for the on-device DINOv2 export wrapper."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from fluke_model.embedders import DINO_V2_REVISION
from fluke_model.mobile_export import MobileDINOv2Wrapper, mobile_model_contract


class FakeVisionModel(torch.nn.Module):
    """Return a deterministic DINOv2-shaped hidden state."""

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        assert pixel_values.shape == (1, 3, 224, 224)
        assert return_dict is False
        hidden = torch.ones((1, 257, 384), dtype=pixel_values.dtype)
        return (hidden,)


def test_mobile_contract_is_exact_and_immutable() -> None:
    contract = mobile_model_contract()
    assert contract.model_id == "facebook/dinov2-small"
    assert contract.revision == DINO_V2_REVISION
    assert contract.input_shape == (1, 3, 224, 224)
    assert contract.output_shape == (1, 384)
    assert contract.preprocessing_version == "dinov2-imagenet-v1"

    with pytest.raises(FrozenInstanceError):
        contract.model_id = "replacement-model"


def test_wrapper_returns_finite_normalized_embedding() -> None:
    wrapper = MobileDINOv2Wrapper(FakeVisionModel()).eval()
    output = wrapper(torch.ones((1, 3, 224, 224), dtype=torch.float32))
    assert output.shape == (1, 384)
    assert torch.isfinite(output).all()
    assert float(torch.linalg.vector_norm(output)) == pytest.approx(1.0, abs=1e-3)
