"""Contract tests for the on-device DINOv2 export wrapper."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

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
    assert contract.revision == "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    assert contract.input_shape == (1, 3, 224, 224)
    assert contract.output_shape == (1, 384)
    assert contract.preprocessing_version == "dinov2-imagenet-v1"

    with pytest.raises(FrozenInstanceError):
        contract.model_id = "replacement-model"


def test_wrapper_owns_eval_copy_without_mutating_caller() -> None:
    caller_model = FakeVisionModel()

    wrapper = MobileDINOv2Wrapper(caller_model)
    owned_model = wrapper.get_submodule("_model")

    assert caller_model.training is True
    assert owned_model is not caller_model
    assert owned_model.training is False


def test_wrapper_returns_finite_normalized_embedding() -> None:
    wrapper = MobileDINOv2Wrapper(FakeVisionModel()).eval()
    output = wrapper(torch.ones((1, 3, 224, 224), dtype=torch.float32))
    assert output.shape == (1, 384)
    assert torch.isfinite(output).all()
    assert float(torch.linalg.vector_norm(output)) == pytest.approx(1.0, abs=1e-3)
