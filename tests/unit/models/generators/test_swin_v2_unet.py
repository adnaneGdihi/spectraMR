"""Unit tests for the swin_v2_unet backbone and its registration."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.swin_v2_unet import SwinV2UNet

TINY = {"base_channels": 8, "depths": (1, 2), "num_heads": (1, 2), "window_size": 4}


def test_forward_maps_channels_and_keeps_resolution():
    model = SwinV2UNet(in_channels=2, out_channels=1, **TINY)
    x = torch.randn(2, 2, 16, 24, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 1, 16, 24)
    out.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_rejects_non_2d_mismatched_stages_and_undivisible_input():
    with pytest.raises(ValueError, match="2D only"):
        SwinV2UNet(spatial_dims=3, **TINY)
    with pytest.raises(ValueError, match="equal length"):
        SwinV2UNet(base_channels=8, depths=(1, 1), num_heads=(1,))
    with pytest.raises(ValueError, match="divisible by 2"):
        SwinV2UNet(**TINY)(torch.randn(1, 1, 15, 16))


def test_unknown_model_kwarg_raises_instead_of_being_swallowed():
    with pytest.raises(TypeError, match="use_fed_bn"):
        SwinV2UNet(use_fed_bn=True, **TINY)


def test_registered_and_resolved_through_the_cross_field_wrapper():
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert MODEL_REGISTRY.get("swin_v2_unet")["class"] is SwinV2UNet
    wrapper = MODEL_REGISTRY.get("cross_field_fm")["class"](
        in_channels=1,
        out_channels=1,
        spatial_dims=2,
        model_kwargs={"backbone": "swin_v2_unet", **TINY},
    )
    assert isinstance(wrapper.backbone, SwinV2UNet)
    assert wrapper(torch.randn(1, 1, 16, 16)).shape == (1, 1, 16, 16)
