"""Regression tests for ``BiasFieldGenerator`` order dispatch.

Pitfall #9 / CLAUDE.md non-negotiable #3: an unknown ``order`` value must
RAISE at construction time, not silently build no head and crash later with
``UnboundLocalError`` in ``forward`` (where ``bias_field`` is never assigned).

Pre-fix: ``BiasFieldGenerator(order="polynomial3")`` constructs without error,
then ``forward`` raises ``UnboundLocalError``. Post-fix: construction raises a
clear ``ValueError`` naming the offending value and the advertised set.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.constants import BIAS_FIELD_MAX_VALUE, BIAS_FIELD_MIN_VALUE
from spectramr.models.generators.disentangled_mri import BiasFieldGenerator


def test_unknown_order_raises_at_construction() -> None:
    # Deliberate typo: 'polynomial3' (missing underscore) must fail loud.
    with pytest.raises(ValueError, match="Unknown bias_field order"):
        BiasFieldGenerator(in_dim=8, order="polynomial3")


@pytest.mark.parametrize("order", ["polynomial_3", "lowrank_unet"])
def test_valid_orders_construct_and_forward(order: str) -> None:
    torch.manual_seed(0)
    gen = BiasFieldGenerator(in_dim=8, order=order, output_resolution=(16, 16))
    features = torch.randn(2, 8, 16, 16, device="cpu")
    field = gen(features)
    assert field.shape == (2, 1, 16, 16)
    # lowrank_unet path is bounded to [MIN, MAX] exactly; polynomial path is a
    # smoothed tanh-scaled field which stays within the same target range.
    assert torch.isfinite(field).all()
    assert field.min().item() >= BIAS_FIELD_MIN_VALUE - 1e-3
    assert field.max().item() <= BIAS_FIELD_MAX_VALUE + 1e-3


def _disentangled():
    from spectramr.models.generators.disentangled_mri import DisentangledMRI

    return DisentangledMRI(in_channels=1, style_dim=8, n_downsample=2)


def test_physics_projection_is_registered_before_forward() -> None:
    """Built lazily it missed the optimizer and a fresh state_dict (#2388)."""
    model = _disentangled()
    assert "phys_to_style_proj.weight" in dict(model.named_parameters())
    assert "phys_to_style_proj.weight" in model.state_dict()


def test_physics_vector_routes_through_the_projection() -> None:
    model = _disentangled().eval()
    phys = torch.rand(1, 4)
    _, mu, _ = model(torch.randn(1, 1, 32, 32), phys)
    assert torch.allclose(mu, model.phys_to_style_proj(phys))


def test_style_vector_of_unknown_width_raises() -> None:
    with pytest.raises(ValueError, match="expected style_dim"):
        _disentangled()(torch.randn(1, 1, 32, 32), torch.rand(1, 5))


def _tissue_model():
    from spectramr.models.generators.disentangled_mri import DisentangledMRI

    return DisentangledMRI(
        in_channels=1, style_dim=8, n_downsample=2, output_mode="tissue_params",
        enable_bias_field=True, num_classes=3,
    ).eval()


def test_decode_tissue_is_the_one_tissue_decode() -> None:
    """Training, validation, forward and decode all run this (one owner)."""
    model = _tissue_model()
    content = model.enc_c(torch.randn(1, 1, 32, 32))
    out = model.decode_tissue(content, torch.zeros(1, 8))
    assert {"rho", "t1", "t2", "predicted_tr", "predicted_te", "bias_field"} <= set(out)


def test_forward_and_decode_agree_with_decode_tissue() -> None:
    model = _tissue_model()
    x = torch.randn(1, 1, 32, 32)
    style = model.encode_style(x)
    via_decode = model.decode(model.enc_c(x), style)
    via_method = model.decode_tissue(model.enc_c(x), style)
    assert set(via_decode) == set(via_method)
    assert torch.allclose(via_decode["predicted_tr"], via_method["predicted_tr"])


def test_decode_tissue_applies_the_bias_field_to_an_image_output() -> None:
    from spectramr.models.generators.disentangled_mri import DisentangledMRI

    model = DisentangledMRI(in_channels=1, style_dim=8, n_downsample=2, enable_bias_field=True).eval()
    # BiasFieldGenerator emits a fixed 256x256 field, so the image must be 256x256.
    content = model.enc_c(torch.randn(1, 1, 256, 256))
    style = torch.zeros(1, 8)
    assert torch.allclose(
        model.decode_tissue(content, style),
        model.gen(content, style) * model.bias_field_generator(content),
    )


def test_condition_style_adds_the_clamped_class_embedding() -> None:
    model = _tissue_model()
    style = torch.zeros(1, 8)
    out = model.condition_style(style, torch.tensor([7]))
    assert torch.allclose(out, model.class_embedding(torch.tensor([2])))
    assert model.condition_style(style, None) is style
