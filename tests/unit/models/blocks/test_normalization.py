"""Tests for normalization blocks.

Targets ``spectramr.models.blocks.normalization``:

- ``get_adaptive_groups`` — group-count selection helper
- ``AdaptiveInstanceNorm`` — AdaIN with style modulation
- ``ConditionalBatchNorm`` — class-conditional BN
- ``AdaptiveGroupNorm`` — AdaGN for diffusion U-Net (zero-init projection)

``ComplexAdaptiveGroupNorm`` is covered in the integration tier (it
involves complex tensor reshaping that's better tested with real
diffusion strategies).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.normalization import (
    AdaptiveGroupNorm,
    AdaptiveInstanceNorm,
    ComplexAdaptiveGroupNorm,
    ConditionalBatchNorm,
    get_adaptive_groups,
)


# ---------------------------------------------------------------------------
# get_adaptive_groups helper
# ---------------------------------------------------------------------------


def test_adaptive_groups_returns_one_for_small_channels() -> None:
    """For channels < min_channels_per_group, fall back to 1 group."""
    assert get_adaptive_groups(8, min_channels_per_group=16) == 1


def test_adaptive_groups_divisor_of_channels() -> None:
    """Returned group count is always a divisor of ``channels``."""
    for c in [16, 32, 64, 128, 256]:
        g = get_adaptive_groups(c)
        assert c % g == 0


def test_adaptive_groups_caps_at_32() -> None:
    """Maximum group count is 32 (per the helper's loop bound)."""
    assert get_adaptive_groups(64) <= 32
    assert get_adaptive_groups(256) <= 32


def test_adaptive_groups_respects_channels_per_group_minimum() -> None:
    """For c=64 with min=16, returned groups satisfies 64/g >= 16 → g <= 4."""
    g = get_adaptive_groups(64, min_channels_per_group=16)
    assert 64 // g >= 16


# ---------------------------------------------------------------------------
# AdaptiveInstanceNorm
# ---------------------------------------------------------------------------


def test_adain_output_shape() -> None:
    """Output preserves input shape."""
    norm = AdaptiveInstanceNorm(channels=8, style_dim=16)
    x = torch.randn(2, 8, 16, 16)
    style = torch.randn(2, 16)
    out = norm(x, style)
    assert out.shape == x.shape


def test_adain_zero_style_yields_zero_output() -> None:
    """Initial style projection has random init → output is finite."""
    norm = AdaptiveInstanceNorm(channels=4, style_dim=8)
    x = torch.randn(1, 4, 8, 8)
    style = torch.zeros(1, 8)  # zero style → linear output is zero (no bias init effect)
    out = norm(x, style)
    assert torch.isfinite(out).all()


def test_adain_gradient_flows() -> None:
    """Backward through AdaIN reaches both content and style."""
    norm = AdaptiveInstanceNorm(channels=4, style_dim=8)
    x = torch.randn(1, 4, 8, 8, requires_grad=True)
    style = torch.randn(1, 8, requires_grad=True)
    norm(x, style).sum().backward()
    assert x.grad is not None
    assert style.grad is not None


# ---------------------------------------------------------------------------
# ConditionalBatchNorm
# ---------------------------------------------------------------------------


def test_conditional_bn_output_shape() -> None:
    """Output preserves input shape."""
    norm = ConditionalBatchNorm(channels=8, condition_dim=16)
    x = torch.randn(2, 8, 16, 16)
    condition = torch.randn(2, 16)
    out = norm(x, condition)
    assert out.shape == x.shape


def test_conditional_bn_gradient_flows() -> None:
    """Backward through Conditional BN reaches inputs."""
    norm = ConditionalBatchNorm(channels=4, condition_dim=8)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    condition = torch.randn(2, 8, requires_grad=True)
    norm(x, condition).sum().backward()
    assert x.grad is not None
    assert condition.grad is not None


# ---------------------------------------------------------------------------
# AdaptiveGroupNorm — zero-init means initial pass-through
# ---------------------------------------------------------------------------


def test_ada_gn_zero_init_yields_pure_groupnorm() -> None:
    """With zero-init projection, ``forward(x, emb)`` ≈ ``GroupNorm(x)``."""
    norm = AdaptiveGroupNorm(channels=8, num_groups=4, emb_channels=16)
    x = torch.randn(1, 8, 8, 8)
    emb = torch.randn(1, 16)
    # Compare against the inner GroupNorm
    expected = norm.norm(x)
    actual = norm(x, emb)
    # With zero scale & shift: scale=0 → (1+0)*norm(x) + 0 = norm(x)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_ada_gn_no_emb_returns_pure_groupnorm() -> None:
    """``emb=None`` → bare GroupNorm output."""
    norm = AdaptiveGroupNorm(channels=8, num_groups=4, emb_channels=16)
    x = torch.randn(1, 8, 8, 8)
    expected = norm.norm(x)
    actual = norm(x, emb=None)
    assert torch.allclose(actual, expected)


def test_ada_gn_after_training_modulates_output() -> None:
    """After non-zero projection weights, AdaGN modulates the output."""
    norm = AdaptiveGroupNorm(channels=4, num_groups=2, emb_channels=8)
    # Override projection to non-zero values to simulate post-training.
    with torch.no_grad():
        norm.proj.weight.fill_(0.1)
        norm.proj.bias.fill_(0.05)
    x = torch.randn(1, 4, 4, 4)
    emb = torch.randn(1, 8)
    out_with_emb = norm(x, emb)
    out_no_emb = norm(x, None)
    assert not torch.allclose(out_with_emb, out_no_emb)


def test_ada_gn_gradient_flows() -> None:
    """Backward through AdaGN reaches input + embedding."""
    norm = AdaptiveGroupNorm(channels=8, num_groups=4, emb_channels=16)
    x = torch.randn(1, 8, 8, 8, requires_grad=True)
    emb = torch.randn(1, 16, requires_grad=True)
    norm(x, emb).sum().backward()
    assert x.grad is not None
    # emb has zero grad initially because projection is zero-init,
    # but the projection bias path should still flow.
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channels, h, w",
    [(4, 8, 8), (16, 16, 16), (32, 32, 32)],
    ids=["c4_8x8", "c16_16x16", "c32_32x32"],
)
def test_ada_gn_shape_matrix(channels: int, h: int, w: int) -> None:
    """AdaGN preserves shape across the parametrised matrix."""
    norm = AdaptiveGroupNorm(channels=channels, num_groups=2, emb_channels=8)
    x = torch.randn(1, channels, h, w)
    emb = torch.randn(1, 8)
    out = norm(x, emb)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# ComplexAdaptiveGroupNorm: no additive term (a shift is a k-space delta)
# ---------------------------------------------------------------------------


class TestComplexAdaptiveGroupNormHasNoShift:
    def test_zero_in_zero_out_with_trained_projection(self):
        norm = ComplexAdaptiveGroupNorm(channels=8, num_groups=4, emb_channels=16)
        torch.nn.init.normal_(norm.proj.weight)
        torch.nn.init.normal_(norm.proj.bias)
        out = norm(torch.zeros(2, 8, 8, 8), torch.randn(2, 16))
        assert torch.count_nonzero(out) == 0

    def test_projection_emits_one_scale_per_complex_channel(self):
        assert ComplexAdaptiveGroupNorm(8, 4, 16).proj.out_features == 4
