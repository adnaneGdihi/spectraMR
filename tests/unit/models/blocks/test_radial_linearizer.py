"""Unit tests for MorphologicalRadialLinearizer and radial permutation indices.

Tests verify:
- Concentric shell permutation validity (2D + 3D)
- Inside-out starts at center, outside-in starts at edge
- Patch-level roundtrip correctness
- Active segmentation mask generation
- Gradient flow through mask
- Integration with MambaLayer2D via linearization_mode
"""

from __future__ import annotations

import torch

from spectramr.models.blocks.radial_linearizer import (
    MorphologicalRadialLinearizer,
    radial_inside_out_indices,
    radial_outside_in_indices,
)
from tests.utils.optional_backends import requires_cuda_for_mamba

# -----------------------------------------------------------------------
# Pixel-level radial index tests
# -----------------------------------------------------------------------


class TestRadialIndices:
    """Test pixel-level concentric permutation generators."""

    def test_inside_out_valid_permutation(self):
        idx = radial_inside_out_indices(8, 8)
        assert set(idx.tolist()) == set(range(64))

    def test_outside_in_valid_permutation(self):
        idx = radial_outside_in_indices(8, 8)
        assert set(idx.tolist()) == set(range(64))

    def test_inside_out_starts_near_center(self):
        """First element in centrifugal scan should be near the grid center."""
        H, W = 16, 16
        idx = radial_inside_out_indices(H, W)
        first_raster = idx[0]
        row, col = first_raster // W, first_raster % W
        center_r, center_c = (H - 1) / 2, (W - 1) / 2
        dist = ((row - center_r) ** 2 + (col - center_c) ** 2) ** 0.5
        # First element should be within 2 pixels of center
        assert dist < 2.0, f"First element at ({row}, {col}), dist={dist:.1f}"

    def test_outside_in_starts_near_edge(self):
        """First element in centripetal scan should be near the grid edge."""
        H, W = 16, 16
        idx = radial_outside_in_indices(H, W)
        first_raster = idx[0]
        row, col = first_raster // W, first_raster % W
        center_r, center_c = (H - 1) / 2, (W - 1) / 2
        dist = ((row - center_r) ** 2 + (col - center_c) ** 2) ** 0.5
        max_dist = ((H - 1) / 2) * (2**0.5)
        # First element should be in outer 30% of radius
        assert dist > max_dist * 0.7, f"dist={dist:.1f}, max={max_dist:.1f}"

    def test_non_square(self):
        idx = radial_inside_out_indices(6, 10)
        assert set(idx.tolist()) == set(range(60))

    def test_modes_are_inverse_order(self):
        """inside-out and outside-in should have inverted radial ordering."""
        H, W = 8, 8
        io = radial_inside_out_indices(H, W)
        oi = radial_outside_in_indices(H, W)
        # The first element of inside-out should be near center
        # The first element of outside-in should be near edges
        # They should NOT produce the same permutation
        assert not (io == oi).all()


# -----------------------------------------------------------------------
# MorphologicalRadialLinearizer tests
# -----------------------------------------------------------------------


class TestMorphologicalRadialLinearizer:
    """Test the patch-level radial linearizer with active segmentation."""

    def test_forward_shape(self):
        lin = MorphologicalRadialLinearizer((32, 32), patch_size=8)
        x = torch.randn(2, 3, 32, 32)
        seq, mask = lin(x)
        n_patches = (32 // 8) * (32 // 8)  # 16
        token_dim = 3 * 8 * 8  # 192
        assert seq.shape == (2, n_patches, token_dim)
        assert mask.shape == (2, n_patches, 1)

    def test_roundtrip(self):
        """forward → reverse should reconstruct the original tensor."""
        lin = MorphologicalRadialLinearizer((32, 32), patch_size=8)
        x = torch.randn(1, 2, 32, 32)
        seq, _ = lin(x)
        restored = lin.reverse(seq)
        assert torch.allclose(x, restored, atol=1e-5)

    def test_mask_is_binary(self):
        lin = MorphologicalRadialLinearizer((32, 32), patch_size=8)
        x = torch.randn(2, 1, 32, 32)
        _, mask = lin(x)
        unique = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_mask_disabled(self):
        lin = MorphologicalRadialLinearizer((32, 32), patch_size=8)
        x = torch.randn(1, 1, 32, 32)
        seq, mask = lin(x, generate_mask=False)
        assert mask is None

    def test_outside_in_mode(self):
        lin = MorphologicalRadialLinearizer((32, 32), patch_size=8, mode="outside_in")
        x = torch.randn(1, 2, 32, 32)
        seq, _ = lin(x)
        restored = lin.reverse(seq)
        assert torch.allclose(x, restored, atol=1e-5)

    def test_3d_forward_shape(self):
        lin = MorphologicalRadialLinearizer((16, 16, 16), patch_size=8, is_3d=True)
        x = torch.randn(1, 2, 16, 16, 16)
        seq, mask = lin(x)
        n_patches = (16 // 8) ** 3  # 8
        token_dim = 2 * 8 * 8 * 8  # 1024
        assert seq.shape == (1, n_patches, token_dim)

    def test_3d_roundtrip(self):
        lin = MorphologicalRadialLinearizer((16, 16, 16), patch_size=8, is_3d=True)
        x = torch.randn(1, 1, 16, 16, 16)
        seq, _ = lin(x)
        restored = lin.reverse(seq)
        assert torch.allclose(x, restored, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through the linearizer."""
        lin = MorphologicalRadialLinearizer((32, 32), patch_size=8)
        x = torch.randn(1, 1, 32, 32, requires_grad=True)
        seq, mask = lin(x)
        loss = (seq * mask).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_buffers_not_trainable(self):
        lin = MorphologicalRadialLinearizer((32, 32), patch_size=8)
        assert not lin.mac_fwd.requires_grad
        assert not lin.mac_inv.requires_grad


# -----------------------------------------------------------------------
# MambaLayer2D radial integration tests
# -----------------------------------------------------------------------


class TestMambaLayerRadialIntegration:
    """Test MambaLayer2D with radial linearization modes."""

    @requires_cuda_for_mamba
    def test_radial_inside_out_mode(self):
        from spectramr.models.generators.mamba_unet import MambaLayer2D

        layer = MambaLayer2D(channels=16, linearization_mode="radial_inside_out")
        x = torch.randn(1, 16, 8, 8)
        out = layer(x)
        assert out.shape == x.shape

    @requires_cuda_for_mamba
    def test_radial_outside_in_mode(self):
        from spectramr.models.generators.mamba_unet import MambaLayer2D

        layer = MambaLayer2D(channels=16, linearization_mode="radial_outside_in")
        x = torch.randn(1, 16, 8, 8)
        out = layer(x)
        assert out.shape == x.shape


def test_patch_count_is_reported_without_a_device_round_trip(caplog) -> None:
    """``total_patches`` comes from ``math.prod`` of the grid, not a tensor ``.item()``."""
    import logging

    with caplog.at_level(logging.INFO):
        MorphologicalRadialLinearizer((32, 32), patch_size=8)
    assert "total_patches=16" in caplog.text
