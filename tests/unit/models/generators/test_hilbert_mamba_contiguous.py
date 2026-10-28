"""Unit tests for _squeeze_volume_to_2d contiguity fix.

Verifies that the depth fold of 5D volumes returns
contiguous tensors, preventing cuDNN CUDNN_STATUS_NOT_SUPPORTED
errors in downstream Conv2d operations.
"""

import torch

from spectramr.models.generators.hilbert_mamba_generators import (
    _squeeze_volume_to_2d,
    _unsqueeze_to_volume,
)


class TestSqueezeVolumeTo2D:
    """Test suite for _squeeze_volume_to_2d contiguity."""

    def test_5d_depth_fold_is_contiguous(self):
        """Depth fold must return a contiguous tensor."""
        x = torch.randn(2, 3, 8, 16, 4)  # [B, C, H, W, D]
        squeezed, original_shape = _squeeze_volume_to_2d(x)
        assert squeezed.is_contiguous(), (
            f"Folded tensor is not contiguous: stride={squeezed.stride()}, shape={squeezed.shape}"
        )
        assert squeezed.shape == (8, 3, 8, 16)
        assert original_shape == (2, 3, 8, 16, 4)

    def test_5d_single_depth_squeeze(self):
        """5D with D=1 should squeeze last dim and be contiguous."""
        x = torch.randn(2, 3, 16, 16, 1)  # [B, C, H, W, 1]
        squeezed, original_shape = _squeeze_volume_to_2d(x)
        assert squeezed.is_contiguous()
        assert squeezed.shape == (2, 3, 16, 16)
        assert original_shape == (2, 3, 16, 16, 1)

    def test_4d_noop(self):
        """4D input should pass through unchanged."""
        x = torch.randn(2, 3, 16, 16)
        squeezed, original_shape = _squeeze_volume_to_2d(x)
        assert squeezed is x  # Same object
        assert original_shape is None

    def test_round_trip_5d_single_depth(self):
        """Squeeze → unsqueeze should restore original shape for D=1."""
        x = torch.randn(2, 3, 16, 16, 1)
        squeezed, original_shape = _squeeze_volume_to_2d(x)
        restored = _unsqueeze_to_volume(squeezed, original_shape)
        assert restored.shape == x.shape

    def test_squeeze_compatible_with_conv2d(self):
        """Squeezed tensor must work with Conv2d without cuDNN errors."""
        x = torch.randn(2, 3, 8, 16, 4)
        squeezed, _ = _squeeze_volume_to_2d(x)
        conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        # This should NOT raise cuDNN errors
        out = conv(squeezed)
        assert out.shape == (8, 8, 8, 16)
