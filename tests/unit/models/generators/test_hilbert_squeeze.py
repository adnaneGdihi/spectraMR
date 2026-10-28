"""Unit tests for the _squeeze_volume_to_2d / _unsqueeze_to_volume depth fold.

Validation hands the Hilbert-Mamba generators full volumes laid out as
``[B, C, H, W, D]``. The fold scores every slice, so a per-slice model must
match the folded output slice by slice.
"""

import torch

from spectramr.models.generators.hilbert_mamba_generators import (
    _squeeze_volume_to_2d,
    _unsqueeze_to_volume,
)


class TestSqueezeVolumeTo2D:
    def test_noop_4d(self) -> None:
        x = torch.randn(2, 1, 64, 64)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert squeezed is x
        assert vol_shape is None

    def test_squeeze_trailing_1(self) -> None:
        x = torch.randn(2, 1, 64, 64, 1)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert squeezed.shape == (2, 1, 64, 64)
        assert vol_shape == (2, 1, 64, 64, 1)

    def test_fold_depth_into_batch(self) -> None:
        x = torch.randn(2, 3, 8, 16, 5)
        folded, vol_shape = _squeeze_volume_to_2d(x)
        assert folded.shape == (10, 3, 8, 16)
        assert folded.is_contiguous()
        assert vol_shape == (2, 3, 8, 16, 5)
        # Sample b, slice d sits at folded index b * D + d.
        assert torch.equal(folded[1 * 5 + 3], x[1, :, :, :, 3])

    def test_fold_odd_depth(self) -> None:
        x = torch.randn(1, 2, 7, 32, 5)
        folded, _ = _squeeze_volume_to_2d(x)
        assert folded.shape == (5, 2, 7, 32)

    def test_fold_depth_one_is_per_patch_squeeze(self) -> None:
        x = torch.randn(2, 1, 8, 8, 1)
        folded, _ = _squeeze_volume_to_2d(x)
        assert torch.equal(folded, x.squeeze(-1))


class TestUnsqueezeToVolume:
    def test_noop_none_shape(self) -> None:
        out = torch.randn(2, 1, 64, 64)
        assert _unsqueeze_to_volume(out, None) is out

    def test_unsqueeze_trailing_1(self) -> None:
        out = torch.randn(2, 1, 64, 64)
        result = _unsqueeze_to_volume(out, (2, 1, 64, 64, 1))
        assert result.shape == (2, 1, 64, 64, 1)
        assert torch.equal(result.squeeze(-1), out)

    def test_unfold_restores_depth_last(self) -> None:
        out = torch.randn(10, 4, 8, 16)  # B*D = 2*5
        result = _unsqueeze_to_volume(out, (2, 3, 8, 16, 5))
        assert result.shape == (2, 4, 8, 16, 5)
        assert torch.equal(result[1, :, :, :, 3], out[1 * 5 + 3])


class TestFoldRoundtrip:
    def test_roundtrip_4d(self) -> None:
        x = torch.randn(2, 1, 64, 64)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert torch.equal(_unsqueeze_to_volume(squeezed, vol_shape), x)

    def test_roundtrip_trailing_1(self) -> None:
        x = torch.randn(2, 1, 64, 64, 1)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert torch.equal(_unsqueeze_to_volume(squeezed, vol_shape), x)

    def test_roundtrip_volume_is_identity(self) -> None:
        x = torch.randn(2, 1, 8, 8, 5)
        folded, vol_shape = _squeeze_volume_to_2d(x)
        assert torch.equal(_unsqueeze_to_volume(folded, vol_shape), x)

    def test_per_slice_model_matches_folded_output(self) -> None:
        """A per-slice model on the folded batch equals slice-by-slice application."""
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
        x = torch.randn(1, 1, 8, 8, 5)
        folded, vol_shape = _squeeze_volume_to_2d(x)
        out = _unsqueeze_to_volume(conv(folded), vol_shape)
        assert out.shape == x.shape
        for d in range(5):
            assert torch.allclose(out[..., d], conv(x[..., d]), atol=1e-6)
