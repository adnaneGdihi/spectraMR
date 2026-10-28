"""Unit tests for Soft-DTW alignment loss."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.soft_dtw_loss import SoftDTWLoss, _soft_dtw_distance


class TestSoftDTWDistance:
    """Test the raw Soft-DTW distance computation."""

    def test_identity(self):
        """DTW(x, x) should be approximately 0."""
        x = torch.randn(2, 16, 4)
        dist = _soft_dtw_distance(x, x, gamma=1.0)
        assert dist.shape == (2,)
        assert (dist < 1.0).all()  # Should be near 0

    def test_symmetry(self):
        """DTW(x, y) ≈ DTW(y, x) for matching lengths."""
        x = torch.randn(2, 16, 4)
        y = torch.randn(2, 16, 4)
        d_xy = _soft_dtw_distance(x, y, gamma=1.0)
        d_yx = _soft_dtw_distance(y, x, gamma=1.0)
        assert torch.allclose(d_xy, d_yx, atol=0.5)

    def test_output_shape(self):
        pred = torch.randn(3, 10, 8)
        tgt = torch.randn(3, 10, 8)
        dist = _soft_dtw_distance(pred, tgt, gamma=0.1)
        assert dist.shape == (3,)


class TestSoftDTWLoss:
    """Test the SoftDTWLoss module."""

    def test_4d_input(self):
        """Should handle [B, C, H, W] input."""
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=8)
        pred = torch.randn(2, 1, 8, 8, requires_grad=True)
        tgt = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, tgt)
        assert loss.shape == ()  # Scalar
        loss.backward()
        assert pred.grad is not None
        # Stronger than existence: a detached/zero grad would pass `is not None`
        # yet mean the differentiable path is inert. Pin real signal.
        assert torch.any(pred.grad != 0)

    def test_3d_input(self):
        """Should handle [B, L, D] input."""
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=8)
        pred = torch.randn(2, 16, 4, requires_grad=True)
        tgt = torch.randn(2, 16, 4)
        loss = loss_fn(pred, tgt)
        assert loss.shape == ()

    def test_short_sequence(self):
        """When sequence < window_size, should compute on full sequence."""
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=128)
        pred = torch.randn(2, 16, 4, requires_grad=True)
        tgt = torch.randn(2, 16, 4)
        loss = loss_fn(pred, tgt)
        assert loss.shape == ()

    def test_invalid_dims(self):
        """Should raise on a 2-D input (no channel/spatial structure).

        5-D volumetric input is now SUPPORTED (folded slice-wise) — see
        :meth:`test_5d_volumetric_input_folds_to_slices`.
        """
        loss_fn = SoftDTWLoss()
        with pytest.raises(ValueError, match="expects 3D, 4D, or 5D"):
            loss_fn(torch.randn(2, 8), torch.randn(2, 8))

    def test_5d_volumetric_input_folds_to_slices(self):
        """A 5-D ``[B, C, H, W, D]`` volume (TorchIO depth-last) is aligned
        slice-wise. Pre-2026-06 this raised and crashed the volumetric
        ``exp_hm_08_sdtw_mamba`` arm (hdsf_mamba_unet backbone, 5-D output).
        """
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=64)
        pred = torch.randn(2, 1, 32, 32, 4)
        target = pred + 0.1 * torch.randn_like(pred)
        out = loss_fn(pred, target)
        assert out.shape == ()
        assert torch.isfinite(out)

    def test_5d_depth1_matches_squeezed_4d(self):
        """A single-slice 5-D volume equals the 4-D Soft-DTW of its squeezed
        slice — the depth fold is a no-op when ``D == 1``."""
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=64)
        pred4 = torch.randn(2, 1, 16, 16)
        target4 = torch.randn(2, 1, 16, 16)
        out5 = loss_fn(pred4.unsqueeze(-1), target4.unsqueeze(-1))
        out4 = loss_fn(pred4, target4)
        assert torch.allclose(out5, out4, atol=1e-5)

    def test_registration(self):
        """Soft-DTW loss should be in the loss registry."""
        from spectramr.models.losses.registry import list_available

        available = list_available()
        assert "soft_dtw" in available


def _cellwise_soft_dtw(pred: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    """The per-cell double loop the anti-diagonal sweep replaced."""
    b, n, m = pred.shape[0], pred.shape[1], target.shape[1]
    dist = torch.cdist(pred, target, p=2.0).pow(2)
    r = torch.full((b, n + 1, m + 1), float("inf"), dtype=pred.dtype)
    r[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            stack = torch.stack([r[:, i - 1, j - 1], r[:, i - 1, j], r[:, i, j - 1]], dim=-1)
            r[:, i, j] = dist[:, i - 1, j - 1] - gamma * torch.logsumexp(-stack / gamma, dim=-1)
    return r[:, n, m]


def _windowed_reference(pred: torch.Tensor, target: torch.Tensor, w: int, stride: int, gamma: float) -> torch.Tensor:
    b, c = pred.shape[0], pred.shape[1]
    ps = pred.reshape(b, c, -1).permute(0, 2, 1)
    ts = target.reshape(b, c, -1).permute(0, 2, 1)
    starts = range(0, ps.shape[1] - w + 1, stride)
    per_window = [_cellwise_soft_dtw(ps[:, s : s + w], ts[:, s : s + w], gamma).mean() for s in starts]
    return torch.stack(per_window).sum() / len(per_window)


class TestAntiDiagonalSweepEquivalence:
    """The vectorised DP and batched windows must reproduce the scalar loops."""

    @pytest.mark.parametrize(("n", "m"), [(5, 7), (7, 5), (6, 6), (1, 4), (4, 1)])
    def test_distance_and_gradient_match_the_cellwise_loop(self, n: int, m: int) -> None:
        torch.manual_seed(n * 10 + m)
        a = torch.randn(3, n, 2, dtype=torch.float64, requires_grad=True)
        b = torch.randn(3, m, 2, dtype=torch.float64, requires_grad=True)
        ours = _soft_dtw_distance(a, b, gamma=0.3)
        ref = _cellwise_soft_dtw(a, b, gamma=0.3)
        torch.testing.assert_close(ours, ref)
        for g_ours, g_ref in zip(
            torch.autograd.grad(ours.sum(), [a, b]),
            torch.autograd.grad(ref.sum(), [a, b]),
            strict=True,
        ):
            torch.testing.assert_close(g_ours, g_ref)

    @pytest.mark.parametrize(("w", "stride"), [(16, 16), (12, 5)])
    def test_batched_windows_match_the_window_loop(self, w: int, stride: int) -> None:
        torch.manual_seed(w)
        pred = torch.rand(2, 2, 6, 8, dtype=torch.float64, requires_grad=True)
        target = torch.rand(2, 2, 6, 8, dtype=torch.float64)
        ours = SoftDTWLoss(gamma=0.1, window_size=w, stride_size=stride)(pred, target)
        ref = _windowed_reference(pred, target, w, stride, 0.1)
        torch.testing.assert_close(ours, ref)
        (g_ours,) = torch.autograd.grad(ours, pred)
        (g_ref,) = torch.autograd.grad(ref, pred)
        torch.testing.assert_close(g_ours, g_ref)

    def test_many_windows_backpropagate_finitely(self) -> None:
        pred = torch.rand(2, 1, 32, 32, requires_grad=True)
        loss = SoftDTWLoss(gamma=0.1, window_size=32)(pred, torch.rand(2, 1, 32, 32))
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
