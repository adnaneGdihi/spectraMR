"""Tests for ``LNCCLoss``.

Targets ``spectramr.models.losses.cross_contrast_losses``. Localised
Normalised Cross-Correlation: ``-mean( cov(p, t)² / (var(p)·var(t)) )``.
Useful for cross-contrast / cross-scanner alignment because it's
invariant to local affine intensity shifts.

Categories:

- Construction: 2D / 3D dispatch, invalid dims raise
- Property: identity (pred == target) → loss = -1 (perfect correlation)
- Property: invariance to additive constant
- Property: invariance to multiplicative scaling
- Sanity-shape: parametrised matrix
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.cross_contrast_losses import LNCCLoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_2d_default() -> None:
    """Default 2D mode persists the kernel + window size."""
    loss = LNCCLoss(window_size=5, dims=2)
    assert loss.window_size == 5
    assert loss.kernel.shape == (1, 1, 5, 5)
    assert loss.window_num_elements == 25


def test_construction_3d() -> None:
    """3D mode produces a 3D kernel."""
    loss = LNCCLoss(window_size=3, dims=3)
    assert loss.kernel.shape == (1, 1, 3, 3, 3)
    assert loss.window_num_elements == 27


def test_construction_invalid_dims_raises() -> None:
    """``dims`` outside {2, 3} → ``ValueError``."""
    with pytest.raises(ValueError, match="only supports 2D and 3D"):
        LNCCLoss(dims=4)


# ---------------------------------------------------------------------------
# Identity / invariance properties
# ---------------------------------------------------------------------------


def test_identity_yields_minus_one() -> None:
    """``loss(x, x) ≈ -1`` (perfect correlation, negated)."""
    loss = LNCCLoss(window_size=3, dims=2)
    torch.manual_seed(0)
    x = torch.rand(1, 1, 16, 16) * 0.5 + 0.25  # avoid degenerate constants
    out = loss(x, x)
    # Should be very close to -1 (the negation of perfect correlation = 1).
    assert out.item() < -0.99


def test_invariant_to_additive_constant() -> None:
    """LNCC is invariant to ``target = pred + c``."""
    loss = LNCCLoss(window_size=3, dims=2)
    torch.manual_seed(0)
    pred = torch.rand(1, 1, 16, 16)
    target_offset = pred + 5.0  # global brightness shift
    out_offset = loss(pred, target_offset)
    out_identity = loss(pred, pred)
    # Both should be close to -1 (perfect correlation up to translation).
    assert abs(out_offset.item() - out_identity.item()) < 0.05


def test_invariant_to_multiplicative_scaling() -> None:
    """LNCC is invariant to ``target = α·pred``."""
    loss = LNCCLoss(window_size=3, dims=2)
    torch.manual_seed(0)
    pred = torch.rand(1, 1, 16, 16)
    target_scaled = pred * 3.0  # global contrast scaling
    out_scaled = loss(pred, target_scaled)
    out_identity = loss(pred, pred)
    assert abs(out_scaled.item() - out_identity.item()) < 0.05


def test_uncorrelated_inputs_give_higher_loss() -> None:
    """Uncorrelated random pairs → loss > -1 (worse than perfect)."""
    loss = LNCCLoss(window_size=3, dims=2)
    torch.manual_seed(0)
    pred = torch.rand(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)  # independent
    out = loss(pred, target)
    assert out.item() > -0.5


# ---------------------------------------------------------------------------
# Multi-channel handling
# ---------------------------------------------------------------------------


def test_multi_channel_collapsed_to_mean() -> None:
    """Inputs with C > 1 are mean-collapsed before structural comparison."""
    loss = LNCCLoss(window_size=3, dims=2)
    torch.manual_seed(0)
    pred = torch.rand(1, 4, 16, 16)
    target = pred.clone()
    out = loss(pred, target)
    # After collapse → identity → ≈ -1
    assert out.item() < -0.95


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 16, 16), (2, 1, 32, 32), (1, 1, 64, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_lncc_2d_shape_matrix(shape: tuple[int, ...]) -> None:
    """LNCC handles the parametrised 2D shape matrix."""
    loss = LNCCLoss(window_size=3, dims=2)
    pred = torch.rand(*shape)
    out = loss(pred, pred.clone())
    assert out.dim() == 0
    assert torch.isfinite(out)


@pytest.mark.parametrize(
    "shape", [(1, 1, 8, 8, 8), (2, 1, 16, 16, 16)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_lncc_3d_shape_matrix(shape: tuple[int, ...]) -> None:
    """LNCC handles the parametrised 3D shape matrix."""
    loss = LNCCLoss(window_size=3, dims=3)
    pred = torch.rand(*shape)
    out = loss(pred, pred.clone())
    assert out.dim() == 0
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_lncc_registered() -> None:
    """``lncc`` is registered in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "lncc" in list_available()


# ---------------------------------------------------------------------------
# Reduced precision
# ---------------------------------------------------------------------------


def test_half_precision_input_is_finite_and_matches_float32() -> None:
    """fp16 inputs are evaluated in float32, where the 1e-8 floor does not underflow."""
    torch.manual_seed(0)
    pred = torch.zeros(1, 1, 32, 32)
    pred[..., 8:24, 8:24] = torch.rand(16, 16) * 1e-2
    target = pred * 0.9 + 1e-3 * torch.rand_like(pred)
    loss = LNCCLoss()
    ref = loss(pred, target)
    half = pred.half().requires_grad_(True)
    out = loss(half, target.half())
    out.backward()
    assert torch.isfinite(out)
    assert half.grad is not None and torch.isfinite(half.grad).all()
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=0.0)


def test_float64_input_is_accepted() -> None:
    """A float64 batch no longer meets a float32 box kernel inside ``conv2d``."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 16, 16, dtype=torch.float64)
    out = LNCCLoss(window_size=3)(x, x)
    assert out.dtype == torch.float64
    assert float(out) < -0.99
