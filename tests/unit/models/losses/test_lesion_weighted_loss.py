"""Tests for ``LesionWeightedLoss``.

Targets ``spectramr.models.losses.lesion_weighted_loss``. Composable wrapper
that boosts the per-voxel loss inside a known lesion mask.

Categories:

- Construction: invalid ``base`` / negative boost rejected
- Identity (``pred == target``) → loss = 0 even with weight map
- Without mask: ``require_mask=True`` raises; ``False`` uses uniform
  weighting
- Mask boosts contributions inside the lesion region
- Soft mask vs hard mask: hard thresholds at 0.5
- Spatial-shape mismatch between mask and prediction raises
- Custom ``base`` callable supported
- Registry: ``lesion_weighted`` resolvable
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.losses.lesion_weighted_loss import LesionWeightedLoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_negative_boost_rejected() -> None:
    """``boost < 0`` raises."""
    with pytest.raises(ValueError, match="boost"):
        LesionWeightedLoss(boost=-1.0)


def test_unknown_base_string_rejected() -> None:
    """Unknown built-in base name raises."""
    with pytest.raises(ValueError, match="unknown base"):
        LesionWeightedLoss(base="cosine")


def test_default_boost() -> None:
    """Default ``boost=4.0`` (i.e. 5× lesion weighting)."""
    loss = LesionWeightedLoss()
    assert loss.boost == 4.0


# ---------------------------------------------------------------------------
# Closed form: identity & uniform weighting
# ---------------------------------------------------------------------------


def test_zero_loss_for_identity_with_mask() -> None:
    """``pred == target`` → loss = 0 regardless of mask."""
    pred = torch.randn(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    loss = LesionWeightedLoss(base="l1", boost=4.0)
    val = loss(pred, pred, context={"lesion_mask": mask}).item()
    assert pytest.approx(val, abs=1e-7) == 0.0


def test_no_mask_with_require_false_uses_uniform_weighting() -> None:
    """Without mask + ``require_mask=False`` reduces to plain mean of base."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    loss = LesionWeightedLoss(base="l1", require_mask=False)
    val = loss(pred, target).item()
    # Plain L1 mean = 1.0
    assert pytest.approx(val) == 1.0


def test_no_mask_with_require_true_raises() -> None:
    """``require_mask=True`` (default) → missing mask raises."""
    loss = LesionWeightedLoss(require_mask=True)
    with pytest.raises(ValueError, match="lesion_mask"):
        loss(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))


# ---------------------------------------------------------------------------
# Mask boosts loss inside lesion
# ---------------------------------------------------------------------------


def test_mask_inside_lesion_boosts_loss() -> None:
    """Per-voxel loss inside the mask is boosted by ``1 + β · mask``.

    Mathematically: ``L = sum(w · |diff|) / sum(w)`` where
    ``w = 1 + β · mask``. Larger β → more weight on the lesion residual.
    """
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    target[..., 0, 0] = 10.0  # only one voxel disagrees, inside the mask
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 0, 0] = 1.0

    loss_low = LesionWeightedLoss(base="l1", boost=0.0, soft_mask=True)
    loss_high = LesionWeightedLoss(base="l1", boost=10.0, soft_mask=True)
    val_low = loss_low(pred, target, context={"lesion_mask": mask}).item()
    val_high = loss_high(pred, target, context={"lesion_mask": mask}).item()
    assert val_high > val_low


def test_hard_mask_thresholds_at_half() -> None:
    """``soft_mask=False`` thresholds the mask at 0.5."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    target[..., 0, 0] = 1.0
    # Mask value 0.6 → hard mask treats this voxel as lesion (1).
    mask = torch.full((1, 1, 4, 4), 0.6)

    loss_hard = LesionWeightedLoss(base="l1", boost=4.0, soft_mask=False)
    loss_soft = LesionWeightedLoss(base="l1", boost=4.0, soft_mask=True)
    val_hard = loss_hard(pred, target, context={"lesion_mask": mask}).item()
    val_soft = loss_soft(pred, target, context={"lesion_mask": mask}).item()
    # Hard treats every voxel above 0.5 as 1 → uniform 5× weight; soft
    # uses 0.6 → uniform 3.4× weight. Different scaling but same relative
    # behaviour. We just check they differ.
    assert val_hard != val_soft


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_spatial_shape_mismatch_rejected() -> None:
    """Mask spatial shape must match prediction's."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros_like(pred)
    bad_mask = torch.ones(1, 1, 8, 8)
    loss = LesionWeightedLoss()
    with pytest.raises(ValueError, match="lesion_mask"):
        loss(pred, target, context={"lesion_mask": bad_mask})


def test_3d_mask_auto_promoted_to_channel_dim() -> None:
    """A ``[B, H, W]`` mask is auto-unsqueezed to ``[B, 1, H, W]``."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros_like(pred)
    mask = torch.ones(1, 4, 4)  # [B, H, W]
    loss = LesionWeightedLoss(base="l1")
    val = loss(pred, target, context={"lesion_mask": mask}).item()
    # No diff → 0.
    assert pytest.approx(val, abs=1e-7) == 0.0


def test_custom_base_module_supported() -> None:
    """A user-supplied ``nn.Module`` returning a per-voxel map works."""

    class _PerVoxelMSE(nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (pred - target) ** 2

    pred = torch.zeros(1, 1, 4, 4)
    target = torch.full((1, 1, 4, 4), 2.0)
    mask = torch.ones(1, 1, 4, 4)
    loss = LesionWeightedLoss(base=_PerVoxelMSE(), boost=0.0)
    val = loss(pred, target, context={"lesion_mask": mask}).item()
    # All voxels in mask, β = 0 → uniform weight 1; mean(4) = 4.
    assert pytest.approx(val) == 4.0


def test_custom_base_returning_scalar_rejected() -> None:
    """A scalar-returning base raises (the wrapper needs per-voxel)."""

    class _Scalar(nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (pred - target).abs().mean()

    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros_like(pred)
    mask = torch.ones(1, 1, 4, 4)
    loss = LesionWeightedLoss(base=_Scalar())
    with pytest.raises(ValueError, match="per-voxel"):
        loss(pred, target, context={"lesion_mask": mask})


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_prediction() -> None:
    """Backward through the wrapper reaches the prediction."""
    pred = torch.randn(1, 1, 4, 4, requires_grad=True)
    target = torch.randn(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 0:2, 0:2] = 1.0
    loss = LesionWeightedLoss()
    loss(pred, target, context={"lesion_mask": mask}).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``lesion_weighted`` resolvable in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "lesion_weighted" in list_available()


def test_multichannel_zero_boost_equals_the_plain_mean() -> None:
    """The weighted mean counts every channel in its denominator: with ``boost=0``
    a ``(B, 1, H, W)`` mask on a 2-channel input reproduces the unmasked mean
    rather than twice it."""
    gen = torch.Generator().manual_seed(0)
    pred = torch.randn(2, 2, 4, 4, generator=gen)
    target = torch.randn(2, 2, 4, 4, generator=gen)
    mask = torch.zeros(2, 1, 4, 4)
    mask[..., :2, :2] = 1.0
    masked = LesionWeightedLoss(base="l1", boost=0.0)(pred, target, context={"lesion_mask": mask})
    unmasked = LesionWeightedLoss(base="l1", require_mask=False)(pred, target)
    torch.testing.assert_close(masked, unmasked)
    torch.testing.assert_close(masked, (pred - target).abs().mean())
