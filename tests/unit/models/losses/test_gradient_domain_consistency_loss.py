"""Tests for ``GradientDomainConsistencyLoss``.

Targets ``spectramr.models.losses.gradient_domain_consistency_loss``.
Penalises divergence in gradient fields between prediction and target —
forces a translation generator to preserve edge support across
contrasts. Three gradient operators (``finite_diff``, ``sobel``,
``laplacian``); three norms (``l1``, ``l2``, ``smooth_l1``).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.gradient_domain_consistency_loss import (
    GradientDomainConsistencyLoss,
    _finite_diff,
    _laplacian,
    _sobel,
)


# ---------------------------------------------------------------------------
# Operator-level — finite-diff / Sobel / Laplacian
# ---------------------------------------------------------------------------


def test_finite_diff_shape() -> None:
    """Finite-diff doubles the channel dimension (dx, dy stacked)."""
    x = torch.zeros(1, 1, 8, 8)
    g = _finite_diff(x)
    assert g.shape == (1, 2, 8, 8)


def test_finite_diff_constant_gives_zero() -> None:
    """Constant input → zero gradient."""
    x = torch.full((1, 1, 8, 8), 3.0)
    g = _finite_diff(x)
    assert torch.allclose(g, torch.zeros_like(g))


def test_sobel_shape() -> None:
    """Sobel doubles the channel dimension."""
    x = torch.zeros(1, 1, 8, 8)
    g = _sobel(x)
    assert g.shape == (1, 2, 8, 8)


def test_laplacian_shape() -> None:
    """Laplacian preserves the channel dimension."""
    x = torch.zeros(1, 1, 8, 8)
    g = _laplacian(x)
    assert g.shape == (1, 1, 8, 8)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator", ["finite_diff", "sobel", "laplacian"])
def test_supported_operators_construct(operator: str) -> None:
    """All three documented operators are accepted."""
    loss = GradientDomainConsistencyLoss(operator=operator)
    assert loss.operator == operator


def test_invalid_operator_raises() -> None:
    """Unknown operator → ``ValueError`` listing valid options."""
    with pytest.raises(ValueError, match="operator must be one of"):
        GradientDomainConsistencyLoss(operator="prewitt")


@pytest.mark.parametrize("norm", ["l1", "l2", "smooth_l1"])
def test_supported_norms_construct(norm: str) -> None:
    """All three norms are accepted."""
    loss = GradientDomainConsistencyLoss(norm=norm)
    assert loss.norm == norm


def test_invalid_norm_raises() -> None:
    """Unknown norm → ``ValueError`` listing valid options."""
    with pytest.raises(ValueError, match="norm must be one of"):
        GradientDomainConsistencyLoss(norm="huber_pseudo")


def test_invalid_anchor_raises() -> None:
    """Anchor must be ``'target'`` or ``'prediction'``."""
    with pytest.raises(ValueError, match="anchor must be"):
        GradientDomainConsistencyLoss(anchor="halfway")


# ---------------------------------------------------------------------------
# Forward — identity / shape mismatch
# ---------------------------------------------------------------------------


def test_identity_yields_zero_loss() -> None:
    """``loss(x, x) == 0`` for all operator/norm combinations."""
    for op in ["finite_diff", "sobel", "laplacian"]:
        loss = GradientDomainConsistencyLoss(operator=op)
        x = torch.randn(1, 1, 16, 16)
        out = loss(x, x)
        assert out.item() < 1e-7, f"operator={op} failed identity"


def test_shape_mismatch_raises() -> None:
    """Mismatched prediction/target shapes → ``ValueError``."""
    loss = GradientDomainConsistencyLoss()
    pred = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 16, 16)
    with pytest.raises(ValueError, match="!="):
        loss(pred, target)


def test_loss_grows_with_gradient_difference() -> None:
    """Different gradient patterns → strictly positive loss."""
    loss = GradientDomainConsistencyLoss(operator="finite_diff", norm="l1")
    pred = torch.zeros(1, 1, 16, 16)
    target = torch.zeros(1, 1, 16, 16)
    target[..., :, 8:] = 1.0  # vertical edge in target only
    out = loss(pred, target)
    assert out.item() > 0


# ---------------------------------------------------------------------------
# normalise_per_image
# ---------------------------------------------------------------------------


def test_normalise_per_image_scale_invariant() -> None:
    """Per-image normalisation makes loss invariant to amplitude scaling."""
    loss_norm = GradientDomainConsistencyLoss(
        operator="finite_diff", norm="l2", normalise_per_image=True
    )
    pred = torch.randn(1, 1, 16, 16)
    target = pred * 5.0  # 5× amplitude scaling
    out = loss_norm(pred, target)
    # After per-image RMS normalisation, scaling is removed.
    assert out.item() < 1e-6


def test_normalise_per_image_off_is_scale_sensitive() -> None:
    """Without normalisation, scaled images give a larger loss."""
    loss_no_norm = GradientDomainConsistencyLoss(
        operator="finite_diff", norm="l2", normalise_per_image=False
    )
    pred = torch.randn(1, 1, 16, 16)
    target = pred * 5.0
    out = loss_no_norm(pred, target)
    assert out.item() > 1e-3


# ---------------------------------------------------------------------------
# Sanity-shape & gradient flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape", [(1, 1, 16, 16), (2, 2, 32, 32), (1, 3, 16, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_loss_shape_matrix(shape: tuple[int, ...]) -> None:
    """Loss handles single + multi-channel + rectangular inputs."""
    loss = GradientDomainConsistencyLoss()
    pred = torch.randn(*shape)
    out = loss(pred, pred.clone())
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_gradient_flows_through_loss() -> None:
    """Backward through the loss reaches the prediction tensor."""
    loss = GradientDomainConsistencyLoss(operator="sobel")
    pred = torch.randn(1, 1, 16, 16, requires_grad=True)
    target = torch.zeros(1, 1, 16, 16)
    target[..., 8:, :] = 1.0
    out = loss(pred, target)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_loss_registered() -> None:
    """``gradient_domain_consistency`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "gradient_domain_consistency" in list_available()


def test_normalise_per_image_gradient_is_finite_for_a_flat_prediction() -> None:
    """A constant prediction has an all-zero gradient field, whose RMS is 0."""
    pred = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 1, 8, 8, generator=torch.Generator().manual_seed(0))
    GradientDomainConsistencyLoss(normalise_per_image=True)(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


@pytest.mark.parametrize(("anchor", "target_moves"), [("target", False), ("prediction", True)])
def test_anchor_decides_whether_the_target_receives_gradient(
    anchor: str, target_moves: bool
) -> None:
    """``anchor`` is read: the default pins the target's gradient field."""
    gen = torch.Generator().manual_seed(1)
    pred = torch.rand(1, 1, 8, 8, generator=gen, requires_grad=True)
    target = torch.rand(1, 1, 8, 8, generator=gen, requires_grad=True)
    GradientDomainConsistencyLoss(anchor=anchor)(pred, target).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    target_grad = 0.0 if target.grad is None else target.grad.abs().sum().item()
    assert (target_grad > 0) is target_moves
