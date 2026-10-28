"""Tests for the standard registry-wrapped losses.

Targets ``spectramr.models.losses.standard_losses``. PyTorch built-ins
(``L1``, ``L2``/``MSE``, ``SmoothL1`` / Huber, ``BCE``, ``BCEWithLogits``,
``CrossEntropy``, ``TotalVariation``) wrapped with the registry plus
optional metrics-aware mixin.

Categories:

- Closed-form: identity inputs → zero (L1, L2, SmoothL1)
- Complex inputs: magnitude-mode L1 / L2 (the wrapped path)
- Metrics: ``compute_metrics=True`` returns a tuple ``(loss, dict)``
- TotalVariation: zero on a constant image, positive on noise
- Registry: all aliases resolvable
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.standard_losses import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    L1Loss,
    MSELoss,
    SmoothL1Loss,
    TotalVariationLoss,
)


# ---------------------------------------------------------------------------
# L1 (real & complex)
# ---------------------------------------------------------------------------


def test_l1_zero_for_identity() -> None:
    """``L1(x, x) == 0``."""
    loss = L1Loss()
    x = torch.randn(1, 1, 4, 4)
    assert pytest.approx(loss(x, x).item(), abs=1e-6) == 0.0


def test_l1_complex_uses_magnitude() -> None:
    """Complex L1 = mean(|pred - target|)."""
    loss = L1Loss()
    pred = torch.complex(torch.tensor([3.0]), torch.tensor([4.0]))
    target = torch.complex(torch.tensor([0.0]), torch.tensor([0.0]))
    # |3 + 4i| = 5
    assert pytest.approx(loss(pred, target).item(), abs=1e-3) == 5.0


def test_l1_metrics_returns_tuple() -> None:
    """``compute_metrics=True`` returns ``(loss, dict)``."""
    loss = L1Loss(compute_metrics=True)
    pred = torch.randn(1, 1, 4, 4)
    target = torch.randn(1, 1, 4, 4)
    out = loss(pred, target)
    assert isinstance(out, tuple)
    val, metrics = out
    assert "loss" in metrics
    assert "error_mean" in metrics


# ---------------------------------------------------------------------------
# L2 / MSE
# ---------------------------------------------------------------------------


def test_l2_zero_for_identity() -> None:
    """``MSE(x, x) == 0``."""
    loss = MSELoss()
    x = torch.randn(1, 1, 4, 4)
    assert pytest.approx(loss(x, x).item(), abs=1e-6) == 0.0


def test_l2_complex_uses_squared_magnitude() -> None:
    """Complex L2 returns mean(|pred - target|²)."""
    loss = MSELoss()
    pred = torch.complex(torch.tensor([3.0]), torch.tensor([4.0]))
    target = torch.complex(torch.tensor([0.0]), torch.tensor([0.0]))
    # |3 + 4i|² = 25
    assert pytest.approx(loss(pred, target).item()) == 25.0


def test_l2_metrics_includes_rmse() -> None:
    """Metrics dict includes ``rmse = sqrt(loss)``."""
    loss = MSELoss(compute_metrics=True)
    pred = torch.full((1, 1, 4, 4), 1.0)
    target = torch.zeros_like(pred)
    val, metrics = loss(pred, target)
    assert pytest.approx(metrics["rmse"]) == 1.0


# ---------------------------------------------------------------------------
# SmoothL1 (Huber)
# ---------------------------------------------------------------------------


def test_smooth_l1_zero_for_identity() -> None:
    """``SmoothL1(x, x) == 0``."""
    loss = SmoothL1Loss()
    x = torch.randn(1, 1, 4, 4)
    assert pytest.approx(loss(x, x).item(), abs=1e-6) == 0.0


def test_smooth_l1_complex_uses_magnitude() -> None:
    """Complex Huber works on the magnitude of the residual."""
    loss = SmoothL1Loss()
    pred = torch.complex(torch.tensor([0.5]), torch.tensor([0.0]))
    target = torch.complex(torch.tensor([0.0]), torch.tensor([0.0]))
    # |0.5| < beta=1, so 0.5 * 0.5² / beta = 0.125
    assert pytest.approx(loss(pred, target).item()) == 0.125


# ---------------------------------------------------------------------------
# BCE family
# ---------------------------------------------------------------------------


def test_bce_zero_for_perfect_prediction() -> None:
    """``BCE(target, target)`` is approximately 0 for binary targets."""
    loss = BCELoss()
    target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    pred = target.clone() * 0.99 + 0.005  # very close
    out = loss(pred, target)
    assert out.item() < 0.05


def test_bce_with_logits_finite_on_random_input() -> None:
    """Logits-version returns a finite loss on random logits."""
    loss = BCEWithLogitsLoss()
    pred = torch.randn(8)
    target = torch.randint(0, 2, (8,)).float()
    out = loss(pred, target)
    assert torch.isfinite(out)


def test_cross_entropy_finite() -> None:
    """``CrossEntropy`` works for ``[N, num_classes]`` + integer targets."""
    loss = CrossEntropyLoss()
    pred = torch.randn(4, 3)  # 3 classes
    target = torch.randint(0, 3, (4,))
    out = loss(pred, target)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# TotalVariation
# ---------------------------------------------------------------------------


def test_total_variation_zero_on_constant() -> None:
    """``TV`` of a constant image is exactly zero."""
    loss = TotalVariationLoss()
    x = torch.full((1, 1, 8, 8), 0.5)
    assert pytest.approx(loss(x).item(), abs=1e-6) == 0.0


def test_total_variation_positive_on_noise() -> None:
    """``TV`` of random noise is strictly positive."""
    torch.manual_seed(0)
    loss = TotalVariationLoss()
    x = torch.randn(1, 1, 8, 8)
    assert loss(x).item() > 0.0


def test_total_variation_grows_with_noise_amplitude() -> None:
    """Doubling the noise std-dev increases TV monotonically."""
    torch.manual_seed(0)
    loss = TotalVariationLoss()
    base = torch.randn(1, 1, 8, 8)
    tv_small = loss(base * 0.1).item()
    tv_big = loss(base * 1.0).item()
    assert tv_big > tv_small


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_standard_losses_registered() -> None:
    """All canonical names resolvable in the registry."""
    from spectramr.models.losses.registry import list_available

    names = list_available()
    for canonical in ["l1", "l2", "smooth_l1", "bce", "bce_with_logits", "cross_entropy", "tv"]:
        assert canonical in names, f"{canonical} missing"


def _reference_tv_4d(x: torch.Tensor) -> torch.Tensor:
    """The two-axis formula this loss has always computed on ``[B, C, H, W]``."""
    count_h = x.size(1) * (x.size(2) - 1) * x.size(3)
    count_w = x.size(1) * x.size(2) * (x.size(3) - 1)
    h_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
    w_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / x.size(0)


def test_total_variation_4d_matches_the_two_axis_formula_exactly() -> None:
    torch.manual_seed(1)
    x = torch.randn(3, 2, 7, 9)
    assert torch.equal(TotalVariationLoss()(x), _reference_tv_4d(x))


def test_total_variation_5d_sees_variation_along_the_last_axis() -> None:
    """On ``[B, C, D, H, W]`` the old code differenced D and H only and never W."""
    x = torch.zeros(1, 1, 4, 4, 4)
    x[..., ::2] = 1.0
    assert TotalVariationLoss()(x).item() == pytest.approx(2.0)


def test_total_variation_5d_treats_every_spatial_axis_alike() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 1, 5, 6, 7)
    permuted = x.permute(0, 1, 4, 3, 2).contiguous()
    torch.testing.assert_close(TotalVariationLoss()(x), TotalVariationLoss()(permuted))


def test_total_variation_skips_a_unit_axis_instead_of_dividing_by_zero() -> None:
    x = torch.randn(2, 1, 8, 1, requires_grad=True)
    out = TotalVariationLoss()(x)
    assert torch.isfinite(out)
    (grad,) = torch.autograd.grad(out, x)
    assert torch.isfinite(grad).all()


def test_total_variation_refuses_a_tensor_with_no_spatial_axis() -> None:
    with pytest.raises(ValueError, match="spatial"):
        TotalVariationLoss()(torch.randn(2, 3))


@pytest.mark.parametrize("cls", [L1Loss, MSELoss])
def test_an_unknown_reduction_is_refused_at_construction(cls: type) -> None:
    """The complex branch used to return the unreduced tensor for any unknown string."""
    with pytest.raises(ValueError, match="reduction"):
        cls(reduction="batchmean")


@pytest.mark.parametrize("cls", [L1Loss, MSELoss])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_the_supported_reductions_still_construct(cls: type, reduction: str) -> None:
    a = torch.randn(2, 3, dtype=torch.complex64)
    out = cls(reduction=reduction)(a, a.conj())
    assert out.shape == (() if reduction != "none" else (2, 3))
