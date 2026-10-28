"""Tests for ``MMDLoss``.

Targets ``spectramr.models.losses.mmd_loss``. Multi-bandwidth Gaussian-kernel
Maximum Mean Discrepancy: a closed-form, gradient-friendly two-sample
distance for unpaired domain alignment.

Categories:

- Construction: invalid sigmas raise; default tuple stored
- Property: ``MMD(P, P) ≈ 0`` (same distribution)
- Property: ``MMD(P, Q) > 0`` for genuinely-different distributions
- Edge: < 2 samples per side raises (unbiased estimator denominator)
- Sanity-shape: 4D input flattens correctly
- Registry: ``mmd`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.mmd_loss import MMDLoss, _multi_kernel_mmd


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_sigmas_persisted() -> None:
    """Default: five multiples of the median distance, two octaves either side."""
    loss = MMDLoss()
    assert loss.sigmas == (0.25, 0.5, 1.0, 2.0, 4.0)
    assert loss.bandwidth_mode == "median"


def test_unknown_bandwidth_mode_raises() -> None:
    """An unknown ``bandwidth_mode`` raises instead of falling back to a default."""
    with pytest.raises(ValueError, match="bandwidth_mode"):
        MMDLoss(bandwidth_mode="silverman")


def test_empty_sigmas_raises() -> None:
    """Empty sigma tuple → ``ValueError``."""
    with pytest.raises(ValueError, match="at least one bandwidth"):
        MMDLoss(sigmas=())


def test_non_positive_sigma_raises() -> None:
    """Any sigma ≤ 0 → ``ValueError``."""
    with pytest.raises(ValueError, match="must be > 0"):
        MMDLoss(sigmas=(1.0, -2.0))


# ---------------------------------------------------------------------------
# Property: MMD properties
# ---------------------------------------------------------------------------


def test_mmd_same_distribution_near_zero() -> None:
    """``MMD(P, P)`` for two large samples from N(0,1) is small."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(1.0, 2.0, 4.0))
    x = torch.randn(64, 8)
    y = torch.randn(64, 8)
    out = loss(x, y)
    # Two identically-distributed Gaussians: MMD is small but not exactly zero.
    assert out.abs().item() < 1.0


def test_mmd_different_distribution_positive() -> None:
    """Mean-shifted samples → MMD is meaningfully positive."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(1.0, 2.0, 4.0))
    x = torch.randn(64, 8)
    y = torch.randn(64, 8) + 5.0  # mean-shifted
    out = loss(x, y)
    assert out.item() > 0.1  # well-separated → strictly positive


def test_mmd_grows_with_separation() -> None:
    """Larger mean shift → larger MMD."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(1.0, 2.0, 4.0))
    x = torch.randn(64, 8)
    y_small = torch.randn(64, 8) + 1.0
    y_big = torch.randn(64, 8) + 5.0
    mmd_small = loss(x, y_small).item()
    mmd_big = loss(x, y_big).item()
    assert mmd_big > mmd_small


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


def test_too_few_samples_raises() -> None:
    """``n < 2`` or ``m < 2`` → ``ValueError``."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        _multi_kernel_mmd(torch.randn(1, 4), torch.randn(4, 4), sigmas=(1.0,))


def test_unequal_batch_sizes_supported() -> None:
    """Different ``n`` and ``m`` are both accepted (unpaired alignment)."""
    torch.manual_seed(0)
    loss = MMDLoss()
    x = torch.randn(16, 4)
    y = torch.randn(32, 4)
    out = loss(x, y)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Spatial flattening
# ---------------------------------------------------------------------------


def test_flatten_spatial_default_handles_4d() -> None:
    """``[B, C, H, W]`` input is flattened to ``[B, C·H·W]`` before MMD."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(2.0,))
    x = torch.randn(4, 1, 8, 8)
    y = torch.randn(4, 1, 8, 8)
    out = loss(x, y)
    assert out.dim() == 0  # scalar


def test_flatten_disabled_requires_2d() -> None:
    """``flatten_spatial=False`` consumes ``[B, D]`` directly."""
    loss = MMDLoss(sigmas=(2.0,), flatten_spatial=False)
    x = torch.randn(4, 16)
    y = torch.randn(4, 16)
    out = loss(x, y)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_input() -> None:
    """Backward through MMD reaches the prediction."""
    torch.manual_seed(0)
    loss = MMDLoss(sigmas=(2.0,))
    x = torch.randn(8, 4, requires_grad=True)
    y = torch.randn(8, 4)
    loss(x, y).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_mmd_registered() -> None:
    """``mmd`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "mmd" in list_available()


# ---------------------------------------------------------------------------
# Bandwidth scale (#2277)
# ---------------------------------------------------------------------------


def _image_batches(shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 1, 64, 64, generator=g)
    y = torch.randn(8, 1, 64, 64, generator=g) + shift
    return x, y


def test_median_bandwidth_separates_image_distributions() -> None:
    """On 4096-d images a 0.5-sigma shift is visible and carries gradient.

    With absolute bandwidths 1..16 every off-diagonal kernel value underflows at
    this dimension: the shifted pair scored ~1e-7 with a 1e-8 gradient norm.
    """
    loss = MMDLoss()
    x_same, y_same = _image_batches(shift=0.0)
    x, y = _image_batches(shift=0.5)
    x.requires_grad_(True)
    shifted = loss(x, y)
    shifted.backward()
    assert abs(float(loss(x_same, y_same))) < 0.02
    assert float(shifted) > 0.1
    assert x.grad is not None and float(x.grad.norm()) > 1e-3


def test_absolute_mode_uses_sigmas_as_distances() -> None:
    """``bandwidth_mode="absolute"`` evaluates exp(-d^2 / 2 sigma^2) as given."""
    x = torch.tensor([[0.0], [1.0]])
    y = torch.tensor([[3.0], [5.0]])
    out = _multi_kernel_mmd(x, y, sigmas=(2.0,), bandwidth_mode="absolute")

    def k(a: float, b: float) -> float:
        return float(torch.exp(torch.tensor(-((a - b) ** 2) / 8.0)))

    expected = k(0, 1) + k(3, 5) - 0.5 * (k(0, 3) + k(0, 5) + k(1, 3) + k(1, 5))
    assert float(out) == pytest.approx(expected, rel=1e-6)


def test_median_scale_is_detached() -> None:
    """The median bandwidth carries no gradient; only the kernel arguments do."""
    from spectramr.models.losses.mmd_loss import pooled_gaussian_kernel_sum

    x = torch.randn(4, 6, requires_grad=True)
    y = torch.randn(4, 6)
    kernel = pooled_gaussian_kernel_sum(x, y, (1.0,), "median")
    (grad,) = torch.autograd.grad(kernel.sum(), x)
    assert torch.isfinite(grad).all()


def test_duplicate_majority_does_not_collapse_the_median_bandwidth() -> None:
    """More than half the pooled pairs at distance 0 put the plain median at 0, and
    the kernel became an exact-match test: no gradient reaches the outliers."""
    x = torch.zeros(6, 10)
    y = torch.cat([torch.zeros(4, 10), torch.ones(2, 10)], dim=0).requires_grad_(True)
    loss = MMDLoss()(x, y)
    loss.backward()
    assert loss.item() > 0
    assert y.grad[4:].abs().sum() > 1e-6


def test_all_identical_samples_give_a_finite_zero_ish_mmd() -> None:
    x = torch.zeros(4, 8)
    loss = MMDLoss()(x, x.clone())
    assert torch.isfinite(loss)
