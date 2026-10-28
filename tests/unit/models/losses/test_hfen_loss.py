"""Tests for ``HFENLoss``.

Targets ``spectramr.models.losses.hfen_loss``. High-Frequency Error Norm:
Laplacian-of-Gaussian filtered L2 distance, suitable as both a
high-pass loss (``normalize=False``) and as a ratio metric
(``normalize=True``).

Categories:

- LoG kernel: zero-mean (DC removed) + odd kernel size enforced
- Property: HFEN(x, x) == 0 (identity)
- Property: HFEN grows with high-frequency disturbance amplitude
- Edge: complex inputs use magnitude
- Sanity-shape: 4D ``[B, C, H, W]`` inputs
- Registry: ``hfen`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.hfen_loss import HFENLoss, _build_log_kernel


# ---------------------------------------------------------------------------
# LoG kernel
# ---------------------------------------------------------------------------


def test_log_kernel_zero_mean() -> None:
    """LoG kernel has zero mean (no DC bias)."""
    kernel = _build_log_kernel(kernel_size=15, sigma=1.5)
    assert pytest.approx(kernel.mean().item(), abs=1e-6) == 0.0


def test_log_kernel_shape() -> None:
    """Kernel has shape ``(1, 1, K, K)``."""
    kernel = _build_log_kernel(kernel_size=11, sigma=1.5)
    assert kernel.shape == (1, 1, 11, 11)


def test_log_kernel_even_size_raises() -> None:
    """Even kernel size raises even under ``python -O``, where an ``assert`` is stripped."""
    with pytest.raises(ValueError, match="odd"):
        _build_log_kernel(kernel_size=14, sigma=1.5)


# ---------------------------------------------------------------------------
# HFEN forward
# ---------------------------------------------------------------------------


def test_hfen_zero_for_identity_unnormalized() -> None:
    """``HFEN(x, x)`` is approximately zero when ``normalize=False``."""
    loss = HFENLoss(normalize=False)
    x = torch.randn(1, 1, 32, 32)
    out = loss(x, x)
    # Tiny epsilon under the sqrt prevents exactly zero.
    assert out.item() < 1e-3


def test_hfen_zero_for_identity_normalized() -> None:
    """``HFEN(x, x)`` ratio is approximately zero (numerator → 0)."""
    loss = HFENLoss(normalize=True)
    x = torch.randn(1, 1, 32, 32)
    out = loss(x, x)
    assert out.item() < 1e-3


def test_hfen_grows_with_perturbation() -> None:
    """A larger high-frequency perturbation gives a larger HFEN."""
    torch.manual_seed(0)
    loss = HFENLoss(normalize=False)
    x = torch.randn(1, 1, 32, 32)
    delta_small = torch.randn_like(x) * 0.1
    delta_big = torch.randn_like(x) * 1.0
    h_small = loss(x, x + delta_small).item()
    h_big = loss(x, x + delta_big).item()
    assert h_big > h_small


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def test_reduction_mean_returns_scalar() -> None:
    """``reduction='mean'`` collapses batch dim."""
    loss = HFENLoss(reduction="mean")
    pred = torch.randn(4, 1, 32, 32)
    target = torch.randn(4, 1, 32, 32)
    out = loss(pred, target)
    assert out.dim() == 0


def test_reduction_sum_returns_scalar() -> None:
    """``reduction='sum'`` also collapses to scalar."""
    loss = HFENLoss(reduction="sum")
    pred = torch.randn(4, 1, 32, 32)
    target = torch.randn(4, 1, 32, 32)
    out = loss(pred, target)
    assert out.dim() == 0


def test_reduction_none_returns_per_sample() -> None:
    """``reduction='none'`` keeps batch dim."""
    loss = HFENLoss(reduction="none")
    pred = torch.randn(4, 1, 32, 32)
    target = torch.randn(4, 1, 32, 32)
    out = loss(pred, target)
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Volumetric (5-D) inputs — slice-wise LoG (regression for exp_hm_07_25d_mamba)
# ---------------------------------------------------------------------------


def test_5d_volumetric_input_returns_scalar() -> None:
    """A 5-D ``[B, C, H, W, D]`` volume (TorchIO depth-last) is handled
    slice-wise. Pre-2026-06 this raised ``ValueError: too many values to
    unpack`` in ``_apply_log`` and crashed every volumetric Hilbert-Mamba arm.
    """
    loss = HFENLoss(reduction="mean")
    pred = torch.randn(2, 1, 32, 32, 4)
    target = pred + 0.1 * torch.randn_like(pred)
    out = loss(pred, target)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_5d_depth1_matches_squeezed_4d() -> None:
    """A single-slice 5-D volume equals the 4-D HFEN of its squeezed slice —
    the depth fold is a no-op transform when ``D == 1``."""
    loss = HFENLoss(reduction="mean")
    pred4 = torch.randn(3, 1, 32, 32)
    target4 = torch.randn(3, 1, 32, 32)
    pred5 = pred4.unsqueeze(-1)  # [B, C, H, W, 1]
    target5 = target4.unsqueeze(-1)
    assert torch.allclose(loss(pred5, target5), loss(pred4, target4), atol=1e-6)


def test_5d_reduction_none_is_per_slice() -> None:
    """``reduction='none'`` over a 5-D volume yields one value per (batch·depth)
    in-plane slice."""
    loss = HFENLoss(reduction="none")
    pred = torch.randn(2, 1, 16, 16, 4)
    out = loss(pred, torch.randn_like(pred))
    assert out.shape == (2 * 4,)


# ---------------------------------------------------------------------------
# Complex input
# ---------------------------------------------------------------------------


def test_complex_input_uses_magnitude() -> None:
    """Complex inputs work — internally taken via ``.abs()``."""
    loss = HFENLoss(normalize=False)
    pred = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
    target = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
    out = loss(pred, target)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Multi-channel
# ---------------------------------------------------------------------------


def test_multichannel_input_supported() -> None:
    """Multiple channels each get the same kernel via grouped conv."""
    loss = HFENLoss(normalize=False)
    pred = torch.randn(1, 4, 32, 32)
    target = torch.randn(1, 4, 32, 32)
    out = loss(pred, target)
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_pred() -> None:
    """Backward reaches the prediction tensor."""
    loss = HFENLoss(normalize=False)
    pred = torch.randn(1, 1, 32, 32, requires_grad=True)
    target = torch.randn(1, 1, 32, 32)
    loss(pred, target).backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_hfen_registered() -> None:
    """``hfen`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "hfen" in list_available()


def test_hfen_of_identical_images_is_zero_with_finite_gradient() -> None:
    """The norm floor keeps sqrt'(0) finite without adding 1e-4 to HFEN(x, x)."""
    target = torch.rand(2, 1, 32, 32, generator=torch.Generator().manual_seed(0))
    pred = target.clone().requires_grad_(True)
    out = HFENLoss()(pred, target)
    out.backward()
    assert out.item() < 1e-8
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_hfen_unknown_reduction_raises() -> None:
    """An unknown reduction used to return the per-sample vector silently."""
    with pytest.raises(ValueError, match="reduction"):
        HFENLoss(reduction="meen")


def test_hfen_repr_names_the_kernel() -> None:
    assert "kernel_size=15, sigma=1.5" in repr(HFENLoss())
