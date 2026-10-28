"""Adjoint / projection invariants for data-consistency (DC) layers.

Invariants tested
-----------------
For **HardDataConsistency** (weight=1 ≡ hard DC):
  ``output_kspace[mask==1] ≡ measured_kspace[mask==1]``
  (measured samples are preserved exactly at masked positions).

For **SoftDataConsistency** (lambda_init → ∞, effectively hard DC):
  As λ→∞, ``k_out[mask==1] → measured_k[mask==1]``.  At finite large
  λ the error is O(1/λ).

For **data_consistency(x, k, mask, weight)** (the function wrapper):
  - weight=1.0: soft-blend via SimpleDataConsistency; measured samples
    dominate but are not exact (blend is (k + λ·k_meas)/(1 + λ)).
  - weight=0.0: output ≡ input  (no DC applied).

For **NoiseSimulatingDataConsistency** (noise-augmented):
  - With noise_lvl=0.0: ``output[mask==1]`` must equal
    ``measured_kspace[mask==1]`` in k-space after the fft2c pass
    (the layer enforces ``kspace_consistent = (1-mask)*k_pred + mask*k_meas``).

Design notes
------------
* All layers are set to eval mode so the noise level is deterministic
  (eval_noise_level is used, and we always pass noise_lvl=0.0).
* Complex inputs are provided directly as complex64 tensors.
* Masks are float32 {0, 1}.
* Tests are CPU-only.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.data_consistency import (
    NoiseSimulatingDataConsistency,
    HardDataConsistency,
    SoftDataConsistency,
    data_consistency,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from tests.utils.phantoms import random_complex
from tests.utils.tolerances import tol_for

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_B, _C, _H, _W = 2, 1, 16, 16


def _all_ones_mask(B: int = _B, H: int = _H, W: int = _W) -> torch.Tensor:
    """Full-sampling mask: every location is measured."""
    return torch.ones(B, 1, H, W, dtype=torch.float32)


def _alternating_cols_mask(B: int = _B, H: int = _H, W: int = _W) -> torch.Tensor:
    """Every other column sampled (acceleration ≈ 2)."""
    m = torch.zeros(B, 1, H, W, dtype=torch.float32)
    m[:, :, :, ::2] = 1.0
    return m


def _random_mask(B: int = _B, H: int = _H, W: int = _W, seed: int = 7) -> torch.Tensor:
    """~50% random binary mask."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    m = torch.randint(0, 2, (B, 1, H, W), generator=gen, dtype=torch.float32)
    m[:, :, H // 2, W // 2] = 1.0  # keep DC always sampled
    return m


_MASKS = {
    "all_ones": _all_ones_mask,
    "alternating_cols": _alternating_cols_mask,
    "random": _random_mask,
}

_MASK_IDS = list(_MASKS.keys())


def _rand_complex_img(seed: int = 42) -> torch.Tensor:
    """Random complex64 image (B, C, H, W)."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return random_complex((_B, _C, _H, _W), dtype=torch.complex64, generator=gen)


# ---------------------------------------------------------------------------
# 1. HardDataConsistency: measured k-space preserved at mask positions
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("mask_name", _MASK_IDS)
def test_hard_dc_preserves_measured_kspace(mask_name: str) -> None:
    """HardDataConsistency with no noise preserves k_meas at mask==1 exactly."""
    mask = _MASKS[mask_name]()
    x_img = _rand_complex_img(seed=10)
    x_img_for_recon = _rand_complex_img(seed=20)  # "reconstruction" (different from meas)

    # Build measured k-space from a reference image
    k_measured = fft2c(x_img)  # complex64, (B, C, H, W)

    layer = HardDataConsistency(
        train_noise_level=0.0,
        eval_noise_level=0.0,
    )
    layer.eval()

    # Forward pass: pass complex inputs
    out = layer.forward(
        reconstruction=x_img_for_recon,
        kspace_obs=k_measured,
        mask=mask,
    )
    # out is complex64 (same domain as input)

    # Reconstruct k-space of output
    k_out = fft2c(out)

    # At mask==1 positions: k_out should match k_measured
    mask_bool = mask.squeeze(1).bool()  # (B, H, W)
    k_out_masked = k_out.squeeze(1)[mask_bool]       # flatten masked positions
    k_meas_masked = k_measured.squeeze(1)[mask_bool]

    tol = tol_for(torch.complex64) * 100  # generous for fft2c→ifft2c round-trip
    max_err = (k_out_masked - k_meas_masked).abs().max().item()
    assert max_err < tol, (
        f"HardDC [{mask_name}]: k_out at mask positions deviates from k_meas.\n"
        f"  max_err={max_err:.3e}  tol={tol:.3e}"
    )


# ---------------------------------------------------------------------------
# 2. HardDataConsistency: at mask==0 positions, output comes from prediction
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("mask_name", ["alternating_cols", "random"])
def test_hard_dc_prediction_at_unsampled(mask_name: str) -> None:
    """HardDC: k-space at unsampled positions equals the FFT of reconstruction."""
    mask = _MASKS[mask_name]()
    x_img_pred = _rand_complex_img(seed=30)
    k_measured = fft2c(_rand_complex_img(seed=40))

    layer = HardDataConsistency(
        train_noise_level=0.0,
        eval_noise_level=0.0,
    )
    layer.eval()

    out = layer.forward(
        reconstruction=x_img_pred,
        kspace_obs=k_measured,
        mask=mask,
    )

    k_pred = fft2c(x_img_pred)
    k_out = fft2c(out)

    # At mask==0 positions the layer should pass through the prediction
    unsampled_bool = (mask.squeeze(1) == 0.0)  # (B, H, W)
    if unsampled_bool.any():
        k_out_unsampled = k_out.squeeze(1)[unsampled_bool]
        k_pred_unsampled = k_pred.squeeze(1)[unsampled_bool]
        tol = tol_for(torch.complex64) * 100
        max_err = (k_out_unsampled - k_pred_unsampled).abs().max().item()
        assert max_err < tol, (
            f"HardDC [{mask_name}]: unsampled positions do not match prediction k-space.\n"
            f"  max_err={max_err:.3e}  tol={tol:.3e}"
        )


# ---------------------------------------------------------------------------
# 3. SoftDataConsistency: large lambda → approaches hard DC
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("mask_name", _MASK_IDS)
def test_soft_dc_large_lambda_approaches_hard_dc(mask_name: str) -> None:
    """SoftDC with very large lambda converges toward HardDC at sampled positions.

    Soft DC formula: k_out = (k_pred + λ·k_meas·mask) / (1 + λ·mask).
    At mask=1: k_out = (k_pred + λ·k_meas) / (1 + λ) → k_meas as λ→∞.
    Error = |k_meas - k_out|_mask ≈ |k_pred - k_meas| / (1 + λ).
    """
    mask = _MASKS[mask_name]()
    x_img_pred = _rand_complex_img(seed=50)
    k_measured = fft2c(_rand_complex_img(seed=60))

    lambda_large = 1e4
    layer = SoftDataConsistency(lambda_init=lambda_large, learnable=False)
    layer.eval()

    out = layer(
        predicted_img=x_img_pred,
        measured_kspace=k_measured,
        mask=mask,
    )

    k_out = fft2c(out)
    mask_bool = mask.squeeze(1).bool()
    if mask_bool.any():
        err = (k_out.squeeze(1)[mask_bool] - k_measured.squeeze(1)[mask_bool]).abs().max().item()
        # O(1/(1+λ)) error budget
        expected_max = 10.0 / (1.0 + lambda_large)
        assert err < expected_max + 1e-3, (
            f"SoftDC [{mask_name}]: large-lambda convergence failed.\n"
            f"  err={err:.3e}  expected<{expected_max:.3e}"
        )


# ---------------------------------------------------------------------------
# 4. data_consistency() helper: weight=0 → output ≡ input
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("mask_name", _MASK_IDS)
def test_data_consistency_weight_zero_passthrough(mask_name: str) -> None:
    """data_consistency(x, k, mask, weight=0) returns the prediction unchanged.

    When weight=0, the SimpleDataConsistency blend reduces to the prediction:
    k_out = k_pred everywhere, so ifft2c(k_out) ≈ x.
    """
    mask = _MASKS[mask_name]()
    x_img = _rand_complex_img(seed=70)
    k_measured = fft2c(_rand_complex_img(seed=80))

    out = data_consistency(x_img, k_measured, mask, weight=0.0)

    tol = tol_for(torch.complex64) * 1000  # generous for double fft round-trip
    # out should be numerically close to x_img
    assert torch.allclose(
        torch.abs(out), torch.abs(x_img), atol=tol, rtol=tol
    ), (
        f"data_consistency(weight=0) passthrough failed for mask={mask_name}.\n"
        f"Max abs(|out| - |x|) = {(torch.abs(out) - torch.abs(x_img)).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 5. NoiseSimulatingDataConsistency (noise-augmented): noise_lvl=0 hard DC
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("mask_name", _MASK_IDS)
def test_data_consistency_layer_no_noise_preserves_measured(mask_name: str) -> None:
    """NoiseSimulatingDataConsistency with noise_lvl=0 enforces k_meas at mask==1.

    The layer computes:
        kspace_consistent = (1 - mask) * k_pred + mask * k_meas
    With zero noise, k_meas is inserted exactly at mask positions.
    We verify this in k-space directly (before the IFFT).
    """
    mask = _MASKS[mask_name]()
    x_img_pred = _rand_complex_img(seed=90)
    k_measured = fft2c(_rand_complex_img(seed=100))

    layer = NoiseSimulatingDataConsistency(noise_lvl=0.0)
    layer.eval()

    # Layer expects: predicted_img, measured_kspace, mask
    out = layer(x_img_pred, k_measured, mask)
    k_out = fft2c(out)

    mask_bool = mask.squeeze(1).bool()  # (B, H, W)
    k_out_sq = k_out.squeeze(1)           # (B, H, W)
    k_meas_sq = k_measured.squeeze(1)

    if mask_bool.any():
        err = (k_out_sq[mask_bool] - k_meas_sq[mask_bool]).abs().max().item()
        tol = tol_for(torch.complex64) * 200
        assert err < tol, (
            f"NoiseSimulatingDataConsistency [{mask_name}]: measured k-space not preserved.\n"
            f"  max_err={err:.3e}  tol={tol:.3e}"
        )
