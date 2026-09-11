"""Tests for TaskBasedDetectability (A-8.1 — CHO model-observer detectability metric).

The load-bearing test is ``test_orthogonal_to_psnr_via_noise_colour``: it proves
the metric reports something PSNR/MSE cannot (noise *colour*, not just energy).
The remaining tests guard the anti-facade properties — noise responsiveness, the
inserted signal being load-bearing, scale-invariance, and graceful degeneracy.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from spectramr.core.metrics.task_based_detectability import TaskBasedDetectability


def _smooth_anatomy(seed: int, h: int = 96, w: int = 96) -> torch.Tensor:
    """A smooth, positive 'anatomy' image [1,1,H,W] (low-freq blobs)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 1, h, w, generator=g)
    k = torch.ones(1, 1, 9, 9) / 81.0
    x = F.conv2d(x, k, padding=4)
    return x.abs() + 0.1


def _unit_rms(n: torch.Tensor) -> torch.Tensor:
    """Rescale each image to unit per-image RMS (so two noises share total energy)."""
    rms = n.flatten(1).pow(2).mean(dim=1).sqrt().clamp_min(1e-8)
    return n / rms.view(-1, 1, 1, 1)


def _lowpass(white: torch.Tensor, patch: int, sigma_frac: float = 0.12) -> torch.Tensor:
    """Convolve with an **L2-normalised** Gaussian blob -> a band-reshaped (low-pass)
    noise at (near-)equal energy.

    L2-normalisation (not sum-normalisation) keeps the filter's energy ~1, so the
    output is a genuine spectral *reshape* rather than energy-loss + amplification.
    Asymmetric reflect-padding makes the (even-kernel) output match the input size,
    avoiding the corner-crop boundary artefact.
    """
    coords = torch.arange(patch, dtype=torch.float32) - (patch - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sigma = sigma_frac * patch
    blob = torch.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma))
    blob = (blob / blob.pow(2).sum().sqrt()).reshape(1, 1, patch, patch)
    lo = (patch - 1) // 2
    hi = (patch - 1) - lo  # lo + hi == patch - 1 -> conv output == input size
    padded = F.pad(white, (lo, hi, lo, hi), mode="reflect")
    return F.conv2d(padded, blob, padding=0)


def test_higher_is_better_and_direction() -> None:
    m = TaskBasedDetectability()
    assert m.higher_is_better is True
    assert m.name == "task_based_detectability"


def test_finite_positive_on_normal_batch() -> None:
    tgt = torch.cat([_smooth_anatomy(s) for s in range(4)], dim=0)
    g = torch.Generator().manual_seed(99)
    pred = tgt + 0.05 * torch.randn(tgt.shape, generator=g)
    d = TaskBasedDetectability()(pred, tgt)
    assert math.isfinite(d) and d > 0.0


def test_responds_to_recon_noise_level() -> None:
    # CORE anti-facade: low recon noise -> lesion stays detectable -> high d';
    # high recon noise masks the lesion -> low d'. A constant/facade metric fails this.
    tgt = torch.cat([_smooth_anatomy(s) for s in range(4)], dim=0)
    g = torch.Generator().manual_seed(7)
    noise = torch.randn(tgt.shape, generator=g)
    m = TaskBasedDetectability()
    d_low = m(tgt + 0.01 * noise, tgt)
    d_high = m(tgt + 0.30 * noise, tgt)
    assert d_low > d_high


def test_inserted_signal_is_load_bearing() -> None:
    # The synthetic lesion must actually drive d' (not be ignored): a stronger
    # lesion contrast -> larger d' on the same recon noise.
    tgt = torch.cat([_smooth_anatomy(s) for s in range(4)], dim=0)
    g = torch.Generator().manual_seed(3)
    pred = tgt + 0.08 * torch.randn(tgt.shape, generator=g)
    d_weak = TaskBasedDetectability(lesion_contrast=0.05)(pred, tgt)
    d_strong = TaskBasedDetectability(lesion_contrast=0.20)(pred, tgt)
    assert d_strong > d_weak


def test_scale_invariance() -> None:
    # B-2.7 lesson: rescaling both pred & target by a constant must not move d'.
    tgt = torch.cat([_smooth_anatomy(s) for s in range(4)], dim=0)
    g = torch.Generator().manual_seed(11)
    pred = tgt + 0.07 * torch.randn(tgt.shape, generator=g)
    m = TaskBasedDetectability()
    d1 = m(pred, tgt)
    d2 = m(pred * 5.0, tgt * 5.0)
    assert abs(d1 - d2) / max(d1, 1e-6) < 0.05


def test_sensitive_to_noise_colour_at_fixed_psnr() -> None:
    # GUARD 1 (colour-sensitivity at fixed PSNR). Two recons with EQUAL residual
    # energy (equal MSE -> equal PSNR) but different noise *colour*: white noise vs
    # band-reshaped (low-pass) noise. The band-reshaped noise overlaps the lesion
    # band and masks it -> strictly lower d'. PSNR is invariant to colour; this is not.
    tgt = torch.cat([_smooth_anatomy(s) for s in range(6)], dim=0)
    g = torch.Generator().manual_seed(123)
    patch = 32
    white = torch.randn(tgt.shape, generator=g)
    band = _lowpass(white, patch)  # L2-normalised blob -> reshape, not energy-loss

    amp = 0.1
    pred_white = tgt + amp * _unit_rms(white)
    pred_band = tgt + amp * _unit_rms(band)
    # Precondition (not the load-bearing assert): _unit_rms equalises energy, so MSE
    # and therefore PSNR match. The real guard is the d' ordering below.
    mse_white = (pred_white - tgt).pow(2).mean().item()
    mse_band = (pred_band - tgt).pow(2).mean().item()
    assert abs(mse_white - mse_band) / mse_white < 0.02  # equal MSE / PSNR precondition

    m = TaskBasedDetectability(patch=patch)
    assert m(pred_band, tgt) < m(pred_white, tgt)  # PSNR is blind to this; the metric is not


def test_task_specific_to_lesion_band() -> None:
    # GUARD 2 (task-specificity, the stronger claim). The lesion is a LOW-frequency
    # blob and the observer weights channels by the lesion's spectrum, so equal-energy
    # noise placed INSIDE the lesion band (low-pass) must mask it MORE (lower d') than
    # noise placed OUTSIDE it (high-pass). A merely structure-aware metric would not
    # separate these; a task-specific observer does.
    tgt = torch.cat([_smooth_anatomy(s) for s in range(6)], dim=0)
    g = torch.Generator().manual_seed(321)
    patch = 32
    white = torch.randn(tgt.shape, generator=g)
    low = _lowpass(white, patch)  # in lesion band
    high = white - low  # complementary (out-of-band) energy

    amp = 0.1
    pred_inband = tgt + amp * _unit_rms(low)
    pred_outband = tgt + amp * _unit_rms(high)
    mse_in = (pred_inband - tgt).pow(2).mean().item()
    mse_out = (pred_outband - tgt).pow(2).mean().item()
    assert abs(mse_in - mse_out) / mse_in < 0.05  # equal MSE / PSNR precondition

    m = TaskBasedDetectability(patch=patch)
    assert m(pred_inband, tgt) < m(pred_outband, tgt)  # lesion-band noise masks the task


def test_perfect_reconstruction_returns_nan() -> None:
    # Pitfall #20: a zero residual has no noise to estimate K_g from, so d' would
    # collapse to a ridge-determined constant independent of the data. The figure of
    # merit must be reported as undefined (nan), never a spurious positive value.
    tgt = torch.cat([_smooth_anatomy(s) for s in range(4)], dim=0)
    assert math.isnan(TaskBasedDetectability()(tgt.clone(), tgt))


def test_degenerate_inputs_return_nan() -> None:
    tgt = _smooth_anatomy(0)
    m = TaskBasedDetectability()
    assert math.isnan(m(tgt, None))  # no reference
    assert math.isnan(m(tgt, _smooth_anatomy(0, h=64, w=64)))  # shape mismatch
    tiny = torch.rand(2, 1, 16, 16)  # smaller than patch=32 -> no ensemble
    assert math.isnan(m(tiny, tiny.clone()))


def test_registered_with_direction() -> None:
    from spectramr.core.metrics.registry import MetricsRegistry

    reg = MetricsRegistry()
    assert "task_based_detectability" in reg.list_available()
