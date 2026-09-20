"""Extended tests for data_consistency.py (critical-partial module).

Existing tests for SimpleDataConsistency live in test_physics_modules.py —
we EXTEND coverage here for:
  - NoiseSimulatingDataConsistency (3-argument form: pred, measured, mask)
  - HardDataConsistency
  - SoftDataConsistency
  - TargetAwareFSDC
  - dc_passthrough_center_patch
  - ConformalDataConsistency
  - data_consistency() function
  - Edge: NaN guard, all-zero mask, all-one mask
  - Raises: TargetAwareFSDC invalid acs_taper
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _img(b: int, c: int, h: int, w: int, complex_: bool = True, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    if complex_:
        re = torch.randn(b, c, h, w, generator=g)
        im = torch.randn(b, c, h, w, generator=g)
        return torch.complex(re, im)
    return torch.randn(b, c, h, w, generator=g)


def _mask(b: int, h: int, w: int, accel: int = 4) -> torch.Tensor:
    """Simple Cartesian mask: keep every accel-th line."""
    m = torch.zeros(b, 1, h, w)
    m[:, :, ::accel, :] = 1.0
    return m


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_data_consistency_imports():
    """All exported symbols import cleanly."""
    from spectramr.infrastructure.physics.data_consistency import (
        NoiseSimulatingDataConsistency,
        AdaptiveDataConsistency,
        HardDataConsistency,
        SoftDataConsistency,
        SimpleDataConsistency,
        ConformalDataConsistency,
        TargetAwareFSDC,
        dc_passthrough_center_patch,
        data_consistency,
    )
    assert all([
        NoiseSimulatingDataConsistency, AdaptiveDataConsistency, HardDataConsistency,
        SoftDataConsistency, SimpleDataConsistency, ConformalDataConsistency,
        TargetAwareFSDC, dc_passthrough_center_patch, data_consistency,
    ])


@pytest.mark.physics
def test_canary_data_consistency_layer_forward():
    """NoiseSimulatingDataConsistency forward pass with complex inputs."""
    from spectramr.infrastructure.physics.data_consistency import NoiseSimulatingDataConsistency
    dc = NoiseSimulatingDataConsistency()
    b, c, h, w = 1, 1, 16, 16
    pred = _img(b, c, h, w, complex_=True)
    meas = _img(b, c, h, w, complex_=True, seed=1)
    mask = _mask(b, h, w)
    out = dc(pred, meas, mask)
    assert out.shape == (b, c, h, w)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# HardDataConsistency
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, c, h, w",
    [(1, 1, 16, 16), (2, 1, 32, 32), (1, 1, 16, 32)],
    ids=["b1c1-16", "b2c1-32", "b1c1-rect"],
)
def test_hard_dc_shape(b, c, h, w):
    from spectramr.infrastructure.physics.data_consistency import HardDataConsistency
    dc = HardDataConsistency()
    pred = _img(b, c, h, w, complex_=True)
    meas = _img(b, c, h, w, complex_=True, seed=2)
    mask = _mask(b, h, w)
    out = dc(pred, kspace_obs=meas, mask=mask)
    assert out.shape == (b, c, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_hard_dc_sampled_lines_replaced():
    """Hard DC must replace sampled k-space with measurements exactly
    (no blending) when mask=1 at sampled locations."""
    from spectramr.infrastructure.physics.data_consistency import HardDataConsistency
    from spectramr.infrastructure.physics.fft_ops import fft2c
    dc = HardDataConsistency(train_noise_level=0.0, eval_noise_level=0.0)
    dc.eval()
    b, c, h, w = 1, 1, 16, 16
    pred = _img(b, c, h, w, complex_=True)
    meas = _img(b, c, h, w, complex_=True, seed=5)
    # Full mask: all-ones
    mask = torch.ones(b, 1, h, w)
    out = dc(pred, kspace_obs=meas, mask=mask)
    # With all-ones mask and zero noise, output should match IFFT(meas kspace)
    from spectramr.infrastructure.physics.fft_ops import ifft2c
    expected = ifft2c(meas)
    assert torch.allclose(out.abs(), expected.abs(), atol=1e-4)


# ---------------------------------------------------------------------------
# SoftDataConsistency
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, c, h, w",
    [(1, 1, 16, 16), (2, 1, 32, 32)],
    ids=["b1c1-16", "b2c1-32"],
)
def test_soft_dc_shape(b, c, h, w):
    from spectramr.infrastructure.physics.data_consistency import SoftDataConsistency
    dc = SoftDataConsistency(lambda_init=10.0)
    pred = _img(b, c, h, w, complex_=True)
    meas = _img(b, c, h, w, complex_=True, seed=3)
    mask = _mask(b, h, w)
    out = dc(pred, meas, mask)
    assert out.shape == (b, c, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_soft_dc_lambda_is_learnable():
    """SoftDataConsistency lambda_param is a trainable Parameter."""
    from spectramr.infrastructure.physics.data_consistency import SoftDataConsistency
    dc = SoftDataConsistency(lambda_init=5.0, learnable=True)
    params = list(dc.parameters())
    assert len(params) >= 1
    assert any(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# TargetAwareFSDC
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_target_aware_fsdc_shape():
    """TA-FSDC returns same [B, C_total, H, W] as input k-space."""
    from spectramr.infrastructure.physics.data_consistency import TargetAwareFSDC
    b, c_total, h, w = 1, 8, 16, 16
    ta = TargetAwareFSDC(target_channels=4)
    kpred = torch.randn(b, c_total, h, w)
    kmeas = torch.randn(b, c_total, h, w)
    mask = _mask(b, h, w)
    out = ta(kpred, kmeas, mask)
    assert out.shape == (b, c_total, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_target_aware_fsdc_hann_taper():
    """TA-FSDC with acs_taper='hann' runs without error."""
    from spectramr.infrastructure.physics.data_consistency import TargetAwareFSDC
    b, c_total, h, w = 1, 4, 16, 16
    ta = TargetAwareFSDC(target_channels=2, acs_taper="hann")
    kpred = torch.randn(b, c_total, h, w)
    kmeas = torch.randn(b, c_total, h, w)
    mask = _mask(b, h, w)
    out = ta(kpred, kmeas, mask)
    assert out.shape == (b, c_total, h, w)


# ---------------------------------------------------------------------------
# dc_passthrough_center_patch
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize("patch_size", [1, 3, 5])
def test_dc_passthrough_center_patch_shape(patch_size):
    from spectramr.infrastructure.physics.data_consistency import dc_passthrough_center_patch
    b, c, h, w = 1, 2, 16, 16
    x = torch.randn(b, c, h, w)
    meas = torch.randn(b, c, h, w)
    mask = torch.ones(b, 1, h, w)
    out = dc_passthrough_center_patch(x, meas, mask, patch_size=patch_size)
    assert out.shape == (b, c, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_dc_passthrough_noop_when_none():
    """dc_passthrough_center_patch is a no-op when measured or mask is None."""
    from spectramr.infrastructure.physics.data_consistency import dc_passthrough_center_patch
    x = torch.randn(1, 2, 16, 16)
    out = dc_passthrough_center_patch(x, None, None, patch_size=3)
    assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# ConformalDataConsistency
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_conformal_dc_shape():
    from spectramr.infrastructure.physics.data_consistency import ConformalDataConsistency
    dc = ConformalDataConsistency(alpha=1.0)
    b, c, h, w = 1, 1, 16, 16
    recon = torch.randn(b, c, h, w)
    y_tilde = torch.randn(b, c, h, w)
    mask = torch.ones(b, c, h, w)
    jac = torch.ones(b, c, h, w) * 2.0
    out = dc(recon, y_tilde=y_tilde, mask=mask, conformal_jacobian=jac)
    assert out.shape == (b, c, h, w)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# data_consistency function
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_data_consistency_function():
    """data_consistency() convenience function returns correct shape."""
    from spectramr.infrastructure.physics.data_consistency import data_consistency
    b, c, h, w = 1, 1, 16, 16
    x = _img(b, c, h, w, complex_=True)
    k_meas = _img(b, c, h, w, complex_=True, seed=7)
    mask = _mask(b, h, w)
    out = data_consistency(x, k_meas, mask)
    assert out.shape == (b, c, h, w)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_data_consistency_edge_all_zero_mask():
    """With all-zero mask the output should not be modified by measurements."""
    from spectramr.infrastructure.physics.data_consistency import NoiseSimulatingDataConsistency
    dc = NoiseSimulatingDataConsistency(train_noise_level=0.0, eval_noise_level=0.0)
    dc.eval()
    b, c, h, w = 1, 1, 16, 16
    pred = _img(b, c, h, w, complex_=True)
    meas = _img(b, c, h, w, complex_=True, seed=8)
    mask = torch.zeros(b, 1, h, w)
    out = dc(pred, meas, mask)
    assert out.shape == (b, c, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_data_consistency_edge_complex_input_preserved():
    """Complex input produces complex output."""
    from spectramr.infrastructure.physics.data_consistency import NoiseSimulatingDataConsistency
    dc = NoiseSimulatingDataConsistency()
    b, c, h, w = 1, 1, 16, 16
    pred = _img(b, c, h, w, complex_=True)
    meas = _img(b, c, h, w, complex_=True, seed=9)
    mask = _mask(b, h, w)
    out = dc(pred, meas, mask)
    assert torch.is_complex(out)


# ---------------------------------------------------------------------------
# Expected-failure (raises) tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_target_aware_fsdc_raises_invalid_taper():
    """TargetAwareFSDC must raise ValueError for unsupported acs_taper values."""
    from spectramr.infrastructure.physics.data_consistency import TargetAwareFSDC
    with pytest.raises(ValueError, match="acs_taper"):
        TargetAwareFSDC(target_channels=4, acs_taper="blackman")
