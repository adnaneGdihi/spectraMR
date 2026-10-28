"""Extended tests for integration.py (critical-partial module).

integration.py composes physics primitives into a OperatorProjectionDataConsistency
(with IForwardOperator) and CoilSensitivityEstimator. The existing canary in
test_physics_modules.py only asserts the module is non-None.

New coverage:
  - OperatorProjectionDataConsistency (integration variant) instantiates with an FFT operator
  - CoilSensitivityEstimator instantiates and estimate_rss works
  - create_operator factory function works for 'fft' type
  - Module imports are clean even when optional sigpy/torchkbnufft absent
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _kspace_complex_last(b: int, c: int, h: int, w: int, seed: int = 0):
    """Build [B, C, H, W, 2] real-last k-space (integration.py layout)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(b, c, h, w, 2, generator=g)
    return t


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_integration_module_imports():
    """integration.py imports without error (optional deps guarded by try/except)."""
    import spectramr.infrastructure.physics.integration as integ
    assert integ is not None


@pytest.mark.physics
def test_canary_create_operator_fft():
    """create_operator('fft') returns an IForwardOperator-compatible object."""
    from spectramr.infrastructure.physics.registry import create_operator
    op = create_operator("fft2d")
    assert hasattr(op, "forward")
    assert hasattr(op, "adjoint")


@pytest.mark.physics
def test_canary_coil_sensitivity_estimator_instantiates():
    """CoilSensitivityEstimator (integration variant) instantiates cleanly."""
    from spectramr.infrastructure.physics.integration import CoilSensitivityEstimator
    cse = CoilSensitivityEstimator(calib_size=24, thresh=0.02)
    assert cse is not None


@pytest.mark.physics
def test_canary_coil_sensitivity_estimator_rss():
    """CoilSensitivityEstimator.estimate_rss returns [B, C, H, W, 2] tensor."""
    from spectramr.infrastructure.physics.integration import CoilSensitivityEstimator
    cse = CoilSensitivityEstimator()
    b, c, h, w = 1, 4, 16, 16
    ks = _kspace_complex_last(b, c, h, w)
    smaps = cse.estimate_rss(ks)
    # estimate_rss calls view_as_real so result is [B, C, H, W, 2]
    assert smaps.shape == (b, c, h, w, 2)
    assert not torch.isnan(smaps).any()


# ---------------------------------------------------------------------------
# OperatorProjectionDataConsistency (integration variant) — instantiation only
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_integration_dc_layer_instantiates():
    """integration.OperatorProjectionDataConsistency can be constructed with an FFT operator."""
    from spectramr.infrastructure.physics.integration import OperatorProjectionDataConsistency
    from spectramr.infrastructure.physics.registry import create_operator
    op = create_operator("fft2d")
    dc = OperatorProjectionDataConsistency(operator=op, lambda_dc=1.0)
    assert dc is not None


# ---------------------------------------------------------------------------
# FFT operator forward / adjoint
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_fft_operator_forward_shape():
    """FFT2DOperator.forward([B, C, H, W] complex) returns same shape."""
    from spectramr.infrastructure.physics.implementations import FFT2DOperator
    op = FFT2DOperator(normalized=True, center=True)
    b, c, h, w = 1, 1, 16, 16
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, c, h, w, generator=g) + 1j * torch.randn(b, c, h, w, generator=g)
    k = op.forward(x)
    assert k.shape == (b, c, h, w)
    assert torch.is_complex(k)


@pytest.mark.physics
def test_fft_operator_round_trip():
    """FFT2DOperator.forward followed by adjoint recovers original."""
    from spectramr.infrastructure.physics.implementations import FFT2DOperator
    op = FFT2DOperator(normalized=True, center=True)
    b, c, h, w = 1, 1, 16, 16
    g = torch.Generator().manual_seed(1)
    x = torch.randn(b, c, h, w, generator=g) + 1j * torch.randn(b, c, h, w, generator=g)
    k = op.forward(x)
    x_rec = op.adjoint(k)
    assert torch.allclose(x, x_rec, atol=1e-5)


# ---------------------------------------------------------------------------
# CoilSensitivityEstimator — forward method (auto)
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_coil_sensitivity_estimator_forward_auto():
    """CoilSensitivityEstimator.forward('auto') runs without error."""
    from spectramr.infrastructure.physics.integration import CoilSensitivityEstimator
    cse = CoilSensitivityEstimator()
    b, c, h, w = 1, 4, 16, 16
    ks = _kspace_complex_last(b, c, h, w)
    smaps = cse.forward(ks, method="auto")
    assert not torch.isnan(smaps).any()


@pytest.mark.physics
def test_coil_sensitivity_estimator_forward_rss():
    """CoilSensitivityEstimator.forward('rss') runs without error."""
    from spectramr.infrastructure.physics.integration import CoilSensitivityEstimator
    cse = CoilSensitivityEstimator()
    b, c, h, w = 1, 4, 16, 16
    ks = _kspace_complex_last(b, c, h, w)
    smaps = cse.forward(ks, method="rss")
    assert not torch.isnan(smaps).any()
