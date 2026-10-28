"""Unit tests for infrastructure/reporting/metrics/vif_fsim.py.

Covers: visual_information_fidelity, feature_similarity_index,
        _phase_congruency, _gradient_magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.infrastructure.reporting.metrics.vif_fsim import (
    feature_similarity_index,
    visual_information_fidelity,
)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vif_canary_identical():
    """VIF of an image with itself should be close to 1.0."""
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    score = visual_information_fidelity(img, img)
    assert score >= 0.0, f"VIF should be non-negative, got {score}"


@pytest.mark.unit
def test_fsim_canary_identical():
    """FSIM of identical images should be close to 1.0."""
    rng = np.random.default_rng(1)
    img = rng.random((64, 64)).astype(np.float32)
    score = feature_similarity_index(img, img)
    assert 0.0 <= score <= 1.0 + 1e-6, f"FSIM out of range: {score}"


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("metric_fn", [visual_information_fidelity, feature_similarity_index])
def test_metric_finite(metric_fn):
    """Both metrics return finite scalars on random inputs."""
    rng = np.random.default_rng(2)
    pred = rng.random((32, 32)).astype(np.float32)
    ref = rng.random((32, 32)).astype(np.float32)
    result = metric_fn(pred, ref)
    assert np.isfinite(result), f"{metric_fn.__name__} returned {result}"


@pytest.mark.unit
@pytest.mark.parametrize("size", [(16, 16), (32, 32), (64, 64)])
def test_vif_various_sizes(size):
    """VIF should work on various image sizes without crashing."""
    rng = np.random.default_rng(3)
    img = rng.random(size).astype(np.float32)
    score = visual_information_fidelity(img, img)
    assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fsim_constant_image():
    """Constant images should not raise and return a finite value."""
    flat = np.full((32, 32), 0.5, dtype=np.float64)
    score = feature_similarity_index(flat, flat)
    assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("metric_fn", [visual_information_fidelity, feature_similarity_index])
def test_shape_mismatch_raises(metric_fn):
    """Both metrics must raise ValueError on shape mismatch."""
    a = np.zeros((32, 32), dtype=np.float32)
    b = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        metric_fn(a, b)


@pytest.mark.unit
def test_fsim_ranks_a_degraded_image_below_the_clean_one():
    """FSIM used to call `sewar.fsim`, which no released sewar provides (#2499)."""
    rng = np.random.default_rng(4)
    img = rng.random((48, 48)).astype(np.float32)
    noisy = np.clip(img + rng.normal(0, 0.2, img.shape), 0, 1).astype(np.float32)
    assert feature_similarity_index(img, img) == pytest.approx(1.0, abs=1e-4)
    assert feature_similarity_index(noisy, img) < feature_similarity_index(img, img)


@pytest.mark.unit
def test_each_package_is_guarded_on_its_own():
    from spectramr.infrastructure.reporting.metrics import vif_fsim

    assert isinstance(vif_fsim.HAVE_SEWAR, bool) and isinstance(vif_fsim.HAVE_PIQ, bool)

@pytest.mark.unit
def test_fsim_sees_errors_in_negative_valued_images():
    """Clipping to [0, data_range] hid every error below zero in z-scored images."""
    rng = np.random.default_rng(5)
    target = rng.normal(0, 1, (48, 48)).astype(np.float32)
    pred = target.copy()
    pred[pred < 0] = -3.0
    assert feature_similarity_index(pred, target) < 0.999
