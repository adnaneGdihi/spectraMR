"""Ring power metrics on per-coil k-space: identity, analytic low-pass, PE exclusion, noise floor."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.ring_power import RING_METRICS, noise_floor, ring_statistics

H = W = 64
_KY = torch.arange(H) - H // 2
_KX = torch.arange(W) - W // 2
_RHO = torch.sqrt((_KY[:, None] ** 2 + _KX[None, :] ** 2).float()) / (min(H, W) / 2)


def _kspace(batch: int = 2, coils: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(batch, coils, H, W, generator=g), torch.randn(batch, coils, H, W, generator=g)
    )


def _value(name: str, pred: torch.Tensor, target: torch.Tensor) -> float:
    return RING_METRICS[name]()(pred, target, context=MetricContext(pred_kspace=pred))


def test_identity_gives_unit_ratio_zero_error_and_finite_snr() -> None:
    k = _kspace()
    for ring in ("mid1", "mid2", "high"):
        assert _value(f"ring_power_ratio_{ring}", k, k) == pytest.approx(1.0, rel=1e-6)
        assert _value(f"ring_nrmse_{ring}", k, k) == pytest.approx(0.0, abs=1e-6)
        assert math.isfinite(_value(f"ring_target_snr_{ring}", k, k))


def test_low_pass_keeps_mid1_and_removes_mid2_and_high_exactly() -> None:
    k = _kspace()
    low = k * (_RHO < 0.25)
    assert _value("ring_power_ratio_mid1", low, k) == pytest.approx(1.0, rel=1e-6)
    assert _value("ring_power_ratio_mid2", low, k) == pytest.approx(0.0, abs=1e-9)
    assert _value("ring_power_ratio_high", low, k) == pytest.approx(0.0, abs=1e-9)
    assert _value("ring_nrmse_mid2", low, k) == pytest.approx(1.0, rel=1e-6)


def test_uniform_half_amplitude_gives_quarter_power_and_half_nrmse() -> None:
    k = _kspace()
    half = 0.5 * k
    for ring in ("mid1", "mid2", "high"):
        assert _value(f"ring_power_ratio_{ring}", half, k) == pytest.approx(0.25, rel=1e-5)
        assert _value(f"ring_nrmse_{ring}", half, k) == pytest.approx(0.5, rel=1e-5)


def test_never_acquired_phase_encode_columns_are_excluded() -> None:
    target = _kspace()
    target[..., 40:] = 0  # never-acquired PE columns: exact zeros in every coil
    clean = 0.5 * target
    polluted = clean.clone()
    polluted[..., 40:] = 1e3  # a hallucination on the unacquired columns must not count
    for name in ("ring_nrmse_mid1", "ring_power_ratio_mid2", "ring_nrmse_high"):
        assert _value(name, polluted, target) == pytest.approx(_value(name, clean, target))


def test_noise_floor_recovers_white_noise_variance() -> None:
    torch.manual_seed(0)
    sigma2 = 2.0
    shape = (4, 8, 256, 256)
    scale = math.sqrt(sigma2 / 2)
    noise = torch.complex(torch.randn(shape) * scale, torch.randn(shape) * scale)
    estimate = noise_floor(noise)
    assert estimate.shape == (4,)
    assert torch.allclose(estimate, torch.full_like(estimate, sigma2), rtol=0.05)


def test_batched_statistics_equal_the_per_sample_sum() -> None:
    target = _kspace(batch=3)
    pred = target * (_RHO < 0.3)
    batched = ring_statistics(pred, target)
    parts = [ring_statistics(pred[i : i + 1], target[i : i + 1]) for i in range(3)]
    for key in ("error", "target", "prediction", "count"):
        assert torch.allclose(batched[key], sum(p[key] for p in parts), rtol=1e-5)


def test_directions_and_names_are_declared() -> None:
    assert RING_METRICS["ring_nrmse_mid2"]().higher_is_better is False
    assert RING_METRICS["ring_power_ratio_mid2"]().higher_is_better is True
    assert RING_METRICS["ring_target_snr_high"]().name == "ring_target_snr_high"
    assert RING_METRICS["ring_nrmse_mid2"]().name == "ring_nrmse_mid2"


def test_without_the_k_space_context_the_value_is_nan() -> None:
    """PLANTED: a context-free image pass must not score rings on images."""
    k = _kspace()
    assert math.isnan(RING_METRICS["ring_nrmse_mid1"]()(k, k))
    assert math.isnan(RING_METRICS["ring_nrmse_mid1"]()(k[:, :2], k))
    assert math.isnan(RING_METRICS["ring_nrmse_mid1"]()(k, None))


def test_the_nine_metrics_share_one_pass_and_an_in_place_edit_invalidates_it(monkeypatch) -> None:
    import spectramr.core.metrics.ring_power as rp

    calls: list[int] = []
    real = rp.ring_statistics

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(rp, "ring_statistics", _spy)
    pred, target = _kspace(seed=1), _kspace(seed=2)
    first = {name: _value(name, pred, target) for name in RING_METRICS}
    assert len(calls) == 1
    pred.mul_(0.5)
    second = {name: _value(name, pred, target) for name in RING_METRICS}
    assert len(calls) == 2
    assert second["ring_power_ratio_mid1"] == pytest.approx(first["ring_power_ratio_mid1"] * 0.25)
    # A different pair is never served the cached statistics.
    other = _kspace(seed=3)
    assert _value("ring_nrmse_mid1", other, target) != second["ring_nrmse_mid1"]
    assert len(calls) == 3
