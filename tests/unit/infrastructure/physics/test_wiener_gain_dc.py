"""Unit tests for :mod:`spectramr.infrastructure.physics.wiener_gain_dc`.

Each test pins one limit the layer's docstring states: hard DC as the noise
vanishes, prediction-only off the support, a planted gain recovered, and the
blend computed on linear rather than log1p-compressed k-space.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.data.transforms.normalization import compress_kspace_log
from spectramr.infrastructure.physics.wiener_gain_dc import WienerGainDataConsistency

B, C, H, W = 2, 2, 32, 32


def _stack(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=2).flatten(1, 2)


def _paired(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0::2], x[:, 1::2])


def _case(seed: int = 0, noise: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, H, W, dtype=torch.complex64, generator=g) * 10.0
    y = x + noise * torch.randn(B, C, H, W, dtype=torch.complex64, generator=g)
    mask = (torch.rand(B, 1, H, W, generator=g) < 0.5).float()
    mask[..., H // 2 - 2 : H // 2 + 2, :] = 1.0  # a measured centre for the gain fit
    return x, y * mask, mask


def test_reduces_to_hard_dc_as_noise_vanishes():
    x, y, mask = _case()
    layer = WienerGainDataConsistency(log_scaled=False, learnable_beta=False)
    out = _paired(layer(_stack(x * 1.3), _stack(y), mask, is_kspace_domain=True, noise_sigma=1e-6))
    measured = mask.bool().expand_as(out)
    torch.testing.assert_close(out[measured], y[measured], rtol=1e-4, atol=1e-4)


def test_trust_weight_carries_no_gradient_into_the_prediction():
    """The signal-power numerator is detached, so w cannot steer k_pred."""
    layer = WienerGainDataConsistency(log_scaled=False, beta=0.5, learnable_beta=False)
    captured = {}
    original = layer._compute_lambda

    def spy(*args, **kwargs):
        lam = original(*args, **kwargs)
        captured["lam"] = lam
        return lam

    layer._compute_lambda = spy
    torch.manual_seed(0)
    y = torch.randn(1, 2, 32, 32)
    obs = torch.zeros(1, 1, 32, 32)
    obs[..., 8:24, :] = 1.0
    k = torch.randn(1, 2, 32, 32, requires_grad=True)
    layer(k, y, obs, is_kspace_domain=True)
    assert not captured["lam"].requires_grad


def test_unmeasured_bins_are_the_gain_scaled_prediction():
    x, y, mask = _case(noise=1.0)
    pred = x * 0.7
    layer = WienerGainDataConsistency(log_scaled=False)
    out = _paired(layer(_stack(pred), _stack(y), mask, is_kspace_domain=True))
    alpha = layer.fit_gain(pred, y, mask.bool())
    unmeasured = ~mask.bool().expand_as(out)
    torch.testing.assert_close(out[unmeasured], (pred * alpha)[unmeasured], rtol=1e-4, atol=1e-4)


def test_fit_gain_recovers_a_planted_gain():
    x, _, mask = _case()
    layer = WienerGainDataConsistency(log_scaled=False)
    alpha = layer.fit_gain(x / 1.6, x, mask.bool())
    torch.testing.assert_close(alpha.flatten(), torch.full((B,), 1.6), rtol=1e-4, atol=1e-4)


def test_gain_is_clamped_and_detached():
    x, _, mask = _case()
    layer = WienerGainDataConsistency(log_scaled=False, gain_bounds=(0.5, 2.0))
    pred = (x / 10.0).requires_grad_(True)
    alpha = layer.fit_gain(pred, x, mask.bool())
    assert torch.all(alpha == 2.0)
    assert not alpha.requires_grad


def test_blend_is_computed_on_linear_kspace():
    """Same data, compressed or not, must give the same LINEAR output."""
    x, y, mask = _case(noise=2.0)
    pred = x * 0.9
    lin = WienerGainDataConsistency(log_scaled=False, learnable_beta=False)
    logd = WienerGainDataConsistency(log_scaled=True, learnable_beta=False)
    out_lin = lin(_stack(pred), _stack(y), mask, is_kspace_domain=True)
    out_log = logd(
        compress_kspace_log(_stack(pred), channel_dim=1),
        compress_kspace_log(_stack(y), channel_dim=1),
        mask,
        is_kspace_domain=True,
    )
    torch.testing.assert_close(
        out_log, compress_kspace_log(out_lin, channel_dim=1), rtol=1e-3, atol=1e-4
    )


def test_noisy_bins_trust_the_prediction_more():
    """At low SNR the output moves toward the prediction, never past the measurement."""
    x, y, mask = _case(noise=20.0)
    layer = WienerGainDataConsistency(log_scaled=False, learnable_beta=False)
    out = _paired(layer(_stack(x), _stack(y), mask, is_kspace_domain=True))
    measured = mask.bool().expand_as(out)
    err_out = (out - x)[measured].abs().pow(2).mean()
    err_hard = (y - x)[measured].abs().pow(2).mean()
    assert err_out < err_hard


def test_beta_receives_a_gradient():
    x, y, mask = _case(noise=5.0)
    layer = WienerGainDataConsistency(log_scaled=False)
    out = layer(_stack(x), _stack(y), mask, is_kspace_domain=True)
    out.square().sum().backward()
    assert layer.beta.grad is not None and torch.isfinite(layer.beta.grad)


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"log_scaled": None}, TypeError),
        ({"log_scaled": False, "gain_radius": 0.0}, ValueError),
        ({"log_scaled": False, "gain_bounds": (1.5, 2.0)}, ValueError),
    ],
)
def test_refuses_undeclared_or_invalid_settings(kwargs, exc):
    with pytest.raises(exc):
        WienerGainDataConsistency(**kwargs)


def test_refuses_image_domain_input():
    x, y, mask = _case()
    layer = WienerGainDataConsistency(log_scaled=False)
    with pytest.raises(ValueError, match="k-space only"):
        layer(_stack(x), _stack(y), mask, is_kspace_domain=False)


def test_noise_variance_is_estimated_once_per_measurement(monkeypatch):
    x, y, mask = _case(noise=1.0)
    y_s = _stack(y)
    layer = WienerGainDataConsistency(log_scaled=False)
    calls = []
    real = layer.estimate_noise_variance
    monkeypatch.setattr(layer, "estimate_noise_variance", lambda *a: calls.append(1) or real(*a))
    first = layer(_stack(x * 0.9), y_s, mask, is_kspace_domain=True)
    second = layer(_stack(x * 0.9), y_s, mask, is_kspace_domain=True)
    assert len(calls) == 1
    assert torch.equal(first, second)
    # A new measurement, or the same one modified in place, is estimated afresh.
    layer(_stack(x * 0.9), y_s.clone(), mask, is_kspace_domain=True)
    y_s.mul_(2.0)
    layer(_stack(x * 0.9), y_s, mask, is_kspace_domain=True)
    assert len(calls) == 3


def test_cached_variance_matches_a_fresh_estimate():
    x, y, mask = _case(noise=1.0)
    y_s = _stack(y)
    warm = WienerGainDataConsistency(log_scaled=False)
    warm(_stack(x * 0.5), y_s, mask, is_kspace_domain=True)
    cached = warm(_stack(x * 0.9), y_s, mask, is_kspace_domain=True)
    fresh = WienerGainDataConsistency(log_scaled=False)(
        _stack(x * 0.9), y_s, mask, is_kspace_domain=True
    )
    assert torch.equal(cached, fresh)
