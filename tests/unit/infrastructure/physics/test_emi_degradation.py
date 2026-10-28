"""EMI degradation (D32): identity at theta=0, determinism, the stripe signature, registration."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    apply_degradation,
    is_identity_at_theta_zero,
)
from spectramr.infrastructure.physics.emi_degradation import degrade_emi
from spectramr.infrastructure.physics.fft_ops import fft2c


def _disc(n: int = 64) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    return (xx**2 + yy**2 < 0.4).float().view(1, 1, n, n)


def test_identity_at_theta_zero() -> None:
    x = _disc()
    assert torch.allclose(degrade_emi(x, 0.0, seed=1).real, x, atol=1e-5)


def test_deterministic_for_a_seed_and_distinct_across_seeds() -> None:
    x = _disc()
    assert torch.equal(degrade_emi(x, 0.7, seed=4), degrade_emi(x, 0.7, seed=4))
    assert not torch.allclose(degrade_emi(x, 0.7, seed=4), degrade_emi(x, 0.7, seed=5))


def test_residual_grows_monotonically_with_theta() -> None:
    x = _disc()
    res = [(degrade_emi(x, t, seed=2) - x).abs().norm().item() for t in (0.0, 0.3, 0.6, 1.0)]
    assert res == sorted(res) and res[-1] > 0.0


def test_theta_does_not_move_the_tone_positions() -> None:
    """The residual at 2*theta is exactly twice the residual at theta: only amplitude moves."""
    x = _disc()
    r1 = fft2c(degrade_emi(x, 0.4, seed=3) - x)
    r2 = fft2c(degrade_emi(x, 0.8, seed=3) - x)
    assert torch.allclose(r2, 2 * r1, atol=1e-4)


def test_residual_is_a_stripe_along_the_phase_encode_axis() -> None:
    """Tones are narrowband in x and independent across lines: the energy sits in a few columns."""
    x = _disc(64)
    r = (degrade_emi(x, 1.0, seed=0, n_tones=1) - x).abs().squeeze()
    col = r.pow(2).sum(dim=0)
    top5 = col.topk(5).values.sum()
    assert (top5 / col.sum()).item() > 0.8


@pytest.mark.parametrize("theta", [0.25, 1.0])
def test_realised_interference_rms_equals_the_declared_ratio(theta: float) -> None:
    x = _disc()
    k = fft2c(x.to(torch.complex64))
    added = fft2c(degrade_emi(x, theta, seed=7)) - k
    declared = DEGRADATION_REGISTRY["emi"].severity.params[0].value_at(theta)
    ratio = added.abs().pow(2).mean().sqrt() / k.abs().pow(2).mean().sqrt()
    assert ratio.item() == pytest.approx(
        declared, rel=0.02
    )  # tone cross terms cancel only on average


def test_n_tones_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_tones"):
        degrade_emi(_disc(), 0.5, n_tones=0)


def test_registered_with_declared_severity_and_identity() -> None:
    spec = DEGRADATION_REGISTRY["emi"]
    assert (spec.code, spec.family) == ("D32", "noise")
    assert spec.severity.params[0].at_theta_min == 0.0
    assert is_identity_at_theta_zero("emi")
    out = apply_degradation("emi", _disc(), 0.5, seed=0)
    assert out.shape == (1, 1, 64, 64) and out.is_complex()
