"""Wood rejection sampler of ``VonMisesFisher``: the polar marginal is right.

``E[w] = I_{d/2}(kappa) / I_{d/2-1}(kappa)`` is the vMF mean resultant length.
The previous acceptance test used ``b`` for ``x0`` and the wrong sign on the
``log(1 - (1-b) eps)`` term; it gave ``E[w] = 0.586`` at ``d=3, kappa=1``
against the true ``0.313``.
"""

from __future__ import annotations

import pytest
import scipy.special as sp
import torch

from spectramr.models.blocks.vae.von_mises_fisher import VonMisesFisher

pytestmark = pytest.mark.unit


def _w(d: int, kappa: float, n: int = 20000) -> torch.Tensor:
    mu = torch.zeros(n, d)
    mu[:, 0] = 1.0
    dist = VonMisesFisher(mu, torch.full((n,), kappa))
    return dist._sample_w(dist.kappa)


@pytest.mark.parametrize(
    ("d", "kappa"), [(3, 1.0), (3, 10.0), (16, 10.0), (16, 100.0), (64, 100.0), (4, 1.0e4)]
)
def test_polar_mean_matches_bessel_ratio(d: int, kappa: float) -> None:
    torch.manual_seed(0)
    expected = sp.ive(d / 2, kappa) / sp.ive(d / 2 - 1, kappa)
    assert _w(d, kappa).mean().item() == pytest.approx(expected, abs=6e-3)


def test_large_kappa_is_not_collapsed_to_the_mean() -> None:
    """float32 cancellation in ``b`` used to return w == 1 (zero noise)."""
    torch.manual_seed(0)
    w = _w(4, 1.0e4)
    gap = (1.0 - w).mean().item()
    assert gap == pytest.approx(3.0 / (2.0e4), rel=0.1)


def test_near_zero_kappa_is_finite_and_near_uniform() -> None:
    torch.manual_seed(0)
    w = _w(8, 1.0e-4)
    assert torch.isfinite(w).all()
    assert abs(w.mean().item()) < 0.02


def test_rsample_on_sphere_and_gradient_reaches_kappa() -> None:
    torch.manual_seed(0)
    kappa = torch.full((32,), 5.0, requires_grad=True)
    mu = torch.nn.functional.normalize(torch.randn(32, 6), dim=-1)
    z = VonMisesFisher(mu, kappa).rsample()
    assert torch.allclose(z.norm(dim=-1), torch.ones(32), atol=1e-5)
    (z * mu).sum().backward()
    assert kappa.grad is not None and torch.isfinite(kappa.grad).all()
