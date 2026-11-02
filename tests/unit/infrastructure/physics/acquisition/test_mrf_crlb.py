"""The IR-FISP Jacobian and relative CRLB that drive the MRF §4 pulse design."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.acquisition.mrf_crlb import (
    expected_relative_crlb,
    ir_fisp_signal_and_jacobian,
    log_grid_prior,
    relative_crlb,
    sample_log_uniform_prior,
)
from spectramr.infrastructure.physics.fisher_efficient_fingerprinting import (
    cramer_rao_lower_bound,
)


def _pulse(n: int = 6) -> torch.Tensor:
    alpha = torch.linspace(0.3, 1.2, n, dtype=torch.float64)
    tr = torch.linspace(9.0, 14.0, n, dtype=torch.float64)
    return torch.stack([alpha, tr], dim=-1)


def _spoiled_reference(pulse: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Closed-form fully spoiled IR-GRE signal, the ``n_states=0`` limit of the EPG."""
    mz, out = -torch.ones_like(t1), []
    for alpha, tr in pulse:
        out.append(mz * alpha.sin() * torch.exp(-2.0 / t2))
        mz = 1.0 + (mz * alpha.cos() - 1.0) * torch.exp(-tr / t1)
    return torch.stack(out, dim=1)


def test_analytic_jacobian_matches_autograd() -> None:
    pulse = _pulse(12)
    t1 = torch.tensor([400.0, 1500.0], dtype=torch.float64)
    t2 = torch.tensor([40.0, 120.0], dtype=torch.float64)
    sig, jac = ir_fisp_signal_and_jacobian(pulse, t1, t2, te_ms=2.0, n_states=8)
    assert torch.equal(jac[..., 2], sig)  # d s / d log M0 at unit M0
    for p in range(t1.shape[0]):

        def signal_of(log_theta: torch.Tensor, p: int = p) -> torch.Tensor:
            t1p, t2p = log_theta.exp().unbind()
            out, _ = ir_fisp_signal_and_jacobian(pulse, t1p[None], t2p[None], te_ms=2.0, n_states=8)
            return out[0]

        log_theta = torch.stack([t1[p], t2[p]]).log()
        expected = torch.autograd.functional.jacobian(signal_of, log_theta)
        assert torch.allclose(jac[p, :, :2], expected, atol=1e-10)


def test_zero_states_is_the_spoiled_limit() -> None:
    pulse = _pulse()
    t1 = torch.tensor([400.0, 1500.0], dtype=torch.float64)
    t2 = torch.tensor([40.0, 120.0], dtype=torch.float64)
    sig, _ = ir_fisp_signal_and_jacobian(pulse, t1, t2, te_ms=2.0, n_states=0)
    assert torch.allclose(sig, _spoiled_reference(pulse, t1, t2), atol=1e-12)


def test_truncated_graph_matches_the_full_graph() -> None:
    pulse = _pulse(80)
    t1, t2 = (x.double() for x in log_grid_prior(9, (100.0, 3000.0), (20.0, 500.0)))
    _, full = ir_fisp_signal_and_jacobian(pulse, t1, t2, te_ms=2.0, n_states=80)
    _, cut = ir_fisp_signal_and_jacobian(pulse, t1, t2, te_ms=2.0)
    assert torch.allclose(relative_crlb(cut), relative_crlb(full), rtol=1e-3)


def test_t2_sensitivity_is_not_collinear_with_the_signal() -> None:
    # A constant e^{-TE/T2} factor makes d s/d log T2 proportional to s, so a free M0 would
    # leave T2 unidentifiable; the EPG pathways must break that, on the mu = 0 schedule too.
    pulse = torch.stack(
        [torch.full((200,), math.pi / 3, dtype=torch.float64), torch.full((200,), 10.0).double()],
        dim=-1,
    )
    t1, t2 = (x.double() for x in log_grid_prior(16, (100.0, 3000.0), (20.0, 500.0)))
    sig, jac = ir_fisp_signal_and_jacobian(pulse, t1, t2, te_ms=2.0)
    basis, _ = torch.linalg.qr(torch.stack([sig, jac[..., 0]], dim=-1))
    d_t2 = jac[..., 1]
    resid = d_t2 - (basis @ (basis.transpose(-1, -2) @ d_t2.unsqueeze(-1))).squeeze(-1)
    assert torch.all(resid.norm(dim=-1) > 1e-2 * d_t2.norm(dim=-1))
    _, spoiled = ir_fisp_signal_and_jacobian(pulse, t1, t2, te_ms=2.0, n_states=0)
    assert relative_crlb(jac).max() < 1e3 < relative_crlb(spoiled).min()


def test_relative_crlb_matches_the_general_certificate() -> None:
    pulse = _pulse(12)
    t1 = torch.tensor([800.0], dtype=torch.float64)
    t2 = torch.tensor([80.0], dtype=torch.float64)
    _, jac = ir_fisp_signal_and_jacobian(pulse, t1, t2, te_ms=2.0)
    general = cramer_rao_lower_bound(jac[0]).double()[:2].sum()
    assert relative_crlb(jac)[0].item() == pytest.approx(general.item(), rel=1e-4)


def test_objective_carries_gradient_to_the_pulse() -> None:
    pulse = _pulse().float().requires_grad_(True)
    t1, t2 = log_grid_prior(9, (100.0, 3000.0), (20.0, 500.0))
    obj = expected_relative_crlb(pulse, t1, t2, te_ms=2.0)
    (grad,) = torch.autograd.grad(obj, pulse)
    assert torch.isfinite(obj) and torch.any(grad != 0)


def test_priors_stay_inside_their_ranges() -> None:
    gen = torch.Generator().manual_seed(0)
    t1, t2 = sample_log_uniform_prior(256, (100.0, 3000.0), (20.0, 500.0), generator=gen)
    assert t1.min() >= 100.0 and t1.max() <= 3000.0
    assert t2.min() >= 20.0 and t2.max() <= 500.0
    g1, g2 = log_grid_prior(64, (100.0, 3000.0), (20.0, 500.0))
    assert g1.numel() == 64 and math.isclose(g1.max().item(), 3000.0, rel_tol=1e-5)
    assert g2.min().item() == pytest.approx(20.0)


@pytest.mark.parametrize("bounds", [(0.0, 1.0), (5.0, 2.0)])
def test_invalid_prior_range_raises(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="prior range"):
        log_grid_prior(4, bounds, (20.0, 500.0))


def test_malformed_pulse_raises() -> None:
    with pytest.raises(ValueError, match=r"\[T, 2\]"):
        ir_fisp_signal_and_jacobian(torch.zeros(4, 3), torch.ones(1), torch.ones(1), te_ms=2.0)
