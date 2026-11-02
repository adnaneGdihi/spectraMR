"""Unit tests for acquisition-stage physics losses (B-4 / Bundle P7).

Covers:
* energy-conservation loss is ~0 for a perfectly conserved trajectory and
  positive for a drifting one,
* hardware-compliance loss is exactly 0 inside the feasible region and
  activates (positive) when G_max / S_max are exceeded,
* both losses are registered under domain="physics" (Phase-0.5 F4),
* gradient flow + no-NaN.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.acquisition_losses import (
    GradientHardwareComplianceLoss,
    HamiltonianEnergyConservationLoss,
)
from spectramr.models.losses.registry import LossRegistry, create_loss


def test_losses_registered_as_physics_domain():
    """Both losses must resolve and carry domain='physics'."""
    assert isinstance(
        create_loss("hamiltonian_energy_conservation"), HamiltonianEnergyConservationLoss
    )
    assert isinstance(
        create_loss("gradient_hardware_compliance", matrix_size=256),
        GradientHardwareComplianceLoss,
    )
    # Domain metadata (Phase-0.5 F4: 'trajectory' is invalid; must be 'physics').
    meta = LossRegistry._loss_domains  # type: ignore[attr-defined]
    assert meta["hamiltonian_energy_conservation"]["domain"] == "physics"
    assert meta["gradient_hardware_compliance"]["domain"] == "physics"


def test_energy_loss_zero_for_constant_energy():
    """Perfectly conserved energy => zero loss."""
    loss = HamiltonianEnergyConservationLoss()
    energy = torch.full((3, 50), 2.5)  # constant across time
    out = loss({"energy": energy})
    assert out.item() < 1e-12


def test_energy_loss_positive_for_drift():
    """Linearly drifting energy => positive loss."""
    loss = HamiltonianEnergyConservationLoss()
    t = torch.linspace(0, 1, 50)
    energy = (2.0 + 0.5 * t).unsqueeze(0)  # drifts from 2.0 to 2.5
    out = loss({"energy": energy})
    assert out.item() > 0.0


def test_energy_loss_accepts_bare_tensor():
    loss = HamiltonianEnergyConservationLoss()
    energy = torch.tensor([1.0, 1.0, 1.0])
    assert loss(energy).item() < 1e-12


def test_hardware_loss_zero_inside_feasible_region():
    """Small gradients/slew => exactly zero penalty."""
    loss = GradientHardwareComplianceLoss(
        g_max_mT_per_m=40.0, s_max_T_per_m_per_s=200.0, dt_seconds=4e-6, matrix_size=256
    )
    # Tiny normalised gradients map to tiny physical gradients => no violation.
    g = torch.zeros(2, 20, 2)
    slew = torch.zeros(2, 20, 2)
    out = loss({"g": g, "slew": slew})
    assert out.item() == 0.0


def test_hardware_loss_activates_when_exceeded():
    """Large normalised gradients exceed G_max => positive penalty."""
    loss = GradientHardwareComplianceLoss(
        g_max_mT_per_m=40.0, s_max_T_per_m_per_s=200.0, dt_seconds=4e-6,
        gamma_hz_per_t=42.577478e6, fov_m=0.22, matrix_size=1,
    )
    # Choose g so that G_phys = g/(gamma*FOV)*1e3 >> 40 mT/m.
    # g such that physical amplitude ~ 1e4 mT/m.
    big = 40.0e-3 * (42.577478e6 * 0.22) * 5.0  # 5x over the bound
    g = torch.full((1, 10, 2), big / (2 ** 0.5))  # per-axis so norm ~ big
    slew = torch.zeros(1, 10, 2)
    out = loss({"g": g, "slew": slew})
    assert out.item() > 0.0


def test_hardware_loss_slew_violation():
    """Large slew exceeds S_max => positive penalty even if amplitude ok."""
    loss = GradientHardwareComplianceLoss(
        g_max_mT_per_m=1e9,  # impossible to exceed amplitude
        s_max_T_per_m_per_s=200.0, dt_seconds=4e-6,
        gamma_hz_per_t=42.577478e6, fov_m=0.22, matrix_size=1,
    )
    g = torch.zeros(1, 10, 2)
    big_slew = 200.0 * (42.577478e6 * 0.22) * 5.0
    slew = torch.full((1, 10, 2), big_slew / (2 ** 0.5))
    out = loss({"g": g, "slew": slew})
    assert out.item() > 0.0


def test_hardware_loss_derives_slew_by_diff():
    """When 'slew' absent, it is finite-differenced from 'g'."""
    loss = GradientHardwareComplianceLoss(dt_seconds=4e-6, matrix_size=256)
    g = torch.randn(1, 10, 2) * 1e-3
    out = loss({"g": g})  # no slew key
    assert torch.isfinite(out).all()


def test_losses_gradient_flow_and_finite():
    """Both losses backprop and stay finite."""
    energy_loss = HamiltonianEnergyConservationLoss()
    hw_loss = GradientHardwareComplianceLoss(dt_seconds=4e-6, matrix_size=256)

    energy = torch.linspace(2.0, 2.3, 30).unsqueeze(0).clone().requires_grad_(True)
    e_out = energy_loss({"energy": energy})
    e_out.backward()
    assert energy.grad is not None and torch.isfinite(energy.grad).all()

    g = (torch.randn(1, 20, 2) * 0.5).requires_grad_(True)
    h_out = hw_loss({"g": g})
    # May be zero (feasible) — guard backward only if it has grad path.
    if h_out.item() > 0:
        h_out.backward()
        assert torch.isfinite(g.grad).all()


def test_hardware_loss_rejects_an_image_instead_of_returning_nan():
    """A 4-D image is not a trajectory: raise, never nan.

    Regression for cluster job 8004252. This loss scores a k-space trajectory
    ``[B, T, d]``, but on a ``[B, C, H, W]`` image ``torch.diff(g, dim=1)``
    differences a length-1 axis, returns an EMPTY tensor, and the ``.mean()``
    in ``forward`` is nan -- reported as the loss value, with a zero gradient.
    Non-negotiable #3 makes that a raise.
    """
    loss = GradientHardwareComplianceLoss(dt_seconds=4e-6, matrix_size=256)
    with pytest.raises(ValueError, match=r"\[B, T, d\]"):
        loss(torch.rand(1, 1, 32, 32))


def test_hardware_loss_rejects_a_single_sample_waveform():
    """Slew needs two samples to finite-difference; T=1 raises."""
    loss = GradientHardwareComplianceLoss(dt_seconds=4e-6, matrix_size=256)
    with pytest.raises(ValueError, match="at least 2 time samples"):
        loss({"g": torch.zeros(1, 1, 2)})


def test_hardware_loss_accepts_a_single_sample_when_slew_is_given():
    """The T>=2 requirement belongs to DERIVING slew, not to the loss itself.

    Guards against over-tightening: the guard sits in ``_derive_slew``, so a
    caller that supplies its own slew is unaffected. A guard that rejected this
    too would be rejecting a legal input to buy a nan fix.
    """
    loss = GradientHardwareComplianceLoss(dt_seconds=4e-6, matrix_size=256)
    out = loss({"g": torch.zeros(1, 1, 2), "slew": torch.zeros(1, 1, 2)})
    assert torch.isfinite(out).all()


def test_hardware_loss_scales_by_matrix_size():
    """#2276: k is cycles/pixel, so G = N * dk/dt / (gamma * FOV).

    The default generator's readout sweeps k from -0.5 to 0.5 over 256 samples
    (255 steps of 4 us). At N = 256 that is 26.8 mT/m; without the N factor it read 0.105 mT/m.
    """
    gamma, fov, dt, n = 42.577478e6, 0.22, 4e-6, 256
    loss = GradientHardwareComplianceLoss(
        g_max_mT_per_m=1.0, dt_seconds=dt, gamma_hz_per_t=gamma, fov_m=fov, matrix_size=n
    )
    dkdt = 1.0 / (255 * dt)
    g = torch.zeros(1, 4, 2)
    g[..., 0] = dkdt
    out = loss({"g": g, "slew": torch.zeros_like(g)})
    g_phys = n * dkdt / (gamma * fov) * 1e3
    assert abs(g_phys - 26.8) < 0.1
    assert out.item() == pytest.approx((g_phys - 1.0) ** 2, rel=1e-5)


def test_hardware_loss_flags_a_readout_67x_over_the_bound():
    """The issue's 2679 mT/m readout scored exactly 0 before the N factor."""
    gamma, fov, n = 42.577478e6, 0.22, 256
    loss = GradientHardwareComplianceLoss(
        g_max_mT_per_m=40.0, gamma_hz_per_t=gamma, fov_m=fov, matrix_size=n
    )
    dkdt = 2679e-3 * gamma * fov / n
    g = torch.zeros(1, 4, 2)
    g[..., 0] = dkdt
    assert loss({"g": g, "slew": torch.zeros_like(g)}).item() > 0.0


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_hardware_loss_rejects_a_bad_matrix_size(bad):
    with pytest.raises(ValueError, match="matrix_size"):
        GradientHardwareComplianceLoss(matrix_size=bad)
