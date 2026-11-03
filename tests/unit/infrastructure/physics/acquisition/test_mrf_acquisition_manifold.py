"""SAR/slew projection of an MRF pulse design onto its :class:`AcquisitionBound`."""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.acquisition.mrf_acquisition_manifold import (
    AcquisitionBound,
    project_pulse_to_bound,
    sar_estimate,
    slew_estimate,
)


def _pulse() -> torch.Tensor:
    return torch.stack([torch.full((5,), 1.5), torch.tensor([10.0, 14.0, 8.0, 12.0, 10.0])], -1)


def test_projection_meets_a_binding_sar_and_slew_bound() -> None:
    pulse = _pulse()
    bound = AcquisitionBound(sar_max_w_per_kg=1.0, gradient_slew_rate_t_per_m_per_s=2.0)
    assert sar_estimate(pulse) > 1.0 and slew_estimate(pulse) > 2.0
    legal = project_pulse_to_bound(pulse, bound)
    assert sar_estimate(legal).item() <= 1.0 + 1e-6
    assert slew_estimate(legal).item() <= 2.0 + 1e-6
    assert (legal[:, 1] > 0).all()


def test_projection_is_identity_inside_a_slack_bound() -> None:
    pulse = _pulse()
    legal = project_pulse_to_bound(pulse, AcquisitionBound(10.0, 100.0))
    assert torch.allclose(legal, pulse)


def test_projection_is_differentiable() -> None:
    pulse = _pulse().requires_grad_(True)
    legal = project_pulse_to_bound(pulse, AcquisitionBound(1.0, 2.0))
    (grad,) = torch.autograd.grad(legal.sum(), pulse)
    assert torch.isfinite(grad).all() and torch.any(grad != 0)
