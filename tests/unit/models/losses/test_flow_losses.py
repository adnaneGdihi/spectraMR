"""Tests for the mri_flow regime losses."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.flow_losses import (
    PhaseContrastVelocityLoss,
    ThroughPlaneFluxConservationLoss,
    VelocityUnwrapConsistencyLoss,
)
from spectramr.models.losses.registry import LossRegistry


def test_flow_losses_registered_and_tagged() -> None:
    from spectramr.config.schemas.enums import Regime

    meta = LossRegistry._loss_domains
    for name in (
        "phase_contrast_velocity",
        "through_plane_flux_conservation",
        "velocity_unwrap_consistency",
    ):
        assert name in meta
        assert Regime.FLOW in (meta[name].get("workflows") or frozenset())


def test_phase_contrast_velocity_zero_on_match() -> None:
    loss = PhaseContrastVelocityLoss(venc=100.0)
    v = torch.randn(2, 3, 8, 8)
    assert loss(v, v).item() == 0.0


def test_phase_contrast_velocity_positive_on_mismatch() -> None:
    loss = PhaseContrastVelocityLoss(venc=100.0)
    v = torch.randn(2, 3, 8, 8)
    assert loss(v, v + 5.0).item() > 0.0


def test_flux_conservation_zero_when_balanced() -> None:
    loss = ThroughPlaneFluxConservationLoss()
    v = torch.ones(1, 1, 4, 4)
    inlet = torch.zeros(1, 1, 4, 4)
    inlet[..., 0, :] = 1.0
    outlet = torch.zeros(1, 1, 4, 4)
    outlet[..., -1, :] = 1.0
    # Equal masks over a uniform field -> balanced flux.
    assert loss(v, inlet, inlet).item() == 0.0
    # Different rows of a uniform field still balance (same count).
    assert loss(v, inlet, outlet).item() == 0.0


def test_unwrap_penalty_only_beyond_venc() -> None:
    loss = VelocityUnwrapConsistencyLoss(venc=10.0)
    within = torch.full((4,), 5.0)
    assert loss(within).item() == 0.0
    beyond = torch.full((4,), 15.0)
    assert loss(beyond).item() > 0.0


def test_phase_contrast_velocity_rejects_an_unknown_norm() -> None:
    """``p`` other than 1 or 2 used to fall through to the L2 branch silently."""
    import pytest

    with pytest.raises(ValueError, match="p must be"):
        PhaseContrastVelocityLoss(venc=1.0, p=3)


def test_phase_contrast_velocity_is_wrap_aware() -> None:
    """v and v - 2 venc give the same encoded signal, so the loss must not separate them."""
    venc = 1.0
    v = torch.full((2, 3, 4, 4), 0.95)
    aliased = v - 2 * venc
    loss = PhaseContrastVelocityLoss(venc=venc)
    assert loss(aliased, v).item() < 1e-5
    assert loss(-v, v).item() > 0.1


def test_phase_contrast_velocity_weight_selects_voxels() -> None:
    loss = PhaseContrastVelocityLoss(venc=1.0)
    v = torch.zeros(1, 3, 4, 4)
    pred = v.clone()
    pred[..., 0, 0] = 0.5
    weight = torch.ones(1, 1, 4, 4)
    weight[..., 0, 0] = 0.0
    assert loss(pred, v, weight=weight).item() == 0.0
    assert loss(pred, v).item() > 0.0


def test_phase_contrast_velocity_places_a_per_direction_venc_on_the_channel_axis() -> None:
    """W == n_dir == 3: a venc broadcast on the width axis would leave a residue."""
    venc = torch.tensor([1.0, 1.5, 2.0])
    v = 0.95 * venc.view(1, 3, 1, 1).expand(1, 3, 3, 3).clone()
    aliased = v - 2 * venc.view(1, 3, 1, 1)
    assert PhaseContrastVelocityLoss()(aliased, v, venc=venc).item() < 1e-5
    assert VelocityUnwrapConsistencyLoss()(v, venc=venc).item() == 0.0


def _encoded(seed: int = 0, n_enc: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mag = torch.rand(2, n_enc, 3, 6, 5, generator=g) + 0.5
    phase = (torch.rand(2, n_enc, 3, 6, 5, generator=g) - 0.5) * 6.0
    return mag * torch.exp(1j * phase)


def test_phasor_loss_ignores_a_shared_background_and_wraps() -> None:
    """A background phasor on each encoding, or a 2 pi phase shift, leaves the phase error unchanged."""
    import math

    from spectramr.models.losses.flow_losses import PhaseContrastPhasorLoss

    loss = PhaseContrastPhasorLoss()
    ref = _encoded()
    assert loss(ref, ref).item() < 1e-6
    background = torch.exp(1j * torch.rand(1, 4, 1, 6, 5))
    assert loss(ref * background, ref * background).item() < 1e-6
    shifted = ref.clone()
    shifted[:, 1:] *= math.e ** (1j * 2 * math.pi)
    assert loss(shifted, ref).item() < 1e-5
    flipped = ref.clone()
    flipped[:, 2] *= -1
    assert loss(flipped, ref).item() > 0.1


def test_phasor_loss_has_finite_gradients_at_a_vanishing_voxel() -> None:
    """angle() would blow up where |x| -> 0; the phasor form stays bounded."""
    from spectramr.models.losses.flow_losses import PhaseContrastPhasorLoss

    pred = _encoded(seed=1).clone()
    pred[0, :, 0, 0, 0] = 1e-12
    pred.requires_grad_(True)
    PhaseContrastPhasorLoss()(pred, _encoded()).backward()
    assert torch.isfinite(torch.view_as_real(pred.grad)).all()


def test_image_nmse_is_scale_free() -> None:
    from spectramr.models.losses.flow_losses import PhaseContrastImageNMSELoss

    ref, pred = _encoded(), _encoded(seed=2)
    loss = PhaseContrastImageNMSELoss()
    assert loss(ref, ref).item() == 0.0
    assert loss(7.0 * pred, 7.0 * ref).item() == pytest.approx(loss(pred, ref).item(), rel=1e-5)
