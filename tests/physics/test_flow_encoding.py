"""Analytic round-trip tests for the phase-contrast / 4D-flow signal model."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.signal_models.flow_encoding import (
    encoding_directions,
    four_point_reference_decode,
    four_point_reference_encode,
    pc_adjoint,
    pc_forward,
    phase_difference,
    phase_to_velocity,
    referenced_decode_images,
    referenced_decode_phases,
    referenced_encode,
    velocity_to_phase,
    venc_to_first_moment,
    wrap_phase,
)


def test_velocity_phase_roundtrip() -> None:
    venc = 150.0
    v = torch.linspace(-venc, venc, 101)
    recovered = phase_to_velocity(velocity_to_phase(v, venc), venc)
    assert torch.allclose(recovered, v, atol=1e-4)


def test_phase_is_pi_at_venc() -> None:
    venc = 80.0
    phi = velocity_to_phase(torch.tensor(venc), venc)
    assert math.isclose(phi.item(), math.pi, rel_tol=1e-6)


def test_pc_forward_adjoint_roundtrip_within_venc() -> None:
    venc = 100.0
    # Stay strictly inside (-venc, venc) so angle() does not wrap.
    v = torch.linspace(-venc * 0.9, venc * 0.9, 50)
    sig = pc_forward(v, venc, magnitude=2.0)
    assert torch.allclose(sig.abs(), torch.full_like(v, 2.0), atol=1e-4)
    v_rec = pc_adjoint(sig, venc)
    assert torch.allclose(v_rec, v, atol=1e-3)


def test_pc_aliases_beyond_venc() -> None:
    venc = 50.0
    v = torch.tensor(1.5 * venc)  # exceeds venc -> should wrap
    v_rec = pc_adjoint(pc_forward(v, venc), venc)
    assert not torch.allclose(v_rec, v, atol=1.0)  # aliased


def test_four_point_reference_roundtrip() -> None:
    venc = 120.0
    v = torch.randn(4, 3) * (venc * 0.3)
    phases = four_point_reference_encode(v, venc)
    assert phases.shape == (4, 4)
    v_rec = four_point_reference_decode(phases, venc)
    assert torch.allclose(v_rec, v, atol=1e-3)


def test_venc_to_first_moment_positive() -> None:
    m1 = venc_to_first_moment(100.0)
    assert m1 > 0
    # M1 scales inversely with venc.
    assert venc_to_first_moment(50.0) > m1


def test_invalid_venc_raises() -> None:
    with pytest.raises(ValueError):
        velocity_to_phase(torch.tensor(1.0), 0.0)


def test_phase_contrast_is_a_signal_model_not_an_operator() -> None:
    """The encoding is nonlinear with no adjoint, so it lives in the signal-model registry."""
    from spectramr.config.schemas.enums import Regime
    from spectramr.infrastructure.physics.integration import create_physics_operator
    from spectramr.infrastructure.physics.registry import list_operators
    from spectramr.infrastructure.physics.signal_models.registry import get_signal_model

    spec = get_signal_model("phase_contrast")
    assert spec.fn is pc_forward
    assert spec.regime is Regime.FLOW
    assert "phase_contrast" not in list_operators()
    with pytest.raises(ValueError, match="is a signal model"):
        create_physics_operator("phase_contrast")


def test_four_point_decode_rewraps_the_difference() -> None:
    """A background offset pushes the encoded phase past pi; the decode must not read 2 venc."""
    venc = 1.5
    v = torch.tensor([[0.9 * venc, -0.8 * venc, 0.1 * venc]])
    background = 2.5
    phases = wrap_phase(four_point_reference_encode(v, venc) + background)
    assert phases.abs().max() <= math.pi
    assert torch.allclose(four_point_reference_decode(phases, venc), v, atol=1e-5)


def test_phase_difference_matches_the_conjugate_product() -> None:
    torch.manual_seed(0)
    mag = torch.rand(8, 8) + 0.5
    ref_phase = (torch.rand(8, 8) - 0.5) * 2 * math.pi
    delta = (torch.rand(8, 8) - 0.5) * 1.8 * math.pi
    x_ref = mag * torch.exp(1j * ref_phase)
    x_enc = 0.7 * mag * torch.exp(1j * (ref_phase + delta))
    assert torch.allclose(phase_difference(x_enc, x_ref), delta, atol=1e-5)


def test_phase_difference_refuses_real_input() -> None:
    with pytest.raises(TypeError, match="complex"):
        phase_difference(torch.ones(3), torch.ones(3))


@pytest.mark.parametrize("scheme", ["four_point_reference", "two_point"])
def test_referenced_images_roundtrip_on_the_encoding_axis(scheme: str) -> None:
    """Encode along dim 1 of [B, E, T, H, W] with a shared background; the decode recovers v."""
    torch.manual_seed(1)
    n_dir = encoding_directions(scheme)
    # Distinct per-direction venc and T == 3: a venc broadcast onto the frame axis
    # instead of the direction axis would fail this round trip.
    venc = torch.tensor([1.0, 1.5, 2.0][:n_dir])
    v = (torch.rand(2, n_dir, 3, 6, 5) - 0.5) * 1.8 * venc.view(1, n_dir, 1, 1, 1)
    phases = referenced_encode(v, venc, scheme, dim=1)
    background = (torch.rand(2, 1, 3, 6, 5) - 0.5) * 2 * math.pi
    mag = torch.rand(2, n_dir + 1, 3, 6, 5) + 0.2
    images = mag * torch.exp(1j * (phases + background))
    v_rec = referenced_decode_images(images, venc, scheme, dim=1)
    assert v_rec.shape == v.shape
    assert torch.allclose(v_rec, v, atol=1e-4)
    assert torch.allclose(
        referenced_decode_phases(wrap_phase(phases + background), venc, scheme, dim=1), v, atol=1e-4
    )


def test_unknown_and_unimplemented_schemes_raise() -> None:
    with pytest.raises(ValueError, match="unknown velocity-encoding scheme"):
        encoding_directions("hadamard")
    with pytest.raises(ValueError, match="encodes 1 direction"):
        referenced_encode(torch.zeros(4, 3), 1.0, "two_point")


def test_venc_must_be_a_float_or_one_dimensional() -> None:
    with pytest.raises(ValueError, match="1-D"):
        referenced_encode(torch.zeros(2, 3), torch.ones(3, 1), "four_point_reference")
