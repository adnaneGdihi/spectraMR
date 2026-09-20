"""Tests for ``MultiCoilForwardOperator`` and helpers.

Targets ``spectramr.infrastructure.physics.forward_operator``. The forward
operator implements ``y = M F S x`` — coil sensitivity, FFT, then
undersampling — and its adjoint. It's the load-bearing physics primitive
that PA-PCFM, unrolled networks, and DC layers all depend on.

Categories:

- Unit: forward → adjoint round-trip on identity coils
- Property: linearity, adjoint test ``<Ax, y> == <x, A†y>`` (within tolerance)
- Sanity-shape: forward output shape ``[B, C_coils, H, W]``
- Edge: B0 phase modulation (static rad mode) preserves magnitude
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.forward_operator import (
    MultiCoilDataConsistency,
    MultiCoilForwardOperator,
    NullSpaceProjection,
    create_cartesian_mask,
    create_radial_mask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_coils(num_coils: int, h: int, w: int) -> torch.Tensor:
    """Coil maps that are unit magnitude everywhere → equivalent to identity."""
    return torch.complex(
        torch.ones(num_coils, h, w) / num_coils**0.5,
        torch.zeros(num_coils, h, w),
    )


def _full_mask(h: int, w: int) -> torch.Tensor:
    """Fully-sampled mask (no undersampling)."""
    return torch.ones(1, 1, h, w)


# ---------------------------------------------------------------------------
# Forward operator — unit / shape
# ---------------------------------------------------------------------------


def test_forward_output_shape_complex() -> None:
    """``forward`` produces ``[B, num_coils, H, W]`` complex output."""
    coils = _identity_coils(4, 16, 16)
    op = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(16, 16))
    x = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
    y = op.forward(x)
    assert y.shape == (2, 4, 16, 16)
    assert torch.is_complex(y)


def test_forward_accepts_real_2channel_layout() -> None:
    """``[B, 2, H, W]`` real input is auto-coerced to complex."""
    coils = _identity_coils(2, 8, 8)
    op = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(8, 8))
    x = torch.randn(1, 2, 8, 8)  # real, two channels
    y = op.forward(x)
    assert y.shape == (1, 2, 8, 8)
    assert torch.is_complex(y)


def test_forward_no_coils_no_mask_passes_through_fft() -> None:
    """Without coils/mask, output magnitude equals FFT magnitude."""
    op = MultiCoilForwardOperator()
    x = torch.complex(torch.randn(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
    y = op.forward(x)
    expected_mag_sum = torch.fft.fft2(x, norm="ortho").abs().sum()
    actual_mag_sum = y.abs().sum()
    # Centred vs uncentred FFT differ in placement, not in sum of magnitudes.
    assert torch.isclose(actual_mag_sum, expected_mag_sum, atol=1e-3)


def test_adjoint_round_trip_with_identity_coils() -> None:
    """``A†A x ≈ x`` when coil maps are unit magnitude and mask is full."""
    coils = _identity_coils(1, 8, 8)
    op = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(8, 8))
    x = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    x_hat = op.normal(x)
    # With C=1 unit-magnitude coils, A†A is identity (within FFT round-trip).
    assert torch.allclose(x_hat, x, atol=1e-4)


def test_adjoint_test_inner_product() -> None:
    """``<A x, y> ≈ <x, A† y>`` (linear-operator adjoint property)."""
    torch.manual_seed(0)
    coils = torch.complex(torch.randn(2, 8, 8), torch.randn(2, 8, 8)) * 0.1
    op = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(8, 8))
    x = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    y = torch.complex(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8))

    Ax = op.forward(x)
    Aty = op.adjoint(y)

    lhs = (Ax * y.conj()).sum()
    rhs = (x * Aty.conj()).sum()
    rel_err = (lhs - rhs).abs().item() / (lhs.abs().item() + 1e-12)
    assert rel_err < 1e-4, f"adjoint test failed: rel_err = {rel_err:.3e}"


def test_undersampling_mask_zeroes_unmeasured_locations() -> None:
    """Where mask = 0, output k-space is exactly zero."""
    h, w = 8, 8
    mask = torch.zeros(1, 1, h, w)
    mask[..., : h // 2, :] = 1.0  # only top half measured
    op = MultiCoilForwardOperator(coil_sensitivities=_identity_coils(2, h, w), mask=mask)
    y = op.forward(torch.complex(torch.randn(1, 1, h, w), torch.randn(1, 1, h, w)))
    assert torch.allclose(y[..., h // 2 :, :].abs(), torch.zeros_like(y[..., h // 2 :, :].abs()))


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------


def test_set_coil_sensitivities_persists() -> None:
    """``set_coil_sensitivities`` updates the buffer."""
    op = MultiCoilForwardOperator()
    new_coils = _identity_coils(3, 8, 8)
    op.set_coil_sensitivities(new_coils)
    assert torch.equal(op.coil_sensitivities, new_coils)


def test_set_mask_unsqueezes_2d_input() -> None:
    """A ``[H, W]`` mask is broadcast to ``[1, 1, H, W]``."""
    op = MultiCoilForwardOperator()
    mask_2d = torch.ones(8, 8)
    op.set_mask(mask_2d)
    assert op.mask.shape == (1, 1, 8, 8)


def test_set_physics_maps_updates_buffers() -> None:
    """B0 / T2* / gradient-NL maps round-trip through ``set_physics_maps``."""
    op = MultiCoilForwardOperator()
    b0 = torch.randn(8, 8)
    t2 = torch.rand(8, 8)
    op.set_physics_maps(b0_map=b0, t2_star_map=t2)
    assert torch.equal(op.b0_map, b0)
    assert torch.equal(op.t2_star_map, t2)


# ---------------------------------------------------------------------------
# B0 / Physics
# ---------------------------------------------------------------------------


def test_b0_static_phase_preserves_magnitude() -> None:
    """Static B0 phase ``exp(i·φ)`` preserves image magnitude before FFT."""
    h, w = 8, 8
    coils = _identity_coils(1, h, w)
    op = MultiCoilForwardOperator(
        coil_sensitivities=coils,
        mask=_full_mask(h, w),
        b0_map=torch.full((h, w), 0.5),  # 0.5 rad phase
        b0_units="rad",
    )
    x = torch.complex(torch.ones(1, 1, h, w), torch.zeros(1, 1, h, w))
    y = op.forward(x)
    # |y| sum should equal |FFT of phased x| sum, which equals |FFT of x| sum.
    y_no_b0 = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(h, w)).forward(x)
    assert torch.allclose(y.abs().sum(), y_no_b0.abs().sum(), atol=1e-4)


def test_simulate_measurement_adds_noise() -> None:
    """SNR-controlled noise is added; high SNR → small relative perturbation."""
    torch.manual_seed(0)
    coils = _identity_coils(2, 8, 8)
    op = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(8, 8))
    x = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    y_clean = op.forward(x)
    y_noisy = op.simulate_measurement(x, snr=60.0)  # very high SNR
    rel_err = (y_noisy - y_clean).abs().mean() / (y_clean.abs().mean() + 1e-8)
    assert rel_err.item() < 0.05  # < 5% perturbation at SNR=60dB


# ---------------------------------------------------------------------------
# NullSpaceProjection
# ---------------------------------------------------------------------------


def test_null_space_plus_range_equals_input() -> None:
    """``P_N(x) + P_R(x) == x``."""
    coils = _identity_coils(1, 8, 8)
    fwd = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(8, 8))
    proj = NullSpaceProjection(fwd)
    x = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    null = proj.forward(x)
    rng = proj.range_projection(x)
    assert torch.allclose(null + rng, x, atol=1e-5)


# ---------------------------------------------------------------------------
# MultiCoilDataConsistency
# ---------------------------------------------------------------------------


def test_data_consistency_hard_replaces_kspace_at_mask() -> None:
    """Hard DC: at measured locations, output k-space matches ``y``."""
    h, w = 8, 8
    mask = torch.zeros(1, 1, h, w)
    mask[..., : h // 2, :] = 1.0
    coils = _identity_coils(1, h, w)
    fwd = MultiCoilForwardOperator(coil_sensitivities=coils, mask=mask)
    dc = MultiCoilDataConsistency(fwd, mode="hard")

    x = torch.complex(torch.randn(1, 1, h, w), torch.randn(1, 1, h, w))
    y_target = torch.complex(torch.randn(1, 1, h, w), torch.randn(1, 1, h, w)) * mask
    x_dc = dc.forward(x, y_target)
    y_dc = fwd.forward(x_dc)
    # Measured locations should match y_target (modulo numerical FFT round-trip).
    diff = ((y_dc - y_target) * mask).abs().max().item()
    assert diff < 1e-4, f"hard DC failed at measured locations: max diff = {diff:.3e}"


def test_data_consistency_soft_returns_residual_when_requested() -> None:
    """``return_residual=True`` returns a ``(x_dc, residual)`` tuple."""
    coils = _identity_coils(1, 8, 8)
    fwd = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(8, 8))
    dc = MultiCoilDataConsistency(fwd, mode="soft", lambda_dc=0.5)
    x = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    y = fwd.forward(x)
    x_dc, residual = dc.forward(x, y, return_residual=True)
    assert torch.is_complex(x_dc)
    assert residual.dim() == 0
    assert residual.item() >= 0


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def test_create_radial_mask_has_centre_dc() -> None:
    """All radial spokes pass through the centre — DC is sampled."""
    mask = create_radial_mask(32, 32, num_spokes=16, golden_angle=True)
    assert mask[16, 16].item() == 1.0
    assert mask.shape == (32, 32)


def test_create_cartesian_mask_includes_centre_lines() -> None:
    """Cartesian mask always includes the central ACS region."""
    mask = create_cartesian_mask(64, 64, acceleration=4, center_fraction=0.08)
    assert mask.shape == (64, 64)
    centre = mask[28:36, :].sum()  # 8 central lines (~0.08 * 64 ≈ 5.12, rounded)
    assert centre.item() > 0


def test_create_cartesian_mask_seed_is_reproducible() -> None:
    """Same seed → same mask."""
    m1 = create_cartesian_mask(32, 32, acceleration=4, random_seed=7)
    m2 = create_cartesian_mask(32, 32, acceleration=4, random_seed=7)
    assert torch.equal(m1, m2)


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, num_coils",
    [
        ((1, 1, 8, 8), 1),
        ((2, 1, 16, 16), 2),
        ((1, 1, 32, 32), 4),
        ((2, 1, 16, 32), 4),  # rectangular
    ],
    ids=["B1C1_8x8_C1", "B2_16x16_C2", "B1_32x32_C4", "rect"],
)
def test_forward_shape_matrix(shape: tuple[int, ...], num_coils: int) -> None:
    """Forward output shape is ``[B, num_coils, H, W]`` across the matrix."""
    b, _, h, w = shape
    coils = _identity_coils(num_coils, h, w)
    op = MultiCoilForwardOperator(coil_sensitivities=coils, mask=_full_mask(h, w))
    x = torch.complex(torch.randn(*shape), torch.randn(*shape))
    y = op.forward(x)
    assert y.shape == (b, num_coils, h, w)
    assert torch.isfinite(y.real).all()
    assert torch.isfinite(y.imag).all()


# ---------------------------------------------------------------------------
# Regression: the broken time-segmented ("hz") B0 path fails loud
# ---------------------------------------------------------------------------
# The b0_units="hz" forward/adjoint were not an adjoint pair and the phase
# tensor mis-broadcast against the coil axis. Rather than silently mis-compute
# (CLAUDE.md #9/#16), constructing with "hz" now raises.


def test_b0_units_hz_is_rejected() -> None:
    """b0_units='hz' (time-segmented) raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="hz"):
        MultiCoilForwardOperator(b0_units="hz", bw_hz=1e4)


def test_b0_units_unknown_is_rejected() -> None:
    """An unrecognised b0_units value raises (no silent fallback to 'rad')."""
    with pytest.raises(ValueError, match="b0_units"):
        MultiCoilForwardOperator(b0_units="bogus")


def test_b0_units_rad_still_constructs() -> None:
    """The supported static-phase mode is unaffected."""
    op = MultiCoilForwardOperator(b0_units="rad")
    assert op.b0_units == "rad"


# ---------------------------------------------------------------------------
# Regression: radial mask uses the radial golden angle (~111.25°), not 137.5°
# ---------------------------------------------------------------------------


def _radial_mask_with_angle(h: int, w: int, num_spokes: int, inc) -> torch.Tensor:
    """Mirror of create_radial_mask's spoke drawing for a given angle increment."""
    mask = torch.zeros(h, w)
    ch, cw = h // 2, w // 2
    for i in range(num_spokes):
        angle = i * inc
        t = torch.linspace(-1, 1, max(h, w))
        x = (cw + t * w * 0.5 * torch.cos(angle)).long()
        y = (ch + t * h * 0.5 * torch.sin(angle)).long()
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        mask[y[valid], x[valid]] = 1.0
    return mask


def test_radial_mask_uses_radial_golden_angle() -> None:
    """golden_angle=True draws spokes at π(√5−1)/2 ≈ 111.25°, not π(3−√5) ≈ 137.5°."""
    h = w = 64
    n = 8
    mask = create_radial_mask(h, w, num_spokes=n, golden_angle=True)
    correct = _radial_mask_with_angle(
        h, w, n, torch.pi * (torch.sqrt(torch.tensor(5.0)) - 1.0) / 2.0
    )
    phyllotaxis = _radial_mask_with_angle(h, w, n, torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0))))
    assert torch.equal(mask, correct)
    assert not torch.equal(mask, phyllotaxis)


# ---- non-negotiable 2: the operator spells no raw torch.fft call (2026-09-03) ----

_RAW_FFT = "torch.fft."


def _raw_fft_lines(source: str) -> list[str]:
    """Lines of ``source`` that call ``torch.fft`` directly, comments excluded."""
    return [line for line in source.splitlines() if _RAW_FFT in line.split("#", 1)[0]]


def test_forward_operator_source_has_no_raw_torch_fft_call() -> None:
    import inspect

    import spectramr.infrastructure.physics.forward_operator as mod

    assert _raw_fft_lines(inspect.getsource(mod)) == []


def test_raw_fft_detector_sees_the_planted_shapes() -> None:
    # Planted: a top-of-file call and an indented one both count; a comment does not.
    planted = "k = torch.fft.fft2(x)\n    y = torch.fft.ifftshift(k, dim=-1)\n# torch.fft.fft2 in a comment\n"
    assert len(_raw_fft_lines(planted)) == 2


def test_readout_transform_pairs_round_trip_on_an_odd_axis() -> None:
    from spectramr.infrastructure.physics.fft_ops import (
        fft1_uncentered,
        fft1c,
        ifft1_uncentered,
        ifft1c,
    )

    k = torch.randn(1, 2, 8, 17, dtype=torch.complex64)
    assert torch.allclose(fft1c(ifft1c(k, dim=-1), dim=-1), k, atol=1e-5)
    assert torch.allclose(fft1_uncentered(ifft1_uncentered(k, dim=-1), dim=-1), k, atol=1e-5)
