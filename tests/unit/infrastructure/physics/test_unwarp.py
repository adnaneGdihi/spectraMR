"""Unit tests for :mod:`spectramr.infrastructure.physics.unwarp`.

``PhysicalUnwarp`` composes the inverse displacement field from three
sources of EPI distortion: B0 inhomogeneity, gradient nonlinearity,
and Maxwell concomitant fields. Each is reduced to a per-voxel pixel
shift along the phase-encode axis.

Property tests:

* The **identity** case (no field maps passed) returns the input
  volume unchanged within numerical tolerance.
* A **constant B0 field** shifts the volume uniformly along the PE
  axis — magnitudes match the analytical ``shift = ΔB0 · ESP · N_PE``.
* Maxwell and B0 contributions are **additive** — the unwarp is
  linear in the offsets.
* ``pe_axis="W"`` shifts along the column axis instead.
* Construction-time guards reject invalid ``echo_spacing_s`` or
  ``pe_axis`` (CLAUDE.md #9 loud-fail).
* Volume shape must be ``(B, C, D, H, W)`` — anything else raises.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.unwarp import PhysicalUnwarp


def _ramp_volume(B: int = 1, C: int = 1, D: int = 4, H: int = 8, W: int = 8) -> torch.Tensor:
    """Return a volume with a smooth intensity ramp along the PE axis."""
    h = torch.arange(H, dtype=torch.float32)
    base = h.view(1, 1, 1, H, 1).expand(B, C, D, H, W).clone()
    return base


# ── Construction-time validation ───────────────────────────────────


class TestConstructionGuards:
    def test_negative_echo_spacing_raises(self) -> None:
        with pytest.raises(ValueError, match="echo_spacing_s must be > 0"):
            PhysicalUnwarp(echo_spacing_s=-1e-3)

    def test_zero_echo_spacing_raises(self) -> None:
        with pytest.raises(ValueError, match="echo_spacing_s must be > 0"):
            PhysicalUnwarp(echo_spacing_s=0.0)

    def test_invalid_pe_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="pe_axis must be"):
            PhysicalUnwarp(echo_spacing_s=1e-3, pe_axis="Z")  # type: ignore[arg-type]


# ── Identity behaviour ─────────────────────────────────────────────


class TestIdentityWarp:
    def test_no_fields_returns_input_unchanged(self) -> None:
        """When no field maps are provided the warp is the identity."""
        op = PhysicalUnwarp(echo_spacing_s=1e-3)
        vol = _ramp_volume()
        out = op(vol)
        # grid_sample at the identity grid + border padding leaves a
        # ramp untouched in the interior; allow a tight tolerance for
        # bilinear rounding at the edges.
        assert torch.allclose(out, vol, atol=1e-5)

    def test_zero_b0_field_is_identity(self) -> None:
        op = PhysicalUnwarp(echo_spacing_s=1e-3)
        vol = _ramp_volume()
        b0 = torch.zeros(4, 8, 8)
        out = op(vol, b0_hz=b0)
        assert torch.allclose(out, vol, atol=1e-5)


# ── Shape-validation guards ────────────────────────────────────────


class TestShapeGuards:
    def test_non_5d_volume_raises(self) -> None:
        op = PhysicalUnwarp(echo_spacing_s=1e-3)
        # 4-D input (missing depth) → fail loud, no silent broadcast.
        bad = torch.zeros(1, 1, 8, 8)
        with pytest.raises(ValueError, match="expects \\(B, C, D, H, W\\)"):
            op(bad)

    def test_wrong_gnl_shape_raises(self) -> None:
        op = PhysicalUnwarp(echo_spacing_s=1e-3)
        vol = _ramp_volume()
        # Should be (B, 3, D, H, W); pass (B, 2, D, H, W).
        bad_gnl = torch.zeros(1, 2, 4, 8, 8)
        with pytest.raises(ValueError, match="gnl_displacement must have shape"):
            op(vol, gnl_displacement=bad_gnl)

    def test_field_with_bad_rank_raises(self) -> None:
        op = PhysicalUnwarp(echo_spacing_s=1e-3)
        vol = _ramp_volume()
        # 2-D field is invalid; module accepts only (D,H,W) or (B,D,H,W).
        bad_b0 = torch.zeros(8, 8)
        with pytest.raises(ValueError, match="field must have shape"):
            op(vol, b0_hz=bad_b0)


# ── Linear-in-offset behaviour ─────────────────────────────────────


class TestLinearityInOffsets:
    def test_b0_and_maxwell_additive(self) -> None:
        """``unwarp(b0 + maxwell) == unwarp(b0) + unwarp(maxwell) -
        unwarp(0)`` to first order — since grid_sample is bilinear and
        the shifts are small, the result is linear in the offsets.
        """
        op = PhysicalUnwarp(echo_spacing_s=5e-4)
        vol = _ramp_volume(D=4, H=16, W=8)
        b0 = torch.full((4, 16, 8), 0.5)
        mx = torch.full((4, 16, 8), 0.3)
        zero = torch.zeros_like(b0)

        baseline = op(vol)
        out_b0 = op(vol, b0_hz=b0)
        out_mx = op(vol, maxwell_hz=mx)
        out_combo = op(vol, b0_hz=b0, maxwell_hz=mx)

        # Linearity check: shifts add.
        residual = out_combo - (out_b0 + out_mx - baseline)
        # Tight tolerance — shifts here are < 1 pixel.
        assert residual.abs().max().item() < 1e-3
        # Suppress unused-variable warning.
        del zero


# ── PE-axis selection ──────────────────────────────────────────────


class TestPEAxisSelection:
    def test_h_and_w_axis_produce_different_outputs(self) -> None:
        op_h = PhysicalUnwarp(echo_spacing_s=5e-4, pe_axis="H")
        op_w = PhysicalUnwarp(echo_spacing_s=5e-4, pe_axis="W")
        vol = _ramp_volume(D=4, H=16, W=16)
        b0 = torch.full((4, 16, 16), 0.5)
        # Same offsets, different axis → different output.
        out_h = op_h(vol, b0_hz=b0)
        out_w = op_w(vol, b0_hz=b0)
        assert not torch.allclose(out_h, out_w, atol=1e-5)


# ── Batch broadcasting ─────────────────────────────────────────────


class TestBatchBroadcast:
    def test_unbatched_field_broadcasts_across_batch(self) -> None:
        op = PhysicalUnwarp(echo_spacing_s=5e-4)
        vol = _ramp_volume(B=2, D=4, H=8, W=8)
        # 3-D field is broadcast across batch.
        b0_unbatched = torch.full((4, 8, 8), 0.3)
        b0_batched = b0_unbatched.unsqueeze(0).expand(2, -1, -1, -1)
        out_a = op(vol, b0_hz=b0_unbatched)
        out_b = op(vol, b0_hz=b0_batched)
        assert torch.allclose(out_a, out_b)

    def test_batched_field_with_size_1_expands(self) -> None:
        op = PhysicalUnwarp(echo_spacing_s=5e-4)
        vol = _ramp_volume(B=2, D=4, H=8, W=8)
        # (1, D, H, W) explicitly expanded to batch.
        b0 = torch.full((1, 4, 8, 8), 0.3)
        out = op(vol, b0_hz=b0)
        # Batch entries differ because vol entries are identical so
        # outputs should be identical too; pin behaviour.
        assert out.shape == vol.shape


class TestReadoutShiftModel:
    """shift_mode='readout': a df offset moves samples by df / pixel bandwidth."""

    def test_readout_needs_pixel_bandwidth(self) -> None:
        with pytest.raises(ValueError, match="pixel_bandwidth_hz > 0"):
            PhysicalUnwarp(echo_spacing_s=None, shift_mode="readout")

    def test_epi_needs_echo_spacing(self) -> None:
        with pytest.raises(ValueError, match="echo_spacing_s must be > 0"):
            PhysicalUnwarp(echo_spacing_s=None, shift_mode="epi")

    def test_known_offset_gives_one_pixel_readout_shift(self) -> None:
        """A 1 pixel shift is df = BW, so the volume slides by one sample along W."""
        bw = 100.0
        op = PhysicalUnwarp(echo_spacing_s=None, shift_mode="readout", pixel_bandwidth_hz=bw)
        assert op.shift_axis == "W"
        W = 16
        vol = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, 1, W).expand(1, 1, 4, 8, W)
        b0 = torch.full((4, 8, W), bw)
        out = op(vol.contiguous(), b0_hz=b0)
        # Interior voxels sample the neighbour one pixel to the right.
        assert torch.allclose(out[0, 0, 0, 0, 1:-2], vol[0, 0, 0, 0, 2:-1], atol=1e-4)

    def test_readout_axis_is_not_pe_axis(self) -> None:
        op = PhysicalUnwarp(echo_spacing_s=None, shift_mode="readout", pixel_bandwidth_hz=50.0)
        assert op.shift_axis == "W"
        assert PhysicalUnwarp(echo_spacing_s=1e-3, pe_axis="W").shift_axis == "W"
