"""gradient_scaling_drift and mechanical_vibration: closed-form physics and knob validation."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from scipy.special import jv

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.fft_ops import fftnc

M = AcquisitionMode
VOXEL = (5.0, 1.0, 2.0)
MS = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL)
VOL = replace(MS, mode=M.VOLUME_3D)


def _box(sizes: tuple[int, int, int], shape=(1, 1, 32, 64, 64)) -> torch.Tensor:
    x = torch.zeros(shape)
    sl = [
        slice(s // 2 - w // 2, s // 2 - w // 2 + w) for s, w in zip(shape[-3:], sizes, strict=True)
    ]
    x[(..., *sl)] = 1.0
    return x


def _extent(out: torch.Tensor, axis: int) -> float:
    """Width of a centred box along ``axis`` (0 = D, 1 = H, 2 = W), from a line through its centre."""
    idx = [out.shape[-3] // 2, out.shape[-2] // 2, out.shape[-1] // 2]
    idx[axis] = slice(None)
    return float(out.abs()[(0, 0, *idx)].sum())


@pytest.mark.parametrize("geometry", [MS, VOL])
def test_static_scale_error_rescales_a_bar_by_one_over_one_plus_eps(geometry) -> None:
    eps = 0.2  # theta 1 * max_scale; anisotropic voxel_mm must not change a pixel-space scale
    sizes = (12, 30, 40)
    out = apply_degradation(
        "gradient_scaling_drift",
        _box(sizes),
        1.0,
        5,
        geometry=geometry,
        max_scale=eps,
        static_fraction=1.0,
    )
    for axis, width in zip((1, 2), sizes[1:], strict=True):
        candidates = (width / (1 + eps), width / (1 - eps))
        assert min(abs(_extent(out, axis) - c) for c in candidates) < 0.5, axis
    z = _extent(out, 0)
    if geometry is MS:
        assert z == pytest.approx(sizes[0], abs=1e-4)  # through-plane axis untouched
    else:
        assert min(abs(z - sizes[0] / (1 + s * eps)) for s in (1, -1)) < 0.5


def test_default_two_percent_range_scales_a_wide_bar() -> None:
    out = apply_degradation(
        "gradient_scaling_drift", _box((32, 50, 60)), 1.0, 1, geometry=VOL, static_fraction=1.0
    )
    for axis, width in ((1, 50), (2, 60)):
        assert min(abs(_extent(out, axis) - width / (1 + s * 0.02)) for s in (1, -1)) < 0.5


def test_drift_leaves_the_first_acquired_lines_unscaled() -> None:
    x = torch.randn(1, 1, 4, 32, 32, dtype=torch.complex64)
    out = apply_degradation("gradient_scaling_drift", x, 1.0, 2, geometry=MS, static_fraction=0.0)
    dims = (-2, -1)
    first = shot_index(line_times(MS, 4, 32, tr_s=1.0), 8) == 0
    kin, kout = fftnc(x, dims=dims), fftnc(out, dims=dims)
    torch.testing.assert_close(kout[..., first, :], kin[..., first, :], rtol=0, atol=2e-5)
    last = shot_index(line_times(MS, 4, 32, tr_s=1.0), 8) == 7
    assert (kout[..., last, :] - kin[..., last, :]).abs().max() > 1e-2


def test_scale_signs_are_seeded_per_sample_and_knobs_are_validated() -> None:
    x = _box((12, 30, 40)).expand(2, 1, 32, 64, 64).contiguous()
    widths = {
        _extent(
            apply_degradation(
                "gradient_scaling_drift",
                x,
                1.0,
                s,
                geometry=VOL,
                max_scale=0.2,
                static_fraction=1.0,
            ),
            2,
        )
        for s in range(6)
    }
    assert len(widths) == 2  # +eps and -eps both occur across seeds
    for kw in ({"static_fraction": 1.5}, {"max_scale": -0.1}, {"n_states": 0}):
        with pytest.raises(ValueError):
            apply_degradation("gradient_scaling_drift", x, 0.5, geometry=VOL, **kw)


def _frac_fold(u: float) -> float:
    """Folds f_v * TR into [0, 1/2]: the stroboscopic (aliased) sideband fraction."""
    return abs(u - round(u))


def _vibrating(
    geometry,
    mode: M,
    theta: float = 1.0,
    amplitude_mm: float = 4.0,
    frequency_hz: float = 5.0,
    tr_s: float = 0.0125,
    n_pe: int = 64,
):
    """Plane wave along readout (cx cycles) times an impulse along PE, vibrated along readout."""
    n_ro, cx, y0 = 32, 4, 20
    x = torch.zeros(1, 1, 1, n_pe, n_ro, dtype=torch.complex64)
    x[..., y0, :] = torch.exp(2j * math.pi * cx * torch.arange(n_ro) / n_ro)
    x = x if mode is not M.SLICE_2D else x[:, :, 0]
    out = apply_degradation(
        "mechanical_vibration",
        x,
        theta,
        7,
        geometry=geometry,
        axis="ro",
        max_amplitude_mm=amplitude_mm,
        frequency_hz=frequency_hz,
        tr_s=tr_s,
    )
    z = 2 * math.pi * (cx / n_ro) * theta * amplitude_mm / VOXEL[2]
    step = _frac_fold(frequency_hz * tr_s) * n_pe
    assert step == pytest.approx(round(step))
    return out.reshape(n_pe, n_ro), y0, z, round(step)


def _assert_sidebands(out, y0, z, step, orders) -> None:
    n_pe = out.shape[0]
    energy = 0.0
    for m in orders:
        amp = float(out[(y0 + step * m) % n_pe].abs().mean())
        assert amp == pytest.approx(abs(jv(m, z)), abs=2e-5), m
        energy += amp**2
    assert energy == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("mode", list(M))
def test_vibration_sidebands_follow_jacobi_anger(mode: M) -> None:
    out, y0, z, step = _vibrating(replace(MS, mode=mode), mode)
    # Sidebands at m * frac_fold(f_v TR) * N_pe = 4 m px; |J_m(z)| is the amplitude of order m.
    assert step == 4
    _assert_sidebands(out, y0, z, step, range(-8, 9))
    assert abs(jv(1, z)) > 0.4  # the first-order ghost is not negligible in this setup


def test_aliased_vibration_puts_the_sideband_at_the_folded_position() -> None:
    # A clinical TR: f_v TR = 20.3 * 0.5 = 10.15 folds to 0.15, so the ghosts sit at 0.15 N_pe.
    out, y0, z, step = _vibrating(VOL, M.VOLUME_3D, frequency_hz=20.3, tr_s=0.5, n_pe=80)
    assert step == 12
    _assert_sidebands(out, y0, z, step, range(-6, 7))
    # no energy between the folded orders
    assert float(out[(y0 + 6) % 80].abs().mean()) < 1e-3


def test_integer_f_v_tr_is_a_static_displacement_without_ghosts() -> None:
    n = 16
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 1, n, n, generator=g).to(torch.complex64)
    kw = {"axis": "ro", "max_amplitude_mm": 4.0, "frequency_hz": 20.0, "tr_s": 0.5}
    out = apply_degradation(
        "mechanical_vibration",
        x,
        1.0,
        11,
        geometry=replace(MS, mode=M.SLICE_2D, voxel_mm=VOXEL),
        **kw,
    )
    # The same draw as the kernel: phi = 2 pi u[:, 1] from the seeded generator.
    from spectramr.infrastructure.physics.degradations._common import _generator
    from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

    u = per_sample_rand(1, 2, generator=_generator(x.device, 11), device=x.device)
    d_px = 4.0 / VOXEL[2] * math.sin(2 * math.pi * float(u[0, 1]))
    # Deliberate raw-FFT oracle (independent of fft_ops): a rigid readout shift by d_px.
    ref = torch.fft.fft(x, dim=-1, norm="ortho")
    ref = torch.fft.ifft(
        ref * torch.exp(-2j * math.pi * torch.fft.fftfreq(n) * d_px), dim=-1, norm="ortho"
    )
    assert abs(d_px) > 0.1
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-5)


def test_small_amplitude_first_sideband_is_the_first_order_term() -> None:
    out, y0, z, step = _vibrating(VOL, M.VOLUME_3D, amplitude_mm=0.1)
    assert z < 0.05
    assert float(out[y0 + step].abs().mean()) == pytest.approx(z / 2, rel=1e-2)


def test_vibration_axis_is_seeded_validated_and_changes_the_result() -> None:
    x = torch.randn(1, 1, 4, 16, 16, dtype=torch.complex64)
    picks = {
        tuple(
            bool(
                (apply_degradation("mechanical_vibration", x, 1.0, s, geometry=VOL, axis=a) - y)
                .abs()
                .max()
                < 1e-6
            )
            for a in ("kz", "pe", "ro")
        )
        for s in range(8)
        for y in [apply_degradation("mechanical_vibration", x, 1.0, s, geometry=VOL)]
    }
    assert len(picks) >= 2  # different seeds draw different axes
    assert all(sum(p) == 1 for p in picks)  # and each draw is exactly one of the three
    with pytest.raises(ValueError, match="axis must be one of"):
        apply_degradation("mechanical_vibration", x, 1.0, geometry=MS, axis="kz")
    for kw in ({"tr_s": 0.0}, {"frequency_hz": -1.0}, {"max_amplitude_mm": -1.0}):
        with pytest.raises(ValueError):
            apply_degradation("mechanical_vibration", x, 1.0, geometry=VOL, **kw)
