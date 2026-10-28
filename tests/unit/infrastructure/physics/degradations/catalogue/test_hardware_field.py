"""b0_drift and gradient_delay: closed-form physics, EPI conditioning and refusals."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.hardware_field import (
    off_resonance_lines,
)
from spectramr.infrastructure.physics.degradations.geometry import SliceOrder
from spectramr.infrastructure.physics.fft_ops import fftnc
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T
from spectramr.infrastructure.physics.scanner_profile import Sequence

M = AcquisitionMode
N = 16
MS = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 2.0))
VOL = replace(MS, mode=M.VOLUME_3D)
EPI = replace(FASTMRI_3T, name="epi_3T", sequence=Sequence.EPI, esp_ms=0.5)


def _plane_wave(axis_cycles: tuple[int, int], d: int = 2) -> torch.Tensor:
    """exp(2 pi i (cy y + cx x) / N), a pure k-space point, so a shift is an exact phase."""
    cy, cx = axis_cycles
    i = torch.arange(N, dtype=torch.float32)
    wave = torch.exp(2j * math.pi * (cy * i[:, None] + cx * i[None, :]) / N)
    return wave.expand(1, 1, d, N, N).clone()


@pytest.mark.parametrize("geometry", [MS, VOL])
def test_constant_offset_is_an_exact_readout_fourier_shift(geometry: AcquisitionGeometry) -> None:
    x = torch.randn(1, 1, 2, N, N, dtype=torch.complex64)
    df, bw = 100.0, 250.0
    out = off_resonance_lines(
        x, torch.full((2, N), df), geometry, bw_hz_per_px=bw, sequence="spin_warp"
    )
    # Deliberate raw-FFT oracle (independent of fft_ops) for the exactness claim.
    k = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(x, -1), dim=-1, norm="ortho"), -1)
    f = (torch.arange(N) - N // 2) / N
    k = k * torch.exp(2j * math.pi * f * (df / bw))  # content moves -df / bw
    ref = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(k, -1), dim=-1, norm="ortho"), -1)
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)
    wave = _plane_wave((0, 3))
    shifted = off_resonance_lines(
        wave, torch.full((2, N), df), geometry, bw_hz_per_px=bw, sequence="spin_warp"
    )
    torch.testing.assert_close(
        shifted, wave * math.e ** (2j * math.pi * 3 / N * 0.4), rtol=0, atol=1e-5
    )


def test_epi_offset_shifts_along_phase_encode_by_df_n_pe_esp() -> None:
    df, esp = 80.0, 0.5e-3
    wave = _plane_wave((3, 0))
    out = off_resonance_lines(
        wave, torch.full((2, N), df), MS, bw_hz_per_px=250.0, sequence="epi", esp_s=esp
    )
    shift_px = df * N * esp
    torch.testing.assert_close(
        out, wave * math.e ** (2j * math.pi * 3 / N * shift_px), rtol=0, atol=1e-5
    )
    ro = off_resonance_lines(
        wave, torch.full((2, N), df), MS, bw_hz_per_px=250.0, sequence="spin_warp"
    )
    assert (ro - wave).abs().max() < 1e-5  # the same offset leaves a PE-only object alone


@pytest.mark.parametrize("geometry", [None, MS, VOL])
def test_drift_ramps_linearly_with_line_time_to_the_declared_offset(geometry) -> None:
    n, cx = 32, 4
    i = torch.arange(n, dtype=torch.float32)
    x = torch.exp(2j * math.pi * cx * i / n).expand(1, 1, n, n).clone()
    x = x * torch.exp(-2j * math.pi * 0.31 * i)[:, None]  # a y-dependent object
    if geometry is not None:
        x = x.unsqueeze(2)
    out = apply_degradation("b0_drift", x, 1.0, geometry=geometry)
    dims = (-2, -1)
    ratio = fftnc(out, dims=dims) / fftnc(x, dims=dims)
    row = ratio[..., n // 2 + cx].reshape(-1, n)[0]  # per ky line at the object's k_x bin
    shift_px = -torch.angle(row) * n / (2 * math.pi * cx)
    ky = torch.arange(n, dtype=torch.float32)
    torch.testing.assert_close(shift_px, -50.0 / 250.0 * ky / (n - 1), rtol=0, atol=2e-4)


@pytest.mark.parametrize("geometry", [replace(MS, slice_order=SliceOrder.SEQUENTIAL), VOL])
def test_epi_drift_is_one_pe_shift_per_shot_not_a_smear(geometry) -> None:
    d, esp, df_max = 4, 0.5e-3, 50.0
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 1, d, N, N, generator=g).to(torch.complex64)
    out = apply_degradation(
        "b0_drift", x, 1.0, geometry=geometry, sequence="epi", esp_s=esp, bw_hz_per_px=250.0
    )
    dims = (-3, -2, -1) if geometry is VOL else (-2, -1)
    ratio = fftnc(out, dims=dims) / fftnc(x, dims=dims)  # exp(-2 pi i f_y shift) per line
    pos = ratio[..., N // 2 + 1 :, :]  # positive ky bins: no phase wrap at <= 0.4 px
    fy = (torch.arange(1, N - N // 2) / N).view(1, 1, 1, -1, 1)
    est = -torch.angle(pos) / (2 * math.pi * fy)
    # Linear order: shot s spans lines s*N .. s*N+N-1, mean time s*N + (N-1)/2, last shot = 1.
    t = torch.arange(d) * N + (N - 1) / 2
    expect = -df_max * (t / t.max()) * N * esp
    # Every ky bin of a shot gives the same shift (no quadratic smear) and it is df(t_shot) N ESP.
    torch.testing.assert_close(est, expect.view(1, 1, d, 1, 1).expand_as(est), rtol=0, atol=2e-4)
    assert abs(float(expect[0])) < abs(float(expect[-1]))


def test_single_epi_slice_takes_the_full_end_of_scan_offset() -> None:
    x = _plane_wave((3, 0), d=1)[:, :, 0]
    out = apply_degradation("b0_drift", x, 1.0, sequence="epi", esp_s=0.5e-3)
    shift = 50.0 * N * 0.5e-3
    torch.testing.assert_close(
        out, x * math.e ** (2j * math.pi * 3 / N * shift), rtol=0, atol=1e-5
    )


def test_epi_requires_the_echo_spacing_and_conditions_from_the_profile() -> None:
    x = torch.randn(1, 1, N, N, dtype=torch.complex64)
    for theta in (0.0, 0.5):  # refused at every severity, not only when it would act
        with pytest.raises(ValueError, match="echo spacing"):
            apply_degradation("b0_drift", x, theta, profile=replace(EPI, esp_ms=None))
    a = apply_degradation("b0_drift", x, 1.0, profile=EPI)
    b = apply_degradation("b0_drift", x, 1.0, geometry=None, sequence="epi", esp_s=0.5e-3)
    assert (a - x).abs().max() > 1e-3
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_spin_warp_shift_scales_inversely_with_the_profile_bandwidth() -> None:
    x = torch.randn(1, 1, N, N, dtype=torch.complex64)
    narrow = replace(FASTMRI_3T, name="narrow", readout_bw_hz_per_px=100.0)
    wide = replace(FASTMRI_3T, name="wide", readout_bw_hz_per_px=400.0)
    e = [
        float((apply_degradation("b0_drift", x, 1.0, profile=p) - x).abs().mean())
        for p in (narrow, wide)
    ]
    assert e[0] > 2.0 * e[1]


def test_non_cartesian_profile_and_bad_arguments_raise() -> None:
    x = torch.randn(1, 1, N, N, dtype=torch.complex64)
    with pytest.raises(ValueError, match="no B0 voxel-shift model"):
        apply_degradation("b0_drift", x, 0.5, profile=replace(FASTMRI_3T, sequence=Sequence.SPIRAL))
    for kw in (
        {"sequence": "radial"},
        {"bw_hz_per_px": 0.0},
        {"max_offset_hz": -1.0},
        {"max_offset_hz": float("nan")},
    ):
        with pytest.raises(ValueError):
            apply_degradation("b0_drift", x, 0.5, **kw)


@pytest.mark.parametrize("geometry", [None, MS, VOL])
def test_gradient_delay_is_a_signed_phase_ramp_per_line_parity(geometry) -> None:
    x = torch.randn(2, 1, 4, N, N, dtype=torch.complex64)
    x = x if geometry is not None else x[:, :, 0]
    out = apply_degradation("gradient_delay", x, 1.0, geometry=geometry)
    dims = (-3, -2) if geometry is VOL else (-2,)
    hx, ho = fftnc(x, dims=dims), fftnc(out, dims=dims)
    tau = 0.5  # theta 1 -> half a dwell
    xc = torch.arange(N, dtype=torch.float32) - N // 2
    expect = hx.clone()
    for ky in range(N):
        sign = 1.0 if ky % 2 else -1.0  # odd lines +tau/2, even lines -tau/2
        phase = torch.exp(-1j * sign * math.pi * tau * xc / N)
        if geometry is VOL:
            expect[..., :, ky, :] = hx[..., :, ky, :] * phase
        else:
            expect[..., ky, :] = hx[..., ky, :] * phase
    torch.testing.assert_close(ho, expect, rtol=0, atol=1e-5)


def test_gradient_delay_ghost_sits_half_a_fov_away_with_sin_amplitude() -> None:
    x = torch.randn(1, 1, N, N, dtype=torch.complex64)
    out = apply_degradation("gradient_delay", x, 1.0)
    xc = torch.arange(N, dtype=torch.float32) - N // 2
    phi = math.pi * 0.5 * xc / N
    expect = torch.cos(phi) * x + 1j * torch.sin(phi) * torch.roll(x, N // 2, dims=-2)
    torch.testing.assert_close(out, expect, rtol=0, atol=1e-5)
    assert (
        apply_degradation("gradient_delay", x, 1.0, max_delay_dwell=0.0).sub(x).abs().max() < 1e-6
    )
