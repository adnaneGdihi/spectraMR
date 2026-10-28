"""offres_blur_noncartesian: off-resonance modulation of a radial readout, via time-segmented NUFFT."""

from __future__ import annotations

import math

import pytest
import torch
import torchkbnufft as tkbn

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.sequence_offres import (
    DEFAULT_T_RO_MS,
    default_trajectory,
    n_segments,
)
from spectramr.infrastructure.physics.gridding import RadialDCF
from spectramr.infrastructure.physics.nufft.time_segmented_nufft import TimeSegmentedNUFFT

from .invariants import GEOMETRIES, assert_axis_invariants

M = AcquisitionMode
NAME = "offres_blur_noncartesian"
H = W = 32


def _apply(x, theta, geometry=None, **kw):
    return apply_degradation(NAME, x, theta, seed=1, geometry=geometry, **kw)


def _x(shape=(1, 1, H, W), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


def test_declaration_and_trajectory_fallback():
    spec = DEGRADATION_REGISTRY[NAME]
    p = spec.severity.primary
    assert (p.units, p.at_theta_min, p.at_theta_max) == ("cycles", 0.0, 2.0)
    assert spec.requires == frozenset()  # a golden-angle trajectory is synthesised when absent
    assert "golden-angle" in spec.severity.notes
    assert spec.realisation[M.MULTISLICE_2D] is Realisation.PER_SLICE_EXACT
    assert spec.realisation[M.VOLUME_3D] is Realisation.NATIVE


def test_uniform_off_resonance_is_the_analytic_per_sample_phase():
    theta, t_ro = 0.8, DEFAULT_T_RO_MS * 1e-3
    traj, t_unit = default_trajectory(H, W)
    df = 2.0 * theta / t_ro  # peak |df| * T_ro = 2 theta cycles
    x = _x()
    fwd, adj = tkbn.KbNufft(im_size=(H, W)), tkbn.KbNufftAdjoint(im_size=(H, W))
    dcf = RadialDCF(traj.shape[1] // max(H, W), max(H, W))(traj.T)
    k0 = fwd(x, traj)
    phase = torch.exp(
        -2j * math.pi * df * t_unit * t_ro
    )  # exp(-i 2 pi df t), t = 0 at the spoke centre
    delta = torch.zeros(1, 1, H, W, dtype=torch.complex64)
    delta[..., H // 2, W // 2] = 1
    gain = adj(fwd(delta, traj) * dcf, traj)[0, 0, H // 2, W // 2].real
    want = x + adj((k0 * phase - k0) * dcf, traj) / gain
    got = _apply(x, theta, b0_map_hz=torch.full((1, 1, H, W), 7.0))
    # Linear interpolation between time segments: error ~ (phase step)^2 / 8.
    assert float((got - want).norm() / (want - x).norm()) < 0.03


def test_time_segmentation_reproduces_the_phase_ramp_on_the_samples():
    traj, t_unit = default_trajectory(H, W)
    t = t_unit * DEFAULT_T_RO_MS * 1e-3
    omega = 2 * math.pi * 2.0 / (DEFAULT_T_RO_MS * 1e-3)
    x = _x()
    seg = TimeSegmentedNUFFT((H, W), n_segments(2.0))(
        x, torch.full((1, 1, H, W), omega), traj, readout_times=t
    )
    k0 = TimeSegmentedNUFFT((H, W), 1)(x, torch.zeros(1, 1, H, W), traj, readout_times=t)
    err = (seg - k0 * torch.exp(-1j * omega * t)).abs().max() / k0.abs().max()
    assert float(err) < 0.03


def test_the_image_depends_on_df_times_readout_only():
    x = _x()
    a = _apply(x, 0.6, t_ro_ms=5.0)
    torch.testing.assert_close(_apply(x, 0.6, t_ro_ms=2.0), a, rtol=0, atol=1e-4)


def test_default_trajectory_is_golden_angle_and_supplying_it_changes_nothing():
    traj, t_unit = default_trajectory(H, W)
    assert traj.shape == (2, math.ceil(math.pi / 2 * H) * W)
    assert float(traj.abs().max()) == pytest.approx(math.pi, abs=1e-5)
    x = _x()
    explicit = _apply(x, 0.5, trajectory=traj, readout_times_s=t_unit * DEFAULT_T_RO_MS * 1e-3)
    torch.testing.assert_close(explicit, _apply(x, 0.5), rtol=0, atol=1e-5)


def test_a_supplied_trajectory_declares_t_ro_by_its_readout_times():
    traj, t_unit = default_trajectory(H, W)
    x = _x()
    ten_ms = _apply(x, 0.5, trajectory=traj, readout_times_s=t_unit * 10e-3)
    torch.testing.assert_close(ten_ms, _apply(x, 0.5, t_ro_ms=10.0), rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="positive duration"):
        _apply(x, 0.5, trajectory=traj, readout_times_s=torch.zeros_like(t_unit))


def test_trajectory_knobs_are_validated():
    traj, t_unit = default_trajectory(H, W)
    x = _x()
    for kw, match in (
        ({"trajectory": traj}, "readout_times_s"),
        ({"readout_times_s": t_unit}, "readout_times_s"),
        ({"trajectory": traj, "readout_times_s": t_unit, "n_spokes": 3}, "readout_times_s"),
        ({"trajectory": traj, "readout_times_s": t_unit[:-1]}, "trajectory must be"),
        ({"trajectory": traj[:, :-1], "readout_times_s": t_unit[:-1]}, "whole number"),
        ({"t_ro_ms": 0.0}, "t_ro_ms"),
        (
            {"trajectory": traj, "readout_times_s": t_unit, "t_ro_ms": 5.0},
            "span of readout_times_s",
        ),
        ({"n_spokes": 0}, "n_spokes"),
    ):
        with pytest.raises(ValueError, match=match):
            _apply(x, 0.5, **kw)


def test_b0_map_sets_the_pattern_and_theta_the_peak():
    x = _x()
    yy = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
    torch.testing.assert_close(
        _apply(x, 0.5, b0_map_hz=yy), _apply(x, 0.5, b0_map_hz=40.0 * yy), rtol=0, atol=1e-5
    )
    assert float((_apply(x, 0.5, b0_map_hz=yy) - _apply(x, 0.5)).abs().max()) > 1e-3
    with pytest.raises(ValueError, match="b0_map_hz"):
        _apply(x, 0.5, b0_map_hz=torch.ones(1, 1, H + 1, W))
    with pytest.raises(RuntimeError):  # an all-zero map: branch-free device assert
        _apply(x, 0.5, b0_map_hz=torch.zeros(1, 1, H, W))


def test_multislice_shares_one_field_and_a_volume_has_its_own_per_slice():
    ms, vol = GEOMETRIES["ms_interleaved"], GEOMETRIES["volume"]
    x = _x((1, 1, 4, H, W))
    out = _apply(x, 0.7, ms)
    for s in range(4):
        torch.testing.assert_close(out[:, :, s], _apply(x[:, :, s], 0.7), rtol=0, atol=1e-5)
    # With the same image in every slice the shared field gives identical slices; a 3-D field does not.
    same = x[:, :, :1].expand_as(x).contiguous()
    ms_out, vol_out = _apply(same, 0.7, ms), _apply(same, 0.7, vol)
    torch.testing.assert_close(ms_out[:, :, 0], ms_out[:, :, 3], rtol=0, atol=1e-5)
    assert float((vol_out[:, :, 0] - vol_out[:, :, 3]).abs().max()) > 1e-3


def test_real_input_returns_a_complex_image():
    assert _apply(torch.rand(1, 1, H, W), 0.5).is_complex()
