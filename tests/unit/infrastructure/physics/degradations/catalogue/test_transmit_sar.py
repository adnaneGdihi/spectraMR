"""sar_limited_flip: a SAR-reduced refocusing angle reweights the TSE echo train (EPG)."""

from __future__ import annotations

import dataclasses
import math
from itertools import pairwise

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.transmit_sar import (
    echo_amplitudes,
    echo_weights,
)
from spectramr.infrastructure.physics.fft_ops import fftnc
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T
from spectramr.infrastructure.physics.scanner_profile import Sequence

from .invariants import GEOMETRIES, THETAS, VOXEL, assert_axis_invariants, phantom, rms

M = AcquisitionMode
NAME = "sar_limited_flip"
FOREVER = 1e12  # ms: no relaxation


@pytest.mark.parametrize("mode", list(M))
def test_every_mode_is_native_and_nothing_is_required(mode):
    spec = DEGRADATION_REGISTRY[NAME]
    assert spec.realisation[mode] is Realisation.NATIVE and spec.requires == frozenset()
    assert spec.severity.physical_value(1.0) == 120.0


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


def test_a_180_degree_train_is_the_cpmg_decay():
    amps = echo_amplitudes(180.0, 12, 10.0, 1000.0, 80.0)
    expected = torch.tensor([math.exp(-n * 10.0 / 80.0) for n in range(1, 13)])
    torch.testing.assert_close(amps, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(echo_weights(180.0, 12, 10.0, 1000.0, 80.0), torch.ones(12))


@pytest.mark.parametrize("alpha", [100.0, 120.0, 150.0])
def test_first_two_echoes_match_the_hand_derived_epg_values(alpha):
    # No relaxation, 90-degree excitation along x, refocusing along y (CPMG): the first echo is
    # sin^2(a/2); the second adds the stimulated pathway, sin^2(a/2) (1 + cos^2(a/2)).
    s2, c2 = math.sin(math.radians(alpha) / 2) ** 2, math.cos(math.radians(alpha) / 2) ** 2
    amps = echo_amplitudes(alpha, 2, 10.0, FOREVER, FOREVER)
    torch.testing.assert_close(amps, torch.tensor([s2, s2 * (1 + c2)]), rtol=1e-5, atol=1e-6)


def test_120_degrees_gives_the_known_second_echo():
    amps = echo_amplitudes(120.0, 2, 10.0, FOREVER, FOREVER)
    torch.testing.assert_close(amps, torch.tensor([0.75, 0.9375]), rtol=1e-5, atol=1e-6)


def test_the_first_echo_falls_as_the_refocusing_angle_drops():
    first = [float(echo_weights(a, 8, 10.0, 1000.0, 80.0)[0]) for a in (180.0, 150.0, 120.0)]
    assert first[0] > first[1] > first[2]


def _delta(shape, *, volume=False):
    """A unit impulse at the centre of each 2-D plane (or volume): flat unit-modulus k-space."""
    x = torch.zeros(shape, dtype=torch.complex64)
    if volume:
        x[..., shape[-3] // 2, shape[-2] // 2, shape[-1] // 2] = 1.0
    else:
        x[..., shape[-2] // 2, shape[-1] // 2] = 1.0
    return x


@pytest.mark.parametrize("te_ms,centre_echo", [(None, 4), (20.0, 2), (80.0, 8)])
def test_kspace_lines_carry_the_echo_weights_of_a_2d_train(te_ms, centre_echo):
    h, etl, theta = 32, 8, 0.7
    kw = {"etl": etl} | ({} if te_ms is None else {"te_ms": te_ms})
    out = apply_degradation(NAME, _delta((1, 2, h, 32)), theta, **kw)
    got = fftnc(out, dims=(-2, -1)).abs()[0, 0, :, 0] * math.sqrt(h * 32)
    w = echo_weights(180.0 - 120.0 * theta, etl, 10.0, 1000.0, 80.0)
    group = torch.arange(h) * etl // h
    expected = w[(group - group[h // 2] + centre_echo - 1) % etl]
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_multislice_gives_every_slice_its_own_ky_train():
    geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=VOXEL)
    shape = (1, 1, 3, 16, 16)
    out = apply_degradation(NAME, _delta(shape), 1.0, geometry=geom, etl=4)
    k = fftnc(out, dims=(-2, -1)).abs()
    assert torch.allclose(k[:, :, 0], k[:, :, 1]) and torch.allclose(k[:, :, 1], k[:, :, 2])
    assert float(k[0, 0, 0, :, 0].max() / k[0, 0, 0, :, 0].min()) > 1.05


def test_volume_gives_every_kz_partition_its_own_ky_train():
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL)
    d, h, etl = 4, 8, 4
    out = apply_degradation(NAME, _delta((1, 1, d, h, 8), volume=True), 1.0, geometry=geom, etl=etl)
    got = fftnc(out, dims=(-3, -2, -1)).abs()[0, 0, :, :, 0] * math.sqrt(d * h * 8)
    w = echo_weights(60.0, etl, 10.0, 1000.0, 80.0)
    group = torch.arange(h) * etl // h
    row = w[(group - group[h // 2] + etl // 2 - 1) % etl]
    # An impulse at the centre partition has flat |kz| spectrum, so each partition shows the row.
    torch.testing.assert_close(got, row.expand(d, h), rtol=1e-4, atol=1e-5)


def test_severity_is_monotone_with_the_effective_echo_from_a_profile_te():
    p = dataclasses.replace(
        FASTMRI_3T, sequence=Sequence.TSE, etl=16, esp_ms=10.0, te_ms=80.0, tr_ms=3000.0
    )
    x = phantom(None)
    res = [rms(x, apply_degradation(NAME, x, t, profile=p)) for t in THETAS]
    assert all(a < b for a, b in pairwise(res)), res


def test_a_profile_without_a_refocusing_train_raises():
    gre = dataclasses.replace(FASTMRI_3T, sequence=Sequence.GRE, fa_deg=15.0, tr_ms=20.0)
    with pytest.raises(ValueError, match="refocused sequence"):
        apply_degradation(NAME, torch.rand(1, 1, 16, 16), 0.5, profile=gre)


def test_profile_conditions_the_train():
    x = torch.rand(1, 1, 32, 32)
    p = dataclasses.replace(
        FASTMRI_3T, sequence=Sequence.TSE, etl=8, esp_ms=12.0, te_ms=36.0, tr_ms=2000.0
    )
    torch.testing.assert_close(
        apply_degradation(NAME, x, 0.6, profile=p),
        apply_degradation(NAME, x, 0.6, geometry=p.geometry, etl=8, esp_ms=12.0, te_ms=36.0),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="outside"):
        apply_degradation(NAME, x, 0.6, etl=8, esp_ms=10.0, te_ms=500.0)


@pytest.mark.parametrize("bad", [{"etl": 0}, {"etl": 2.5}, {"esp_ms": 0.0}, {"t2_ms": -1.0}])
def test_bad_train_arguments_raise(bad):
    with pytest.raises(ValueError):
        apply_degradation(NAME, torch.rand(1, 1, 32, 32), 0.5, **bad)


def test_a_train_longer_than_the_line_count_leaves_echoes_unused():
    x = torch.rand(1, 1, 8, 8)
    out = apply_degradation(NAME, x, 0.5, etl=16)
    assert out.shape == x.shape and not torch.allclose(out.abs(), x)


def test_the_measured_dip_near_theta_0_2_to_0_3_is_pinned():
    # rms falls from 0.0069 at theta 0.20 to 0.0045 at 0.30, then rises: stimulated-echo recovery.
    x = phantom(None)
    thetas = [round(0.05 * i, 2) for i in range(21)]
    r = dict(zip(thetas, [rms(x, apply_degradation(NAME, x, t)) for t in thetas], strict=True))
    assert r[0.2] == pytest.approx(0.00685, rel=0.03) and r[0.3] == pytest.approx(0.00452, rel=0.03)
    assert all(r[a] < r[b] for a, b in pairwise([t for t in thetas if t <= 0.2]))
    assert all(r[a] > r[b] for a, b in pairwise([0.2, 0.25, 0.3]))
    assert all(r[a] < r[b] for a, b in pairwise([t for t in thetas if t >= 0.35]))
