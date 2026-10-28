"""tse_t2_blur: exp(-TE_e / T2_eff) weights on the phase-encode lines of a turbo-spin-echo train."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.fft_ops import fftnc
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T
from spectramr.infrastructure.physics.scanner_profile import Sequence

from .invariants import GEOMETRIES, assert_axis_invariants

M = AcquisitionMode
NAME = "tse_t2_blur"
N, ETL, ESP, THETA = 32, 8, 10.0, 0.5


def _rich(*shape):
    """Seeded complex image with a strong mean, so every k-space ratio is well conditioned."""
    g = torch.Generator().manual_seed(0)
    return 3 + 0.3 * torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _apply(x, theta, geometry=None, **kw):
    return apply_degradation(NAME, x, theta, seed=1, geometry=geometry, **kw)


def _ratio(out, x, dims=(-2, -1)):
    return fftnc(out, dims) / fftnc(x, dims)


def _expected(echo_of_line):
    """exp(-(TE_e - TE_centre) / T2_eff) with TE_e = (e + 1) ESP, from the formula as written."""
    t2_eff = ETL * ESP / (2 * THETA)
    te = {n: (e + 1) * ESP for n, e in echo_of_line.items()}
    return {n: math.exp(-(t - te[N // 2]) / t2_eff) for n, t in te.items()}


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


def test_declaration():
    spec = DEGRADATION_REGISTRY[NAME]
    p = spec.severity.primary
    assert (p.units, p.at_theta_min, p.at_theta_max) == ("ETL x ESP / T2", 0.0, 2.0)
    assert spec.requires == frozenset() and spec.condition is not None
    assert spec.realisation[M.MULTISLICE_2D] is Realisation.PER_SLICE_EXACT


@pytest.mark.parametrize("order", ["linear", "centric"])
def test_pe_weighting_is_exactly_exp_minus_te_over_t2_eff(order):
    x = _rich(1, 1, N, N)
    if order == "linear":  # lines in ky order, ETL equal segments
        echo = {n: n * ETL // N for n in range(N)}
    else:  # by increasing |ky|, ties by index
        ranked = sorted(range(N), key=lambda n: (abs(n - N // 2), n))
        echo = {n: r * ETL // N for r, n in enumerate(ranked)}
    want = torch.tensor([_expected(echo)[n] for n in range(N)], dtype=torch.float64)
    got = _ratio(_apply(x, THETA, etl=ETL, esp_ms=ESP, order=order), x)
    torch.testing.assert_close(got.real[0, 0, :, N // 2].double(), want, rtol=2e-4, atol=1e-5)
    assert float(got.imag.abs().max()) < 1e-4
    assert float(got[0, 0, N // 2, 0].real) == pytest.approx(1.0, abs=1e-5)  # centre keeps TE_eff


def test_a_train_longer_than_the_lines_gives_each_line_its_own_echo():
    # 12 lines, ETL 16: lines take echoes 0..11 in acquisition order, the last 4 echoes unused.
    n, etl = 12, 16
    x = _rich(1, 1, n, n)
    got = _ratio(_apply(x, THETA, etl=etl, esp_ms=ESP, order="linear"), x).real[0, 0, :, n // 2]
    t2_eff = etl * ESP / (2 * THETA)
    want = torch.tensor([math.exp(-(k - n // 2) * ESP / t2_eff) for k in range(n)])
    torch.testing.assert_close(got.float(), want, rtol=2e-4, atol=1e-5)


def test_linear_is_asymmetric_and_centric_decays_from_the_centre():
    x = _rich(1, 1, N, N)
    lin = _ratio(_apply(x, 1.0, etl=ETL, esp_ms=ESP, order="linear"), x).real[0, 0, :, N // 2]
    cen = _ratio(_apply(x, 1.0, etl=ETL, esp_ms=ESP, order="centric"), x).real[0, 0, :, N // 2]
    assert lin[0] > 1.0 > lin[-1]  # edge enhancement on one side, attenuation on the other
    assert float(cen.max()) == pytest.approx(1.0, abs=1e-5)
    assert cen[0] < 0.5 and cen[-1] < 0.5  # low-pass: the blur


def test_volume_weights_the_ky_kz_plane_by_line_order():
    d, h, etl = 6, 16, 4
    x = _rich(1, 1, d, h, 8)
    for order, rank in (
        (LineOrder.LINEAR, lambda kz, ky: kz * h + ky),
        (LineOrder.KZ_INNER, lambda kz, ky: ky * d + kz),
    ):
        g = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0), line_order=order)
        got = _ratio(_apply(x, 0.5, g, etl=etl, esp_ms=ESP), x, (-3, -2, -1)).real[0, 0, :, :, 4]
        echo = lambda kz, ky, rank=rank: rank(kz, ky) * etl // (d * h)  # noqa: E731
        te_c = (echo(d // 2, h // 2) + 1) * ESP
        want = torch.tensor(
            [
                [
                    math.exp(-((echo(kz, ky) + 1) * ESP - te_c) * 2 * 0.5 / (etl * ESP))
                    for ky in range(h)
                ]
                for kz in range(d)
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(got.double(), want, rtol=2e-4, atol=1e-5)


def test_order_kwarg_overrides_the_volume_geometry():
    g = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None, line_order=LineOrder.CENTRIC)
    x = torch.randn(1, 1, 4, 16, 8)
    assert torch.equal(
        _apply(x, 0.5, g, order="linear"), _apply(x, 0.5, replace(g, line_order=LineOrder.LINEAR))
    )
    assert not torch.equal(_apply(x, 0.5, g), _apply(x, 0.5, g, order="linear"))


def test_a_tse_profile_supplies_etl_and_esp_and_other_sequences_leave_the_defaults():
    x = torch.randn(1, 1, 24, 24, dtype=torch.complex64)
    tse = replace(FASTMRI_3T, sequence=Sequence.TSE, etl=6, esp_ms=7.0)
    assert torch.equal(
        apply_degradation(NAME, x, 0.5, profile=tse), _apply(x, 0.5, etl=6, esp_ms=7.0)
    )
    gre = replace(FASTMRI_3T, sequence=Sequence.GRE, etl=6, esp_ms=7.0)
    assert torch.equal(apply_degradation(NAME, x, 0.5, profile=gre), _apply(x, 0.5))


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"etl": 1}, "etl"),
        ({"etl": 2.5}, "etl"),
        ({"esp_ms": 0.0}, "esp_ms"),
        ({"order": "spiral"}, "order"),
    ],
)
def test_bad_knobs_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        _apply(torch.randn(1, 1, 16, 16), 0.5, **kw)
