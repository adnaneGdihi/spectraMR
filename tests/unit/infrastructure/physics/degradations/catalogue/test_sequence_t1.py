"""t1_shot_blur: Mz recovery across an inversion-recovery readout train, as a k-space line filter."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

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
from spectramr.infrastructure.physics.degradations.catalogue.sequence_t1 import (
    DEFAULT_FA_DEG,
    mz_recursion,
)
from spectramr.infrastructure.physics.fft_ops import fftnc
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T
from spectramr.infrastructure.physics.scanner_profile import Sequence

from .invariants import GEOMETRIES, assert_axis_invariants

M = AcquisitionMode
NAME = "t1_shot_blur"
T1, TI, FA, TR = 1000.0, 900.0, 5.0, 7.0


def _rich(*shape):
    """Seeded complex image with a strong mean, so every k-space ratio is well conditioned."""
    g = torch.Generator().manual_seed(0)
    return 3 + 0.3 * torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _apply(x, theta, geometry=None, **kw):
    return apply_degradation(NAME, x, theta, seed=1, geometry=geometry, **kw)


def _hand_recursion(n_lines):
    """M_0 = 1 - 2 exp(-TI/T1), then M_{n+1} = M_ss + (M_n - M_ss) E1 cos(alpha), stepped line by line."""
    e1 = math.exp(-TR / T1)
    m_ss = (1 - e1) / (1 - e1 * math.cos(math.radians(FA)))
    m = [1 - 2 * math.exp(-TI / T1)]
    for _ in range(n_lines - 1):
        m.append(m_ss + (m[-1] - m_ss) * e1 * math.cos(math.radians(FA)))
    return m


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


def test_declaration():
    spec = DEGRADATION_REGISTRY[NAME]
    p = spec.severity.primary
    assert (p.units, p.at_theta_min, p.at_theta_max) == ("N_train x TR_ro / T1", 0.0, 1.0)
    assert spec.requires == frozenset() and spec.condition is not None
    assert spec.realisation[M.MULTISLICE_2D] is Realisation.PER_SLICE_EXACT


def test_weights_are_the_hand_recursion_at_n_0_1_5():
    h = 32
    theta = h * TR / T1  # N_train == lines, so pulse p == line n
    x = _rich(1, 1, h, 16)
    got = fftnc(_apply(x, theta, t1_ms=T1, ti_ms=TI, fa_deg=FA, tr_ro_ms=TR), (-2, -1))
    ratio = (got / fftnc(x, (-2, -1))).real[0, 0, :, 8]
    m = _hand_recursion(h)
    e1c = math.exp(-TR / T1) * math.cos(math.radians(FA))
    m_ss = (1 - math.exp(-TR / T1)) / (1 - e1c)
    for n in (0, 1, 5):
        assert float(ratio[n]) == pytest.approx(1 + (m[n] - m[h // 2]) / m_ss, rel=2e-4)
    assert float(ratio[h // 2]) == pytest.approx(1.0, abs=1e-5)  # centre line normalised


def test_recursion_closed_form_matches_the_stepped_recursion():
    p = torch.arange(40.0)
    torch.testing.assert_close(
        mz_recursion(p, T1, TI, FA, TR), torch.tensor(_hand_recursion(40), dtype=torch.float64)
    )


def test_inner_loop_is_ky_for_linear_and_kz_for_kz_inner():
    d, h = 6, 16
    x = _rich(1, 1, d, h, 8)
    for order, axis in ((LineOrder.LINEAR, "ky"), (LineOrder.KZ_INNER, "kz")):
        g = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None, line_order=order)
        ratio = (fftnc(_apply(x, 0.5, g), (-3, -2, -1)) / fftnc(x, (-3, -2, -1))).real[
            0, 0, :, :, 4
        ]
        varies_along_ky = float((ratio - ratio[:, :1]).abs().max()) > 1e-3  # ratio is [kz, ky]
        varies_along_kz = float((ratio - ratio[:1]).abs().max()) > 1e-3
        assert (varies_along_ky, varies_along_kz) == (
            (True, False) if axis == "ky" else (False, True)
        )


@pytest.mark.parametrize("t1", [1500.0, 4000.0])
def test_long_t1_sweeps_are_finite_and_monotone(t1):
    x = _rich(1, 1, 32, 32)
    devs = []
    for theta in torch.linspace(0.02, 1.0, 50).tolist():
        out = _apply(x, theta, t1_ms=t1)
        assert torch.isfinite(torch.view_as_real(out)).all()
        devs.append(float((out - x).norm()))
    assert all(a < b for a, b in pairwise(devs)), devs


def test_theta_zero_weights_are_one():
    x = torch.randn(1, 1, 16, 16)
    assert torch.equal(_apply(x, 0.0, t1_ms=4000.0), x)


def test_the_declared_default_flip_angle_is_the_documented_one():
    assert DEFAULT_FA_DEG == 5.0
    assert "fa_deg (5)" in DEGRADATION_REGISTRY[NAME].severity.notes


def test_a_profile_supplies_ti_flip_angle_and_readout_repetition():
    x = torch.randn(1, 1, 24, 24, dtype=torch.complex64)
    mp = replace(
        FASTMRI_3T, sequence=Sequence.MPRAGE, ti_ms=800.0, fa_deg=7.0, esp_ms=6.0, tr_ms=2300.0
    )
    want = _apply(x, 0.5, ti_ms=800.0, fa_deg=7.0, tr_ro_ms=6.0)
    assert torch.equal(apply_degradation(NAME, x, 0.5, profile=mp), want)
    gre = replace(mp, sequence=Sequence.GRE, tr_ms=9.0, te_ms=3.0)
    assert torch.equal(
        apply_degradation(NAME, x, 0.5, profile=gre),
        _apply(x, 0.5, ti_ms=800.0, fa_deg=7.0, tr_ro_ms=9.0),
    )


@pytest.mark.parametrize("sequence", [Sequence.SPIN_ECHO, Sequence.TSE, Sequence.IR_TSE])
def test_a_non_gradient_echo_profile_keeps_the_defaults_and_stamps_it(sequence):
    x = torch.randn(1, 1, 24, 24, dtype=torch.complex64)
    se = replace(FASTMRI_3T, sequence=sequence, ti_ms=800.0, fa_deg=90.0, etl=4, esp_ms=6.0)
    cond = DEGRADATION_REGISTRY[NAME].condition(se, 0.5, se.geometry, tuple(x.shape))
    assert cond == {"profile_ti_fa_ignored": str(sequence)}
    out = apply_degradation(NAME, x, 0.5, profile=se)  # fa 90 does not raise: it is not used
    assert torch.equal(out, _apply(x, 0.5))


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"fa_deg": 90.0}, "fa_deg"),
        ({"fa_deg": 0.0}, "fa_deg"),
        ({"t1_ms": -1.0}, "t1_ms"),
        ({"ti_ms": 0.0}, "ti_ms"),
        ({"tr_ro_ms": 0.0}, "tr_ro_ms"),
        ({"order": "random"}, "order"),
    ],
)
def test_bad_knobs_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        _apply(torch.randn(1, 1, 16, 16), 0.5, **kw)
