"""chemical_shift_phase: fat-water interference |W + F exp(i 2 pi df TE)|."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.chemical_shift_phase import (
    fat_offset_hz,
    swept_te_ms,
)
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

from .invariants import GEOMETRIES, assert_axis_invariants, phantom

M = AcquisitionMode
NAME = "chemical_shift_phase"


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


def test_declaration():
    spec = DEGRADATION_REGISTRY[NAME]
    assert all(r is Realisation.NATIVE for r in spec.realisation.values())
    assert set(spec.implementations) == set(M) and spec.requires == frozenset()


@pytest.mark.parametrize(
    "b0,df,te_in,te_opp", [(3.0, -447.06, 2.2368, 3.3552), (1.5, -223.53, 4.4736, 6.7104)]
)
def test_in_and_opposed_phase_te(b0, df, te_in, te_opp):
    assert fat_offset_hz(b0) == pytest.approx(df, abs=0.05)
    assert swept_te_ms(0.0, b0) == pytest.approx(te_in, abs=1e-3)
    assert swept_te_ms(1.0, b0) == pytest.approx(te_opp, abs=1e-3)
    # In phase, an echo is a whole number of fat-water cycles; opposed phase, a half.
    assert abs(fat_offset_hz(b0)) * swept_te_ms(0.0, b0) * 1e-3 == pytest.approx(1.0)
    assert abs(fat_offset_hz(b0)) * swept_te_ms(1.0, b0) * 1e-3 == pytest.approx(1.5)


def test_opposed_phase_with_half_fat_is_zero():
    for geom in ("slice_2d", "volume"):
        g = GEOMETRIES[geom]
        x = phantom(g)
        fat = torch.full((1, 1, *x.shape[-2:]) if g is None else (1, 1, *x.shape[-3:]), 0.5)
        out = apply_degradation(NAME, x, 1.0, geometry=g, fat_map=fat)
        assert out.abs().max() < 1e-6


@pytest.mark.parametrize("b0", [1.5, 3.0])
@pytest.mark.parametrize("theta", [0.25, 0.7, 1.0])
def test_matches_direct_complex_sum_and_keeps_phase(b0, theta):
    g = GEOMETRIES["volume"]
    x = phantom(g)
    f = torch.rand(1, 1, *x.shape[-3:], generator=torch.Generator().manual_seed(0))
    out = apply_degradation(NAME, x, theta, geometry=g, b0_T=b0, fat_map=f)
    phi = 2 * math.pi * fat_offset_hz(b0) * swept_te_ms(theta, b0) * 1e-3
    direct = (1 - f) * x + f * x * complex(math.cos(phi), math.sin(phi))
    torch.testing.assert_close(out.abs(), direct.abs(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out.angle(), x.angle(), atol=1e-6, rtol=0)


def test_opposed_phase_fallback_map_is_bounded_declared_and_seeded():
    x = phantom(None)
    out = apply_degradation(NAME, x, 1.0, seed=2, fat_fraction=0.4)
    gain = out.abs() / x.abs()
    # Peak fat fraction 0.4: the opposed-phase gain is bounded below by |1 - 2 f_max|.
    assert gain.min() == pytest.approx(0.2, abs=1e-5)  # map spans [0, 0.4]
    assert apply_degradation(NAME, x, 0.0, seed=2, fat_fraction=0.4) is not None
    assert torch.equal(apply_degradation(NAME, x, 0.0, seed=2), x)
    again = apply_degradation(NAME, x, 1.0, seed=2, fat_fraction=0.4)
    assert torch.equal(out, again)
    assert not torch.equal(out, apply_degradation(NAME, x, 1.0, seed=3, fat_fraction=0.4))


def test_profile_sets_fat_fraction_and_field():
    x = phantom(None)
    base = apply_degradation(NAME, x, 1.0, seed=1)
    fatty = apply_degradation(NAME, x, 1.0, seed=1, profile=replace(FASTMRI_3T, fat_fraction=0.4))
    assert (x - fatty).abs().mean() > (x - base).abs().mean()
    with pytest.raises(ValueError, match="fat_fraction"):
        apply_degradation(NAME, x, 1.0, seed=1, fat_fraction=0.2, profile=FASTMRI_3T)


@pytest.mark.parametrize(
    "kw", [{"fat_fraction": 1.5}, {"b0_T": 0.0}, {"fat_map": torch.full((1, 1, 1, 32, 32), 2.0)}]
)
def test_bad_inputs_raise(kw):
    with pytest.raises(ValueError):
        apply_degradation(NAME, phantom(None), 0.5, **kw)


def test_both_fat_axes_read_one_ppm():
    from spectramr.infrastructure.physics.scanner_profile import DEFAULT_CHI_PPM

    b0, bw = FASTMRI_3T.b0_T, FASTMRI_3T.readout_bw_hz_per_px
    # chemical_shift's px at the profile's chi_ppm is this axis's |df| / BW.
    assert FASTMRI_3T.chi_ppm == DEFAULT_CHI_PPM
    assert abs(fat_offset_hz(b0)) / bw == pytest.approx(FASTMRI_3T.chemical_shift_max_px())
    x = phantom(None)
    bare = apply_degradation(NAME, x, 0.6, seed=1, fat_fraction=FASTMRI_3T.fat_fraction)
    prof = apply_degradation(NAME, x, 0.6, seed=1, profile=FASTMRI_3T)
    torch.testing.assert_close(bare, prof, rtol=0, atol=0)


def test_non_default_chi_ppm_reaches_the_axis():
    x = phantom(None)
    p = replace(FASTMRI_3T, chi_ppm=3.0)
    assert abs(fat_offset_hz(p.b0_T, p.chi_ppm)) / p.readout_bw_hz_per_px == pytest.approx(
        p.chemical_shift_max_px()
    )
    via_profile = apply_degradation(NAME, x, 0.6, seed=1, profile=p)
    explicit = apply_degradation(NAME, x, 0.6, seed=1, fat_fraction=p.fat_fraction, sigma_ppm=3.0)
    torch.testing.assert_close(via_profile, explicit, rtol=0, atol=0)
    # TE is tuned to df, so the image is ppm-invariant by construction; the ppm shows in the TE.
    cond = DEGRADATION_REGISTRY[NAME].condition(p, 0.6, p.geometry, (1, 1, 8, 8))
    assert cond["sigma_ppm"] == 3.0
    assert swept_te_ms(0.6, p.b0_T, 3.0) != pytest.approx(swept_te_ms(0.6, p.b0_T), rel=1e-3)
