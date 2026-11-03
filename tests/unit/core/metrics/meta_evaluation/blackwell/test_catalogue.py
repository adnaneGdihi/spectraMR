"""The Fourier-diagonal catalogue: profiles, realisation, shells and mixtures."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell.catalogue import (
    FAMILIES,
    DiagonalExperiment,
    build_catalogue,
    info_groups,
    mixture,
    radial_frequency,
)

pytestmark = pytest.mark.unit
SHAPE = (32, 32)


def test_default_catalogue_is_five_families_by_five_severities() -> None:
    cat = build_catalogue(SHAPE)
    assert len(cat) == 25
    assert {e.family for e in cat} == set(FAMILIES)
    assert all(e.log_info.dtype == torch.float64 and e.log_info.shape == SHAPE for e in cat)


def test_realisation_recovers_the_declared_noise_and_unit_peak_gain() -> None:
    cat = {e.name: e for e in build_catalogue(SHAPE, base_noise=0.1)}
    assert cat["noise_0.2"].noise_std == pytest.approx(0.2)
    assert cat["blur_2"].noise_std == pytest.approx(0.1)  # unit gain at DC
    assert cat["blur_low_noise_2"].noise_std == pytest.approx(0.05)
    for e in cat.values():
        assert float(e.log_transfer().max()) == pytest.approx(0.0)


def test_truncation_leaves_high_frequencies_unacquired() -> None:
    e = next(e for e in build_catalogue(SHAPE) if e.name == "truncation_0.15")
    rho = radial_frequency(SHAPE)
    assert torch.isneginf(e.log_info[rho > 0.15]).all()
    assert torch.isfinite(e.log_info[rho <= 0.15]).all()


def test_strong_blurs_stay_distinct_below_float32_range() -> None:
    """Two blurs that both underflow in float32 must still differ in the profile."""
    cat = {e.name: e for e in build_catalogue(SHAPE)}
    a, b = cat["blur_2"].log_info, cat["blur_3"].log_info
    corner = (0, 0)
    assert math.exp(float(b[corner])) < 1e-38  # beyond float32
    assert float(a[corner]) > float(b[corner])


def test_every_profile_is_constant_on_each_info_group() -> None:
    cat = build_catalogue(SHAPE)
    groups = info_groups(cat).reshape(-1)
    for e in cat:
        flat = e.log_info.reshape(-1)
        for g in torch.unique(groups)[:60]:
            vals = flat[groups == g]
            assert (vals == vals[0]).all()


def test_info_groups_split_on_a_non_radial_profile() -> None:
    cat = build_catalogue(SHAPE)
    radial = int(info_groups(cat).max()) + 1
    line = cat[0].log_info.clone()
    line[:, 3] = -math.inf  # one dropped column breaks radial symmetry
    masked = DiagonalExperiment("line", "mask", 0.0, line)
    assert int(info_groups([*cat, masked]).max()) + 1 > radial


def test_default_transfers_zero_fill_unacquired_coefficients() -> None:
    e = next(e for e in build_catalogue(SHAPE) if e.name == "truncation_0.15")
    h, g = e.transfers()
    unacquired = torch.isneginf(e.log_info)
    assert (h[unacquired] == 0).all() and (g[unacquired] == 0).all()
    assert (g[~unacquired] == 1).all()


def test_explicit_transfers_and_sigma_take_precedence() -> None:
    log_info = torch.zeros(SHAPE, dtype=torch.float64)
    h = torch.full(SHAPE, 0.5, dtype=torch.float64)
    e = DiagonalExperiment("x", "x", 0.0, log_info, h, h, 0.5)
    assert e.noise_std == 0.5 and torch.equal(e.transfers()[0], h)


def test_mixture_information_is_the_time_shared_average() -> None:
    cat = {e.name: e for e in build_catalogue(SHAPE)}
    a, b = cat["noise_0.05"], cat["truncation_0.2"]
    mix = mixture(a, b, 0.25)
    expect = 0.25 * torch.exp(a.log_info) + 0.75 * torch.exp(b.log_info)
    assert torch.allclose(torch.exp(mix.log_info), expect, rtol=1e-12)
    with pytest.raises(ValueError):
        mixture(a, b, 1.0)


def test_invalid_noise_arguments_raise() -> None:
    with pytest.raises(ValueError):
        build_catalogue(SHAPE, base_noise=0.0)
    with pytest.raises(ValueError):
        build_catalogue(SHAPE, low_noise_factor=1.5)
