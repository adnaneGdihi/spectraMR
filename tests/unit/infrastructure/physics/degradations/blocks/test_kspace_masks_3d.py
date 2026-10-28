"""Exact-count (kz, ky) masks and partial-Fourier masks."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations.blocks.kspace_masks_3d import (
    cartesian_mask_3d,
    partial_fourier_mask_3d,
)

NZ, NY = 16, 32


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("density", ["uniform", "variable"])
@pytest.mark.parametrize("R", [1, 2, 4, 6.5])
def test_realised_acceleration_equals_declared(R, density):
    m = cartesian_mask_3d(NZ, NY, R, density=density, generator=_gen())
    assert m.dtype == torch.bool and m.shape == (NZ, NY)
    assert int(m.sum()) == max(2 * 8, round(NZ * NY / R))
    assert NZ * NY / int(m.sum()) == pytest.approx(R, rel=0.02)


def test_r1_all_true():
    assert cartesian_mask_3d(NZ, NY, 1.0).all()


@pytest.mark.parametrize("density", ["uniform", "variable"])
def test_acs_block_present_and_centred(density):
    m = cartesian_mask_3d(NZ, NY, 6.0, acs=(2, 8), density=density, generator=_gen())
    assert m[NZ // 2, NY // 2]
    z0, y0 = NZ // 2 - 1, NY // 2 - 4
    assert m[z0 : z0 + 2, y0 : y0 + 8].all()


def test_acs_larger_than_budget_gives_acs_only():
    m = cartesian_mask_3d(NZ, NY, 100.0, acs=(2, 8), generator=_gen())
    z0, y0 = NZ // 2 - 1, NY // 2 - 4
    expected = torch.zeros(NZ, NY, dtype=torch.bool)
    expected[z0 : z0 + 2, y0 : y0 + 8] = True
    assert torch.equal(m, expected)


def test_seeded_deterministic_and_seed_sensitive():
    a = cartesian_mask_3d(NZ, NY, 4, generator=_gen(1))
    assert torch.equal(a, cartesian_mask_3d(NZ, NY, 4, generator=_gen(1)))
    assert not torch.equal(a, cartesian_mask_3d(NZ, NY, 4, generator=_gen(2)))


def test_variable_density_denser_at_centre():
    acc = torch.zeros(NZ, NY)
    for s in range(40):
        acc += cartesian_mask_3d(
            NZ, NY, 4, acs=(1, 1), density="variable", alpha=4.0, generator=_gen(s)
        )
    centre = acc[NZ // 2 - 3 : NZ // 2 + 3, NY // 2 - 6 : NY // 2 + 6].mean()
    edge = torch.cat([acc[:3].flatten(), acc[-3:].flatten()]).mean()
    assert centre > 2 * edge


@pytest.mark.parametrize(
    "kwargs",
    [
        {"R": 0.5},
        {"R": 2, "acs": (0, 4)},
        {"R": 2, "acs": (99, 4)},
        {"R": 2, "density": "x"},
        {"R": 2, "alpha": -1},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        cartesian_mask_3d(NZ, NY, **kwargs)


def test_partial_fourier_axes_keep_low_side():
    ky = partial_fourier_mask_3d(10, 20, 0.7, axis="ky")
    assert ky[:, :14].all() and not ky[:, 14:].any()
    kz = partial_fourier_mask_3d(10, 20, 0.7, axis="kz")
    assert kz[:7].all() and not kz[7:].any()


def test_partial_fourier_both_is_intersection():
    m = partial_fourier_mask_3d(10, 20, 0.7, axis="ky+kz")
    assert int(m.sum()) == 7 * 14 and m[:7, :14].all()


@pytest.mark.parametrize("axis", ["ky", "kz", "ky+kz"])
def test_partial_fourier_fraction_one_all_true(axis):
    assert partial_fourier_mask_3d(10, 20, 1.0, axis=axis).all()


@pytest.mark.parametrize("fraction", [0.5, 0.3, 1.2])
def test_partial_fourier_bad_fraction_raises(fraction):
    with pytest.raises(ValueError):
        partial_fourier_mask_3d(10, 20, fraction, axis="ky")


def test_partial_fourier_bad_axis_raises():
    with pytest.raises(ValueError):
        partial_fourier_mask_3d(10, 20, 0.8, axis="kx")  # type: ignore[arg-type]
