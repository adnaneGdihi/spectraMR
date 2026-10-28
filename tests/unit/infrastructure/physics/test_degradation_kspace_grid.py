"""Regression tests for the degradations that built k-space grids off the fft2c grid,
and for those whose random draw or scale depended on the batch or the grid size."""

from __future__ import annotations

import pytest
import torch

import spectramr.infrastructure.physics.degradations.slice2d.fields as fields
from spectramr.infrastructure.physics.degradations import degrade_iq_ghost, degrade_susceptibility
from spectramr.infrastructure.physics.degradations.slice2d.motion import _apply_pe_line_shift
from spectramr.infrastructure.physics.digital_twin_simulator import (
    simulate_chemical_shift,
    simulate_gradient_nonlinearity,
    simulate_rigid_motion_kspace,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


def _img(n: int, b: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(b, 1, n, n, generator=g, dtype=torch.complex64)


@pytest.mark.parametrize("n", [64, 65])
def test_pe_line_shift_by_an_integer_is_a_roll(n: int) -> None:
    x = _img(n)
    out = _apply_pe_line_shift(x, torch.full((n,), 7.0))
    assert torch.allclose(out, torch.roll(x, 7, dims=-1), atol=1e-5)


@pytest.mark.parametrize("n", [64, 65])
def test_chemical_shift_moves_fat_by_exactly_the_declared_pixels(n: int) -> None:
    x = _img(n)
    k = simulate_chemical_shift(fft2c(x), (n, n), shift_pixels=7.0, fat_fraction=1.0)
    assert torch.allclose(ifft2c(k), torch.roll(x, 7, dims=-1), atol=1e-5)


@pytest.mark.parametrize("n", [64, 65, 128])
def test_susceptibility_field_has_no_discarded_imaginary_part(
    n: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dipole kernel must be Hermitian on the fft2c grid; on an even linspace grid
    the imaginary part dropped by ``.real`` was ~half the real part."""
    ratios: list[float] = []
    orig = fields.ifft2c

    def spy(k: torch.Tensor) -> torch.Tensor:
        out = orig(k)
        ratios.append((out.imag.norm() / out.real.norm()).item())
        return out

    monkeypatch.setattr(fields, "ifft2c", spy)
    degrade_susceptibility(_img(n), 1.0, seed=1)
    assert ratios[0] < 1e-5


def test_rigid_motion_sample_does_not_depend_on_batch_size() -> None:
    x = _img(64, b=2)
    one = simulate_rigid_motion_kspace(fft2c(x[:1]), generator=torch.Generator().manual_seed(5))
    two = simulate_rigid_motion_kspace(fft2c(x), generator=torch.Generator().manual_seed(5))
    assert torch.equal(two[:1], one)


def test_iq_ghost_spike_is_set_per_sample() -> None:
    x = _img(64, b=2)
    x[1] *= 10
    assert torch.equal(
        degrade_iq_ghost(x, 1.0, seed=0)[:1], degrade_iq_ghost(x[:1], 1.0, seed=0)
    )


@pytest.mark.parametrize("n", [64, 65])
def test_iq_ghost_mirrors_through_the_fov_centre(n: int) -> None:
    c = n // 2
    imp = torch.zeros(1, 1, n, n, dtype=torch.complex64)
    imp[..., c + 5, c - 3] = 1.0
    ghost = (degrade_iq_ghost(imp, 1.0, seed=0, max_dc_offset=0.0) - imp).abs().squeeze()
    ghost[c + 5, c - 3] = 0.0
    assert divmod(int(ghost.argmax()), n) == (c - 5, c + 3)


def test_gradient_nonlinearity_displacement_is_a_fixed_fraction_of_the_fov() -> None:
    peaks = []
    for n in (64, 256):
        _, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
        ramp = xx.view(1, 1, n, n).to(torch.complex64)
        torch.manual_seed(11)
        out = simulate_gradient_nonlinearity(ramp, (n, n), severity=7e-5)
        t = n // 8
        peaks.append((out.real - ramp.real).abs()[..., t:-t, t:-t].max().item())
    assert peaks[0] == pytest.approx(peaks[1], rel=0.02)
