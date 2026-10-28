"""The coil-manifold identity, measured rather than assumed.

``||Pf - x||^2 + ||P_perp f||^2 = ||f - x||^2`` per pixel for any ``f`` and a
rank-one ``x`` is what makes both the projection (never increases error) and the
null-space residual (a reference-free lower bound) true. It only holds if ``P``
is a real projector at every pixel -- including where the maps vanish -- so that
edge is pinned too.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.coil_manifold import (
    as_complex_coils,
    as_interleaved,
    null_space_energy,
    project_kspace_onto_manifold,
    project_onto_manifold,
    require_complex_maps,
    unit_maps,
    weighted_mean,
)
from spectramr.infrastructure.physics.fft_ops import fft2c

_C, _N = 4, 24


def _maps(support_only: bool = False) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _N), torch.linspace(-1, 1, _N), indexing="ij")
    centres = [(-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7), (0.7, 0.7)]
    maps = torch.stack(
        [
            torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) * torch.exp(1j * (a * xx + b * yy))
            for a, b in centres
        ]
    ).to(torch.complex64)
    maps = maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()
    if support_only:
        maps = maps * ((xx**2 + yy**2) < 0.5)
    return maps.unsqueeze(0)


def _randn(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=gen), torch.randn(*shape, generator=gen))


def _rank_one(maps: torch.Tensor, seed: int = 0) -> torch.Tensor:
    return maps * _randn(1, 1, _N, _N, seed=seed)


def _per_pixel(z: torch.Tensor) -> torch.Tensor:
    return z.abs().pow(2).sum(1)


class TestTheIdentity:
    @pytest.mark.parametrize("support_only", [False, True], ids=["full-support", "vanishing-maps"])
    def test_projection_error_plus_residual_is_the_error(self, support_only: bool) -> None:
        maps = _maps(support_only)
        x = _rank_one(maps)
        f = _randn(1, _C, _N, _N, seed=1)
        residual, _ = null_space_energy(f, maps)
        lhs = _per_pixel(project_onto_manifold(f, maps) - x) + residual
        assert torch.allclose(lhs, _per_pixel(f - x), rtol=1e-4, atol=1e-4)

    def test_projecting_never_increases_the_error(self) -> None:
        maps = _maps(support_only=True)
        x = _rank_one(maps)
        f = x + 0.3 * _randn(1, _C, _N, _N, seed=2)
        assert (_per_pixel(project_onto_manifold(f, maps) - x) <= _per_pixel(f - x) + 1e-5).all()

    def test_the_projector_is_idempotent(self) -> None:
        maps = _maps(support_only=True)
        f = _randn(1, _C, _N, _N, seed=3)
        once = project_onto_manifold(f, maps)
        assert torch.allclose(project_onto_manifold(once, maps), once, atol=1e-5)

    def test_unit_maps_vanish_where_the_maps_do(self) -> None:
        unit, norm2 = unit_maps(_maps(support_only=True))
        outside = norm2 == 0
        assert outside.any()
        assert (unit.abs().sum(1)[outside] == 0).all()
        assert torch.allclose(unit.abs().pow(2).sum(1)[~outside], torch.ones(1), atol=1e-5)


class TestTheResidual:
    def test_zero_on_a_rank_one_image(self) -> None:
        maps = _maps()
        x = _rank_one(maps)
        residual, _ = null_space_energy(x, maps)
        assert residual.max() < 1e-5 * _per_pixel(x).max()

    def test_isotropic_noise_puts_c_minus_one_over_c_off_the_manifold(self) -> None:
        maps = _maps()
        noise = _randn(64, _C, _N, _N, seed=4)
        residual, weights = null_space_energy(noise, maps.expand(64, -1, -1, -1))
        fraction = weighted_mean(residual, weights) / weighted_mean(_per_pixel(noise), weights)
        assert fraction.item() == pytest.approx((_C - 1) / _C, rel=0.02)

    def test_a_zero_weight_total_is_nan_not_zero(self) -> None:
        """A zero would read as a perfect certificate."""
        assert torch.isnan(weighted_mean(torch.ones(2, 3), torch.zeros(2, 3)))


class TestLayoutsAndGuards:
    def test_interleaving_round_trips(self) -> None:
        z = _randn(2, _C, 5, 6, seed=5)
        real = as_interleaved(z)
        assert real.shape == (2, 2 * _C, 5, 6)
        assert torch.equal(real[:, 0], z[:, 0].real) and torch.equal(real[:, 1], z[:, 0].imag)
        assert torch.equal(as_complex_coils(real), z)

    def test_kspace_projection_keeps_the_layout_and_matches_the_image_one(self) -> None:
        maps = _maps(support_only=True)
        kspace = fft2c(_randn(1, _C, _N, _N, seed=6))
        projected = project_kspace_onto_manifold(as_interleaved(kspace), maps)
        assert projected.shape == (1, 2 * _C, _N, _N) and not torch.is_complex(projected)
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        expected = project_onto_manifold(ifft2c(kspace), maps)
        assert torch.allclose(ifft2c(as_complex_coils(projected)), expected, atol=1e-5)

    def test_an_odd_channel_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="odd channel count"):
            as_complex_coils(torch.zeros(1, 7, 4, 4))

    @pytest.mark.parametrize(
        ("maps", "match"),
        [
            (None, "none reached"),
            (torch.ones(1, _C, _N, _N), "COMPLEX"),
            (torch.ones(1, _C, _N, 3, dtype=torch.complex64), "disagree"),
        ],
        ids=["missing", "magnitude", "mis-shaped"],
    )
    def test_maps_are_checked(self, maps, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            require_complex_maps(
                maps, torch.zeros(1, _C, _N, _N, dtype=torch.complex64), metric="probe"
            )
