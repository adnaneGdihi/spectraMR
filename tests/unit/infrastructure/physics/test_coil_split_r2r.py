"""Null-space R2R: two independent halves from ONE multi-coil acquisition.

Every statistical claim is measured on noise drawn with a known covariance, and
each guard that claims to matter has a planted violation that the same
measurement catches.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.coil_split_r2r import (
    check_split_angle,
    draw_null_direction,
    null_residual,
    prepare_coil_split,
    split_coil_images,
    split_halves,
    split_kspace,
    split_kspace_prepared,
    unit_directions,
    whitener,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

_C, _H, _W = 4, 96, 96
_COMMITTED = single_repetition_covariance().to(torch.complex128)


def _smooth_maps() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _H), torch.linspace(-1, 1, _W), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    maps = torch.stack(
        [
            torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) * torch.exp(1j * k * xx)
            for k, (a, b) in enumerate(centres)
        ]
    ).to(torch.complex128)
    return (maps / maps.abs().pow(2).sum(0, keepdim=True).sqrt()).unsqueeze(0)


def _anatomy() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _H), torch.linspace(-1, 1, _W), indexing="ij")
    return ((xx**2 + yy**2) < 0.5).to(torch.complex128) * 5.0


def _noise(cov: torch.Tensor, batch: int = 1, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, _C, _H, _W)
    white = torch.complex(
        torch.randn(shape, generator=gen, dtype=torch.float64),
        torch.randn(shape, generator=gen, dtype=torch.float64),
    ) / math.sqrt(2)
    return torch.einsum("ij,bjhw->bihw", torch.linalg.cholesky(cov), white)


def _combine(maps: torch.Tensor, coil_images: torch.Tensor) -> torch.Tensor:
    return (maps.conj() * coil_images).sum(1)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a * b.conj()).mean().abs()
    return float(num / (a.abs().pow(2).mean() * b.abs().pow(2).mean()).sqrt())


def _split(y, maps, cov, angle=45.0, seed=5):
    return split_coil_images(
        y,
        maps,
        whitening=whitener(cov, _C),
        angle_deg=angle,
        generator=torch.Generator().manual_seed(seed),
    )


class TestTheCleanComponent:
    @pytest.mark.parametrize("cov", [None, _COMMITTED], ids=["none", "committed"])
    @pytest.mark.parametrize("angle", [20.0, 45.0, 70.0])
    def test_a_noiseless_rank_one_image_round_trips(self, cov, angle) -> None:
        """(iv) For y = s m, each half re-expands to s (s^H y) = y exactly."""
        maps = _smooth_maps()
        clean = maps * _anatomy()
        first, second = _split(clean, maps, cov, angle)
        assert torch.allclose(first, clean, atol=1e-10)
        assert torch.allclose(second, clean, atol=1e-10)

    def test_the_split_is_linear_so_both_halves_carry_the_same_clean_image(self) -> None:
        """(i) With v fixed by the generator, split(x + n) = split(x) + split(n)."""
        maps = _smooth_maps()
        clean, noise = maps * _anatomy(), _noise(torch.eye(_C, dtype=torch.complex128))
        whole = _split(clean + noise, maps, None)
        parts = _split(noise, maps, None)
        for half, noise_half in zip(whole, parts, strict=True):
            assert torch.allclose(half - noise_half, clean, atol=1e-10)


class TestTheNoiseSplit:
    def test_white_noise_halves_are_uncorrelated(self) -> None:
        """(i) Exact maps, white noise, W = I: the halves' noise cross-covariance ~ 0."""
        maps = _smooth_maps()
        first, second = _split(_noise(torch.eye(_C, dtype=torch.complex128)), maps, None)
        assert _corr(_combine(maps, first), _combine(maps, second)) < 0.02

    @pytest.mark.parametrize("angle", [30.0, 45.0])
    def test_each_half_carries_its_r2r_share_of_the_sense_variance(self, angle) -> None:
        """(ii) 1/cos^2 and 1/sin^2 of the SENSE-combined variance: 2x each at 45."""
        maps = _smooth_maps()
        noise = _noise(torch.eye(_C, dtype=torch.complex128))
        first, second = _split(noise, maps, None, angle)
        sense = _combine(maps, noise).abs().pow(2).mean()
        theta = math.radians(angle)
        for half, share in ((first, 1 / math.cos(theta) ** 2), (second, 1 / math.sin(theta) ** 2)):
            ratio = float(_combine(maps, half).abs().pow(2).mean() / sense)
            assert ratio == pytest.approx(share, rel=0.03), (angle, ratio, share)


class TestWhiteningNeedsOnlyTheShape:
    """Sigma = g^2 * committed: `committed` stays exact, `none` does not."""

    @staticmethod
    def _cross_correlation(gain: float, cov: torch.Tensor | None) -> float:
        # Maps along Sigma's leading eigenvector: the geometry where an
        # unwhitened split is furthest from orthogonal in the noise metric.
        top = torch.linalg.eigh(_COMMITTED).eigenvectors[:, -1]
        maps = top.view(1, _C, 1, 1).expand(1, _C, _H, _W).clone()
        first, second = _split(_noise(_COMMITTED * gain**2), maps, cov)
        return _corr(_combine(maps, first), _combine(maps, second))

    @pytest.mark.parametrize("gain", [0.5, 3.0])
    def test_a_pure_gain_error_costs_nothing_under_committed_whitening(self, gain) -> None:
        assert self._cross_correlation(gain, _COMMITTED) < 0.02

    @pytest.mark.parametrize("gain", [0.5, 3.0])
    def test_without_whitening_the_same_noise_leaves_correlated_halves(self, gain) -> None:
        """The planted violation: W = I on M4Raw's coupled, unequal coils."""
        assert self._cross_correlation(gain, None) > 0.2


class TestTheInputKeepsTheRemainingNullDirections:
    """The input half carries C-1 whitened noise directions; the target half one."""

    @staticmethod
    def _basis(maps: torch.Tensor, cov: torch.Tensor, seed: int = 5):
        w = whitener(cov, _C)
        a = torch.einsum("ij,bjhw->bihw", w, unit_directions(maps))
        s_tilde = a / a.abs().pow(2).sum(1, keepdim=True).sqrt()
        return w, s_tilde, draw_null_direction(s_tilde, torch.Generator().manual_seed(seed))

    def test_the_input_half_keeps_exactly_the_residual_null_noise(self) -> None:
        """P_perp{s, v} W y_1 == P_perp{s, v} W y: nothing else is added or lost."""
        maps = _smooth_maps()
        y = maps * _anatomy() + _noise(_COMMITTED)
        first, _ = _split(y, maps, _COMMITTED)
        w, s_tilde, v = self._basis(maps, _COMMITTED)
        whiten = lambda x: torch.einsum("ij,bjhw->bihw", w, x)  # noqa: E731
        got, want = null_residual(whiten(first), s_tilde, v), null_residual(whiten(y), s_tilde, v)
        assert torch.allclose(got, want, atol=1e-9)
        assert float(want.abs().pow(2).mean()) > 0.0

    def test_the_target_half_has_no_null_space_component(self) -> None:
        """The clean signal is zero off the coil direction, so its target is too."""
        maps = _smooth_maps()
        y = maps * _anatomy() + _noise(_COMMITTED)
        _, second = self._split_with_basis(y, maps)
        w, s_tilde, v = self._basis(maps, _COMMITTED)
        whitened = torch.einsum("ij,bjhw->bihw", w, second)
        assert torch.allclose(
            null_residual(whitened, s_tilde, v), torch.zeros_like(whitened), atol=1e-9
        )
        assert torch.allclose(
            (v.conj() * whitened).sum(1), torch.zeros(1, _H, _W, dtype=whitened.dtype), atol=1e-9
        )

    @staticmethod
    def _split_with_basis(y, maps):
        return _split(y, maps, _COMMITTED)

    def test_the_residual_does_not_touch_the_signal_direction_half(self) -> None:
        """s^H W y_1 is m_1 exactly, so the variance shares of (ii) are unchanged."""
        maps = _smooth_maps()
        y = maps * _anatomy() + _noise(_COMMITTED)
        first, _ = _split(y, maps, _COMMITTED)
        w, s_tilde, v = self._basis(maps, _COMMITTED)
        whitened = torch.einsum("ij,bjhw->bihw", w, y)
        m1, _ = split_halves(whitened, s_tilde, v, math.radians(45.0))
        got = (s_tilde.conj() * torch.einsum("ij,bjhw->bihw", w, first)).sum(1)
        assert torch.allclose(got, m1, atol=1e-9)

    def test_the_residual_is_independent_of_the_target_noise(self) -> None:
        """Isotropic after whitening: the kept directions do not leak into y_2."""
        maps = _smooth_maps()
        first, second = _split(_noise(_COMMITTED, seed=3), maps, _COMMITTED)
        w, s_tilde, v = self._basis(maps, _COMMITTED)
        residual = null_residual(torch.einsum("ij,bjhw->bihw", w, first), s_tilde, v)
        target = (s_tilde.conj() * torch.einsum("ij,bjhw->bihw", w, second)).sum(1)
        for c in range(_C):
            assert _corr(residual[:, c], target) < 0.02, c


class TestTheNullDirectionIsSmooth:
    """Band-limited noise is correlated across pixels, so v must not jump between them."""

    @staticmethod
    def _neighbour_correlation(v_per_pixel: bool) -> float:
        noise_k = fft2c(_noise(torch.eye(_C, dtype=torch.complex128), batch=4, seed=3))
        noise_k[..., 73:] = 0  # ~24% of phase-encode columns never acquired
        noise = ifft2c(noise_k)
        s_hat = unit_directions(_smooth_maps()).expand(4, -1, -1, -1)
        if v_per_pixel:
            gen = torch.Generator().manual_seed(9)
            g = torch.complex(
                torch.randn(noise.shape, generator=gen, dtype=torch.float64),
                torch.randn(noise.shape, generator=gen, dtype=torch.float64),
            )
            v = g - s_hat * (s_hat.conj() * g).sum(1, keepdim=True)
            v = v / v.abs().pow(2).sum(1, keepdim=True).sqrt()
        else:
            v = draw_null_direction(s_hat, torch.Generator().manual_seed(9))
        first, second = split_halves(noise, s_hat, v, math.radians(45.0))
        return _corr(first[..., :-1], second[..., 1:])

    def test_the_drawn_direction_decorrelates_neighbouring_pixels(self) -> None:
        assert self._neighbour_correlation(v_per_pixel=False) < 0.02

    def test_a_per_pixel_direction_would_leak_across_the_phase_encode_axis(self) -> None:
        """The planted violation: independent per pixel, correlated with its neighbour."""
        assert self._neighbour_correlation(v_per_pixel=True) > 0.08

    def test_v_is_a_unit_vector_orthogonal_to_the_coil_direction(self) -> None:
        s_hat = unit_directions(_smooth_maps())
        v = draw_null_direction(s_hat, torch.Generator().manual_seed(1))
        assert torch.allclose(v.abs().pow(2).sum(1), torch.ones(1, _H, _W, dtype=torch.float64))
        assert float((s_hat.conj() * v).sum(1).abs().max()) < 1e-10


class TestKSpaceAndGuards:
    def test_unsampled_columns_stay_exactly_zero(self) -> None:
        maps = _smooth_maps()
        kspace = fft2c(maps * _anatomy() + _noise(torch.eye(_C, dtype=torch.complex128)))
        support = torch.ones(_W, dtype=torch.bool)
        support[73:] = False
        kspace = kspace * support
        halves = split_kspace(
            kspace, maps, support, whitening=whitener(_COMMITTED, _C), angle_deg=45.0
        )
        for half in halves:
            assert torch.count_nonzero(half[..., 73:]) == 0
            assert torch.count_nonzero(half[..., :73]) > 0

    @pytest.mark.parametrize("angle", [0.0, 90.0, -5.0, 120.0])
    def test_an_angle_outside_the_open_quarter_turn_raises(self, angle) -> None:
        """(iii) At either end one half divides by zero."""
        with pytest.raises(ValueError, match=r"outside \(0, 90\)"):
            check_split_angle(angle)

    def test_the_whitener_whitens(self) -> None:
        w = whitener(_COMMITTED, _C)
        assert torch.allclose(w @ _COMMITTED @ w.conj().T, torch.eye(_C, dtype=torch.complex128))

    def test_a_covariance_that_cannot_whiten_raises(self) -> None:
        singular = torch.zeros(_C, _C, dtype=torch.complex128)
        with pytest.raises(ValueError, match="positive-definite"):
            whitener(singular, _C)

    def test_pixels_outside_the_support_get_a_unit_direction(self) -> None:
        maps = _smooth_maps()
        maps[..., :10, :] = 0
        s_hat = unit_directions(maps)
        assert torch.allclose(s_hat.abs().pow(2).sum(1), torch.ones(1, _H, _W, dtype=torch.float64))
        assert torch.allclose(s_hat[..., 0, 0], s_hat[..., 5, 7])


def test_prepared_split_reuse_is_bit_identical_to_independent_calls():
    """Reusing one PreparedSplit across draws changes no value of the per-call path."""
    maps = _smooth_maps()
    y = fft2c(_anatomy() * maps + _noise(_COMMITTED))
    support = torch.ones(_W, dtype=torch.bool)
    w = whitener(_COMMITTED, _C)
    prepared = prepare_coil_split(ifft2c(y), maps, whitening=w)
    ref_gen = torch.Generator().manual_seed(7)
    new_gen = torch.Generator().manual_seed(7)
    for _ in range(3):
        expected, _unused = split_kspace(
            y, maps, support, whitening=w, angle_deg=45.0, generator=ref_gen
        )
        got = split_kspace_prepared(prepared, support, angle_deg=45.0, generator=new_gen)
        assert torch.equal(got, expected)
