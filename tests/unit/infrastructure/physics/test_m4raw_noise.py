"""Tests for the measured M4Raw noise model and its R2R recorruption.

Every test here is synthetic and seeded: the M4Raw archive is not available to
CI, and the properties under test are algebraic, so they do not need it. The
constants themselves were measured against real data (7 subjects, 32 repetition
pairs); what these tests pin is that the module still IMPLEMENTS them.
"""

import pytest
import torch

from spectramr.infrastructure.physics.m4raw_noise import (
    M4RAW_NUM_COILS,
    M4RawNoiseSampler,
    _coil_covariance,
    coil_sigmas,
    estimate_covariance_from_repetitions,
    measured_difference_covariance,
    single_repetition_covariance,
)

# Per-coil, per-component sigma of one repetition, pooled over 32 phase-aligned
# repetition pairs on the outer k-space band.
MEASURED_SIGMAS = (3.777, 3.265, 4.033, 3.877)


def _full_mask(h: int = 32, w: int = 32) -> torch.Tensor:
    return torch.ones(h, w, dtype=torch.bool)


def _sample_covariance(x: torch.Tensor) -> torch.Tensor:
    """``E[x x^H]`` across the coil axis of a ``(..., C, H, W)`` tensor."""
    return _coil_covariance(x.to(torch.complex128), _full_mask(*x.shape[-2:]))


class TestMeasuredConstants:
    def test_coil_sigmas_reproduce_the_measured_values(self) -> None:
        assert coil_sigmas().tolist() == pytest.approx(MEASURED_SIGMAS, abs=1e-3)

    def test_single_rep_covariance_is_half_the_difference_covariance(self) -> None:
        """The factor that cannot be got wrong quietly.

        A wrong factor leaves ``Cov(input, target) = (Sigma_n - Sigma_z) != 0``
        and training still converges -- to a fixed point part-way to the
        identity map. Nothing downstream reports it.
        """
        diff = measured_difference_covariance(dtype=torch.complex128)
        single = single_repetition_covariance(dtype=torch.complex128)
        assert torch.allclose(single * 2.0, diff)

    def test_covariance_is_hermitian(self) -> None:
        cov = single_repetition_covariance(dtype=torch.complex128)
        assert torch.allclose(cov, cov.conj().T)

    def test_covariance_is_positive_definite(self) -> None:
        torch.linalg.cholesky(single_repetition_covariance(dtype=torch.complex128))

    def test_inter_coil_coupling_carries_phase(self) -> None:
        """A real-only covariance would drop most of the strongest coupling.

        Coils 0-3 correlate at |rho| = 0.25 with ~55 deg of phase, so the
        imaginary part is load-bearing rather than decorative.
        """
        cov = single_repetition_covariance(dtype=torch.complex128)
        entry = cov[0, 3]
        assert entry.imag.abs() > entry.real.abs()

    def test_declared_coil_count_matches_the_matrix(self) -> None:
        assert single_repetition_covariance().shape == (M4RAW_NUM_COILS, M4RAW_NUM_COILS)


class TestSampler:
    def test_draw_reproduces_the_covariance(self) -> None:
        """Cholesky closed loop, including the off-diagonal coupling."""
        sampler = M4RawNoiseSampler()
        gen = torch.Generator().manual_seed(0)
        z = sampler.sample((400, M4RAW_NUM_COILS, 32, 32), _full_mask(), generator=gen)
        empirical = _sample_covariance(z)
        expected = single_repetition_covariance(dtype=torch.complex128)
        assert torch.allclose(empirical, expected, rtol=0.05, atol=0.15)

    def test_difference_of_two_draws_reproduces_the_measured_matrix(self) -> None:
        """The end-to-end pin on the /2 factor, in the direction it was measured."""
        sampler = M4RawNoiseSampler()
        gen = torch.Generator().manual_seed(1)
        mask = _full_mask()
        a = sampler.sample((400, M4RAW_NUM_COILS, 32, 32), mask, generator=gen)
        b = sampler.sample((400, M4RAW_NUM_COILS, 32, 32), mask, generator=gen)
        empirical = _sample_covariance(b - a)
        expected = measured_difference_covariance(dtype=torch.complex128)
        assert torch.allclose(empirical, expected, rtol=0.05, atol=0.3)

    def test_noise_is_circular(self) -> None:
        sampler = M4RawNoiseSampler()
        gen = torch.Generator().manual_seed(2)
        z = sampler.sample((200, M4RAW_NUM_COILS, 32, 32), _full_mask(), generator=gen)
        for c in range(M4RAW_NUM_COILS):
            re, im = z[:, c].real.flatten(), z[:, c].imag.flatten()
            assert re.var().item() / im.var().item() == pytest.approx(1.0, abs=0.05)
            corr = (re * im).mean() / (re.std() * im.std())
            assert corr.abs().item() < 0.02

    def test_unsampled_columns_stay_exactly_zero(self) -> None:
        """Roughly 24 % of M4Raw phase-encode columns are exact zeros.

        Writing noise into one fabricates a line the scanner never acquired.
        """
        mask = _full_mask()
        mask[:, 8:16] = False
        gen = torch.Generator().manual_seed(3)
        z = M4RawNoiseSampler().sample((4, M4RAW_NUM_COILS, 32, 32), mask, generator=gen)
        assert torch.count_nonzero(z[..., 8:16]) == 0
        assert torch.count_nonzero(z[..., :8]) > 0

    def test_draws_are_reproducible_under_a_seeded_generator(self) -> None:
        sampler = M4RawNoiseSampler()
        shape = (2, M4RAW_NUM_COILS, 8, 8)
        first = sampler.sample(shape, _full_mask(8, 8), torch.Generator().manual_seed(4))
        second = sampler.sample(shape, _full_mask(8, 8), torch.Generator().manual_seed(4))
        assert torch.equal(first, second)


class TestRecorruption:
    @staticmethod
    def _decorrelation_residual(sampler: M4RawNoiseSampler, seed: int = 5) -> float:
        """``max|Cov(input, target)|`` normalised by the noise variance.

        Measured on pure noise: with signal present, ``E|x|^2`` survives the
        cross-covariance and the statistic measures signal power instead --
        which shows up as a residual that varies with alpha, since alpha cancels
        from the true identity.
        """
        gen = torch.Generator().manual_seed(seed)
        truth = M4RawNoiseSampler()
        y = truth.sample((300, M4RAW_NUM_COILS, 32, 32), _full_mask(), generator=gen)
        inp, tgt = sampler.recorrupt(y, _full_mask(), generator=gen)
        c = inp.shape[-3]
        a = inp.movedim(-3, 0).reshape(c, -1).to(torch.complex128)
        b = tgt.movedim(-3, 0).reshape(c, -1).to(torch.complex128)
        cross = a @ b.conj().T / a.shape[1]
        return (cross.abs().max() / _sample_covariance(y).diagonal().real.mean()).item()

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_halves_decorrelate_for_every_alpha(self, alpha: float) -> None:
        """``Cov(y + a*z, y - z/a) = Sigma_n - Sigma_z = 0``, independent of a.

        Alpha cancels from the identity; it only trades variance between the two
        halves. A residual that TRACKS alpha means the statistic is measuring
        something else (signal power, typically), not a broken model.
        """
        assert self._decorrelation_residual(M4RawNoiseSampler(alpha=alpha)) < 0.06

    @pytest.mark.parametrize(
        ("factor", "why"),
        [(0.5, "halved once too often"), (2.0, "not halved"), (1.06, "6 % gain drift")],
    )
    def test_a_wrong_covariance_is_detected(self, factor: float, why: str) -> None:
        """Planted violations: the decorrelation check must go red for each.

        A detector is only a detector for the violation shape it has been
        watched failing on (CLAUDE.md non-negotiable 15). The 1.06 case is the
        one that matters in practice -- it is the size of a receiver-gain
        difference elsewhere in the corpus.
        """
        wrong = M4RawNoiseSampler(covariance=single_repetition_covariance() * factor)
        assert self._decorrelation_residual(wrong) > 0.06, why

    def test_variance_split_follows_alpha(self) -> None:
        """Input carries ``(1 + a^2) Sigma``; target carries ``(1 + 1/a^2) Sigma``."""
        gen = torch.Generator().manual_seed(6)
        y = M4RawNoiseSampler().sample((300, M4RAW_NUM_COILS, 32, 32), _full_mask(), generator=gen)
        alpha = 2.0
        inp, tgt = M4RawNoiseSampler(alpha=alpha).recorrupt(y, _full_mask(), generator=gen)
        base = _sample_covariance(y).diagonal().real
        assert (_sample_covariance(inp).diagonal().real / base).mean().item() == pytest.approx(
            1 + alpha**2, rel=0.1
        )
        assert (_sample_covariance(tgt).diagonal().real / base).mean().item() == pytest.approx(
            1 + 1 / alpha**2, rel=0.1
        )

    def test_a_shared_artefact_survives_in_both_halves(self) -> None:
        """R2R is a second-moment identity; it does not remove a common spike.

        Both halves carry the artefact with the SAME sign, so the minimiser is
        ``x + s`` and the network is trained to KEEP it. This is why the cohort
        also trains a real-repetition-pair arm.
        """
        y = torch.zeros(1, M4RAW_NUM_COILS, 32, 32, dtype=torch.complex64)
        y[0, 0, 16, 16] = 500.0 + 0j
        inp, tgt = M4RawNoiseSampler().recorrupt(
            y, _full_mask(), generator=torch.Generator().manual_seed(7)
        )
        assert inp[0, 0, 16, 16].real > 400.0
        assert tgt[0, 0, 16, 16].real > 400.0


class TestCoilCovarianceReduction:
    def test_mask_rank_does_not_change_the_result(self) -> None:
        """Regression: the coil axis must not depend on the mask's rank.

        Boolean fancy-indexing collapses a different number of trailing axes for
        a 1-D column mask than for a 2-D one, so indexing BEFORE moving the coil
        axis to the front silently reduced over the readout axis instead and
        returned a near-uniform, wrong covariance.
        """
        gen = torch.Generator().manual_seed(8)
        z = M4RawNoiseSampler().sample((60, M4RAW_NUM_COILS, 32, 32), _full_mask(), generator=gen)
        columns = torch.zeros(32, dtype=torch.bool)
        columns[4:20] = True
        two_d = torch.zeros(32, 32, dtype=torch.bool)
        two_d[:, 4:20] = True
        from_1d = _coil_covariance(z.to(torch.complex128), columns)
        from_2d = _coil_covariance(z.to(torch.complex128), two_d)
        assert torch.allclose(from_1d, from_2d)
        assert from_1d.shape == (M4RAW_NUM_COILS, M4RAW_NUM_COILS)


class TestReMeasurement:
    def test_round_trips_the_constant_from_a_synthetic_repetition_pair(self) -> None:
        """Two synthetic repetitions of one signal must recover ``Sigma_n``."""
        sampler = M4RawNoiseSampler()
        gen = torch.Generator().manual_seed(9)
        mask = _full_mask()
        signal = torch.randn(200, M4RAW_NUM_COILS, 32, 32, generator=gen) * 50.0
        signal = signal.to(torch.complex64)
        rep_a = signal + sampler.sample(signal.shape, mask, generator=gen)
        rep_b = signal + sampler.sample(signal.shape, mask, generator=gen)
        recovered = estimate_covariance_from_repetitions(rep_a, rep_b, mask)
        expected = single_repetition_covariance(dtype=torch.complex128)
        assert torch.allclose(recovered, expected, rtol=0.1, atol=0.4)


class TestGuards:
    def test_non_positive_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be > 0"):
            M4RawNoiseSampler(alpha=0.0)

    def test_real_covariance_raises(self) -> None:
        cov = single_repetition_covariance().real
        with pytest.raises(TypeError, match="covariance must be complex"):
            M4RawNoiseSampler(covariance=cov)

    def test_non_square_covariance_raises(self) -> None:
        with pytest.raises(ValueError, match="square matrix"):
            M4RawNoiseSampler(covariance=torch.zeros(2, 3, dtype=torch.complex64))

    def test_coil_count_mismatch_raises(self) -> None:
        """A coil-combined tensor must not broadcast into the wrong noise model."""
        with pytest.raises(ValueError, match="does not match the covariance"):
            M4RawNoiseSampler().sample((2, 1, 32, 32), _full_mask())

    def test_non_complex_kspace_raises(self) -> None:
        with pytest.raises(TypeError, match="expects complex k-space"):
            M4RawNoiseSampler().recorrupt(torch.zeros(1, 4, 32, 32), _full_mask())

    def test_non_bool_mask_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a bool tensor"):
            M4RawNoiseSampler().sample((2, M4RAW_NUM_COILS, 32, 32), torch.ones(32, 32))
