"""Per-scan coil calibration from ONE repetition (#2322).

Pins the three things the self-calibrated R2R draw and the null-space split
both rest on: the slice axis is ESPIRiT's batch axis, the image-domain fit is
corrected by the sampled phase-encode fraction, and a support that is not one
contiguous band is refused.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.coil_noise_fit import estimate_covariance_from_nullspace
from spectramr.infrastructure.physics.coil_sensitivity import estimate_csm_espirit
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance
from spectramr.infrastructure.physics.scan_coil_calibration import (
    NonBandSupportError,
    as_slice_batch,
    calibrate_scan,
    fit_scan_covariance,
    pe_band_fraction,
)

_SIZE = 128
_ZERO_COLS = 31  # 97 of 128 sampled: M4Raw's ~24% partial phase FOV


def _volume(slices: int = 2, zero_cols: int = _ZERO_COLS) -> torch.Tensor:
    """``(S, 4, H, W)`` k-space at the committed noise level, PE tail zero-filled."""
    gen = torch.Generator().manual_seed(11)
    chol = torch.linalg.cholesky(single_repetition_covariance().to(torch.complex128))
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, _SIZE), torch.linspace(-1, 1, _SIZE), indexing="ij"
    )
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
    sens = sens * torch.exp(1j * torch.stack([k * xx for k in range(4)]))
    sens = (sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex128)
    anatomy = ((xx**2 + yy**2) < 0.5).double() * 30.0
    white = (
        torch.complex(
            torch.randn(slices, 4, _SIZE, _SIZE, generator=gen, dtype=torch.float64),
            torch.randn(slices, 4, _SIZE, _SIZE, generator=gen, dtype=torch.float64),
        )
        / 2**0.5
    )
    kspace = fft2c(sens * anatomy + torch.einsum("ij,bjhw->bihw", chol, white))
    if zero_cols:
        kspace[..., _SIZE - zero_cols :] = 0
    return kspace.to(torch.complex64)


def _support(kspace: torch.Tensor) -> torch.Tensor:
    return kspace.abs().sum(dim=(0, 1, 2)) > 0


def _relative_error(fitted: torch.Tensor) -> float:
    truth = single_repetition_covariance().to(torch.complex128)
    return float(torch.linalg.norm(fitted.to(torch.complex128) - truth) / torch.linalg.norm(truth))


class TestTheSlicesAreTheBatchAxis:
    def test_flattening_them_into_coils_is_rank_deficient(self) -> None:
        """The #2322 crash, reproduced: 3 slices x 4 coils read as 12 "coils".

        M4Raw's 18 slices make it 72; 12 is already past the 361 patches a
        24x24 ACS offers against 36 unknowns per coil.
        """
        volume = _volume(slices=3)
        flattened = volume.reshape(-1, _SIZE, _SIZE).unsqueeze(0)
        with pytest.raises(ValueError, match="rank-deficient"):
            estimate_csm_espirit(flattened, num_coils=flattened.shape[1])

    def test_one_map_per_slice_and_one_covariance_per_scan(self) -> None:
        volume = _volume()
        calibration = calibrate_scan(volume, _support(volume), fit_covariance=True)
        assert calibration.maps.shape == volume.shape
        assert calibration.covariance is not None
        assert calibration.covariance.shape == (4, 4)

    def test_a_single_slice_is_accepted(self) -> None:
        assert as_slice_batch(_volume(slices=1)[0]).shape == (1, 4, _SIZE, _SIZE)

    @pytest.mark.parametrize(
        "bad",
        [torch.zeros(1, 1, 4, 8, 8, dtype=torch.complex64), torch.zeros(4, 8, 8)],
        ids=["five_d", "real"],
    )
    def test_anything_without_a_coil_axis_where_espirit_reads_it_raises(self, bad) -> None:
        with pytest.raises((ValueError, TypeError)):
            as_slice_batch(bad)


class TestTheSampledFractionCorrection:
    def test_the_fit_recovers_the_single_repetition_sigma(self) -> None:
        volume = _volume()
        fitted = fit_scan_covariance(volume, _support(volume))
        assert _relative_error(fitted) < 0.03

    def test_without_it_the_fit_is_low_by_the_sampled_fraction(self) -> None:
        """The planted defect: the raw image-domain fit reads f * Sigma."""
        volume = _volume()
        subspace = estimate_csm_espirit(volume, num_coils=4, return_subspace=True)
        raw = estimate_covariance_from_nullspace(ifft2c(volume), subspace.null_projector)
        truth = single_repetition_covariance().to(torch.complex128)
        ratio = float(raw.diagonal().real.sum() / truth.diagonal().real.sum())
        fraction = (_SIZE - _ZERO_COLS) / _SIZE
        assert ratio == pytest.approx(fraction, abs=0.03), (ratio, fraction)

    def test_a_fully_sampled_scan_is_left_as_fitted(self) -> None:
        volume = _volume(zero_cols=0)
        calibration = calibrate_scan(volume, _support(volume), fit_covariance=True)
        assert calibration.sampled_fraction == 1.0
        assert calibration.covariance is not None
        assert _relative_error(calibration.covariance) < 0.03


class TestTheSupportMustBeABand:
    def test_the_band_fraction(self) -> None:
        support = torch.zeros(10, dtype=torch.bool)
        support[2:9] = True
        assert pe_band_fraction(support) == pytest.approx(0.7)

    @pytest.mark.parametrize(
        "support",
        [
            torch.tensor([True, False, True, True]),
            torch.tensor([False, False, False, False]),
        ],
        ids=["gap", "empty"],
    )
    def test_a_gap_or_nothing_is_refused(self, support) -> None:
        with pytest.raises(NonBandSupportError):
            pe_band_fraction(support)

    def test_scattered_undersampling_is_refused_before_any_fit(self) -> None:
        volume = _volume(zero_cols=0)
        volume[..., 3::4] = 0
        with pytest.raises(NonBandSupportError, match="contiguous band"):
            calibrate_scan(volume, _support(volume), fit_covariance=True)

    def test_a_support_of_the_wrong_width_is_refused(self) -> None:
        volume = _volume()
        with pytest.raises(ValueError, match="phase-encode columns"):
            calibrate_scan(volume, torch.ones(7, dtype=torch.bool), fit_covariance=False)


def _undersampled(gain: float = 1.0, seed: int = 0, size: int = 256) -> torch.Tensor:
    """One textured 4-coil slice, 25 % random columns around a 24-column ACS run.

    M4Raw's matrix width, so ``f_ACS = 24/256`` and the Dirichlet kernel is the
    ~11 px one Proposition 14 is about.
    """
    gen = torch.Generator().manual_seed(seed)
    truth = single_repetition_covariance().to(torch.complex128)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
    sens = sens * torch.exp(1j * torch.stack([k * xx for k in range(4)]))
    sens = (sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex128)
    texture = 0.6 + 0.4 * torch.rand(size, size, generator=gen, dtype=torch.float64)
    anatomy = ((xx**2 + yy**2) < 0.5).double() * 30.0 * gain * texture
    white = (
        torch.complex(
            torch.randn(1, 4, size, size, generator=gen, dtype=torch.float64),
            torch.randn(1, 4, size, size, generator=gen, dtype=torch.float64),
        )
        / 2**0.5
    )
    chol = torch.linalg.cholesky(truth)
    kspace = fft2c(sens * anatomy + torch.einsum("ij,bjhw->bihw", chol, white))
    keep = torch.rand(size, generator=torch.Generator().manual_seed(1000 + seed)) < 0.25
    c = size // 2
    keep[c - 12 : c + 12] = True
    keep[c - 13] = False
    keep[c + 12] = False
    return (kspace * keep).to(torch.complex64)


def _trace_ratio(covariance: torch.Tensor) -> float:
    truth = single_repetition_covariance().to(torch.complex128)
    return float(
        covariance.to(torch.complex128).diagonal().real.sum() / truth.diagonal().real.sum()
    )


class TestTheAcsPolicy:
    """``support_policy="acs"`` (PAPER.md Proposition 14) on random undersampling."""

    @pytest.mark.parametrize("seed", [0, 1])
    def test_the_acs_fit_recovers_the_trace_at_the_committed_level(self, seed) -> None:
        """Within 3 %: measured 0.991-1.009 over four noise draws (mean 1.003)."""
        volume = _undersampled(seed=seed)
        calibration = calibrate_scan(
            volume, _support(volume), fit_covariance=True, support_policy="acs"
        )
        assert calibration.covariance is not None
        assert _trace_ratio(calibration.covariance) == pytest.approx(1.0, abs=0.03)
        assert calibration.fit_fraction == pytest.approx(24 / 256)
        assert calibration.support_policy == "acs"
        assert 0.05 < calibration.leak_bound < 0.2

    def test_the_whole_support_fit_still_raises_by_default(self) -> None:
        volume = _undersampled()
        with pytest.raises(NonBandSupportError, match="contiguous band"):
            calibrate_scan(volume, _support(volume), fit_covariance=True)

    def test_the_leak_bound_is_logged(self, caplog) -> None:
        import logging

        volume = _undersampled()
        with caplog.at_level(logging.INFO):
            calibrate_scan(volume, _support(volume), fit_covariance=True, support_policy="acs")
        assert any("leak bound" in r.message for r in caplog.records)

    def test_a_band_is_fitted_whole_under_acs_too(self) -> None:
        volume = _volume()
        band = calibrate_scan(volume, _support(volume), fit_covariance=True)
        acs = calibrate_scan(volume, _support(volume), fit_covariance=True, support_policy="acs")
        assert acs.leak_bound is None and band.leak_bound is None
        torch.testing.assert_close(acs.covariance, band.covariance)

    def test_the_bound_is_on_amplitude_so_high_snr_still_leaks(self) -> None:
        """Not an accuracy claim: the documented limit, pinned so no reader takes the
        ACS policy as SNR-independent. At ten times the signal the ACS fit reads
        ~2.6x the true trace (the whole random support ~4.4x)."""
        volume = _undersampled(gain=10.0)
        calibration = calibrate_scan(
            volume, _support(volume), fit_covariance=True, support_policy="acs"
        )
        assert _trace_ratio(calibration.covariance) > 1.5
