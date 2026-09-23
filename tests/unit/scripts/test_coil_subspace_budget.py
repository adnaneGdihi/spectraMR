"""The budget script reads 1 on a repetition, which is what makes it a check.

A single M4Raw repetition is exactly "clean image plus one noise draw", so the
budget's mean over the ESPIRiT support must come out at 1 -- not by convention
but because the denominator is the expectation of the null-space residual under
the committed covariance. That one number validates the projector, the
covariance and the scale simultaneously, which is why the script exists and why
it reports per study series: the covariance is scoped to one series
(20220610xx) and a drift elsewhere shows up here as a mean away from 1.

``budget_for_slice`` is exercised directly rather than through the CLI. The
cluster has the data; this pins the arithmetic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/analysis/coil_subspace_budget.py"


def _load():
    spec = importlib.util.spec_from_file_location("coil_subspace_budget", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUDGET = _load()

_C, _SIZE = 4, 64


@pytest.fixture(scope="module")
def slice_parts() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, _SIZE), torch.linspace(-1, 1, _SIZE), indexing="ij"
    )
    magnitude = ((xx**2 + yy**2) < 0.6).float() * (1.0 + 0.3 * torch.cos(4 * xx))
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres]).to(
        torch.complex64
    )
    sens = sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt().clamp(min=1e-6)
    covariance = single_repetition_covariance().to(torch.complex64)
    clean = sens * magnitude.to(torch.complex64) * (covariance.diagonal().real.mean().sqrt() * 8)
    return {
        "clean": clean,
        "covariance": covariance,
        "cholesky": torch.linalg.cholesky(covariance.to(torch.complex128)).to(torch.complex64),
    }


def _noisy_repetition(parts: dict[str, torch.Tensor], seed: int) -> torch.Tensor:
    """Image-domain clean + one coloured noise draw, returned as k-space."""
    torch.manual_seed(seed)
    white = (torch.randn(_SIZE * _SIZE, _C) + 1j * torch.randn(_SIZE * _SIZE, _C)).to(
        torch.complex64
    )
    noise = (
        ((white / 2**0.5) @ parts["cholesky"].transpose(-1, -2))
        .reshape(_SIZE, _SIZE, _C)
        .permute(2, 0, 1)
    )
    return fft2c(parts["clean"] + noise)


def _call(kspace: torch.Tensor, parts: dict[str, torch.Tensor], **kwargs):
    return BUDGET.budget_for_slice(
        kspace,
        parts["covariance"],
        acs_size=kwargs.pop("acs_size", 24),
        kernel_size=kwargs.pop("kernel_size", 6),
        eigen_threshold=kwargs.pop("eigen_threshold", 0.95),
        **kwargs,
    )


class TestTheSelfValidatingNumber:
    def test_a_repetition_reads_one(self, slice_parts) -> None:
        """Clean + one draw is the null hypothesis the denominator encodes."""
        means = [_call(_noisy_repetition(slice_parts, seed), slice_parts)[0] for seed in range(6)]
        mean = sum(means) / len(means)
        assert 0.90 < mean < 1.10, f"budget is not calibrated on a repetition: {mean}"

    def test_a_noise_free_slice_reads_near_zero(self, slice_parts) -> None:
        """A rank-one signal has no null-space component at all."""
        result = _call(fft2c(slice_parts["clean"]), slice_parts)
        assert result is not None
        assert result[0] < 0.05

    def test_a_wrong_scale_shows_up_as_a_wrong_budget(self, slice_parts) -> None:
        """The failure the `scale` argument exists to prevent, end to end.

        Serve the data at half scale and leave the covariance alone and the
        budget falls by four -- a plausible number, and silently wrong.
        """
        kspace = _noisy_repetition(slice_parts, seed=11)
        honest = _call(kspace / 2.0, slice_parts, scale=2.0)[0]
        naive = _call(kspace / 2.0, slice_parts)[0]
        assert 0.85 < honest < 1.15, honest
        assert naive < 0.4, naive
        assert honest / naive > 3.0


class TestTheRankAssumptionIsReported:
    def test_the_rank_fraction_is_returned_and_in_range(self, slice_parts) -> None:
        result = _call(_noisy_repetition(slice_parts, seed=3), slice_parts)
        assert result is not None
        assert 0.0 <= result[3] <= 1.0

    def test_a_looser_tolerance_cannot_report_more_violations(self, slice_parts) -> None:
        """Monotone in the tolerance -- otherwise the flag is not measuring rank."""
        kspace = _noisy_repetition(slice_parts, seed=4)
        strict = _call(kspace, slice_parts, rank_one_tol=0.2)[3]
        loose = _call(kspace, slice_parts, rank_one_tol=0.8)[3]
        assert loose <= strict


class TestTheReportingContract:
    def test_a_background_only_slice_returns_none_rather_than_nan(self, slice_parts) -> None:
        """No support is a property of the slice, not an error, and not a number."""
        empty = fft2c(torch.zeros(_C, _SIZE, _SIZE, dtype=torch.complex64))
        assert _call(empty, slice_parts) is None

    @pytest.mark.parametrize(
        ("file_id", "expected"),
        [
            ("2022061007_T101", "202206"),
            ("2022070112_FLAIR02", "202207"),
            ("odd_name", "unknown"),
        ],
    )
    def test_series_grouping(self, file_id: str, expected: str) -> None:
        """The per-series split is the whole point; a wrong key merges the groups."""
        assert BUDGET.series_of(file_id) == expected

    def test_the_script_reads_through_the_dataset_not_h5py(self) -> None:
        """Non-negotiable 7: one owner for file -> tensor."""
        source = SCRIPT.read_text()
        assert "h5py" not in source
        assert "M4RawRepetitionDataset" in source

    def test_the_coil_mode_preserves_the_coil_vector(self) -> None:
        """Any combination destroys the measurement this script is built on."""
        source = SCRIPT.read_text()
        assert 'coil_processing_mode="none"' in source


class TestThePerScanCovarianceDrift:
    """The fifth element answers the question the committed constant cannot.

    ``m4raw_noise.py`` scopes its covariance to one study series and says
    receiver gain may differ elsewhere. R2R's recorruption is exact only where
    ``Sigma_z == Sigma_n``, so reporting the per-scan fit beside the constant is
    what tells you where that holds.
    """

    def test_it_is_small_when_the_constant_describes_the_data(self, slice_parts) -> None:
        result = _call(_noisy_repetition(slice_parts, seed=21), slice_parts)
        assert result is not None
        drift = result[4]
        assert drift == drift, "drift came back NaN on identifiable data"
        assert drift < 0.30, drift

    def test_it_is_large_when_the_data_has_a_different_covariance(self, slice_parts) -> None:
        """A scan whose noise is 3x the constant must not read as consistent."""
        torch.manual_seed(22)
        white = (torch.randn(_SIZE * _SIZE, _C) + 1j * torch.randn(_SIZE * _SIZE, _C)).to(
            torch.complex64
        )
        loud = (
            ((white / 2**0.5) @ (slice_parts["cholesky"] * 3.0).transpose(-1, -2))
            .reshape(_SIZE, _SIZE, _C)
            .permute(2, 0, 1)
        )
        result = _call(fft2c(slice_parts["clean"] + loud), slice_parts)
        assert result is not None
        assert result[4] > 1.0, result[4]

    def test_the_script_reports_it(self) -> None:
        """Computing it and not printing it would be a measurement nobody sees."""
        source = SCRIPT.read_text()
        assert "estimate_covariance_from_nullspace" in source
        assert "Sigma drift" in source
        assert "_sigma_drift" in source
