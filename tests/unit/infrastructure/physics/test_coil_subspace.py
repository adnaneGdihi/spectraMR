"""The coil null space is a pure-noise channel, and the budget is calibrated.

Two claims carry every use of this module, and neither is safe to assume:

* a **physical** multi-coil image is rank one in coil space, so its null-space
  component is exactly zero, while noise is full rank and lands there;
* the budget's denominator is an **expectation**, so the mean of the statistic
  under pure noise is 1 -- not the median, which is below it because the null
  distribution is a right-skewed chi-squared.

Both are measured here against the real committed M4Raw covariance rather than
a hand-written fixture, because the whole construction fails quietly if the
covariance is wrong by a scale factor.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.coil_sensitivity import estimate_csm_espirit
from spectramr.infrastructure.physics.coil_subspace import (
    coil_subspace_energy_budget,
    coil_subspace_residual,
    expected_null_residual,
    scale_covariance,
)
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

_C, _H, _W = 4, 64, 64


@pytest.fixture(scope="module")
def phantom() -> dict[str, torch.Tensor]:
    """A rank-one multi-coil phantom and its ESPIRiT subspace."""
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _H), torch.linspace(-1, 1, _W), indexing="ij")
    magnitude = ((xx**2 + yy**2) < 0.6).float() * (1.0 + 0.3 * torch.cos(4 * xx))
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres]).to(
        torch.complex64
    )
    sens = sens / sens.abs().pow(2).sum(0, keepdim=True).sqrt().clamp(min=1e-6)

    covariance = single_repetition_covariance().to(torch.complex64)
    # Put the phantom at a realistic SNR against the measured covariance; the
    # budget is a ratio, but the ESPIRiT support depends on the signal level.
    clean = (sens * magnitude.to(torch.complex64)).unsqueeze(0) * (
        covariance.diagonal().real.mean().sqrt() * 8
    )
    subspace = estimate_csm_espirit(
        fft2c(clean[0]).unsqueeze(0), num_coils=_C, acs_size=24, return_subspace=True
    )
    return {
        "clean": clean,
        "covariance": covariance,
        "projector": subspace.null_projector,
        "support": subspace.leading_eigenvalue >= 0.95,
        "cholesky": torch.linalg.cholesky(covariance.to(torch.complex128)).to(torch.complex64),
    }


def _noise(cholesky: torch.Tensor) -> torch.Tensor:
    white = (torch.randn(_H * _W, _C) + 1j * torch.randn(_H * _W, _C)).to(torch.complex64)
    coloured = (white / 2**0.5) @ cholesky.transpose(-1, -2)
    return coloured.reshape(_H, _W, _C).permute(2, 0, 1).unsqueeze(0)


class TestTheProjectorIsAProjector:
    def test_it_is_hermitian_and_idempotent(self, phantom) -> None:
        inside = phantom["projector"][phantom["support"]]
        assert (inside - inside.conj().transpose(-2, -1)).abs().max() < 1e-6
        assert (inside @ inside - inside).abs().max() < 1e-5

    def test_its_rank_is_the_coil_count_minus_one(self, phantom) -> None:
        """A rank-one signal model leaves C-1 null directions per pixel."""
        inside = phantom["projector"][phantom["support"]]
        trace = inside.diagonal(dim1=-2, dim2=-1).sum(-1).real
        assert torch.allclose(trace, torch.full_like(trace, _C - 1.0), atol=1e-4)


class TestTheAsymmetryTheWholeThingRestsOn:
    def test_a_physical_image_has_no_null_component(self, phantom) -> None:
        """x = s*m is rank one in coil space, so P x is zero identically."""
        support = phantom["support"]
        residual = coil_subspace_residual(phantom["clean"], phantom["projector"])
        energy = phantom["clean"].abs().pow(2).sum(1)
        assert (residual[support].sum() / energy[support].sum()) < 1e-4

    def test_noise_lands_mostly_in_the_null_space(self, phantom) -> None:
        """Roughly (C-1)/C of it, because the noise is near-isotropic in coils."""
        torch.manual_seed(1)
        noise = _noise(phantom["cholesky"])
        support = phantom["support"]
        residual = coil_subspace_residual(noise, phantom["projector"])
        energy = noise.abs().pow(2).sum(1)
        fraction = (residual[support].sum() / energy[support].sum()).item()
        assert 0.6 < fraction < 0.9, f"expected ~{(_C - 1) / _C}, got {fraction}"


class TestTheBudgetIsCalibrated:
    def test_pure_noise_has_unit_mean(self, phantom) -> None:
        """The denominator is an expectation, so this is the claim that matters."""
        torch.manual_seed(2)
        means = []
        for _ in range(12):
            budget = coil_subspace_energy_budget(
                _noise(phantom["cholesky"]),
                phantom["projector"],
                phantom["covariance"],
                support=phantom["support"],
            )
            means.append(budget[phantom["support"]].mean().item())
        mean = torch.tensor(means).mean().item()
        assert 0.93 < mean < 1.07, f"budget is not calibrated: mean {mean}"

    def test_the_median_sits_below_one(self, phantom) -> None:
        """Anti-misreading: 0.87 is the null median, not evidence of over-smoothing."""
        torch.manual_seed(3)
        budget = coil_subspace_energy_budget(
            _noise(phantom["cholesky"]),
            phantom["projector"],
            phantom["covariance"],
            support=phantom["support"],
        )
        median = budget[phantom["support"]].median().item()
        assert 0.80 < median < 0.95

    def test_a_noise_free_image_reads_far_below_one(self, phantom) -> None:
        budget = coil_subspace_energy_budget(
            phantom["clean"],
            phantom["projector"],
            phantom["covariance"],
            support=phantom["support"],
        )
        assert budget[phantom["support"]].mean().item() < 0.05

    def test_extra_noise_raises_it_proportionally(self, phantom) -> None:
        """Two independent draws carry twice the variance, so E roughly doubles."""
        torch.manual_seed(4)
        one = _noise(phantom["cholesky"])
        two = one + _noise(phantom["cholesky"])
        budgets = [
            coil_subspace_energy_budget(
                img, phantom["projector"], phantom["covariance"], support=phantom["support"]
            )[phantom["support"]]
            .mean()
            .item()
            for img in (one, two)
        ]
        assert 1.7 < budgets[1] / budgets[0] < 2.3, budgets


class TestTheFailureModesAreLoud:
    def test_a_magnitude_image_raises(self, phantom) -> None:
        """RSS or magnitude has already destroyed the coil vector."""
        with pytest.raises(TypeError, match="COMPLEX"):
            coil_subspace_residual(phantom["clean"].abs(), phantom["projector"])

    def test_a_coil_count_mismatch_raises(self, phantom) -> None:
        wrong = torch.eye(3, dtype=torch.complex64)
        with pytest.raises(ValueError, match="coil"):
            expected_null_residual(phantom["projector"], wrong)

    def test_a_shape_mismatch_raises(self, phantom) -> None:
        with pytest.raises(ValueError, match=r"\(B, H, W\)"):
            coil_subspace_residual(phantom["clean"][:, :, :32], phantom["projector"])

    def test_scaling_the_covariance_is_quadratic(self, phantom) -> None:
        """A covariance scales as the SQUARE of the divisor applied to the data."""
        scaled = scale_covariance(phantom["covariance"], 2.0)
        assert torch.allclose(scaled, phantom["covariance"] / 4.0)

    def test_a_non_positive_scale_raises(self, phantom) -> None:
        with pytest.raises(ValueError, match="positive"):
            scale_covariance(phantom["covariance"], 0.0)

    def test_normalizing_the_data_without_the_covariance_breaks_the_budget(self, phantom) -> None:
        """The failure this module's `scale_covariance` exists to prevent.

        Divide the images and forget the covariance and the budget falls by the
        square of the divisor -- a plausible-looking number, four times wrong.
        """
        torch.manual_seed(5)
        noise = _noise(phantom["cholesky"])
        naive = coil_subspace_energy_budget(
            noise / 2.0,
            phantom["projector"],
            phantom["covariance"],
            support=phantom["support"],
        )[phantom["support"]].mean()
        corrected = coil_subspace_energy_budget(
            noise / 2.0,
            phantom["projector"],
            scale_covariance(phantom["covariance"], 2.0),
            support=phantom["support"],
        )[phantom["support"]].mean()
        assert naive < 0.35, naive
        assert 0.93 < corrected < 1.07, corrected


def test_masked_pixels_are_nan_not_zero(phantom) -> None:
    """A zero would average in silently and drag the summary down."""
    budget = coil_subspace_energy_budget(
        phantom["clean"],
        phantom["projector"],
        phantom["covariance"],
        support=phantom["support"],
    )
    assert torch.isnan(budget[~phantom["support"]]).all()
    assert not torch.isnan(budget[phantom["support"]]).any()
