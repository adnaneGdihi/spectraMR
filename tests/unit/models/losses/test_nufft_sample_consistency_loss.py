"""Contract tests for the sample-domain consistency term."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.diffusion.noncartesian_spoke_process import (
    NonCartesianSpokeProcess,
)
from spectramr.models.losses.nufft_sample_consistency_loss import (
    NUFFTSampleConsistencyLoss,
)

IM = (32, 32)
LADDER = [1.0, 2.0, 4.0, 16.0]


def _process() -> NonCartesianSpokeProcess:
    return NonCartesianSpokeProcess(
        num_spokes=128,
        samples_per_spoke=64,
        im_size=IM,
        num_timesteps=len(LADDER),
        max_acceleration=16.0,
        base_acceleration=1.0,
        schedule_kwargs={"acceleration_range": list(LADDER)},
    )


def _phantom() -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, IM[0]), torch.linspace(-1.0, 1.0, IM[1]), indexing="ij"
    )
    radius = (xx.pow(2) + yy.pow(2)).sqrt()
    magnitude = torch.nn.functional.avg_pool2d(
        ((radius < 0.7).float() * (1.0 - 0.5 * radius))[None, None], 5, 1, 2
    )[0, 0]
    return (magnitude * torch.exp(1j * (1.2 * xx + 0.8 * yy)))[None, None]


def _degrade(rung: int) -> tuple[torch.Tensor, torch.Tensor, object]:
    """``(x_0, x_t, measurement)`` at one rung of a fresh process."""
    process = _process()
    x0 = _phantom()
    x_t, _ = process.q_sample(x0, torch.tensor([rung]))
    return x0, x_t, process.last_sample_measurement


def test_a_perfect_reconstruction_scores_zero() -> None:
    """The term must not penalise the answer it is asking for."""
    x0, _, measurement = _degrade(1)
    assert float(NUFFTSampleConsistencyLoss()(x0, x0, sample_measurement=measurement)) < 1e-4


def test_the_gridded_state_is_not_sample_consistent() -> None:
    """Gridding is one adjoint step, so ``x_t`` does not reproduce its own samples.

    This is the gap the term exists to close; a zero here would mean the
    non-Cartesian arm had nothing to learn.
    """
    x0, x_t, measurement = _degrade(2)
    assert float(NUFFTSampleConsistencyLoss()(x_t, x0, sample_measurement=measurement)) > 1e-3


def test_unacquired_samples_carry_no_weight() -> None:
    """Dropped spokes must not enter the score at any rung."""
    process = _process()
    x0 = _phantom()
    process.q_sample(x0, torch.tensor([3]))
    measurement = process.last_sample_measurement

    loss = NUFFTSampleConsistencyLoss()
    before = float(loss(x0, x0, sample_measurement=measurement))

    dropped = measurement.mask[0] < 0.5
    assert bool(dropped.any()), "rung 3 must drop spokes for this test to watch anything"
    poisoned = measurement.samples.clone()
    poisoned[:, :, dropped] += 1e3
    after = float(
        loss(
            x0,
            x0,
            sample_measurement=type(measurement)(
                samples=poisoned,
                mask=measurement.mask,
                trajectory=measurement.trajectory,
                projector=measurement.projector,
            ),
        )
    )
    assert after == pytest.approx(before, abs=1e-5)


def test_the_value_is_comparable_across_rungs() -> None:
    """Normalising by the acquired count keeps rungs on one scale."""
    scores = [
        float(
            NUFFTSampleConsistencyLoss()(
                _degrade(r)[1], _degrade(r)[0], sample_measurement=_degrade(r)[2]
            )
        )
        for r in range(len(LADDER))
    ]
    assert all(s > 0 for s in scores)
    assert max(scores) / min(scores) < 100.0


def test_the_samples_are_scored_on_the_ortho_scale() -> None:
    """``A`` carries no ``1/sqrt(HW)``; the term must supply it.

    Without this the score would grow with image size alone, and the arm's
    declared weight would mean a different thing at every resolution. Measured
    against ``fft2c``'s own DC, the raw operator sits at exactly ``sqrt(H*W)``.
    """
    image = torch.randn(1, 1, *IM, dtype=torch.complex64)
    process = _process()
    grid_dc = fft2c(image)[0, 0, IM[0] // 2, IM[1] // 2].abs()
    raw_dc = process.nufft.forward_project(image, torch.zeros(2, 1))[0, 0, 0].abs()
    assert float(raw_dc / grid_dc) == pytest.approx(float(IM[0] ** 0.5 * IM[1] ** 0.5), rel=1e-3)


def test_an_unthreaded_measurement_writes_no_component() -> None:
    """The validation loop scores a sampled reconstruction with no rung in scope.

    Returning ``None`` leaves the component out entirely; a zero would read as
    a perfectly sample-consistent reconstruction.
    """
    pred = torch.randn(1, 2, *IM)
    assert NUFFTSampleConsistencyLoss()(pred, pred) is None


def test_a_threaded_none_measurement_raises() -> None:
    """PLANTED VIOLATION: the wiring breaking must not read as zero error.

    A caller that threads the sample axis and supplies nothing has a broken
    stash. Contributing zero there would train the arm with no measurement
    pressure at all -- the DC-blob shape (CLAUDE.md pitfalls 9/15).
    """
    pred = torch.randn(1, 2, *IM)
    with pytest.raises(ValueError, match="sample_measurement=None"):
        NUFFTSampleConsistencyLoss()(pred, pred, sample_measurement=None)


def test_an_unknown_norm_raises() -> None:
    """An unrecognised norm must not degrade to a default (non-negotiable 3)."""
    with pytest.raises(ValueError, match="norm must be"):
        NUFFTSampleConsistencyLoss(norm="huber")


def test_coil_count_disagreement_raises() -> None:
    """A prediction with the wrong coil count must not broadcast into a score."""
    _, _, measurement = _degrade(1)
    with pytest.raises(ValueError, match="must have one shape"):
        NUFFTSampleConsistencyLoss()(
            torch.randn(1, 4, *IM, dtype=torch.complex64),
            torch.randn(1, 4, *IM, dtype=torch.complex64),
            sample_measurement=measurement,
        )
