"""The one owner of "the rung for R" and "the rung mask over the acquired lines" (#2883)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator
from spectramr.models.diffusion.cold_rung import (
    RungStatus,
    acceleration_over_acquired,
    level_is_declared,
    rung_for_acceleration,
    rung_mask,
    schedule_accelerator,
)
from tests.unit.models.diffusion import _reverse_phantom as P


def _step(num_timesteps: int, ladder: list[float]):
    gen = KSpaceMaskGenerator(
        num_timesteps=num_timesteps,
        default_pattern="random",
        accelerator_kwargs={
            "acceleration_schedule": "step",
            "acceleration_range": list(ladder),
            "base_acceleration": min(ladder),
            "max_acceleration": max(ladder),
        },
    )
    return gen._get_accelerator(None)


def _linear(num_timesteps: int, max_acceleration: float):
    gen = KSpaceMaskGenerator(
        num_timesteps=num_timesteps,
        default_pattern="random",
        accelerator_kwargs={
            "acceleration_schedule": "linear",
            "base_acceleration": 1.0,
            "max_acceleration": max_acceleration,
        },
    )
    return gen._get_accelerator(None)


class TestRungForAcceleration:
    def test_a_declared_rung_round_trips(self) -> None:
        accel = _step(29, [2.0, 4.0, 8.0, 16.0, 32.0])
        rung = rung_for_acceleration(accel, 8, 29)
        assert rung.status is RungStatus.OK and rung.ok
        assert rung.timestep == accel.timestep_for_acceleration(8.0)
        assert rung.realized == pytest.approx(8.0)

    def test_a_rung_absent_from_the_ladder_is_a_status_not_an_error(self) -> None:
        rung = rung_for_acceleration(_step(10, [2.0, 4.0, 16.0, 32.0]), 8, 10)
        assert rung.status is RungStatus.NOT_DECLARED
        assert rung.timestep is None and isinstance(rung.error, ValueError)

    def test_a_declared_but_unrealised_rung_raises(self) -> None:
        """#1171: a config defect; turning it into a skip is what the guard prevents."""
        accel = _step(2, [2.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0])
        with pytest.raises(ValueError, match="realised at NO"):
            rung_for_acceleration(accel, 8, 2)

    def test_saturation_above_max_is_a_round_trip_mismatch(self) -> None:
        """A linear T=1000 arm capped at R=8 returns t=999 for R=64, which decodes to 8."""
        rung = rung_for_acceleration(_linear(1000, 8.0), 64, 1000)
        assert rung.status is RungStatus.ROUND_TRIP_MISMATCH
        assert rung.timestep == 999 and rung.realized == pytest.approx(8.0)

    def test_the_identity_rung_is_refused(self) -> None:
        rung = rung_for_acceleration(_linear(100, 8.0), 1.0, 100)
        assert rung.status is RungStatus.IDENTITY and not rung.ok


def test_level_is_declared_reads_through_the_wrapper() -> None:
    accel = _step(4, [2.0, 8.0, 16.0, 32.0])
    assert getattr(accel, "acceleration_range", None) is None
    assert level_is_declared(accel, 8) is True
    assert level_is_declared(accel, 4) is False


def test_the_schedule_accelerator_is_the_processs_own() -> None:
    process = P.process()
    assert schedule_accelerator(process) is process.mask_generator._get_accelerator(None)


class TestRungMask:
    def test_it_is_the_processs_cascade_tiled_to_the_channels(self) -> None:
        process = P.process()
        kspace = P.kspace()
        acquired = torch.zeros(2, P.H, dtype=torch.bool)
        acquired[:, 5:57] = True
        steps = torch.full((2,), 4, dtype=torch.long)
        mask = rung_mask(process, steps, kspace, acquired)
        expected = process.cascade_batch_masks(steps, (P.H, P.W), "cpu", acquired=acquired)
        assert mask.shape == kspace.shape and mask.dtype == torch.float32
        assert torch.equal(mask[:, :1], expected.float())
        assert torch.equal(mask[:, -1:], expected.float())

    def test_without_acquired_lines_it_is_the_full_plane_cascade(self) -> None:
        process = P.process()
        steps = torch.full((2,), 4, dtype=torch.long)
        mask = rung_mask(process, steps, P.kspace(), None)
        assert torch.equal(mask[:, :1], process.cascade_batch_masks(steps, (P.H, P.W), "cpu"))


def test_acceleration_over_acquired_counts_only_acquired_lines() -> None:
    acquired = torch.zeros(1, 16, dtype=torch.bool)
    acquired[:, 2:14] = True  # 12 lines
    sampled = torch.zeros(1, 16, dtype=torch.bool)
    sampled[:, 0:8:2] = True  # lines 0, 2, 4, 6; line 0 was never acquired
    assert float(acceleration_over_acquired(sampled, acquired)[0]) == pytest.approx(4.0)
