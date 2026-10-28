"""Tests for the nested trajectory ladder: one ordering, a prefix per rung (#612)."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.physics.sampling import MaskGenerator
from spectramr.infrastructure.physics.trajectory_ladder import (
    COVERAGE_CEILING,
    NestedTrajectoryLadder,
    golden_radial_ordering,
    golden_spiral_ordering,
    validate_ladder_reachable,
)


def _ladder(ordering, height: int, width: int, *, core_bins: int = 1) -> NestedTrajectoryLadder:
    traj, quantum = ordering(height, width)
    rows, cols = MaskGenerator.trajectory_bin_indices(traj, height, width)
    core = torch.zeros(height, width, dtype=torch.bool)
    core.view(-1)[:core_bins] = True
    return NestedTrajectoryLadder(rows, cols, height, width, quantum, core, "test")


def _spiral(height: int, width: int) -> tuple[torch.Tensor, int]:
    return golden_spiral_ordering(height, width, num_turns=4.0, samples_per_arm=1024)


ORDERINGS = {"radial": golden_radial_ordering, "spiral": _spiral}


class TestFirstHitRanking:
    def test_prefix_masks_nest(self) -> None:
        rows = torch.tensor([0, 1, 0, 2, 3, 1])
        cols = torch.tensor([0, 1, 0, 2, 3, 2])
        ladder = NestedTrajectoryLadder(
            rows, cols, 4, 4, 1, torch.zeros(4, 4, dtype=torch.bool), "toy"
        )
        masks = [ladder.mask(n) for n in range(7)]
        for small, big in itertools.pairwise(masks):
            assert not bool((small & ~big).any())
        # The repeated sample (index 2) adds no bin.
        assert int(masks[2].sum()) == int(masks[3].sum()) == 2

    def test_quantum_rounds_up_to_whole_spokes(self) -> None:
        rows = torch.arange(8) % 4
        cols = torch.arange(8) // 4
        ladder = NestedTrajectoryLadder(
            rows, cols, 4, 4, 4, torch.zeros(4, 4, dtype=torch.bool), "toy"
        )
        assert ladder.sample_count(1 / 16) == 4
        assert ladder.sample_count(5 / 16) == 8

    def test_unreachable_fraction_raises_naming_the_ceiling(self) -> None:
        """Planted: the whole ordering hits 2 of 16 bins, so 50% cannot be met."""
        ladder = NestedTrajectoryLadder(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            4,
            4,
            1,
            torch.zeros(4, 4, dtype=torch.bool),
            "toy",
        )
        with pytest.raises(ValueError, match=r"covers at most 12\.5%"):
            ladder.sample_count(0.5)


class TestOrderings:
    @pytest.mark.parametrize("family", sorted(ORDERINGS))
    @pytest.mark.parametrize("grid", [(32, 32), (48, 80), (64, 64), (128, 128)])
    def test_saturated_coverage_clears_the_build_ceiling(self, family: str, grid) -> None:
        """A rung the build check accepts must be one the grid can deliver."""
        assert _ladder(ORDERINGS[family], *grid).ceiling >= COVERAGE_CEILING

    @pytest.mark.parametrize("family", sorted(ORDERINGS))
    def test_spiral_and_radial_hit_their_budgets(self, family: str) -> None:
        ladder = _ladder(ORDERINGS[family], 64, 64)
        for r in (2.0, 8.0, 32.0):
            realised = float(ladder.mask(ladder.sample_count(1 / r)).float().mean())
            assert realised >= 1 / r
            if family == "spiral":  # mid-arm truncation hits the budget to the bin
                assert realised * 64 * 64 == pytest.approx(-(-64 * 64 // r))


class TestBuildCheck:
    @staticmethod
    def _accel(rungs: list[float]) -> SimpleNamespace:
        return SimpleNamespace(num_timesteps=len(rungs), get_acceleration_factor=lambda t: rungs[t])

    def test_identity_and_reachable_rungs_pass(self) -> None:
        validate_ladder_reachable(self._accel([1.0, 1.0 / COVERAGE_CEILING, 32.0]), "radial")

    def test_rung_between_identity_and_ceiling_raises(self) -> None:
        """Planted: R=1.2 needs 83% of k-space."""
        with pytest.raises(ValueError, match=r"rung t=1 declares R=1\.2"):
            validate_ladder_reachable(self._accel([1.0, 1.2, 32.0]), "spiral")


class TestAngleOffset:
    """A seeded rotation is how a second realisation of one ordering is drawn."""

    @pytest.mark.parametrize("family", sorted(ORDERINGS))
    def test_zero_offset_is_the_ladder_ordering(self, family: str) -> None:
        kwargs = {} if family == "radial" else {"num_turns": 4.0, "samples_per_arm": 1024}
        ordering = golden_radial_ordering if family == "radial" else golden_spiral_ordering
        base, _ = ordering(32, 32, **kwargs)
        explicit, _ = ordering(32, 32, angle_offset=0.0, **kwargs)
        assert torch.equal(base, explicit)

    def test_an_offset_rotates_every_spoke_by_that_angle(self) -> None:
        base, quantum = golden_radial_ordering(32, 32)
        turned, _ = golden_radial_ordering(32, 32, angle_offset=0.3)
        assert torch.allclose(turned.norm(dim=0), base.norm(dim=0), atol=1e-5)
        last = quantum - 1  # the spoke's +k end: angle(turned) = angle(base) + 0.3
        a0 = torch.atan2(base[1, last], base[0, last])
        a1 = torch.atan2(turned[1, last], turned[0, last])
        assert float(torch.remainder(a1 - a0, 2 * torch.pi)) == pytest.approx(0.3, abs=1e-5)

    def test_spiral_offset_moves_the_interleaves(self) -> None:
        base, _ = golden_spiral_ordering(32, 32, num_turns=4.0, samples_per_arm=256)
        turned, _ = golden_spiral_ordering(
            32, 32, num_turns=4.0, samples_per_arm=256, angle_offset=1.0
        )
        assert not torch.allclose(base, turned)
        assert torch.allclose(turned.norm(dim=0), base.norm(dim=0), atol=1e-5)
