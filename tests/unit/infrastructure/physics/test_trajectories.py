"""Unit tests for the non-Cartesian trajectory factory.

Targets :mod:`spectramr.infrastructure.physics.trajectories`. The module
provides analytic generators for radial / spiral / EPI trajectories
used by the NUFFT pipeline. Coordinates must live in ``[-π, π]`` and
the density-compensation function (DCF) must be non-zero everywhere so
the gridding adjoint stays numerically stable.

CPU-only; no NUFFT library required.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.trajectories import (
    TrajectoryFactory,
    get_trajectory,
)


# ── Radial ─────────────────────────────────────────────────────────


class TestRadialTrajectory:
    def test_shape_is_two_by_n(self) -> None:
        traj, dcf = TrajectoryFactory.get_radial_trajectory(im_size=(64, 64), num_spokes=16)
        # samples_per_spoke defaults to max(im_size) = 64.
        # 16 spokes × 64 samples = 1024 points.
        assert traj.shape == (2, 16 * 64)
        assert dcf.shape == (16 * 64,)

    def test_coordinates_in_minus_pi_to_pi(self) -> None:
        traj, _ = TrajectoryFactory.get_radial_trajectory(im_size=(64, 64), num_spokes=16)
        # Spokes go r ∈ [-0.5, 0.5] × 2π → [-π, π].
        assert traj.abs().max().item() <= math.pi + 1e-5

    def test_dcf_is_strictly_positive(self) -> None:
        _, dcf = TrajectoryFactory.get_radial_trajectory(im_size=(64, 64), num_spokes=16)
        # Zero weight at DC is patched by the source — every entry > 0.
        assert torch.all(dcf > 0)

    def test_golden_angle_changes_trajectory(self) -> None:
        uni, _ = TrajectoryFactory.get_radial_trajectory(
            im_size=(64, 64), num_spokes=16, golden_angle=False
        )
        gold, _ = TrajectoryFactory.get_radial_trajectory(
            im_size=(64, 64), num_spokes=16, golden_angle=True
        )
        # Both produce the same number of points but at different
        # angles → trajectories diverge.
        assert uni.shape == gold.shape
        assert not torch.allclose(uni, gold)

    def test_samples_per_spoke_override(self) -> None:
        traj, _ = TrajectoryFactory.get_radial_trajectory(
            im_size=(64, 64), num_spokes=4, samples_per_spoke=32
        )
        # 4 × 32 = 128.
        assert traj.shape == (2, 128)


# ── Spiral ─────────────────────────────────────────────────────────


class TestSpiralTrajectory:
    def test_default_shape(self) -> None:
        traj, dcf = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=4, samples_per_arm=128
        )
        # 4 arms × 128 samples.
        assert traj.shape == (2, 512)
        assert dcf.shape == (512,)

    def test_coordinates_in_bounds(self) -> None:
        traj, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=4, samples_per_arm=128
        )
        assert traj.abs().max().item() <= math.pi + 1e-5

    def test_dcf_normalised(self) -> None:
        _, dcf = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=4, samples_per_arm=128
        )
        # Source normalises sum(dcf) ≈ N.
        assert dcf.sum().item() == pytest.approx(dcf.numel(), abs=1e-3)
        # Strictly positive (the floor patches zero at the centre).
        assert torch.all(dcf > 0)

    def test_variable_density_changes_radius_profile(self) -> None:
        traj_var, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64),
            num_arms=1,
            samples_per_arm=128,
            variable_density=True,
        )
        traj_lin, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64),
            num_arms=1,
            samples_per_arm=128,
            variable_density=False,
        )
        # Different radial growth → different trajectories.
        assert not torch.allclose(traj_var, traj_lin)


# ── EPI ────────────────────────────────────────────────────────────


class TestGoldenAngleSpiral:
    """Interleaves rotated by the full-circle golden angle nest across counts (#612)."""

    def test_first_interleaves_do_not_depend_on_the_count(self) -> None:
        few, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=3, samples_per_arm=128, golden_angle=True
        )
        many, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=7, samples_per_arm=128, golden_angle=True
        )
        assert torch.equal(few, many[:, : few.shape[1]])

    def test_rotation_is_the_full_circle_golden_angle(self) -> None:
        from spectramr.infrastructure.physics.trajectories import SPIRAL_GOLDEN_ANGLE

        assert math.degrees(SPIRAL_GOLDEN_ANGLE) == pytest.approx(137.508, abs=1e-3)
        traj, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=2, samples_per_arm=128, golden_angle=True
        )
        end0 = torch.atan2(traj[1, 127], traj[0, 127])
        end1 = torch.atan2(traj[1, 255], traj[0, 255])
        turn = float(torch.remainder(end1 - end0, 2 * math.pi))
        assert turn == pytest.approx(SPIRAL_GOLDEN_ANGLE, abs=1e-4)

    def test_every_interleave_is_the_first_one_rotated(self) -> None:
        from spectramr.infrastructure.physics.trajectories import SPIRAL_GOLDEN_ANGLE

        traj, dcf = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=3, samples_per_arm=128, golden_angle=True
        )
        assert dcf.numel() == traj.shape[1] == 3 * 128
        first = torch.complex(traj[0, :128], traj[1, :128])
        for a in (1, 2):
            arm = torch.complex(traj[0, 128 * a : 128 * (a + 1)], traj[1, 128 * a : 128 * (a + 1)])
            rotation = torch.polar(torch.tensor(1.0), torch.tensor(a * SPIRAL_GOLDEN_ANGLE))
            assert torch.allclose(arm, first * rotation, atol=1e-4)

    def test_default_spacing_is_unchanged(self) -> None:
        """NUFFT consumers keep the uniform interleave spacing unless they opt in."""
        uniform, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=4, samples_per_arm=128
        )
        golden, _ = TrajectoryFactory.get_spiral_trajectory(
            im_size=(64, 64), num_arms=4, samples_per_arm=128, golden_angle=True
        )
        assert torch.equal(uniform[:, :128], golden[:, :128])
        assert not torch.equal(uniform, golden)


class TestEPITrajectory:
    def test_shape(self) -> None:
        traj, dcf = TrajectoryFactory.get_epi_trajectory(im_size=(32, 32))
        # H/acceleration × W = 32 × 32 = 1024.
        assert traj.shape == (2, 1024)
        assert dcf.shape == (1024,)

    def test_acceleration_reduces_pe_lines(self) -> None:
        traj1, _ = TrajectoryFactory.get_epi_trajectory(im_size=(32, 32))
        traj2, _ = TrajectoryFactory.get_epi_trajectory(im_size=(32, 32), acceleration=2)
        # 16 pe lines × 32 readout samples = 512.
        assert traj2.shape == (2, 512)
        assert traj1.shape[1] > traj2.shape[1]

    def test_dcf_is_uniform(self) -> None:
        _, dcf = TrajectoryFactory.get_epi_trajectory(im_size=(32, 32))
        # Cartesian EPI → uniform weights.
        assert torch.allclose(dcf, torch.ones_like(dcf))

    def test_alternating_readout_direction(self) -> None:
        """EPI alternates left-right / right-left on consecutive PE
        lines; verify by checking kx of two adjacent rows is flipped."""
        traj, _ = TrajectoryFactory.get_epi_trajectory(im_size=(8, 8))
        kx = traj[0].reshape(8, 8)
        row0 = kx[0]
        row1 = kx[1]
        # Row 1 is the reverse of row 0.
        assert torch.allclose(row1, row0.flip(0))

    def test_acceleration_below_one_raises(self) -> None:
        """acceleration < 1 (e.g. 0 → ZeroDivisionError, or negative) must
        raise a clear ValueError rather than crash or silently produce a
        degenerate trajectory (pitfall #15: validate the knob)."""
        with pytest.raises(ValueError, match="acceleration must be >= 1"):
            TrajectoryFactory.get_epi_trajectory(im_size=(32, 32), acceleration=0)
        with pytest.raises(ValueError, match="acceleration must be >= 1"):
            TrajectoryFactory.get_epi_trajectory(im_size=(32, 32), acceleration=-2)

    def test_acceleration_one_is_numerically_unchanged(self) -> None:
        """The guard is inert for the only production path (acceleration=1):
        full PE coverage is preserved byte-identically."""
        traj, _ = TrajectoryFactory.get_epi_trajectory(im_size=(16, 16), acceleration=1)
        # 16 PE lines × 16 readout samples = 256 points.
        assert traj.shape == (2, 256)


# ── Dispatcher ─────────────────────────────────────────────────────


class TestDispatcher:
    @pytest.mark.parametrize(
        "kind",
        ["radial", "golden_angle", "spiral", "epi"],
    )
    def test_dispatcher_yields_valid_pair(self, kind: str) -> None:
        if kind == "spiral":
            extra: dict[str, object] = {"num_arms": 2, "samples_per_arm": 64}
        elif kind in ("radial", "golden_angle"):
            extra = {"num_spokes": 8, "samples_per_spoke": 32}
        else:
            extra = {}
        traj, dcf = get_trajectory(trajectory_type=kind, im_size=(32, 32), **extra)
        # Trajectory always (2, N), DCF (N,).
        assert traj.shape[0] == 2
        assert dcf.shape == (traj.shape[1],)
        # No NaNs / infs leak from a buggy formula.
        assert torch.isfinite(traj).all()
        assert torch.isfinite(dcf).all()

    def test_unknown_kind_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown trajectory type"):
            get_trajectory(trajectory_type="not_a_thing")  # type: ignore[arg-type]
