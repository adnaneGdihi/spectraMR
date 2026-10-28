"""Unit tests for ImageTopologyLinearizer and K-Space Kinematic Generators.

Tests verify:
- All image topology modes produce valid permutations
- All modes support roundtrip (forward → reverse = identity)
- K-space trajectory shapes are correct
- Registry dispatch works
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics.kspace_trajectory_generator import (
    TRAJECTORY_REGISTRY,
    archimedean_spiral,
    cartesian_sequential,
    create_trajectory,
    epi_boustrophedon,
    radial_golden_angle,
)
from spectramr.models.blocks.topology_linearizer import (
    ImageTopologyLinearizer,
    _hilbert_2d_indices,
    _hilbert_3d_indices,
    _rect_hilbert_2d_indices,
    _rect_hilbert_3d_indices,
    _snake_2d_indices,
    _zigzag_2d_indices,
)

# -----------------------------------------------------------------------
# Image Topology Linearizer Tests
# -----------------------------------------------------------------------


class TestHilbert2D:
    """Test Hilbert 2D curve index generation."""

    def test_valid_permutation_4x4(self):
        idx = _hilbert_2d_indices(4, 4)
        assert set(idx.tolist()) == set(range(16))

    def test_valid_permutation_8x8(self):
        idx = _hilbert_2d_indices(8, 8)
        assert set(idx.tolist()) == set(range(64))

    def test_rejects_non_power_of_2(self):
        with pytest.raises(AssertionError):
            _hilbert_2d_indices(6, 6)

    def test_rejects_non_square(self):
        with pytest.raises(AssertionError):
            _hilbert_2d_indices(4, 8)


class TestRectHilbert:
    """Resolution-agnostic Hilbert (``hilbert_2d_rect`` / ``hilbert_3d_rect``).

    The strict square/cubic power-of-2 contract above is preserved; these
    variants generalise it to any extent so the CS-MNO operator can validate
    on full-resolution non-power-of-2 volumes without re-raising (F37 sequel).
    """

    def test_2d_reduces_to_strict_on_square_pow2(self):
        """On a square power-of-2 grid the rect curve is byte-identical to the
        strict one — guarantees zero behavioural change for existing callers."""
        for n in (2, 4, 8, 16):
            assert np.array_equal(
                _rect_hilbert_2d_indices(n, n), _hilbert_2d_indices(n, n)
            )

    def test_2d_non_pow2_is_valid_permutation(self):
        idx = _rect_hilbert_2d_indices(6, 6)
        assert sorted(idx.tolist()) == list(range(36))

    def test_2d_non_square_is_valid_permutation(self):
        idx = _rect_hilbert_2d_indices(221, 200)
        assert sorted(idx.tolist()) == list(range(221 * 200))

    def test_2d_preserves_locality(self):
        """Consecutive Hilbert tokens stay spatially adjacent on average —
        the mean step between successive visited cells is far below the
        raster row-stride ``w``. Guards against the rect restriction
        degenerating into a raster scan."""
        h, w = 32, 24
        fwd = _rect_hilbert_2d_indices(h, w)
        coords = np.stack(np.unravel_index(fwd, (h, w)), axis=1).astype(np.float64)
        mean_step = np.linalg.norm(np.diff(coords, axis=0), axis=1).mean()
        assert mean_step < 2.0  # Hilbert ~1.x; raster would be ~w on each wrap

    def test_3d_reduces_to_strict_on_cubic_pow2(self):
        for n in (2, 4, 8):
            assert np.array_equal(
                _rect_hilbert_3d_indices(n, n, n), _hilbert_3d_indices(n, n, n)
            )

    def test_3d_non_cubic_is_valid_permutation(self):
        idx = _rect_hilbert_3d_indices(5, 6, 7)
        assert sorted(idx.tolist()) == list(range(5 * 6 * 7))

    def test_linearizer_roundtrip_non_square(self):
        """Full forward→reverse round-trip through the module on a non-square,
        non-power-of-2 extent."""
        lin = ImageTopologyLinearizer((221, 200), mode="hilbert_2d_rect")
        x = torch.randn(1, 1, 221, 200)
        assert torch.allclose(lin.reverse(lin(x)), x)
        # arc-length deltas are finite + start at 0 (used by ContinuousSFCMamba)
        delta = lin.physical_arc_delta()
        assert delta.shape == (221 * 200,)
        assert float(delta[0]) == 0.0
        assert bool(torch.isfinite(delta).all())


class TestResolutionAgnosticMode:
    """The shared strict->rect Hilbert remap SSOT."""

    def test_maps_strict_hilbert_to_rect(self):
        from spectramr.models.blocks.topology_linearizer import resolution_agnostic_mode

        assert resolution_agnostic_mode("hilbert_2d") == "hilbert_2d_rect"
        assert resolution_agnostic_mode("hilbert_3d") == "hilbert_3d_rect"

    def test_passthrough_for_other_modes(self):
        from spectramr.models.blocks.topology_linearizer import resolution_agnostic_mode

        for m in ("morton_2d", "snake_2d", "raster_2d", "hilbert_2d_rect"):
            assert resolution_agnostic_mode(m) == m


class TestHilbert3DLocality:
    """The 3D Hilbert curve must be locality-preserving, not merely a bijection.

    The old recursive octant construction returned a valid permutation whose
    consecutive voxels could be up to L1 distance 12 apart (n=8) — the locality
    property is the entire scientific justification for a Hilbert scan, so a
    step > 1 silently defeats the mechanism. These tests pin max L1 step == 1,
    which the previous ``_hilbert_3d_matrix`` could never satisfy.
    """

    @staticmethod
    def _max_l1_step(n: int) -> int:
        perm = _hilbert_3d_indices(n, n, n)
        coords = np.stack(np.unravel_index(perm, (n, n, n)), axis=1)
        return int(np.abs(np.diff(coords, axis=0)).sum(axis=1).max())

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_max_l1_step_is_one(self, n: int) -> None:
        assert self._max_l1_step(n) == 1

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_bijection(self, n: int) -> None:
        perm = _hilbert_3d_indices(n, n, n)
        assert sorted(perm.tolist()) == list(range(n**3))

    def test_rect_crop_bounded_locality(self) -> None:
        """A cropped (non-cubic) 3D Hilbert stays local on average — cropping
        only breaks strict adjacency at the enclosing-cube boundary, so the
        mean L1 step must stay well below a raster plane-stride."""
        d, h, w = 5, 6, 7
        perm = _rect_hilbert_3d_indices(d, h, w)
        coords = np.stack(np.unravel_index(perm, (d, h, w)), axis=1).astype(np.float64)
        mean_step = np.abs(np.diff(coords, axis=0)).sum(axis=1).mean()
        assert mean_step < 3.0  # raster would wrap by ~w=7 each row


class TestSnake2D:
    """Test boustrophedon scan."""

    def test_valid_permutation(self):
        idx = _snake_2d_indices(4, 4)
        assert set(idx.tolist()) == set(range(16))

    def test_odd_rows_reversed(self):
        idx = _snake_2d_indices(4, 4)
        # Row 0: [0, 1, 2, 3], Row 1 should be [7, 6, 5, 4]
        assert idx[4] == 7
        assert idx[5] == 6
        assert idx[6] == 5
        assert idx[7] == 4


class TestZigzag2D:
    """Test diagonal zig-zag scan."""

    def test_valid_permutation(self):
        idx = _zigzag_2d_indices(4, 4)
        assert set(idx.tolist()) == set(range(16))

    def test_non_square(self):
        idx = _zigzag_2d_indices(3, 5)
        assert set(idx.tolist()) == set(range(15))


class TestImageTopologyLinearizer:
    """Test the unified linearizer module."""

    @pytest.mark.parametrize(
        "mode", ["hilbert_2d", "morton_2d", "snake_2d", "zigzag_2d"]
    )
    def test_roundtrip(self, mode):
        """forward → reverse should reconstruct the original tensor."""
        if mode == "hilbert_2d":
            shape = (8, 8)  # Power of 2 required
        else:
            shape = (6, 8)

        lin = ImageTopologyLinearizer(shape, mode=mode)
        x = torch.randn(2, 3, *shape)
        seq = lin(x)
        restored = lin.reverse(seq)
        assert torch.allclose(x, restored, atol=1e-6)

    @pytest.mark.parametrize(
        "mode", ["hilbert_2d", "morton_2d", "snake_2d", "zigzag_2d"]
    )
    def test_output_shape(self, mode):
        """Flattened sequence should have correct length."""
        if mode == "hilbert_2d":
            shape = (8, 8)
        else:
            shape = (6, 8)

        lin = ImageTopologyLinearizer(shape, mode=mode)
        x = torch.randn(2, 3, *shape)
        seq = lin(x)
        expected_len = shape[0] * shape[1]
        assert seq.shape == (2, 3, expected_len)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown linearization mode"):
            ImageTopologyLinearizer((8, 8), mode="nonexistent")

    def test_extra_repr(self):
        lin = ImageTopologyLinearizer((8, 8), mode="hilbert_2d")
        assert "hilbert_2d" in lin.extra_repr()

    def test_indices_are_buffers(self):
        lin = ImageTopologyLinearizer((8, 8), mode="hilbert_2d")
        assert not lin.forward_idx.requires_grad
        assert not lin.inverse_idx.requires_grad


# -----------------------------------------------------------------------
# K-Space Trajectory Generator Tests
# -----------------------------------------------------------------------


class TestArchimedeanSpiral:
    def test_output_shape(self):
        traj = archimedean_spiral(num_arms=4, points_per_arm=128)
        assert traj.shape == (2, 4 * 128)

    def test_center_starts_near_zero(self):
        traj = archimedean_spiral(num_arms=1, points_per_arm=128)
        assert traj[:, 0].abs().max() < 0.01  # Starts at center


class TestRadialGoldenAngle:
    def test_output_shape(self):
        traj = radial_golden_angle(num_spokes=10, points_per_spoke=64)
        assert traj.shape == (2, 10 * 64)

    def test_all_spokes_cross_dc(self):
        """Each spoke should pass through k-space center (kx=0, ky=0)."""
        traj = radial_golden_angle(num_spokes=10, points_per_spoke=64)
        # The midpoint of each spoke should be at (0, 0)
        for i in range(10):
            mid = i * 64 + 32
            assert traj[:, mid].abs().max() < 0.01


class TestEPIBoustrophedon:
    def test_output_shape(self):
        traj = epi_boustrophedon(num_lines=16, points_per_line=32)
        assert traj.shape == (2, 16 * 32)

    def test_alternating_direction(self):
        """Odd lines should be read in reverse direction."""
        traj = epi_boustrophedon(num_lines=4, points_per_line=8)
        # Line 0: kx goes from -max_k to +max_k
        line0_kx = traj[0, :8]
        assert line0_kx[0] < line0_kx[-1]  # Increasing
        # Line 1: kx goes from +max_k to -max_k (reversed)
        line1_kx = traj[0, 8:16]
        assert line1_kx[0] > line1_kx[-1]  # Decreasing


class TestCartesianSequential:
    def test_output_shape(self):
        traj = cartesian_sequential(num_pe_lines=16, num_fe_samples=32)
        assert traj.shape == (2, 16 * 32)


class TestTrajectoryRegistry:
    def test_all_registered(self):
        expected = {
            "archimedean_spiral",
            "radial_golden_angle",
            "epi_boustrophedon",
            "cartesian_sequential",
        }
        assert set(TRAJECTORY_REGISTRY.keys()) == expected

    def test_create_by_name(self):
        traj = create_trajectory("archimedean_spiral", num_arms=2, points_per_arm=64)
        assert traj.shape == (2, 128)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown trajectory"):
            create_trajectory("nonexistent")



@pytest.mark.parametrize(
    ("fine_hw", "coarse_hw"),
    [
        ((256, 256), (128, 128)),
        ((64, 64), (32, 32)),
        ((48, 64), (24, 32)),
        ((66, 60), (33, 30)),
        ((40, 24), (20, 12)),
        ((100, 100), (50, 50)),
    ],
)
def test_hilbert_2d_rect_children_of_a_coarse_token_are_contiguous(fine_hw, coarse_hw):
    """Fine token j sits under coarse token j // 4 (hm_02/03/04 rely on this)."""
    fine_perm = _rect_hilbert_2d_indices(*fine_hw)
    coarse_perm = _rect_hilbert_2d_indices(*coarse_hw)
    coarse_pos = np.empty_like(coarse_perm)
    coarse_pos[coarse_perm] = np.arange(coarse_perm.size)
    rows, cols = np.divmod(fine_perm, fine_hw[1])
    parent = coarse_pos[(rows // 2) * coarse_hw[1] + cols // 2]
    np.testing.assert_array_equal(parent, np.arange(fine_perm.size) // 4)
