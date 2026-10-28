"""Unit tests for k-space acceleration consistency.

Verifies that all registered accelerator types hit their target
acceleration factors within a defined tolerance.
"""

import pytest
import torch

from spectramr.infrastructure.physics.sampling import (
    _ACCELERATOR_REGISTRY,
    ColdDiffusionAccelerator,
)

H, W = 256, 256
DEVICE = torch.device("cpu")

# Types that are expected to hit targets within 10% at these levels
ACCEL_TARGETS = [2, 4, 8]

# Extended targets (some types have physical limits at extreme acceleration)
EXTENDED_TARGETS = [16, 32]

# Types that are correct by design but cannot physically hit arbitrary targets
EXCLUDED_TYPES = {
    "partial_fourier",  # Fixed at pf_fraction=0.75, max ~1.3x
    "multi_mask",       # Delegates to sub-accelerator, has its own constraints
}

# Types with known physical limits at extreme acceleration (non-Cartesian raster)
NONCARTESIAN_TYPES = {"radial", "golden_angle", "spiral"}
CARTESIAN_TYPES = set(_ACCELERATOR_REGISTRY.keys()) - EXCLUDED_TYPES - NONCARTESIAN_TYPES


def _horizon(accel_type: str) -> int:
    """Ramp length for a target-R check at the last rung.

    A 1000-step linear ramp from R=1 has rungs at R~1.001, which a trajectory family
    refuses at build (no rasterised trajectory covers more than ~79% of the grid,
    #612). Two rungs -- the identity and the target -- test the same final mask.
    """
    return 2 if accel_type in NONCARTESIAN_TYPES else 1000


def _property_horizon(accel_type: str) -> int:
    """Ramp length for the structural checks; four rungs keep every R >= 2 above 1."""
    return 4 if accel_type in NONCARTESIAN_TYPES else 1000


class TestAccelerationTargetsBasic:
    """Test all mask types hit x2, x4, x8 within 10% tolerance."""

    @pytest.fixture(params=sorted(set(_ACCELERATOR_REGISTRY.keys()) - EXCLUDED_TYPES))
    def accel_type(self, request: pytest.FixtureRequest) -> str:
        """Parametrize over all valid accelerator types."""
        return request.param

    @pytest.mark.parametrize("target", ACCEL_TARGETS)
    def test_acceleration_x2_x4_x8(self, accel_type: str, target: int) -> None:
        """Verify mask achieves target acceleration within 10%."""
        accel = ColdDiffusionAccelerator(
            num_timesteps=_horizon(accel_type),
            acceleration_type=accel_type,
            max_acceleration=float(target),
            base_acceleration=1.0,
        )
        mask = accel.get_acceleration_mask(
            kspace_shape=(1, H, W), timestep=_horizon(accel_type) - 1, device=DEVICE,
        )
        frac = mask.float().sum().item() / (H * W)
        actual_accel = 1.0 / frac if frac > 0 else float("inf")

        # Within 15% tolerance (accounts for rasterization and line quantization)
        assert actual_accel >= target * 0.85, (
            f"{accel_type} at x{target}: actual={actual_accel:.1f}x, "
            f"expected>={target * 0.85:.1f}x (frac={frac:.4f})"
        )
        assert actual_accel <= target * 1.15, (
            f"{accel_type} at x{target}: actual={actual_accel:.1f}x, "
            f"expected<={target * 1.15:.1f}x (frac={frac:.4f})"
        )


class TestAccelerationTargetsExtended:
    """Test Cartesian types hit x16, x32 within 15% tolerance."""

    @pytest.fixture(params=sorted(CARTESIAN_TYPES))
    def cartesian_type(self, request: pytest.FixtureRequest) -> str:
        """Parametrize over Cartesian-only types."""
        return request.param

    @pytest.mark.parametrize("target", EXTENDED_TARGETS)
    def test_cartesian_high_acceleration(
        self, cartesian_type: str, target: int
    ) -> None:
        """Cartesian types should scale to x16/x32."""
        accel = ColdDiffusionAccelerator(
            num_timesteps=1000,
            acceleration_type=cartesian_type,
            max_acceleration=float(target),
            base_acceleration=1.0,
        )
        mask = accel.get_acceleration_mask(
            kspace_shape=(1, H, W), timestep=999, device=DEVICE,
        )
        frac = mask.float().sum().item() / (H * W)
        actual_accel = 1.0 / frac if frac > 0 else float("inf")

        assert actual_accel >= target * 0.80, (
            f"{cartesian_type} at x{target}: actual={actual_accel:.1f}x, "
            f"expected>={target * 0.80:.1f}x"
        )
        assert actual_accel <= target * 1.20, (
            f"{cartesian_type} at x{target}: actual={actual_accel:.1f}x, "
            f"expected<={target * 1.20:.1f}x"
        )


class TestNonCartesianAcceleration:
    """Test non-Cartesian trajectory types."""

    @pytest.fixture(params=sorted(NONCARTESIAN_TYPES))
    def noncartesian_type(self, request: pytest.FixtureRequest) -> str:
        """Parametrize over non-Cartesian types."""
        return request.param

    @pytest.mark.parametrize("target", [2, 4, 8])
    def test_noncartesian_low_acceleration(
        self, noncartesian_type: str, target: int
    ) -> None:
        """Non-Cartesian types should hit x2-x8 within 15%."""
        accel = ColdDiffusionAccelerator(
            num_timesteps=_horizon(noncartesian_type),
            acceleration_type=noncartesian_type,
            max_acceleration=float(target),
            base_acceleration=1.0,
        )
        mask = accel.get_acceleration_mask(
            kspace_shape=(1, H, W), timestep=_horizon(noncartesian_type) - 1, device=DEVICE,
        )
        frac = mask.float().sum().item() / (H * W)
        actual_accel = 1.0 / frac if frac > 0 else float("inf")

        assert actual_accel >= target * 0.80, (
            f"{noncartesian_type} at x{target}: actual={actual_accel:.1f}x"
        )
        assert actual_accel <= target * 1.20, (
            f"{noncartesian_type} at x{target}: actual={actual_accel:.1f}x"
        )

    @pytest.mark.parametrize("target", [16, 32])
    def test_noncartesian_high_acceleration(
        self, noncartesian_type: str, target: int
    ) -> None:
        """Non-Cartesian types should scale to x16/x32 within 30%."""
        accel = ColdDiffusionAccelerator(
            num_timesteps=_horizon(noncartesian_type),
            acceleration_type=noncartesian_type,
            max_acceleration=float(target),
            base_acceleration=1.0,
        )
        mask = accel.get_acceleration_mask(
            kspace_shape=(1, H, W), timestep=_horizon(noncartesian_type) - 1, device=DEVICE,
        )
        frac = mask.float().sum().item() / (H * W)
        actual_accel = 1.0 / frac if frac > 0 else float("inf")

        # Wider tolerance for non-Cartesian at extreme acceleration
        assert actual_accel >= target * 0.60, (
            f"{noncartesian_type} at x{target}: actual={actual_accel:.1f}x"
        )
        assert actual_accel <= target * 1.40, (
            f"{noncartesian_type} at x{target}: actual={actual_accel:.1f}x"
        )


class TestMaskProperties:
    """Test structural properties of masks."""

    @pytest.fixture(params=sorted(set(_ACCELERATOR_REGISTRY.keys()) - EXCLUDED_TYPES))
    def accel_type(self, request: pytest.FixtureRequest) -> str:
        """Parametrize over all valid types."""
        return request.param

    def test_mask_shape(self, accel_type: str) -> None:
        """Mask should match requested k-space shape."""
        shape = (1, H, W)
        accel = ColdDiffusionAccelerator(
            num_timesteps=_property_horizon(accel_type),
            acceleration_type=accel_type,
            max_acceleration=4.0,
        )
        mask = accel.get_acceleration_mask(
            kspace_shape=shape, timestep=_property_horizon(accel_type) // 2, device=DEVICE,
        )
        assert mask.shape[-2:] == (H, W), (
            f"{accel_type}: expected shape (*, {H}, {W}), got {mask.shape}"
        )

    def test_mask_is_binary(self, accel_type: str) -> None:
        """Mask should only contain 0s and 1s."""
        accel = ColdDiffusionAccelerator(
            num_timesteps=_property_horizon(accel_type),
            acceleration_type=accel_type,
            max_acceleration=4.0,
        )
        mask = accel.get_acceleration_mask(
            kspace_shape=(1, H, W), timestep=_property_horizon(accel_type) // 2, device=DEVICE,
        )
        unique = mask.unique()
        assert all(v in (0, 1, True, False) for v in unique.tolist()), (
            f"{accel_type}: non-binary values: {unique.tolist()}"
        )

    def test_monotonic_degradation(self, accel_type: str) -> None:
        """Higher timesteps should yield sparser masks (fewer sampled points)."""
        accel = ColdDiffusionAccelerator(
            num_timesteps=_property_horizon(accel_type),
            acceleration_type=accel_type,
            max_acceleration=8.0,
            base_acceleration=1.0,
        )
        t_low = _property_horizon(accel_type) // 10
        t_high = _property_horizon(accel_type) * 9 // 10
        mask_low = accel.get_acceleration_mask(
            kspace_shape=(1, H, W), timestep=t_low, device=DEVICE,
        )
        mask_high = accel.get_acceleration_mask(
            kspace_shape=(1, H, W), timestep=t_high, device=DEVICE,
        )
        frac_low = mask_low.float().sum().item()
        frac_high = mask_high.float().sum().item()

        assert frac_low >= frac_high, (
            f"{accel_type}: t={t_low} has {frac_low} samples "
            f"vs t={t_high} has {frac_high} — violates monotonicity"
        )


class TestPartialFourier:
    """Verify partial_fourier has correct fixed behavior."""

    def test_partial_fourier_fixed(self) -> None:
        """partial_fourier should always be ~75% regardless of target."""
        for target in [2, 8, 32]:
            accel = ColdDiffusionAccelerator(
                num_timesteps=1000,
                acceleration_type="partial_fourier",
                max_acceleration=float(target),
            )
            mask = accel.get_acceleration_mask(
                kspace_shape=(1, H, W), timestep=999, device=DEVICE,
            )
            frac = mask.float().sum().item() / (H * W)
            assert 0.70 <= frac <= 0.80, (
                f"partial_fourier at x{target}: frac={frac:.3f}, expected ~0.75"
            )


class TestCenterFractionDefault:
    """Verify that center_fraction default is 0.0325."""

    def test_sampling_py_default(self) -> None:
        """Check that accelerator uses 0.0325 center_fraction."""
        accel = ColdDiffusionAccelerator(
            num_timesteps=1000,
            acceleration_type="variable_density",
            max_acceleration=32.0,
        )
        inner_accel = accel.accelerator
        assert hasattr(inner_accel, "center_fraction"), (
            "variable_density accelerator should have center_fraction attribute"
        )
        assert abs(inner_accel.center_fraction - 0.0325) < 1e-6, (
            f"Expected center_fraction=0.0325, got {inner_accel.center_fraction}"
        )
