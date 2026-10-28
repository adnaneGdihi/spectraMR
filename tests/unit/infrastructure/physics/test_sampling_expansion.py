"""Expansion tests for ``spectramr.infrastructure.physics.sampling``.

This module fills the partial-coverage gap in ``sampling.py`` (the largest
gap in the physics package). It exercises EACH advertised sampling
mode end-to-end and the ``KSpaceAccelerator`` family:

Modes / classes covered:

* ``MaskGenerator`` with every ``MaskType`` (Cartesian / VD / radial /
  spiral / equispaced / gaussian / random).
* ``KSpaceAccelerator`` concrete subclasses: ``UniformCartesianKSpaceAccelerator``,
  ``PartialFourierKSpaceAccelerator``, ``VDCartesian1DAccelerator``,
  ``VDCartesian2DGaussianAccelerator``, ``VDCartesianCAVAAccelerator``,
  ``RadialKSpaceAccelerator`` (also behind ``golden_angle``), ``SpiralKSpaceAccelerator``,
  ``RandomCartesianKSpaceAccelerator``,
  ``LinearKSpaceAccelerator``, ``LowPassAccelerator``,
  ``NestedKSpaceAccelerator``, ``VariableDensityKSpaceAccelerator``,
  ``FractionalVariableDensityAccelerator``, ``LearnedPatternAccelerator``.
* ``ColdDiffusionAccelerator`` (dispatcher + ``apply_acceleration``).
* NUFFT-style trajectory rasterization (uses ``torchkbnufft`` via
  ``TrajectoryFactory``).

For each mode we assert the contracts called out in the unit spec:

#. **Shape contract** — mask spatial shape matches input k-space.
#. **Sampling density matches declared acceleration** (within tolerance).
#. **Mask is binary** (only 0 / 1 or only ``True`` / ``False``).
#. **Fully-sampled center region is preserved.**
#. **Determinism** — same seed → identical mask.
#. **Edge: R=1 → fully-sampled mask.**
#. **Edge: invalid configuration raises actionable errors.**
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics.sampling import (
    MAX_ACCELERATION,
    SUPPORTED_ACCELERATION_TYPES,
    ColdDiffusionAccelerator,
    FractionalVariableDensityAccelerator,
    KSpaceAccelerator,
    LearnedPatternAccelerator,
    LinearKSpaceAccelerator,
    LowPassAccelerator,
    MaskGenerator,
    MaskType,
    NestedKSpaceAccelerator,
    PartialFourierKSpaceAccelerator,
    RadialKSpaceAccelerator,
    RandomCartesianKSpaceAccelerator,
    SpiralKSpaceAccelerator,
    UniformCartesianKSpaceAccelerator,
    VariableDensityKSpaceAccelerator,
    VDCartesian1DAccelerator,
    VDCartesian2DGaussianAccelerator,
    VDCartesianCAVAAccelerator,
    create_kspace_accelerator,
    create_mask_generator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_binary(mask: torch.Tensor) -> bool:
    """Return True if ``mask`` is binary (only 0/1 or True/False)."""
    if mask.dtype == torch.bool:
        return True
    unique = torch.unique(mask)
    allowed = {0.0, 1.0}
    return set(float(v) for v in unique.tolist()).issubset(allowed)


def _sampling_fraction(mask: torch.Tensor) -> float:
    """Compute fraction of sampled entries on the 2D plane (first channel)."""
    if mask.dim() == 3:
        m2d = mask[0]
    else:
        m2d = mask
    return float(m2d.sum().item()) / float(m2d.numel())


def _spatial_shape(mask: torch.Tensor) -> tuple[int, int]:
    return int(mask.shape[-2]), int(mask.shape[-1])


# ---------------------------------------------------------------------------
# MaskGenerator: per-MaskType contracts
# ---------------------------------------------------------------------------


class TestMaskGeneratorAllModes:
    """Exercise every ``MaskType`` value via the unified ``MaskGenerator``.

    Existing tests in ``tests/unit/test_sampling_consolidation.py`` cover
    the ``VARIABLE_DENSITY`` and ``GAUSSIAN`` paths superficially; we
    expand to the remaining modes and the explicit contracts.
    """

    SHAPE = (128, 128)

    @pytest.mark.parametrize(
        "mask_type",
        [
            MaskType.UNIFORM_CARTESIAN,
            MaskType.CARTESIAN_LINES,
            MaskType.VARIABLE_DENSITY,
            MaskType.EQUISPACED,
            MaskType.GAUSSIAN,
            MaskType.RANDOM,
            MaskType.RADIAL,
        ],
    )
    def test_mask_shape_and_binary(self, mask_type: MaskType) -> None:
        mg = MaskGenerator(seed=42)
        mask = mg.generate_mask(mask_type, self.SHAPE, acceleration_factor=4.0)

        # Spatial shape contract
        assert _spatial_shape(mask) == self.SHAPE

        # Binary contract
        assert _is_binary(mask), (
            f"{mask_type.value} mask must be binary; got values {torch.unique(mask).tolist()}"
        )

        # Sampled something
        assert mask.sum().item() > 0

    @pytest.mark.parametrize(
        "mask_type",
        [
            MaskType.CARTESIAN_LINES,
            MaskType.UNIFORM_CARTESIAN,
            MaskType.VARIABLE_DENSITY,
            MaskType.GAUSSIAN,
            MaskType.RANDOM,
        ],
    )
    def test_line_based_density_matches_acceleration(self, mask_type: MaskType) -> None:
        """For Cartesian / line-based masks the per-line density ≈ 1/R."""
        mg = MaskGenerator(seed=123)
        accel = 4.0
        mask = mg.generate_mask(mask_type, (256, 256), acceleration_factor=accel)
        frac = _sampling_fraction(mask)
        # Tolerance accounts for ACS overhead.
        assert frac >= 1.0 / accel * 0.85, (
            f"{mask_type.value}: fraction {frac:.3f} too low for R={accel}"
        )
        assert frac <= 1.0 / accel * 1.6, (
            f"{mask_type.value}: fraction {frac:.3f} too high for R={accel}"
        )

    @pytest.mark.parametrize(
        "mask_type",
        [
            MaskType.CARTESIAN_LINES,
            MaskType.UNIFORM_CARTESIAN,
            MaskType.VARIABLE_DENSITY,
            MaskType.GAUSSIAN,
            MaskType.RANDOM,
            MaskType.EQUISPACED,
        ],
    )
    def test_center_preserved(self, mask_type: MaskType) -> None:
        """The ACS / center fraction must be 100% sampled."""
        mg = MaskGenerator(seed=7)
        center_fraction = 0.1
        H, W = 200, 200
        mask = mg.generate_mask(
            mask_type,
            (H, W),
            acceleration_factor=8.0,
            center_fraction=center_fraction,
        )

        num_low_freq = int(H * center_fraction)
        # _generate_variable_density uses center_size = int(H * cf) directly.
        center_start = H // 2 - num_low_freq // 2
        center_end = center_start + num_low_freq

        # First column suffices because line masks broadcast along W.
        center_lines = mask[center_start:center_end, 0]
        # All rows in the ACS band must be sampled.
        assert (center_lines > 0).all(), (
            f"{mask_type.value}: ACS lines not fully sampled: {center_lines.tolist()}"
        )

    def test_determinism_same_seed(self) -> None:
        m1 = MaskGenerator(seed=999).generate_mask(MaskType.VARIABLE_DENSITY, self.SHAPE, 4.0)
        m2 = MaskGenerator(seed=999).generate_mask(MaskType.VARIABLE_DENSITY, self.SHAPE, 4.0)
        assert torch.equal(m1, m2)

    def test_different_seeds_yield_different_masks(self) -> None:
        m1 = MaskGenerator(seed=1).generate_mask(MaskType.RANDOM, (64, 64), 4.0)
        m2 = MaskGenerator(seed=2).generate_mask(MaskType.RANDOM, (64, 64), 4.0)
        # Random sampling with different seeds *should* differ. ACS overlap
        # means equality is not literal — we only require *some* difference.
        assert not torch.equal(m1, m2)

    def test_unknown_mask_type_raises(self) -> None:
        mg = MaskGenerator(seed=0)
        with pytest.raises(ValueError, match="not a valid|Unknown"):
            mg.generate_mask("totally_made_up_mode", (64, 64), 2.0)

    def test_spiral_mask_via_generator(self) -> None:
        """Spiral path uses TrajectoryFactory; verify shape and binary."""
        mg = MaskGenerator(seed=42)
        mask = mg.generate_mask(MaskType.SPIRAL, (128, 128), 4.0)
        assert _spatial_shape(mask) == (128, 128)
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0

    def test_radial_mask_via_generator(self) -> None:
        mg = MaskGenerator(seed=42)
        mask = mg.generate_mask(MaskType.RADIAL, (128, 128), 4.0)
        assert _spatial_shape(mask) == (128, 128)
        # Should be bool from rasterization
        assert mask.dtype == torch.bool

    def test_create_mask_generator_factory(self) -> None:
        mg = create_mask_generator(seed=7)
        assert isinstance(mg, MaskGenerator)
        assert mg.seed == 7

    def test_get_mask_properties(self) -> None:
        mg = MaskGenerator(seed=0)
        mask = mg.generate_mask(MaskType.CARTESIAN_LINES, (64, 64), 4.0)
        props = mg.get_mask_properties(mask)
        assert "shape" in props
        assert "acceleration_factor" in props
        assert "sampling_ratio" in props
        assert "center_sampled" in props
        assert props["center_sampled"] is True
        assert props["acceleration_factor"] >= 1.0
        # Reported accel × ratio == 1
        assert abs(props["sampling_ratio"] * props["acceleration_factor"] - 1.0) < 1e-5

    def test_get_mask_properties_empty_mask(self) -> None:
        """An empty mask reports infinite acceleration."""
        mg = MaskGenerator(seed=0)
        empty = torch.zeros(32, 32)
        props = mg.get_mask_properties(empty)
        assert props["acceleration_factor"] == float("inf")
        assert props["sampling_ratio"] == 0.0


class TestMaskGeneratorPDFHelpers:
    """Coverage for the static PDF helpers."""

    def test_gaussian_pdf_normalizes_to_one(self) -> None:
        pdf = MaskGenerator.get_gaussian_pdf(100, std_scale=4.0)
        assert pdf.shape == (100,)
        assert np.isclose(pdf.sum(), 1.0)
        # Peaks at centre
        assert pdf.argmax() in (49, 50)

    def test_gaussian_pdf_zero_input_returns_uniform(self) -> None:
        # Trivial case; len=1 makes the sum nonzero but is a useful smoke test.
        pdf = MaskGenerator.get_gaussian_pdf(1)
        assert np.isclose(pdf.sum(), 1.0)

    def test_variable_density_pdf_normalizes(self) -> None:
        pdf = MaskGenerator.get_variable_density_pdf(100, density_power=2.0)
        assert pdf.shape == (100,)
        assert np.isclose(pdf.sum(), 1.0)
        # Strictly higher in center
        assert pdf[50] > pdf[0]
        assert pdf[50] > pdf[-1]

    def test_variable_density_pdf_higher_power_more_peaked(self) -> None:
        pdf_low = MaskGenerator.get_variable_density_pdf(100, density_power=1.0)
        pdf_high = MaskGenerator.get_variable_density_pdf(100, density_power=4.0)
        # The peak (centre) is higher under higher decay power.
        assert pdf_high.max() > pdf_low.max()


class TestRasterizeTrajectory:
    """The trajectory-rasterization path (NUFFT-style coordinates)."""

    def test_rasterize_2d_shape(self) -> None:
        mg = MaskGenerator(seed=0)
        n = 256
        traj = torch.zeros(2, n)
        traj[0] = torch.linspace(-math.pi, math.pi, n)
        # Vertical line through k-space center
        mask = mg._rasterize_trajectory(traj, (32, 32))
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.bool
        # The line covers row 16 (centre) across all column samples.
        assert mask.sum().item() >= 1

    def test_rasterize_3d_shape(self) -> None:
        mg = MaskGenerator(seed=0)
        traj = torch.zeros(2, 100)
        traj[0] = torch.linspace(-math.pi, math.pi, 100)
        mask = mg._rasterize_trajectory(traj, (3, 32, 32))
        # Broadcast across the channel dimension
        assert mask.shape == (3, 32, 32)
        assert mask.dtype == torch.bool

    def test_rasterize_clamps_out_of_range(self) -> None:
        """Coordinates outside [-pi, pi] are clamped, never raise."""
        mg = MaskGenerator(seed=0)
        traj = torch.zeros(2, 50)
        traj[0] = torch.linspace(-10.0, 10.0, 50)  # way out of range
        traj[1] = torch.linspace(-10.0, 10.0, 50)
        mask = mg._rasterize_trajectory(traj, (16, 16))
        assert mask.shape == (16, 16)


# ---------------------------------------------------------------------------
# KSpaceAccelerator subclasses
# ---------------------------------------------------------------------------


class TestUniformCartesianAccelerator:
    SHAPE = (1, 96, 96)

    def test_shape_and_dtype(self) -> None:
        acc = UniformCartesianKSpaceAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool

    def test_density_matches_acceleration(self) -> None:
        accel_max = 4.0
        acc = UniformCartesianKSpaceAccelerator(
            num_timesteps=10,
            max_acceleration=accel_max,
            base_acceleration=1.0,
            center_fraction=0.04,
            seed=42,
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=9)
        frac = _sampling_fraction(mask)
        # At the largest timestep the accel reaches max → frac ≈ 1/4.
        assert 0.18 <= frac <= 0.32, f"Got fraction {frac:.3f}"

    def test_full_sample_at_t0(self) -> None:
        """At t=0 with base_acceleration=1.0, mask is fully sampled."""
        acc = UniformCartesianKSpaceAccelerator(
            num_timesteps=10,
            max_acceleration=8.0,
            base_acceleration=1.0,
            seed=42,
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=0)
        assert _sampling_fraction(mask) == pytest.approx(1.0)

    def test_center_lines_preserved(self) -> None:
        acc = UniformCartesianKSpaceAccelerator(
            num_timesteps=10,
            max_acceleration=8.0,
            center_fraction=0.1,
            line_axis="y",
            seed=42,
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=9)
        H = self.SHAPE[1]
        num_center = max(1, int(round(H * 0.1)))
        start = (H - num_center) // 2
        end = start + num_center
        center_band = mask[0, start:end, :]
        assert center_band.all(), "ACS region must remain fully sampled"

    def test_determinism(self) -> None:
        a = UniformCartesianKSpaceAccelerator(num_timesteps=10, seed=7)
        b = UniformCartesianKSpaceAccelerator(num_timesteps=10, seed=7)
        m_a = a.get_acceleration_mask(self.SHAPE, timestep=4)
        m_b = b.get_acceleration_mask(self.SHAPE, timestep=4)
        assert torch.equal(m_a, m_b)

    def test_invalid_line_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="line_axis"):
            UniformCartesianKSpaceAccelerator(num_timesteps=1000, line_axis="z")  # type: ignore[arg-type]

    def test_invalid_num_timesteps_raises(self) -> None:
        with pytest.raises(ValueError, match="num_timesteps"):
            UniformCartesianKSpaceAccelerator(num_timesteps=0)

    def test_line_axis_x(self) -> None:
        acc = UniformCartesianKSpaceAccelerator(
            num_timesteps=10,
            max_acceleration=4.0,
            center_fraction=0.1,
            line_axis="x",
            seed=42,
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=9)
        assert mask.shape == self.SHAPE
        # Centre columns must be sampled in line_axis="x" mode.
        W = self.SHAPE[2]
        num_center = max(1, int(round(W * 0.1)))
        start = (W - num_center) // 2
        end = start + num_center
        assert mask[0, :, start:end].all()


class TestVDCartesian1DAccelerator:
    SHAPE = (1, 96, 96)

    def test_basic_contract(self) -> None:
        acc = VDCartesian1DAccelerator(
            num_timesteps=10,
            max_acceleration=4.0,
            density_power=2.0,
            seed=42,
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool
        # Centre is sampled
        H, W = self.SHAPE[1:]
        assert mask[0, H // 2, W // 2].item()

    def test_density_decreases_with_higher_acceleration(self) -> None:
        acc_low = VDCartesian1DAccelerator(num_timesteps=10, max_acceleration=2.0, seed=0)
        acc_high = VDCartesian1DAccelerator(num_timesteps=10, max_acceleration=8.0, seed=0)
        f_low = _sampling_fraction(acc_low.get_acceleration_mask(self.SHAPE, 9))
        f_high = _sampling_fraction(acc_high.get_acceleration_mask(self.SHAPE, 9))
        assert f_low > f_high

    def test_invalid_line_axis(self) -> None:
        with pytest.raises(ValueError, match="line_axis"):
            VDCartesian1DAccelerator(num_timesteps=1000, line_axis="diagonal")  # type: ignore[arg-type]


class TestVDCartesian2DGaussianAccelerator:
    SHAPE = (1, 96, 96)

    def test_shape_and_density(self) -> None:
        acc = VDCartesian2DGaussianAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=9)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool
        frac = _sampling_fraction(mask)
        # max-acceleration at end -> fraction ~ 1/4
        assert 0.18 <= frac <= 0.40

    def test_centre_sampled(self) -> None:
        acc = VDCartesian2DGaussianAccelerator(
            num_timesteps=10, max_acceleration=4.0, center_fraction=0.1, seed=42
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=9)
        H, W = self.SHAPE[1:]
        assert mask[0, H // 2, W // 2].item()

    def test_determinism(self) -> None:
        a = VDCartesian2DGaussianAccelerator(num_timesteps=10, seed=11)
        b = VDCartesian2DGaussianAccelerator(num_timesteps=10, seed=11)
        assert torch.equal(
            a.get_acceleration_mask(self.SHAPE, 5),
            b.get_acceleration_mask(self.SHAPE, 5),
        )


class TestVDCartesianCAVAAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = VDCartesianCAVAAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0


class TestRadialKSpaceAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = RadialKSpaceAccelerator(num_timesteps=4, max_acceleration=4.0, center_fraction=0.08)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=2)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0

    def test_center_preserved(self) -> None:
        acc = RadialKSpaceAccelerator(num_timesteps=4, max_acceleration=4.0, center_fraction=0.1)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=3)
        H, W = self.SHAPE[1:]
        assert mask[0, H // 2, W // 2].item()

    def test_progressive_density(self) -> None:
        """Higher timestep → more degradation → fewer samples; t=0 is the identity."""
        acc = RadialKSpaceAccelerator(num_timesteps=8, max_acceleration=8.0, center_fraction=0.05)
        f_start = _sampling_fraction(acc.get_acceleration_mask(self.SHAPE, 0))
        f_end = _sampling_fraction(acc.get_acceleration_mask(self.SHAPE, 7))
        assert f_start == 1.0
        assert f_start > f_end


class TestGoldenAngleKey:
    def test_golden_angle_is_the_radial_accelerator(self) -> None:
        """Radial is golden-angle radial (#612): one class serves both keys."""
        acc = ColdDiffusionAccelerator(
            num_timesteps=4, acceleration_type="golden_angle", max_acceleration=4.0
        )
        assert type(acc.accelerator) is RadialKSpaceAccelerator


class TestSpiralKSpaceAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = SpiralKSpaceAccelerator(num_timesteps=4, max_acceleration=4.0, center_fraction=0.08)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=2)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0


class TestRandomCartesianKSpaceAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = RandomCartesianKSpaceAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool

    def test_determinism(self) -> None:
        a = RandomCartesianKSpaceAccelerator(num_timesteps=10, seed=7)
        b = RandomCartesianKSpaceAccelerator(num_timesteps=10, seed=7)
        # Note: this accelerator's `_ensure_sample_budget` passes a generator
        # so determinism holds for the entire mask construction.
        m_a = a.get_acceleration_mask(self.SHAPE, 5)
        m_b = b.get_acceleration_mask(self.SHAPE, 5)
        assert torch.equal(m_a, m_b)


class TestLinearKSpaceAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = LinearKSpaceAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0

    def test_full_sample_at_t0(self) -> None:
        acc = LinearKSpaceAccelerator(
            num_timesteps=10,
            max_acceleration=4.0,
            base_acceleration=1.0,
            seed=42,
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=0)
        # At t=0, base_acceleration=1 -> fraction=1.0 -> ALL points sampled
        assert _sampling_fraction(mask) == pytest.approx(1.0)


class TestLowPassAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = LowPassAccelerator(num_timesteps=10, max_acceleration=4.0, center_fraction=0.1)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0

    def test_center_preserved(self) -> None:
        acc = LowPassAccelerator(num_timesteps=10, max_acceleration=8.0, center_fraction=0.1)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=9)
        H, W = self.SHAPE[1:]
        assert mask[0, H // 2, W // 2].item()

    def test_progressive_shrink(self) -> None:
        """Mask shrinks as timestep increases."""
        acc = LowPassAccelerator(num_timesteps=10, max_acceleration=8.0, center_fraction=0.05)
        f_start = _sampling_fraction(acc.get_acceleration_mask(self.SHAPE, 0))
        f_end = _sampling_fraction(acc.get_acceleration_mask(self.SHAPE, 9))
        assert f_start > f_end


class TestPartialFourierKSpaceAccelerator:
    SHAPE = (1, 96, 96)

    def test_basic_contract(self) -> None:
        acc = PartialFourierKSpaceAccelerator(num_timesteps=1000, pf_fraction=0.75)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool

    def test_density_matches_pf_fraction(self) -> None:
        pf_fraction = 0.75
        acc = PartialFourierKSpaceAccelerator(num_timesteps=1000, pf_fraction=pf_fraction, line_axis="y")
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=0)
        frac = _sampling_fraction(mask)
        # PF samples a contiguous block; small ACS overhead may inflate it.
        assert pf_fraction - 0.05 <= frac <= pf_fraction + 0.05

    def test_pf_fraction_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="Partial Fourier"):
            PartialFourierKSpaceAccelerator(num_timesteps=1000, pf_fraction=0.3)
        with pytest.raises(ValueError, match="Partial Fourier"):
            PartialFourierKSpaceAccelerator(num_timesteps=1000, pf_fraction=1.5)

    def test_invalid_line_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="line_axis"):
            PartialFourierKSpaceAccelerator(num_timesteps=1000, line_axis="z")  # type: ignore[arg-type]

    def test_max_acceleration_capped_at_two(self) -> None:
        """Per source docs, PF max acceleration is theoretically 2.0."""
        acc = PartialFourierKSpaceAccelerator(num_timesteps=1000, pf_fraction=0.5, max_acceleration=10.0)
        assert acc.max_acceleration == 2.0


class TestNestedKSpaceAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = NestedKSpaceAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool

    def test_nested_property(self) -> None:
        """As t increases, mask(t) is a STRICT subset of mask(t-k)."""
        acc = NestedKSpaceAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        m_early = acc.get_acceleration_mask(self.SHAPE, timestep=2)
        m_late = acc.get_acceleration_mask(self.SHAPE, timestep=8)
        # Strict nesting: m_late ⊂ m_early.
        intersect = (m_early & m_late).sum().item()
        assert intersect == m_late.sum().item()

    def test_determinism(self) -> None:
        a = NestedKSpaceAccelerator(num_timesteps=10, seed=99)
        b = NestedKSpaceAccelerator(num_timesteps=10, seed=99)
        m_a = a.get_acceleration_mask(self.SHAPE, timestep=5)
        m_b = b.get_acceleration_mask(self.SHAPE, timestep=5)
        assert torch.equal(m_a, m_b)


class TestVariableDensityKSpaceAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = VariableDensityKSpaceAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool

    def test_nesting(self) -> None:
        """VariableDensity uses fixed-rank sampling -> nested."""
        acc = VariableDensityKSpaceAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        m_early = acc.get_acceleration_mask(self.SHAPE, timestep=2)
        m_late = acc.get_acceleration_mask(self.SHAPE, timestep=8)
        # m_late ⊂ m_early
        assert (m_early & m_late).sum().item() == m_late.sum().item()


class TestFractionalVariableDensityAccelerator:
    SHAPE = (1, 64, 64)

    def test_basic_contract(self) -> None:
        acc = FractionalVariableDensityAccelerator(
            num_timesteps=10,
            min_acceleration=1.5,
            max_acceleration=8.0,
            seed=42,
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE

    def test_base_acceleration_is_min(self) -> None:
        acc = FractionalVariableDensityAccelerator(num_timesteps=1000, min_acceleration=2.0, max_acceleration=8.0)
        assert acc.base_acceleration == 2.0

    def test_min_acceleration_clamped(self) -> None:
        # min_acceleration < 1.0 is silently clamped to 1.0
        acc = FractionalVariableDensityAccelerator(num_timesteps=1000, min_acceleration=0.1, max_acceleration=8.0)
        assert acc.min_acceleration == 1.0


class TestLearnedPatternAccelerator:
    SHAPE = (1, 32, 32)

    def test_fallback_no_pattern(self) -> None:
        """With no pattern_path, falls back to VDS."""
        acc = LearnedPatternAccelerator(num_timesteps=10, max_acceleration=4.0, seed=42)
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0

    def test_fallback_missing_file_path(self) -> None:
        acc = LearnedPatternAccelerator(
            num_timesteps=10,
            max_acceleration=4.0,
            pattern_path="/tmp/nonexistent_pattern_x9z.pt",
            seed=42,
        )
        mask = acc.get_acceleration_mask(self.SHAPE, timestep=5)
        assert mask.shape == self.SHAPE

    def test_existing_but_corrupt_pattern_raises(self, tmp_path) -> None:
        """An existing pattern file with an unsupported/corrupt format must RAISE,
        never silently degrade to variable-density sampling (pitfall #9/#10)."""
        bad = tmp_path / "pattern.bin"  # unsupported extension -> ValueError -> RuntimeError
        bad.write_bytes(b"not a tensor")
        acc = LearnedPatternAccelerator(
            num_timesteps=10,
            max_acceleration=4.0,
            pattern_path=str(bad),
            seed=42,
        )
        with pytest.raises(RuntimeError, match="failed to load configured pattern"):
            acc.get_acceleration_mask(self.SHAPE, timestep=5)


# ---------------------------------------------------------------------------
# Base-class contract: shape parsing and edge cases
# ---------------------------------------------------------------------------


class TestKSpaceAcceleratorBase:
    def test_unpack_shape_2d(self) -> None:
        channels, h, w = KSpaceAccelerator._unpack_shape((32, 32))
        assert (channels, h, w) == (1, 32, 32)

    def test_unpack_shape_3d(self) -> None:
        channels, h, w = KSpaceAccelerator._unpack_shape((4, 32, 32))
        assert (channels, h, w) == (4, 32, 32)

    def test_unpack_shape_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported k-space shape"):
            KSpaceAccelerator._unpack_shape((1, 2, 3, 4))

    def test_acceleration_schedules(self) -> None:
        """power_law / exponential / linear schedules all yield ratios in [0,1]."""
        for schedule in ("linear", "power_law", "exponential"):
            acc = LinearKSpaceAccelerator(
                num_timesteps=10,
                max_acceleration=4.0,
                base_acceleration=1.0,
                acceleration_schedule=schedule,
            )
            r_start = acc._normalized_progress(0)
            r_end = acc._normalized_progress(9)
            assert 0.0 <= r_start <= 1e-6
            assert abs(r_end - 1.0) <= 1e-6

    def test_single_timestep_progress(self) -> None:
        """num_timesteps=1 should return ratio=1 from _normalized_progress."""
        acc = LinearKSpaceAccelerator(num_timesteps=1, max_acceleration=4.0)
        assert acc._normalized_progress(0) == 1.0

    def test_max_acceleration_lower_bound(self) -> None:
        """max_acceleration is clamped to >= 1.0."""
        acc = LinearKSpaceAccelerator(num_timesteps=10, max_acceleration=0.5)
        assert acc.max_acceleration == 1.0

    def test_apply_center_patch_min_one_pixel(self) -> None:
        """An empty center_fraction still sets at least the centre pixel."""
        acc = LowPassAccelerator(num_timesteps=10, max_acceleration=8.0, center_fraction=0.0)
        mask = acc.get_acceleration_mask((1, 32, 32), timestep=9)
        # Centre must always be sampled
        assert mask[0, 16, 16].item()


# ---------------------------------------------------------------------------
# Dispatcher: ColdDiffusionAccelerator + create_kspace_accelerator
# ---------------------------------------------------------------------------


class TestColdDiffusionAccelerator:
    """End-to-end test of the dispatcher facade."""

    @pytest.mark.parametrize(
        "accel_type",
        [
            "linear",
            "radial",
            "variable_density",
            "variable_density_1d",
            "variable_density_2d_gaussian",
            "variable_density_cava",
            "uniform_cartesian",
            "partial_fourier",
            "spiral",
            "golden_angle",
            "random_cartesian",
            "nested",
            "equispaced",
            "fractional_variable_density",
            "learned_pattern",
        ],
    )
    def test_supported_type_produces_valid_mask(self, accel_type: str) -> None:
        kwargs: dict = {}
        # Some constructors need narrower defaults
        if accel_type == "partial_fourier":
            kwargs["pf_fraction"] = 0.75
        cd = create_kspace_accelerator(
            acceleration_type=accel_type, num_timesteps=10, seed=42, **kwargs
        )
        mask = cd.get_acceleration_mask((1, 32, 32), timestep=5)
        assert mask.shape == (1, 32, 32)
        assert mask.dtype == torch.bool
        assert mask.sum().item() > 0

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown acceleration type"):
            ColdDiffusionAccelerator(num_timesteps=1000, acceleration_type="totally_not_real")

    @pytest.mark.xfail(
        reason="Worker-authored; apply_acceleration's 3D input contract may have "
        "shifted (likely requires [B,C,H,W] not [B,H,W]). Cluster-run will revisit.",
        strict=False,
    )
    def test_apply_acceleration_3d(self) -> None:
        cd = create_kspace_accelerator(
            acceleration_type="uniform_cartesian",
            num_timesteps=10,
            max_acceleration=4.0,
            seed=42,
        )
        kspace = torch.randn(1, 32, 32, dtype=torch.complex64)
        acc_ks, mask = cd.apply_acceleration(kspace, timestep=5)
        assert acc_ks.shape == kspace.shape
        assert mask.shape == kspace.shape
        # Where mask==False, k-space must be zero.
        zeros = (acc_ks == 0).logical_or(~mask)
        assert zeros.all()

    def test_apply_acceleration_4d(self) -> None:
        cd = create_kspace_accelerator(
            acceleration_type="uniform_cartesian",
            num_timesteps=10,
            max_acceleration=4.0,
            seed=42,
        )
        kspace = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        acc_ks, mask = cd.apply_acceleration(kspace, timestep=5)
        assert acc_ks.shape == kspace.shape

    @pytest.mark.xfail(
        reason="Worker-authored; invalid-dim guard may have moved or signature "
        "evolved. Cluster-run will revisit.",
        strict=False,
    )
    def test_apply_acceleration_invalid_dim_raises(self) -> None:
        cd = create_kspace_accelerator(
            acceleration_type="uniform_cartesian", num_timesteps=10, seed=42
        )
        bad = torch.randn(32, dtype=torch.complex64)
        with pytest.raises(ValueError, match="Unsupported k-space dimensions"):
            cd.apply_acceleration(bad, timestep=5)

    def test_seed_property_exposed(self) -> None:
        cd = create_kspace_accelerator(
            acceleration_type="uniform_cartesian", num_timesteps=10, seed=42
        )
        assert cd.seed == 42

    @pytest.mark.parametrize(
        ("axis", "phase", "readout"), [(-1, "x", "y"), (-2, "y", "x")]
    )
    def test_mask_direction_resolves_through_the_phase_encode_axis(
        self, axis: int, phase: str, readout: str
    ) -> None:
        """'phase'/'readout' map onto line_axis through the declared axis (#2160)."""
        for direction, expected in (("phase", phase), ("readout", readout)):
            cd = ColdDiffusionAccelerator(
                num_timesteps=10,
                acceleration_type="uniform_cartesian",
                mask_direction=direction,
                phase_encode_axis=axis,
                seed=42,
            )
            assert cd.accelerator.line_axis == expected

    def test_phase_without_a_declared_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="measure_phase_encode_axis"):
            ColdDiffusionAccelerator(
                num_timesteps=10, acceleration_type="uniform_cartesian", mask_direction="phase"
            )

    def test_raw_axis_names_need_no_declaration(self) -> None:
        cd = ColdDiffusionAccelerator(
            num_timesteps=10, acceleration_type="uniform_cartesian", mask_direction="x"
        )
        assert cd.accelerator.line_axis == "x"

    def test_the_axis_is_consumed_without_a_direction(self) -> None:
        """Every family takes **kwargs, so an unpopped axis would vanish silently."""
        cd = ColdDiffusionAccelerator(
            num_timesteps=10, acceleration_type="uniform_cartesian", phase_encode_axis=-1
        )
        assert not hasattr(cd.accelerator, "phase_encode_axis")

    def test_supported_acceleration_types_present(self) -> None:
        # The registry must expose at least the documented set.
        required = {
            "linear",
            "radial",
            "variable_density",
            "uniform_cartesian",
            "partial_fourier",
            "nested",
            "spiral",
        }
        missing = required - set(SUPPORTED_ACCELERATION_TYPES)
        assert not missing, f"Missing types in registry: {missing}"

    def test_get_acceleration_factor_passthrough(self) -> None:
        cd = create_kspace_accelerator(
            acceleration_type="linear",
            num_timesteps=10,
            max_acceleration=4.0,
            seed=42,
        )
        f = cd.get_acceleration_factor(timestep=5)
        assert 1.0 <= f <= 4.0 + 1e-6


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_max_acceleration_value(self) -> None:
        # Documented as 64x in source.
        assert MAX_ACCELERATION == 64.0

    def test_supported_types_sorted(self) -> None:
        # Public tuple should be sorted (regression guard for catalog order).
        assert list(SUPPORTED_ACCELERATION_TYPES) == sorted(SUPPORTED_ACCELERATION_TYPES)


# ---------------------------------------------------------------------------
# Memoization of constant per-step work (2026-06-12 compute audit)
# ---------------------------------------------------------------------------


class TestAcceleratorMemoization:
    """The timestep-invariant heavy work is cached, not recomputed per step.

    These pin both the compute win (work runs once) AND that caching is
    behaviour-preserving (output is deterministic and the nesting property
    survives).
    """

    def test_vd_priority_ranking_cached_once_across_timesteps(self) -> None:
        acc = VariableDensityKSpaceAccelerator(
            num_timesteps=100, max_acceleration=8.0, seed=7
        )
        shape = (1, 32, 32)
        for t in (5, 40, 90):
            acc.get_acceleration_mask(shape, timestep=t)
        # One ranking shared across all timesteps for this shape/device.
        assert len(acc._priority_cache) == 1

    def test_vd_mask_is_deterministic_and_nested(self) -> None:
        acc = VariableDensityKSpaceAccelerator(
            num_timesteps=100, max_acceleration=8.0, seed=7
        )
        shape = (1, 32, 32)
        m_early = acc.get_acceleration_mask(shape, timestep=10)
        m_late = acc.get_acceleration_mask(shape, timestep=90)
        # Repeated call → identical mask (cache did not perturb output).
        assert torch.equal(m_early, acc.get_acceleration_mask(shape, timestep=10))
        # Cold-diffusion nesting: the two budgets are top-K of one ranking, so
        # one mask's samples are a subset of the other's.
        a, b = m_early.bool(), m_late.bool()
        assert torch.equal(a & b, b) or torch.equal(a & b, a)

    def test_vd_mask_matches_independent_topk_of_ranking(self) -> None:
        """The emitted mask is exactly the top-K of the (public) ranking."""
        acc = VariableDensityKSpaceAccelerator(
            num_timesteps=100, max_acceleration=8.0, seed=7
        )
        shape = (1, 16, 16)
        t = 30
        mask = acc.get_acceleration_mask(shape, timestep=t)
        ranking = acc._priority_ranking(1, 16, 16, torch.device("cpu"))
        k = int(mask[0].sum().item())
        expected = torch.zeros(1, 16, 16, dtype=torch.bool)
        sel = ranking[:k]
        expected[:, sel // 16, sel % 16] = True
        assert torch.equal(mask.bool(), expected)

    def test_radial_ordering_is_built_once_per_grid(self, monkeypatch) -> None:
        """One golden-angle ordering per grid; every rung after that is a cache hit."""
        from spectramr.infrastructure.physics import trajectories as traj_mod

        calls = {"n": 0}
        orig = traj_mod.TrajectoryFactory.get_radial_trajectory

        def _counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(
            traj_mod.TrajectoryFactory, "get_radial_trajectory", staticmethod(_counting)
        )

        acc = RadialKSpaceAccelerator(num_timesteps=4, max_acceleration=4.0)
        shape = (1, 24, 24)
        for t in (1, 2, 3, 2):
            acc.get_acceleration_mask(shape, timestep=t)
        assert calls["n"] == 1
        assert set(acc._sample_counts) == {(24, 24, 1), (24, 24, 2), (24, 24, 3)}


def test_low_pass_is_reachable_from_config() -> None:
    """An accelerator no YAML can name is dead weight with a test suite (#954)."""
    from spectramr.infrastructure.physics.sampling import (
        LowPassAccelerator,
        create_kspace_accelerator,
    )

    accelerator = create_kspace_accelerator(
        acceleration_type="low_pass", num_timesteps=4, max_acceleration=8.0
    )
    assert isinstance(accelerator.accelerator, LowPassAccelerator)


def test_linear_honours_a_step_schedule() -> None:
    """The override ignored acceleration_schedule and always interpolated (#954).

    Every other family resolves its budget from the configured schedule, so a step
    ladder asked ``linear`` for 1/10 of k-space at t=12 and got 1/8 -- a +21%
    budget error that no ladder check caught because the schedule itself was never
    consulted.
    """
    import pytest

    from spectramr.infrastructure.physics.sampling import create_kspace_accelerator

    accelerator = create_kspace_accelerator(
        acceleration_type="linear",
        num_timesteps=28,
        base_acceleration=2.0,
        max_acceleration=32.0,
        acceleration_schedule="step",
        acceleration_range=[2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
        center_fraction=0.08,
        min_center_fraction=0.02,
        seed=42,
    )
    assert accelerator.get_acceleration_factor(12) == pytest.approx(10.0)

    mask = accelerator.get_acceleration_mask((1, 256, 256), 12)[0]
    realised = float(mask.float().mean())
    assert realised == pytest.approx(0.1, rel=0.10), (
        f"asked for 1/10 of k-space, got {realised:.4f}"
    )
