"""The magnitude clamp that was not a clamp (issue #1281).

``experiment_11_attention_none`` declares ``reverse_clip_ratio: 1.3`` and
realised **29.8x** on real M4Raw k-space. Two independent defects compounded:

1. the clamp bounded ``|Re|`` and ``|Im`` SEPARATELY on interleaved real-stacked
   complex k-space -- a square of half-width ``ceiling``, admitting ``sqrt(2) *
   ceiling`` at the corner and ROTATING phase, under a source comment that
   called it "phase-preserving";
2. the ratio was applied in ``log1p`` units while being documented and declared
   as a physical multiplier, which makes the physical bound
   ``expm1(f*m)/expm1(m)`` -- exponential in the dynamic range.

Each test below fails against exactly one of those defects, so a partial revert
cannot go unnoticed.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.diffusion.kspace_process import (
    apply_ceiling_ratio,
    clamp_to_magnitude_ceiling,
    paired_magnitude,
    paired_squared_magnitude,
)


class TestPairedMagnitude:
    def test_matches_torch_complex_abs(self) -> None:
        """The paired reading IS the complex modulus, not an approximation."""
        torch.manual_seed(0)
        re, im = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
        interleaved = torch.stack([re, im], dim=2).reshape(2, 6, 8, 8)
        expected = torch.complex(re, im).abs()
        assert torch.allclose(paired_magnitude(interleaved), expected, atol=1e-6)

    def test_elementwise_abs_understates_the_modulus(self) -> None:
        """The exact defect: at |Re| == |Im| the elementwise read is off by sqrt(2)."""
        x = torch.tensor([[[[3.0]], [[3.0]]]])  # one coefficient, Re == Im == 3
        assert float(x.abs().max()) == pytest.approx(3.0)
        assert float(paired_magnitude(x).max()) == pytest.approx(3.0 * math.sqrt(2))

    def test_complex_input_passes_through(self) -> None:
        z = torch.complex(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
        assert torch.equal(paired_magnitude(z), z.abs())

    def test_odd_channel_count_raises_rather_than_guessing(self) -> None:
        """CLAUDE.md #3: an undefined layout must raise, never degrade quietly."""
        with pytest.raises(ValueError, match="even channel count"):
            paired_magnitude(torch.randn(1, 3, 4, 4))


class TestRadialClamp:
    @staticmethod
    def _corner(mag: float) -> torch.Tensor:
        """One coefficient on the Re == Im diagonal -- the box clamp's worst case."""
        c = mag / math.sqrt(2)
        return torch.tensor([[[[c]], [[c]]]])

    def test_box_clamp_admitted_sqrt2_times_the_ceiling(self) -> None:
        """Characterises the DEFECT, so a revert re-reddens this file."""
        x = self._corner(100.0)
        ceiling = torch.ones(1, 1, 1, 1)
        boxed = x * (ceiling / x.abs().clamp_min(1e-8)).clamp(max=1.0)
        assert float(paired_magnitude(boxed).max()) == pytest.approx(
            math.sqrt(2), rel=1e-6
        )

    def test_radial_clamp_admits_exactly_the_ceiling(self) -> None:
        x = self._corner(100.0)
        out = clamp_to_magnitude_ceiling(x, torch.ones(1, 1, 1, 1))
        assert float(paired_magnitude(out).max()) == pytest.approx(1.0, rel=1e-6)

    def test_radial_clamp_preserves_phase_exactly(self) -> None:
        """The box clamp moved 5.71 deg -> 26.57 deg; in MRI phase IS the image."""
        x = torch.tensor([[[[10.0]], [[1.0]]]])  # arg = atan2(1, 10) = 5.71 deg
        before = math.atan2(1.0, 10.0)
        out = clamp_to_magnitude_ceiling(x, torch.ones(1, 1, 1, 1))
        after = math.atan2(float(out[0, 1, 0, 0]), float(out[0, 0, 0, 0]))
        assert after == pytest.approx(before, abs=1e-7)

        boxed = x * (torch.ones(1, 1, 1, 1) / x.abs().clamp_min(1e-8)).clamp(max=1.0)
        boxed_arg = math.atan2(float(boxed[0, 1, 0, 0]), float(boxed[0, 0, 0, 0]))
        assert abs(boxed_arg - before) > math.radians(20.0)

    def test_below_the_ceiling_is_a_no_op(self) -> None:
        """A bound must not perturb data that already satisfies it."""
        torch.manual_seed(0)
        x = torch.randn(2, 4, 6, 6) * 0.01
        out = clamp_to_magnitude_ceiling(x, torch.ones(2, 1, 6, 6))
        assert torch.allclose(out, x, atol=1e-7)

    def test_accepts_both_production_ceiling_shapes(self) -> None:
        """band_local gives [B,1,H,W]; global_max gives [B,1,1,1]. Both broadcast."""
        x = torch.randn(2, 8, 6, 6)
        for ceiling in (torch.full((2, 1, 6, 6), 0.5), torch.full((2, 1, 1, 1), 0.5)):
            out = clamp_to_magnitude_ceiling(x, ceiling)
            assert out.shape == x.shape
            assert float(paired_magnitude(out).max()) <= 0.5 + 1e-6


class TestCeilingRatioDomain:
    def test_physical_domain_is_a_plain_product(self) -> None:
        ref = torch.tensor([2.0])
        assert torch.allclose(
            apply_ceiling_ratio(ref, 1.3, log_scaled=False), torch.tensor([2.6])
        )

    def test_log_domain_ceiling_restores_the_declared_physical_ratio(self) -> None:
        """log1p(ratio * expm1(ref)) is the compression of the physical ceiling."""
        ref = torch.tensor([4.0284])  # measured compressed |z|max, real M4Raw
        ceil = apply_ceiling_ratio(ref, 1.3, log_scaled=True)
        realised = torch.expm1(ceil) / torch.expm1(ref)
        assert float(realised) == pytest.approx(1.3, rel=1e-6)

    def test_the_old_log_domain_ratio_was_exponential(self) -> None:
        """Characterises defect (2) and shows the two defects COMPOUND.

        At this arm's measured compressed dynamic range (``|z|max = 4.0284``,
        real M4Raw through the production loader) a declared 1.3 realised:

            log1p ratio alone            expm1(1.3*m)/expm1(m)          =  3.39x
            box clamp alone              sqrt(2) * 1.3                  =  1.84x
            both, as shipped             expm1(sqrt(2)*1.3*m)/expm1(m)  = 29.8x

        29.8 is far more than 3.39 * 1.84 = 6.2, which is the point: the
        sqrt(2) enters the EXPONENT, so the defects multiply super-linearly and
        fixing either one alone leaves a badly-violated bound.
        """
        ref = 4.0284
        log_only = math.expm1(1.3 * ref) / math.expm1(ref)
        assert log_only == pytest.approx(3.39, rel=0.01)

        box_only = math.sqrt(2) * 1.3
        assert box_only == pytest.approx(1.84, rel=0.01)

        compounded = math.expm1(math.sqrt(2) * 1.3 * ref) / math.expm1(ref)
        assert compounded == pytest.approx(29.8, rel=0.01)
        assert compounded > log_only * box_only * 4  # super-linear, not a product


class TestTheClampHasAFiniteGradientAtAnExactZero:
    """Hard DC copies the M4Raw readout window's zeros into the generator output.

    Flooring the modulus AFTER ``sqrt`` guarded the forward divide and not the
    backward: ``clamp_min`` returns 0 into sqrt's infinite gradient, 0 * inf is
    NaN, and on experiment_11_attention_dual_domain every one of the generator's
    204 gradient tensors came back NaN while the loss stayed finite. The first
    optimizer step then made every weight NaN and every validation sample
    non-finite.
    """

    @staticmethod
    def _grad(x: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        out = clamp_to_magnitude_ceiling(x, torch.tensor(0.5))
        (torch.view_as_real(out) if out.is_complex() else out).sum().backward()
        return x.grad

    def test_interleaved_zeros(self) -> None:
        x = torch.randn(1, 4, 4, 4)
        x[..., :2] = 0.0  # both halves of every coefficient in two columns
        assert torch.isfinite(self._grad(x)).all()

    def test_complex_zeros(self) -> None:
        """Complex ``abs()`` was never NaN-prone; the fix takes ``sqrt`` here too."""
        z = torch.complex(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
        z[..., :2] = 0
        assert torch.isfinite(self._grad(z)).all()

    def test_the_forward_value_is_unchanged(self) -> None:
        """Only the gradient moves: the floored modulus is still max(|z|, eps)."""
        torch.manual_seed(0)
        x = torch.randn(2, 6, 8, 8) * 3
        x[..., :3] = 0.0
        ceiling = torch.tensor(1.0)
        legacy = x * (ceiling / paired_magnitude(x).clamp_min(1e-8)).clamp(max=1.0).repeat_interleave(2, dim=-3)
        assert torch.equal(clamp_to_magnitude_ceiling(x, ceiling, 1e-8), legacy)

    def test_the_squared_modulus_agrees_with_the_modulus(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 6, 8, 8)
        assert torch.allclose(paired_squared_magnitude(x).sqrt(), paired_magnitude(x), atol=1e-6)
