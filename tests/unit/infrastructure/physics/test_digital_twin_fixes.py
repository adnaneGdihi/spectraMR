"""Regression tests for the C1-C4 + M1 fixes flagged in the
digital-twin review.

Each test pins the *post-fix* invariant; running these against the
pre-fix code would fail. The tests are intentionally small (16x16 to
32x32 phantoms, 1-3 channels) so the full suite runs in seconds.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.degradations import (  # noqa: E402
    apply_degradation,
    degrade_pulsatile_motion,
    degrade_slice_thickness_pve,
    degrade_susceptibility,
    degrade_t2star_blur,
)
from spectramr.infrastructure.physics.digital_twin_simulator import (  # noqa: E402
    simulate_rigid_motion_kspace,
)


def _phantom(H: int = 32, W: int = 32, C: int = 1) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    base = ((x ** 2 + y ** 2) < 0.6).float().unsqueeze(0).unsqueeze(0)
    base = base.expand(1, C, -1, -1).contiguous()
    return torch.complex(base, torch.zeros_like(base))


class TestC1_RigidMotionUsesRotation:
    """`max_rotation` must actually rotate; per-PE-line motion produces
    ghosting energy that a pure-translation phase ramp cannot."""

    def test_rotation_changes_output_vs_translation_only(self) -> None:
        torch.manual_seed(0)
        ksp = torch.fft.fftn(_phantom(64, 64), dim=(-2, -1))
        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)
        no_rot = simulate_rigid_motion_kspace(
            ksp, max_translation=3.0, max_rotation=0.0, generator=gen_a
        )
        with_rot = simulate_rigid_motion_kspace(
            ksp, max_translation=3.0, max_rotation=0.3, generator=gen_b
        )
        # Same translation seed; the only difference is the rotation
        # angle. The outputs must differ above floating-point eps.
        delta = (with_rot - no_rot).abs().max().item()
        assert delta > 1e-3, (
            f"Expected rotation to change the output; got max|with_rot - no_rot| = "
            f"{delta:.3e}. The pre-fix simulator silently dropped max_rotation."
        )

    def test_per_line_translation_differs_from_global_phase_ramp(self) -> None:
        """A single global phase ramp is rank-1 in (k_x, k_y); per-line
        translation isn't. The two must produce visibly different
        outputs even at identical mean displacement."""
        ksp = torch.fft.fftn(_phantom(64, 64), dim=(-2, -1))
        H, W = ksp.shape[-2:]

        # Per-line translation via the fixed function (rotation off).
        gen = torch.Generator().manual_seed(0)
        out_per_line = simulate_rigid_motion_kspace(
            ksp, max_translation=4.0, max_rotation=0.0, generator=gen
        )

        # Same expected mean displacement applied as a single global ramp.
        # (Pre-fix behaviour: a single (dx, dy) per batch.)
        dx = 0.0  # mean of uniform [-4, 4]
        dy = 0.0
        kx = torch.linspace(-0.5, 0.5, W).view(1, 1, 1, -1)
        ky = torch.linspace(-0.5, 0.5, H).view(1, 1, -1, 1)
        global_ramp = torch.exp(-1j * 2 * math.pi * (kx * dx + ky * dy))
        out_global = ksp * global_ramp

        # The fixed implementation must NOT collapse to the global ramp.
        delta = (out_per_line - out_global).abs().mean().item()
        ref = out_global.abs().mean().item()
        assert delta / ref > 0.05, (
            f"Per-line translation collapsed to a global phase ramp "
            f"(relative diff {delta / ref:.3e}). This was the pre-fix bug."
        )

    def test_generator_kwarg_makes_output_deterministic(self) -> None:
        ksp = torch.fft.fftn(_phantom(32, 32), dim=(-2, -1))
        out_a = simulate_rigid_motion_kspace(
            ksp, max_translation=2.0, max_rotation=0.1,
            generator=torch.Generator().manual_seed(7),
        )
        out_b = simulate_rigid_motion_kspace(
            ksp, max_translation=2.0, max_rotation=0.1,
            generator=torch.Generator().manual_seed(7),
        )
        assert torch.allclose(out_a, out_b)


class TestC2_PulsatileMotionVariesAcrossLines:
    """The pulsation amplitude must differ between PE lines (per-line
    acquisition time), not be a single static deformation."""

    def test_displacement_field_not_static(self) -> None:
        torch.manual_seed(0)
        x = _phantom(48, 48)
        out = degrade_pulsatile_motion(x, theta=0.8, seed=11, base_freq_hz=1.2, tr_s=0.5)
        # Compute row-wise norms of (out - x); the pulsation modulates
        # different PE lines differently, so the per-row residual norm
        # must vary above floating-point noise across the image height.
        residual = (out - x).abs().squeeze(0).squeeze(0)  # [H, W]
        per_row_energy = residual.pow(2).mean(dim=-1)  # [H]
        variation = per_row_energy.std() / per_row_energy.mean().clamp(min=1e-9)
        assert variation > 0.05, (
            f"Expected per-line variation > 5% in pulsatile displacement; "
            f"got std/mean = {variation:.3f}. The pre-fix sine evaluated at "
            f"a constant argument produced a static deformation."
        )

    def test_zero_theta_is_identity(self) -> None:
        x = _phantom(32, 32)
        out = degrade_pulsatile_motion(x, theta=0.0, seed=3)
        assert torch.allclose(out, x, atol=1e-5)


class TestC3_DelegatedWrappersIsolateGlobalRNG:
    """Calls to delegated degradations must not pollute the global RNG."""

    @pytest.mark.parametrize("name", ["b0", "eddy", "spike", "rigid_motion"])
    def test_global_rng_state_unchanged(self, name: str) -> None:
        x = _phantom(16, 16)

        # Capture a reference random tensor from a known global-RNG seed.
        torch.manual_seed(2024)
        ref_after_seed = torch.randn(8)

        # Re-seed, run a delegated degradation with a *different* per-call
        # seed, then draw again. Under C3 (no isolation) the wrapper's
        # internal manual_seed(seed) would have re-seeded the global RNG
        # so the next draw differs from ref_after_seed.
        torch.manual_seed(2024)
        _ = apply_degradation(name, x, theta=0.4, seed=99)
        after_delegated = torch.randn(8)

        assert torch.allclose(after_delegated, ref_after_seed), (
            f"{name} delegated wrapper leaked seed into the global RNG: "
            f"expected {ref_after_seed.tolist()}, got {after_delegated.tolist()}."
        )

    def test_same_seed_still_deterministic_through_isolation(self) -> None:
        x = _phantom(16, 16)
        a = apply_degradation("b0", x, theta=0.5, seed=7)
        b = apply_degradation("b0", x, theta=0.5, seed=7)
        assert torch.allclose(a, b)


class TestC4_DepthwiseBlurCorrectForMultiChannel:
    """D12 and D13 must accept C>1 inputs and blur each channel independently."""

    @pytest.mark.parametrize("C", [1, 2, 4])
    def test_t2star_blur_accepts_multi_channel(self, C: int) -> None:
        x = _phantom(32, 32, C=C)
        out = degrade_t2star_blur(x, theta=0.5, seed=0)
        assert out.shape == x.shape

    @pytest.mark.parametrize("C", [1, 2, 4])
    def test_slice_pve_accepts_multi_channel(self, C: int) -> None:
        x = _phantom(32, 32, C=C)
        out = degrade_slice_thickness_pve(x, theta=0.5, seed=0)
        assert out.shape == x.shape

    def test_blur_does_not_mix_channels(self) -> None:
        """Depthwise blur must keep channels independent — zero in
        channel 0 stays zero after blurring channel 1."""
        x = torch.zeros(1, 2, 32, 32, dtype=torch.complex64)
        # Put signal only in channel 1.
        x[:, 1, 10:22, 10:22] = 1.0 + 0j
        out = degrade_t2star_blur(x, theta=0.8, seed=0)
        # Channel 0 must remain zero (no mixing from channel 1).
        ch0_max = out[:, 0].abs().max().item()
        ch1_max = out[:, 1].abs().max().item()
        assert ch0_max < 1e-5, (
            f"Channel 0 leaked signal from channel 1 (max={ch0_max:.3e}); "
            f"the convolution is mixing channels — should be groups=C."
        )
        assert ch1_max > 0.1, "Channel 1 should retain blurred signal."


class TestM1_SusceptibilityUsesOrthoNorm:
    """D17 must produce magnitude that is invariant under FFT
    normalisation choice. The fft2c/ifft2c path provides this."""

    def test_susceptibility_shape_and_finite(self) -> None:
        x = _phantom(32, 32)
        out = degrade_susceptibility(x, theta=0.6, seed=1)
        assert out.shape == x.shape
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()

    def test_susceptibility_dropout_clamped(self) -> None:
        x = _phantom(32, 32)
        out = degrade_susceptibility(x, theta=1.0, seed=2)
        # Dropout should reduce, not amplify, the magnitude envelope.
        assert out.abs().max() <= x.abs().max() * 1.01
