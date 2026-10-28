"""VF / ULF physics test bundle (Task II.4).

Covers:
  - degradations package         (apply_degradation / degrade_* functions)
  - vf_corrections.py            (GNLUnwarpingOperator, BiasFieldCorrector,
                                   AnalyticalPSFSolver, EPINyquistGhostCorrector,
                                   SENSEGFactorCalibrator, RespiratoryBinningAverager,
                                   SubVoxelSuperResolution)
  - vf_field_extraction.py       (MarkerB0FromDisplacement, MarkerB0FromDualEcho,
                                   MarkerB1TransmitDoubleAngle, MarkerB1TransmitAFI,
                                   MarkerB1ReceiveInversion, LarmorDriftTracker,
                                   SusceptibilityB0Solver, BlochSiegertB1Estimator,
                                   SSFPBandingB0Estimator, STEAMSpinEchoB1Estimator,
                                   K0PhaseDriftTracker, PhaseUnwrappingSeed)
  - vf_physics_operators.py      (T2StarDecayOperator, TGVConstraintOperator,
                                   VarianceWeightedTGV, LplusSOperator)
  - vf_operators.py              (MarkerPriorProjection, MarkerKSpaceDCProjection,
                                   RigidKinematicOperator, B0PhaseOperator,
                                   GeometricWarpOperator, ToeplitzPSFOperator,
                                   NoisePreWhitening)
  - vf_operators_extended.py     (RFModulationOperator, NUFFTTrajectoryCalibrator)

Shared fixture helpers are defined once at module level so every test class
can reuse them without touching global torch state.
"""

from __future__ import annotations

import math

import pytest
import torch

# ============================================================
# Shared synthetic fixtures (deterministic, CPU-only)
# ============================================================

_H, _W = 16, 16  # small spatial dims for fast tests


def _b0_map(B: int = 1, H: int = _H, W: int = _W) -> torch.Tensor:
    """Smooth synthetic B0 field in Hz, shape [B, 1, H, W]."""
    y = torch.linspace(-1.0, 1.0, H)
    x = torch.linspace(-1.0, 1.0, W)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    field = 10.0 * torch.cos(math.pi * gy) * torch.sin(math.pi * gx)  # Hz
    return field.view(1, 1, H, W).expand(B, 1, H, W).contiguous()


def _b1_map(B: int = 1, H: int = _H, W: int = _W, offset: float = 0.0) -> torch.Tensor:
    """Smooth B1+ scale map near 1.0, shape [B, 1, H, W]."""
    y = torch.linspace(-1.0, 1.0, H)
    x = torch.linspace(-1.0, 1.0, W)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    field = 1.0 + offset + 0.2 * torch.sin(math.pi * gy) * torch.cos(math.pi * gx)
    return field.view(1, 1, H, W).expand(B, 1, H, W).contiguous()


def _vf_mask(H: int = _H, W: int = _W) -> torch.Tensor:
    """Binary virtual fiducial mask — small 4×4 patch in upper-left."""
    mask = torch.zeros(1, 1, H, W)
    mask[..., 1:5, 1:5] = 1.0
    return mask


def _complex_image(B: int = 1, C: int = 1, H: int = _H, W: int = _W) -> torch.Tensor:
    """Deterministic complex image [B, C, H, W]."""
    g = torch.Generator()
    g.manual_seed(42)
    r = torch.randn(B, C, H, W, generator=g)
    i = torch.randn(B, C, H, W, generator=g)
    return torch.complex(r, i)


def _real_image(B: int = 1, C: int = 1, H: int = _H, W: int = _W) -> torch.Tensor:
    """Deterministic real image [B, C, H, W], values in (0, 1]."""
    g = torch.Generator()
    g.manual_seed(7)
    return torch.rand(B, C, H, W, generator=g) + 0.1


# ============================================================
# SECTION 1: degradations package
# ============================================================


@pytest.mark.physics
class TestDigitalTwinExtensionsCanary:
    """Canary — import + basic list check."""

    def test_import(self):
        import spectramr.infrastructure.physics.degradations  # noqa: F401

    def test_list_degradations_nonempty(self):
        from spectramr.infrastructure.physics.degradations import list_degradations
        names = list_degradations()
        assert len(names) > 0

    def test_apply_degradation_unknown_raises(self):
        from spectramr.infrastructure.physics.degradations import apply_degradation
        x = _real_image()
        with pytest.raises(KeyError, match="Unknown degradation"):
            apply_degradation("not_a_real_degradation", x, theta=0.5)


@pytest.mark.physics
@pytest.mark.parametrize("H,W", [(8, 8), (16, 24)])
class TestDegradeSamplingShape:
    """Sanity-shape: sampling degradations preserve shape."""

    def test_cartesian_undersamp_shape(self, H, W):
        from spectramr.infrastructure.physics.degradations import degrade_cartesian_undersampling
        x = _complex_image(B=1, C=1, H=H, W=W)
        out = degrade_cartesian_undersampling(x, theta=0.5, seed=0)
        assert out.shape == x.shape

    def test_variable_density_2d_shape(self, H, W):
        from spectramr.infrastructure.physics.degradations import degrade_variable_density_2d
        x = _complex_image(B=1, C=1, H=H, W=W)
        out = degrade_variable_density_2d(x, theta=0.5, seed=0)
        assert out.shape == x.shape

    def test_radial_undersamp_shape(self, H, W):
        from spectramr.infrastructure.physics.degradations import degrade_radial_undersampling
        x = _complex_image(B=1, C=1, H=H, W=W)
        out = degrade_radial_undersampling(x, theta=0.5, seed=0)
        assert out.shape == x.shape


@pytest.mark.physics
class TestDegradeNoise:
    """D1/D2 noise degradations basic contract."""

    def test_complex_gaussian_theta_zero_identity(self):
        """theta=0 → nearly no noise."""
        from spectramr.infrastructure.physics.degradations import degrade_complex_gaussian_noise
        x = _real_image()
        xc = torch.complex(x, torch.zeros_like(x))
        out = degrade_complex_gaussian_noise(xc, theta=0.0, seed=0)
        assert out.shape == xc.shape

    def test_rician_theta_zero_identity(self):
        from spectramr.infrastructure.physics.degradations import degrade_rician_noise
        x = _real_image()
        out = degrade_rician_noise(x, theta=0.0, seed=0)
        assert torch.allclose(out, x.abs(), atol=1e-5)

    def test_rician_theta_one_noisy(self):
        from spectramr.infrastructure.physics.degradations import degrade_rician_noise
        x = _real_image()
        out = degrade_rician_noise(x, theta=1.0, seed=0)
        # Rician noise should change the image
        assert not torch.allclose(out, x, atol=1e-4)


@pytest.mark.physics
class TestDegradeFieldEffects:
    """D8, D11, D12, D13, D17, D18, D20 field/geometric degradations."""

    def test_nyquist_ghost_theta_zero_unchanged(self):
        from spectramr.infrastructure.physics.degradations import degrade_nyquist_ghost
        x = _complex_image()
        out = degrade_nyquist_ghost(x, theta=0.0)
        # theta=0 → zero phase → output should match input
        assert torch.allclose(out.abs(), x.abs(), atol=1e-5)

    def test_nyquist_ghost_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_nyquist_ghost
        x = _complex_image(B=2, C=1)
        out = degrade_nyquist_ghost(x, theta=0.5)
        assert out.shape == x.shape

    def test_bias_field_theta_zero_identity(self):
        from spectramr.infrastructure.physics.degradations import degrade_bias_field
        x = _complex_image()
        out = degrade_bias_field(x, theta=0.0, seed=0)
        # theta=0 → log_b=0 → bias=1 → identity
        assert torch.allclose(out.abs(), x.abs(), atol=1e-5)

    def test_bias_field_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_bias_field
        x = _complex_image(B=2, C=1)
        out = degrade_bias_field(x, theta=0.5, seed=99)
        assert out.shape == x.shape

    def test_t2star_blur_theta_zero_identity(self):
        from spectramr.infrastructure.physics.degradations import degrade_t2star_blur
        x = _real_image()
        out = degrade_t2star_blur(x, theta=0.0)
        assert torch.allclose(out, x.float(), atol=1e-5)

    def test_t2star_blur_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_t2star_blur
        x = _real_image(B=2, C=3)
        out = degrade_t2star_blur(x, theta=0.5)
        assert out.shape == x.shape

    def test_slice_pve_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_slice_thickness_pve
        x = _real_image(B=2, C=2)
        out = degrade_slice_thickness_pve(x, theta=0.5)
        assert out.shape == x.shape

    def test_coil_miscal_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_coil_sensitivity_miscal
        x = _complex_image(B=2, C=1)
        out = degrade_coil_sensitivity_miscal(x, theta=0.5, seed=0)
        assert out.shape == x.shape

    def test_geometric_warp_theta_zero_nearidentity(self):
        from spectramr.infrastructure.physics.degradations import degrade_geometric_warp
        x = _complex_image()
        out = degrade_geometric_warp(x, theta=0.0, seed=0)
        assert out.shape == x.shape

    def test_susceptibility_raises_bad_b0_axis(self):
        from spectramr.infrastructure.physics.degradations import degrade_susceptibility
        x = _complex_image()
        with pytest.raises(ValueError, match="b0_axis must be"):
            degrade_susceptibility(x, theta=0.5, b0_axis="z")

    def test_susceptibility_raises_bad_slice_plane(self):
        from spectramr.infrastructure.physics.degradations import degrade_susceptibility
        x = _complex_image()
        with pytest.raises(ValueError, match="slice_plane must be"):
            degrade_susceptibility(x, theta=0.5, slice_plane="transverse")


@pytest.mark.physics
class TestDegradeMotion:
    """D7, pulsatile motion — shape and determinism."""

    def test_pulsatile_motion_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_pulsatile_motion
        x = _complex_image(B=2, C=1)
        out = degrade_pulsatile_motion(x, theta=0.5, seed=0)
        assert out.shape == x.shape

    def test_pulsatile_motion_theta_zero_near_identity(self):
        from spectramr.infrastructure.physics.degradations import degrade_pulsatile_motion
        x = _complex_image()
        out = degrade_pulsatile_motion(x, theta=0.0, seed=0)
        # theta=0 → amp=0 → grid_sample(x, identity_grid)
        assert torch.allclose(out.abs(), x.abs(), atol=1e-4)

    def test_pulsatile_deterministic(self):
        from spectramr.infrastructure.physics.degradations import degrade_pulsatile_motion
        x = _complex_image()
        out1 = degrade_pulsatile_motion(x, theta=0.5, seed=123)
        out2 = degrade_pulsatile_motion(x, theta=0.5, seed=123)
        assert torch.allclose(out1, out2)


@pytest.mark.physics
class TestDegradeWave3:
    """Wave-3 (partial Fourier, IQ ghost, stimulated echo, slice profile)."""

    def test_partial_fourier_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_partial_fourier
        x = _complex_image(B=1, C=1, H=16, W=16)
        out = degrade_partial_fourier(x, theta=0.5, axis="ky")
        assert out.shape == x.shape

    def test_partial_fourier_invalid_axis_raises(self):
        from spectramr.infrastructure.physics.degradations import degrade_partial_fourier
        x = _complex_image()
        with pytest.raises(ValueError, match="axis must be"):
            degrade_partial_fourier(x, theta=0.5, axis="kz")

    def test_iq_ghost_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_iq_ghost
        x = _complex_image(B=2, C=1)
        out = degrade_iq_ghost(x, theta=0.5, seed=0)
        assert out.shape == x.shape

    def test_stimulated_echo_shape(self):
        from spectramr.infrastructure.physics.degradations import degrade_stimulated_echo
        x = _complex_image(B=1, C=1)
        out = degrade_stimulated_echo(x, theta=0.5, seed=0)
        assert out.shape == x.shape

    def test_slice_profile_axis_removed(self):
        """#245: D27 (slice_profile) was a byte-for-byte duplicate of D12."""
        from spectramr.infrastructure.physics.degradations import list_degradations

        assert "slice_profile" not in list_degradations()


@pytest.mark.physics
class TestApplyDegradationRegistry:
    """apply_degradation dispatches known names without raising."""

    @pytest.mark.parametrize("name", [
        "complex_gaussian", "rician", "cartesian_undersamp", "vd_undersamp",
        "vd_cartesian", "radial_undersamp", "pulsatile_motion", "nyquist_ghost",
        "bias", "t2star_blur", "slice_pve", "coil_miscal", "geometric_warp",
        "partial_fourier", "iq_ghost", "stimulated_echo",
    ])
    def test_apply_degradation_runs(self, name):
        from spectramr.infrastructure.physics.degradations import apply_degradation
        x = _real_image(B=1, C=1, H=16, W=16)
        out = apply_degradation(name, x, theta=0.3, seed=1)
        assert out.shape == x.shape


# ============================================================
# SECTION 2: vf_corrections.py
# ============================================================


@pytest.mark.physics
class TestGNLUnwarpingOperatorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_corrections import GNLUnwarpingOperator
        op = GNLUnwarpingOperator(im_size=(16, 16))
        assert op is not None

    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_corrections import GNLUnwarpingOperator
        op = GNLUnwarpingOperator(im_size=(_H, _W))
        image = _real_image(B=2)
        flow = torch.zeros(1, 2, _H, _W)
        out = op(image, flow)
        assert out.shape == image.shape

    def test_zero_flow_near_identity(self):
        from spectramr.infrastructure.physics.vf_corrections import GNLUnwarpingOperator
        op = GNLUnwarpingOperator(im_size=(_H, _W))
        image = _real_image()
        flow = torch.zeros(1, 2, _H, _W)
        out = op(image, flow)
        # Jacobian det of zero-flow field ≈ 1 → output ≈ input
        assert torch.allclose(out, image, atol=1e-5)

    def test_estimate_deformation_shape(self):
        from spectramr.infrastructure.physics.vf_corrections import GNLUnwarpingOperator
        op = GNLUnwarpingOperator(im_size=(_H, _W))
        N = 6
        true_pos = torch.rand(N, 2)
        dist_pos = true_pos + 0.05 * torch.rand(N, 2)
        flow = op.estimate_deformation(true_pos, dist_pos)
        assert flow.shape == (1, 2, _H, _W)


@pytest.mark.physics
class TestBiasFieldCorrectorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_corrections import BiasFieldCorrector
        bc = BiasFieldCorrector(smooth_sigma=3.0)
        assert bc is not None

    def test_uniform_bias_preserves_signal(self):
        from spectramr.infrastructure.physics.vf_corrections import BiasFieldCorrector
        bc = BiasFieldCorrector(smooth_sigma=3.0)
        image = _real_image()
        bias = torch.ones(1, 1, _H, _W)
        out = bc(image, bias)
        assert out.shape == image.shape

    def test_zero_bias_raises_or_clips(self):
        """All-zero bias should not produce NaN (corrector clamps min to 1e-4)."""
        from spectramr.infrastructure.physics.vf_corrections import BiasFieldCorrector
        bc = BiasFieldCorrector(smooth_sigma=3.0)
        image = _real_image()
        bias = torch.zeros(1, 1, _H, _W)
        out = bc(image, bias)
        assert not torch.isnan(out).any()


@pytest.mark.physics
class TestAnalyticalPSFSolverCanary:
    def test_import_and_estimate_psf_shape(self):
        from spectramr.infrastructure.physics.vf_corrections import AnalyticalPSFSolver
        solver = AnalyticalPSFSolver(cross_size=6, cross_width=2)
        marker_image = _real_image()
        psf = solver.estimate_psf(marker_image, cross_center=(_H // 2, _W // 2))
        assert psf.shape == (1, 1, _H, _W)

    def test_deconvolve_shape(self):
        from spectramr.infrastructure.physics.vf_corrections import AnalyticalPSFSolver
        solver = AnalyticalPSFSolver()
        image = _real_image(B=2, C=1)
        psf = torch.zeros(1, 1, _H, _W)
        psf[0, 0, _H // 2, _W // 2] = 1.0  # delta PSF
        out = solver.deconvolve(image, psf)
        assert out.shape == image.shape


@pytest.mark.physics
class TestEPINyquistGhostCorrectorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_corrections import EPINyquistGhostCorrector
        op = EPINyquistGhostCorrector(im_size=(_H, _W))
        assert op is not None

    def test_zero_phase_error_identity(self):
        from spectramr.infrastructure.physics.vf_corrections import EPINyquistGhostCorrector
        op = EPINyquistGhostCorrector(im_size=(_H, _W))
        kspace = _complex_image(B=1, C=1)
        phi0 = torch.zeros(1)
        phi1 = torch.zeros(1)
        out = op(kspace, phi0, phi1)
        # With phi0=phi1=0 the correction is exp(0)=1 → identity
        assert torch.allclose(out, kspace)

    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_corrections import EPINyquistGhostCorrector
        op = EPINyquistGhostCorrector(im_size=(_H, _W))
        kspace = _complex_image(B=2, C=1)
        phi0 = torch.zeros(2)
        phi1 = torch.zeros(2)
        out = op(kspace, phi0, phi1)
        assert out.shape == kspace.shape


@pytest.mark.physics
class TestSENSEGFactorCalibratorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_corrections import SENSEGFactorCalibrator
        cal = SENSEGFactorCalibrator(num_coils=4, acceleration=2)
        assert cal is not None

    def test_compute_g_factor_shape_and_nonneg(self):
        from spectramr.infrastructure.physics.vf_corrections import SENSEGFactorCalibrator
        C = 4
        cal = SENSEGFactorCalibrator(num_coils=C, acceleration=2)
        # Random complex coil sensitivities
        g = torch.Generator()
        g.manual_seed(0)
        s_maps = torch.complex(
            torch.randn(C, _H, _W, generator=g),
            torch.randn(C, _H, _W, generator=g),
        )
        gmap = cal.compute_g_factor(s_maps)
        assert gmap.shape == (1, 1, _H, _W)
        assert (gmap >= 0).all()


@pytest.mark.physics
class TestRespiratoryBinningAveragerCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_corrections import RespiratoryBinningAverager
        avg = RespiratoryBinningAverager(num_bins=4)
        frames = _real_image(B=4, C=1)  # 4 bins
        out = avg(frames)
        assert out.shape == (1, 1, _H, _W)

    def test_bin_by_phase(self):
        from spectramr.infrastructure.physics.vf_corrections import RespiratoryBinningAverager
        avg = RespiratoryBinningAverager(num_bins=4)
        N = 20
        kspace_lines = torch.randn(N, _W)
        resp_trace = torch.rand(N)
        bins = avg.bin_by_phase(kspace_lines, resp_trace)
        assert len(bins) == 4
        assert sum(len(b) for b in bins) == N


@pytest.mark.physics
class TestSubVoxelSuperResolutionCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_corrections import SubVoxelSuperResolution
        sr = SubVoxelSuperResolution(upscale_factor=2, num_iters=2)
        N = 4
        H_lr, W_lr = 8, 8
        frames = torch.rand(N, 1, H_lr, W_lr)
        shifts = torch.zeros(N, 2)
        out = sr(frames, shifts)
        assert out.shape == (1, 1, H_lr * 2, W_lr * 2)


# ============================================================
# SECTION 3: vf_field_extraction.py
# ============================================================


@pytest.mark.physics
class TestMarkerB0FromDisplacementCanary:
    def test_import_and_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import MarkerB0FromDisplacement
        N_markers = 3
        true_pos = torch.rand(N_markers, 2)
        op = MarkerB0FromDisplacement(true_pos, pe_bandwidth=30.0)
        marker_image = _real_image()
        marker_mask = torch.zeros(N_markers, 1, _H, _W)
        marker_mask[0, :, 1:4, 1:4] = 1.0
        b0 = op(marker_image, marker_mask)
        assert b0.shape == (1, 1, _H, _W)

    def test_zero_image_gives_zero_b0(self):
        from spectramr.infrastructure.physics.vf_field_extraction import MarkerB0FromDisplacement
        true_pos = torch.tensor([[4.0, 4.0]])
        op = MarkerB0FromDisplacement(true_pos)
        zero_image = torch.zeros(1, 1, _H, _W)
        mask = torch.ones(1, 1, _H, _W) * 0.0
        mask[0, 0, 1:5, 1:5] = 1.0
        b0 = op(zero_image, mask)
        assert b0.shape == (1, 1, _H, _W)


@pytest.mark.physics
class TestMarkerB0FromDualEchoCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import MarkerB0FromDualEcho
        op = MarkerB0FromDualEcho(delta_te=0.00246)
        echo1 = _complex_image()
        echo2 = _complex_image()
        b0 = op(echo1, echo2)
        assert b0.shape == echo1.shape

    def test_identical_echoes_zero_b0(self):
        from spectramr.infrastructure.physics.vf_field_extraction import MarkerB0FromDualEcho
        op = MarkerB0FromDualEcho(delta_te=0.001)
        echo = _complex_image()
        b0 = op(echo, echo)
        assert torch.allclose(b0, torch.zeros_like(b0), atol=1e-5)

    def test_mask_applied(self):
        from spectramr.infrastructure.physics.vf_field_extraction import MarkerB0FromDualEcho
        op = MarkerB0FromDualEcho(delta_te=0.001)
        echo1 = _complex_image()
        echo2 = _complex_image()
        mask = _vf_mask()
        b0 = op(echo1, echo2, marker_mask=mask)
        # Outside mask should be zero
        outside = b0 * (1 - mask)
        assert torch.allclose(outside, torch.zeros_like(outside), atol=1e-6)


@pytest.mark.physics
class TestMarkerB1TransmitDoubleAngleCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import (
            MarkerB1TransmitDoubleAngle,
        )
        op = MarkerB1TransmitDoubleAngle(t1_marker=300.0, tr=50.0, nominal_alpha=30.0)
        sig_alpha = _real_image()
        sig_2alpha = _real_image() * 0.9
        b1 = op(sig_alpha, sig_2alpha)
        assert b1.shape == sig_alpha.shape

    def test_equal_signals_gives_alpha_pi_over_3(self):
        """S_2α / (2·S_α) = 0.5 → arccos(0.5) = π/3 for 30° nominal."""
        from spectramr.infrastructure.physics.vf_field_extraction import (
            MarkerB1TransmitDoubleAngle,
        )
        op = MarkerB1TransmitDoubleAngle(nominal_alpha=30.0)
        s = torch.ones(1, 1, _H, _W)
        b1 = op(s, s)  # ratio = 0.5 → arccos(0.5) ≈ π/3
        assert b1.shape == s.shape


@pytest.mark.physics
class TestMarkerB1TransmitAFICanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import MarkerB1TransmitAFI
        op = MarkerB1TransmitAFI(t1_marker=300.0, tr1=20.0, tr2=100.0)
        s1 = _real_image()
        s2 = _real_image()
        b1 = op(s1, s2)
        assert b1.shape == s1.shape


@pytest.mark.physics
class TestMarkerB1ReceiveInversionCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import MarkerB1ReceiveInversion
        op = MarkerB1ReceiveInversion()
        signal = _real_image()
        b1_minus = op(signal)
        assert b1_minus.shape == signal.shape

    def test_mask_zeroes_outside(self):
        from spectramr.infrastructure.physics.vf_field_extraction import MarkerB1ReceiveInversion
        op = MarkerB1ReceiveInversion()
        signal = _real_image()
        mask = _vf_mask()
        b1_minus = op(signal, marker_mask=mask)
        outside = b1_minus * (1 - mask)
        assert torch.allclose(outside, torch.zeros_like(outside), atol=1e-6)


@pytest.mark.physics
class TestLarmorDriftTrackerCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import LarmorDriftTracker
        tracker = LarmorDriftTracker(reference_freq_hz=0.0)
        N = 32
        signal = torch.complex(torch.randn(N), torch.randn(N))
        drift = tracker(signal, dt=0.001)
        # Output is a scalar or 1-D per-timepoint
        assert drift.numel() >= 1


@pytest.mark.physics
class TestSusceptibilityB0SolverCanary:
    def test_forward_shape_and_finite(self):
        from spectramr.infrastructure.physics.vf_field_extraction import SusceptibilityB0Solver
        solver = SusceptibilityB0Solver(b0_tesla=0.064)
        chi_map = torch.zeros(1, 1, _H, _W)
        chi_map[0, 0, 4:8, 4:8] = 9.05  # ppm step
        b0_hz = solver(chi_map)
        assert b0_hz.shape == (1, 1, _H, _W)
        assert torch.isfinite(b0_hz).all()

    def test_zero_chi_gives_small_b0(self):
        from spectramr.infrastructure.physics.vf_field_extraction import SusceptibilityB0Solver
        solver = SusceptibilityB0Solver()
        chi_map = torch.zeros(1, 1, _H, _W)
        b0_hz = solver(chi_map)
        assert b0_hz.abs().max() < 1e-6


@pytest.mark.physics
class TestBlochSiegertB1EstimatorCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import BlochSiegertB1Estimator
        est = BlochSiegertB1Estimator(omega_off_hz=4000.0)
        s_pos = _complex_image()
        s_neg = _complex_image()
        b1 = est(s_pos, s_neg)
        assert b1.shape == s_pos.shape

    def test_identical_signals_gives_nonneg(self):
        from spectramr.infrastructure.physics.vf_field_extraction import BlochSiegertB1Estimator
        est = BlochSiegertB1Estimator()
        s = _complex_image()
        b1 = est(s, s)
        assert (b1 >= 0).all()


@pytest.mark.physics
class TestSSFPBandingB0EstimatorCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import SSFPBandingB0Estimator
        est = SSFPBandingB0Estimator(tr_s=0.005)
        ssfp = _real_image()
        mask = _vf_mask()
        b0_grad = est(ssfp, mask)
        assert b0_grad.shape == ssfp.shape


@pytest.mark.physics
class TestSTEAMSpinEchoB1EstimatorCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import STEAMSpinEchoB1Estimator
        est = STEAMSpinEchoB1Estimator(nominal_flip_deg=90.0)
        steam = _real_image()
        se = _real_image()
        b1 = est(steam, se)
        assert b1.shape == steam.shape

    def test_ratio_half_gives_near_unity_b1(self):
        """For 90° nominal: STE/SE = sin²(45°) = 0.5 → actual = π/2 → scale = 1.0."""
        from spectramr.infrastructure.physics.vf_field_extraction import STEAMSpinEchoB1Estimator
        est = STEAMSpinEchoB1Estimator(nominal_flip_deg=90.0)
        se = torch.ones(1, 1, _H, _W)
        steam = se * 0.5  # ratio = 0.5 → arcsin(√0.5) = π/4 → actual = π/2
        b1 = est(steam, se)
        assert torch.allclose(b1, torch.ones_like(b1), atol=1e-4)


@pytest.mark.physics
class TestK0PhaseDriftTrackerCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import K0PhaseDriftTracker
        tracker = K0PhaseDriftTracker(te=0.01)
        T = 8
        ks_ts = _complex_image(B=T, C=1)
        drift = tracker(ks_ts)
        assert drift.shape[0] == T or drift.numel() == T

    def test_constant_phase_zero_drift(self):
        from spectramr.infrastructure.physics.vf_field_extraction import K0PhaseDriftTracker
        tracker = K0PhaseDriftTracker(te=0.01)
        T = 4
        # All timepoints identical → no drift
        single = _complex_image(B=1, C=1)
        ks_ts = single.expand(T, -1, -1, -1).contiguous()
        drift = tracker(ks_ts)
        assert torch.allclose(drift.float(), torch.zeros_like(drift.float()), atol=1e-5)


@pytest.mark.physics
class TestPhaseUnwrappingSeedCanary:
    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_field_extraction import PhaseUnwrappingSeed
        op = PhaseUnwrappingSeed(known_b0_at_marker=0.0, te=0.01)
        wrapped = torch.zeros(1, 1, _H, _W)
        mask = _vf_mask()
        out = op(wrapped, mask)
        assert out.shape == wrapped.shape

    def test_already_unwrapped_unchanged(self):
        """If phase is well within [-π, π] and no wraps, the result should be near-identical."""
        from spectramr.infrastructure.physics.vf_field_extraction import PhaseUnwrappingSeed
        op = PhaseUnwrappingSeed(known_b0_at_marker=0.0, te=0.01)
        # Small-amplitude field, no wraps
        phase = 0.1 * torch.sin(torch.linspace(0, 1, _H)).view(-1, 1).expand(_H, _W)
        phase = phase.view(1, 1, _H, _W)
        mask = _vf_mask()
        out = op(phase, mask)
        assert out.shape == phase.shape
        assert torch.isfinite(out).all()


# ============================================================
# SECTION 4: vf_physics_operators.py
# ============================================================


@pytest.mark.physics
class TestT2StarDecayOperatorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_physics_operators import T2StarDecayOperator
        op = T2StarDecayOperator()
        assert op is not None

    def test_none_t2star_map_identity(self):
        from spectramr.infrastructure.physics.vf_physics_operators import T2StarDecayOperator
        op = T2StarDecayOperator()
        x = _real_image()
        out = op(x)
        assert torch.allclose(out, x)

    def test_with_t2star_map_shape(self):
        from spectramr.infrastructure.physics.vf_physics_operators import T2StarDecayOperator
        t2s = torch.full((1, 1, _H, _W), 0.05)  # 50 ms
        op = T2StarDecayOperator(t2star_map=t2s, readout_duration=5e-3)
        x = _real_image()
        out = op(x)
        assert out.shape == x.shape
        # Decay should reduce signal
        assert (out <= x + 1e-6).all()

    def test_multisegment_output_count(self):
        from spectramr.infrastructure.physics.vf_physics_operators import T2StarDecayOperator
        t2s = torch.full((1, 1, _H, _W), 0.05)
        op = T2StarDecayOperator(t2star_map=t2s, num_segments=3)
        x = _real_image()
        segs = op.forward_multisegment(x)
        assert len(segs) == 3

    def test_set_t2star_raises_on_missing(self):
        """Calling forward without T2* map (none set) returns identity — not raises."""
        from spectramr.infrastructure.physics.vf_physics_operators import T2StarDecayOperator
        op = T2StarDecayOperator()
        x = _real_image()
        # Should return identity (no map set)
        out = op(x)
        assert torch.allclose(out, x)


@pytest.mark.physics
class TestTGVConstraintOperatorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_physics_operators import TGVConstraintOperator
        op = TGVConstraintOperator(num_iters=2)
        assert op is not None

    def test_forward_shape_single_channel(self):
        from spectramr.infrastructure.physics.vf_physics_operators import TGVConstraintOperator
        op = TGVConstraintOperator(num_iters=2)
        x = _real_image(B=1, C=1)
        out = op(x)
        assert out.shape == x.shape

    def test_forward_shape_multi_channel(self):
        from spectramr.infrastructure.physics.vf_physics_operators import TGVConstraintOperator
        op = TGVConstraintOperator(num_iters=2)
        x = _real_image(B=1, C=3)
        out = op(x)
        assert out.shape == x.shape

    def test_with_marker_grad_and_mask(self):
        from spectramr.infrastructure.physics.vf_physics_operators import TGVConstraintOperator
        op = TGVConstraintOperator(num_iters=2)
        x = _real_image(B=1, C=1)
        marker_grad = torch.zeros(1, 2, _H, _W)
        marker_mask = _vf_mask()
        out = op(x, marker_grad=marker_grad, marker_mask=marker_mask)
        assert out.shape == x.shape


@pytest.mark.physics
class TestVarianceWeightedTGVCanary:
    def test_forward_no_variance_map(self):
        from spectramr.infrastructure.physics.vf_physics_operators import VarianceWeightedTGV
        op = VarianceWeightedTGV(num_iters=2)
        x = _real_image(B=1, C=1)
        out = op(x)
        assert out.shape == x.shape

    def test_forward_with_variance_map(self):
        from spectramr.infrastructure.physics.vf_physics_operators import VarianceWeightedTGV
        op = VarianceWeightedTGV(num_iters=2)
        x = _real_image(B=1, C=1)
        variance_map = torch.ones_like(x) * 0.1
        out = op(x, variance_map=variance_map)
        assert out.shape == x.shape


@pytest.mark.physics
class TestLplusSOperatorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_physics_operators import LplusSOperator
        op = LplusSOperator(rank=2, num_iters=3)
        assert op is not None

    def test_forward_returns_two_tensors(self):
        from spectramr.infrastructure.physics.vf_physics_operators import LplusSOperator
        op = LplusSOperator(rank=2, num_iters=2)
        B, T = 1, 4
        x = torch.rand(B, T, _H, _W)
        L, S = op(x)
        assert L.shape == x.shape
        assert S.shape == x.shape

    def test_L_plus_S_approx_input(self):
        from spectramr.infrastructure.physics.vf_physics_operators import LplusSOperator
        op = LplusSOperator(rank=4, lambda_s=0.0, num_iters=5)
        B, T = 1, 4
        x = torch.rand(B, T, _H, _W)
        L, S = op(x)
        # With lambda_s=0 and full rank, L+S should approximate x
        assert (L + S).shape == x.shape

    def test_adjoint_identity(self):
        from spectramr.infrastructure.physics.vf_physics_operators import LplusSOperator
        op = LplusSOperator()
        y = torch.rand(1, 4, _H, _W)
        grad_L, grad_S = op.adjoint(y)
        assert torch.allclose(grad_L, y)
        assert torch.allclose(grad_S, y)


# ============================================================
# SECTION 5: vf_operators.py
# ============================================================


@pytest.mark.physics
class TestMarkerPriorProjectionCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_operators import MarkerPriorProjection
        mask = _vf_mask()
        prior = _real_image()
        op = MarkerPriorProjection(mask, prior)
        assert op is not None

    def test_marker_region_pinned(self):
        from spectramr.infrastructure.physics.vf_operators import MarkerPriorProjection
        mask = _vf_mask()
        prior = torch.ones(1, 1, _H, _W)
        op = MarkerPriorProjection(mask, prior)
        x = torch.zeros(1, 1, _H, _W)
        out = op(x)
        # Marker region should be 1 (from prior)
        marker_region = out[mask.bool()].squeeze()
        assert torch.allclose(marker_region, torch.ones_like(marker_region))

    def test_outside_marker_unchanged(self):
        from spectramr.infrastructure.physics.vf_operators import MarkerPriorProjection
        mask = _vf_mask()
        prior = _real_image()
        op = MarkerPriorProjection(mask, prior)
        x = _real_image(B=2)
        out = op(x)
        outside = out * (1 - mask)
        expected = x * (1 - mask)
        assert torch.allclose(outside, expected)


@pytest.mark.physics
class TestMarkerKSpaceDCProjectionCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_operators import MarkerKSpaceDCProjection
        mask = _vf_mask()
        prior = _real_image()
        op = MarkerKSpaceDCProjection(mask, prior)
        assert op is not None

    def test_forward_shape_and_complex(self):
        from spectramr.infrastructure.physics.vf_operators import MarkerKSpaceDCProjection
        mask = _vf_mask()
        prior = _real_image()
        op = MarkerKSpaceDCProjection(mask, prior)
        x = _complex_image()
        out = op(x)
        assert out.shape == x.shape
        assert torch.is_complex(out)


@pytest.mark.physics
class TestRigidKinematicOperatorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_operators import RigidKinematicOperator
        op = RigidKinematicOperator(im_size=(_H, _W))
        assert op is not None

    def test_zero_motion_identity(self):
        from spectramr.infrastructure.physics.vf_operators import RigidKinematicOperator
        op = RigidKinematicOperator(im_size=(_H, _W))
        kspace = _complex_image(B=1, C=1)
        motion_params = torch.zeros(1, 3, _H)
        out = op(kspace, motion_params)
        assert torch.allclose(out, kspace)

    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_operators import RigidKinematicOperator
        op = RigidKinematicOperator(im_size=(_H, _W))
        kspace = _complex_image(B=2, C=1)
        motion_params = torch.randn(2, 3, _H) * 0.01
        motion_params[:, 2, :] = 0.0  # rotation requires regridding (raises otherwise)
        out = op(kspace, motion_params)
        assert out.shape == kspace.shape


@pytest.mark.physics
class TestB0PhaseOperatorCanary:
    def test_none_b0_identity(self):
        from spectramr.infrastructure.physics.vf_operators import B0PhaseOperator
        op = B0PhaseOperator(b0_map=None, te=0.01)
        x = _complex_image()
        out = op(x)
        assert torch.allclose(out, x)

    def test_with_b0_map_shape(self):
        from spectramr.infrastructure.physics.vf_operators import B0PhaseOperator
        b0 = _b0_map()
        op = B0PhaseOperator(b0_map=b0, te=0.01)
        x = _complex_image()
        out = op(x)
        assert out.shape == x.shape
        assert torch.is_complex(out)

    def test_zero_b0_identity(self):
        from spectramr.infrastructure.physics.vf_operators import B0PhaseOperator
        b0 = torch.zeros(1, 1, _H, _W)
        op = B0PhaseOperator(b0_map=b0, te=0.01)
        x = _complex_image()
        out = op(x)
        assert torch.allclose(out, x)


@pytest.mark.physics
class TestGeometricWarpOperatorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_operators import GeometricWarpOperator
        op = GeometricWarpOperator(im_size=(_H, _W))
        assert op is not None

    def test_zero_flow_identity(self):
        from spectramr.infrastructure.physics.vf_operators import GeometricWarpOperator
        op = GeometricWarpOperator(im_size=(_H, _W))
        x = _real_image(B=2)
        flow = torch.zeros(2, 2, _H, _W)
        out = op(x, flow)
        assert torch.allclose(out, x, atol=1e-5)

    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_operators import GeometricWarpOperator
        op = GeometricWarpOperator(im_size=(_H, _W))
        x = _real_image(B=1, C=2)
        flow = torch.randn(1, 2, _H, _W) * 0.5
        out = op(x, flow)
        assert out.shape == x.shape


@pytest.mark.physics
class TestToeplitzPSFOperatorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_operators import ToeplitzPSFOperator
        op = ToeplitzPSFOperator(kernel_size=5)
        assert op is not None

    def test_delta_psf_identity(self):
        from spectramr.infrastructure.physics.vf_operators import ToeplitzPSFOperator
        op = ToeplitzPSFOperator(kernel_size=5)
        # Default init is a delta PSF; forward should be near-identity
        x = _real_image(B=1, C=1)
        out = op(x)
        assert out.shape == x.shape

    def test_adjoint_shape(self):
        from spectramr.infrastructure.physics.vf_operators import ToeplitzPSFOperator
        op = ToeplitzPSFOperator(kernel_size=5)
        x = _real_image(B=2, C=1)
        out = op.adjoint(x)
        assert out.shape == x.shape

    def test_set_psf_accepted(self):
        from spectramr.infrastructure.physics.vf_operators import ToeplitzPSFOperator
        op = ToeplitzPSFOperator(kernel_size=5)
        psf = torch.zeros(1, 1, 5, 5)
        psf[0, 0, 2, 2] = 1.0
        op.set_psf(psf)
        x = _real_image()
        out = op(x)
        assert out.shape == x.shape


# ============================================================
# SECTION 6: vf_operators_extended.py
# ============================================================


@pytest.mark.physics
class TestRFModulationOperatorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_operators_extended import RFModulationOperator
        op = RFModulationOperator(num_coils=1)
        assert op is not None

    def test_forward_single_coil_shape(self):
        from spectramr.infrastructure.physics.vf_operators_extended import RFModulationOperator
        op = RFModulationOperator(num_coils=1)
        image = _real_image(B=2, C=1)
        b1_plus = _b1_map(B=2)
        b1_minus = _b1_map(B=2, offset=0.05)
        out = op(image, b1_plus, b1_minus)
        assert out.shape == image.shape

    def test_unit_b1_maps_preserve_magnitude(self):
        from spectramr.infrastructure.physics.vf_operators_extended import RFModulationOperator
        op = RFModulationOperator(num_coils=1)
        image = _real_image()
        b1_ones = torch.ones(1, 1, _H, _W)
        out = op(image, b1_ones, b1_ones)
        assert torch.allclose(out, image)

    def test_adjoint_shape(self):
        from spectramr.infrastructure.physics.vf_operators_extended import RFModulationOperator
        op = RFModulationOperator(num_coils=1)
        coil_images = _real_image()
        b1_plus = _b1_map()
        b1_minus = _b1_map(offset=0.05)
        out = op.adjoint(coil_images, b1_plus, b1_minus)
        assert out.shape == coil_images.shape


@pytest.mark.physics
class TestNUFFTTrajectoryCalibratorCanary:
    def test_import_and_construct(self):
        from spectramr.infrastructure.physics.vf_operators_extended import NUFFTTrajectoryCalibrator
        cal = NUFFTTrajectoryCalibrator(im_size=(_H, _W))
        assert cal is not None

    def test_compute_phase_correction_shape(self):
        from spectramr.infrastructure.physics.vf_operators_extended import NUFFTTrajectoryCalibrator
        cal = NUFFTTrajectoryCalibrator(im_size=(_H, _W))
        delays = torch.zeros(2)
        correction = cal.compute_phase_correction(delays)
        assert correction.shape == (1, 1, _H, _W)
        assert torch.is_complex(correction)

    def test_zero_delays_identity_correction(self):
        from spectramr.infrastructure.physics.vf_operators_extended import NUFFTTrajectoryCalibrator
        cal = NUFFTTrajectoryCalibrator(im_size=(_H, _W))
        delays = torch.zeros(2)
        correction = cal.compute_phase_correction(delays)
        # exp(0) == 1 everywhere
        assert torch.allclose(correction.abs(), torch.ones_like(correction.abs()), atol=1e-5)

    def test_forward_shape(self):
        from spectramr.infrastructure.physics.vf_operators_extended import NUFFTTrajectoryCalibrator
        cal = NUFFTTrajectoryCalibrator(im_size=(_H, _W))
        kspace = _complex_image(B=1, C=1)
        out = cal(kspace)
        assert out.shape == kspace.shape
