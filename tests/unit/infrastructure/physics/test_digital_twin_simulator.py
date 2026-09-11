"""Unit tests for Digital Twin Simulator.

Tests marker builders, motion models, corruption functions,
and the full DigitalTwinSimulator pipeline.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.digital_twin_simulator import (
    _QUANTILE_MAX_ELEMS,
    CornerFiducialEmbedder,
    DigitalTwinSimulator,
    _build_corner_crosses,
    _build_corner_gaussians,
    _robust_quantile,
    simulate_b0_geometric_distortion,
    simulate_b0_inhomogeneity,
    simulate_b1_bias_field,
    simulate_chemical_shift,
    simulate_eddy_current,
    simulate_elastic_motion,
    simulate_gibbs_ringing,
    simulate_periodic_motion,
    simulate_random_shot_motion,
    simulate_rf_crosstalk,
    simulate_rigid_motion_kspace,
    simulate_spike_noise,
    simulate_undersampling,
)
from spectramr.infrastructure.physics.fft_ops import fft2c

# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

IM_SIZE = (64, 64)


@pytest.fixture()
def complex_image() -> torch.Tensor:
    """Complex [1, 1, H, W] image with a bright disc."""
    H, W = IM_SIZE
    # N806: upper-case here matches H, W on the line above and the image-space
    # convention used throughout the physics tree. Lower-casing only these two
    # would be less readable, not more.
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    mag = (X**2 + Y**2 < 0.5).float()
    return torch.complex(mag, torch.zeros_like(mag)).unsqueeze(0).unsqueeze(0)


@pytest.fixture()
def complex_kspace(complex_image: torch.Tensor) -> torch.Tensor:
    """K-space of the disc phantom."""
    return fft2c(complex_image)


# ──────────────────────────────────────────────────────────────────────
# B0 geometric distortion — real-field injection (VF real-reference seam)
# ──────────────────────────────────────────────────────────────────────


class TestB0ExternalField:
    """``external_b0_field`` makes the twin apply a REAL B0 map instead of a
    randomly-generated one — the seam that turns VF field-scoring from
    self-consistency into real-reference grading."""

    def test_external_field_applied_verbatim(self) -> None:
        H = W = 16
        img = torch.complex(torch.randn(1, 1, H, W), torch.randn(1, 1, H, W))
        real_b0 = torch.full((1, H, W), 123.0)  # known real field, 123 Hz
        _, b0_map, shift_y = simulate_b0_geometric_distortion(
            img, (H, W), return_field=True, external_b0_field=real_b0
        )
        assert torch.allclose(b0_map, real_b0, atol=1e-3)  # REAL field, not random
        assert shift_y.abs().mean() > 0  # a non-zero field induces a displacement

    def test_external_field_4d_and_resize_accepted(self) -> None:
        H = W = 16
        img = torch.complex(torch.randn(1, 1, H, W), torch.randn(1, 1, H, W))
        # supply [B, 1, h, w] at a different resolution → squeezed + resized
        real_b0 = torch.full((1, 1, 8, 8), 50.0)
        _, b0_map, _ = simulate_b0_geometric_distortion(
            img, (H, W), return_field=True, external_b0_field=real_b0
        )
        assert tuple(b0_map.shape) == (1, H, W)
        assert torch.allclose(b0_map, torch.full((1, H, W), 50.0), atol=1.0)

    def test_without_external_field_is_random_default(self) -> None:
        # regression: omitting external_b0_field keeps the existing random model
        H = W = 16
        img = torch.complex(torch.randn(1, 1, H, W), torch.randn(1, 1, H, W))
        _, b0_map, _ = simulate_b0_geometric_distortion(img, (H, W), return_field=True)
        assert b0_map.std() > 0  # spatially-varying random field

    def test_forward_threads_external_field_to_last_b0(self) -> None:
        """The simulator's forward threads external_b0_field → last_b0_field, so
        scoring grades against the real field, not a random one."""
        sim = DigitalTwinSimulator(
            im_size=IM_SIZE,
            enable_motion=False,
            enable_b0_distortion=True,
            snr_range=(100.0, 100.0),
        )
        img = torch.complex(torch.randn(1, 1, *IM_SIZE), torch.randn(1, 1, *IM_SIZE))
        real_b0 = torch.full((1, *IM_SIZE), 77.0)
        sim(img, external_b0_field=real_b0)
        assert sim.last_b0_field is not None
        assert abs(float(sim.last_b0_field.mean()) - 77.0) < 1.0  # the REAL field

    def test_phase_path_external_field_sets_last_b0(self) -> None:
        """The PHASE B0 path (enable_b0, no geometric distortion) also exposes the
        real B0 as last_b0_field — so the phase-domain arms can be graded."""
        sim = DigitalTwinSimulator(
            im_size=IM_SIZE,
            enable_motion=False,
            enable_b0=True,
            enable_b0_distortion=False,
            snr_range=(100.0, 100.0),
        )
        img = torch.complex(torch.randn(1, 1, *IM_SIZE), torch.randn(1, 1, *IM_SIZE))
        real_b0 = torch.full((1, *IM_SIZE), 60.0)
        sim(img, external_b0_field=real_b0)
        assert sim.last_b0_field is not None
        assert abs(float(sim.last_b0_field.mean()) - 60.0) < 1.0

    def test_simulate_b0_inhomogeneity_accepts_external_field(self) -> None:
        from spectramr.infrastructure.physics.fft_ops import fft2c

        img = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        out = simulate_b0_inhomogeneity(
            fft2c(img), (16, 16), external_b0_field=torch.full((1, 1, 16, 16), 50.0)
        )
        assert out.shape == img.shape  # runs end-to-end with a real field


# ──────────────────────────────────────────────────────────────────────
# Marker Builder Tests
# ──────────────────────────────────────────────────────────────────────


class TestMarkerBuilders:
    """Tests for Gaussian and Cross marker builders."""

    def test_gaussian_shape(self) -> None:
        """Gaussian builder returns [1, 1, H, W] complex."""
        m = _build_corner_gaussians(IM_SIZE, offset=0.8, sigma=0.03)
        assert m.shape == (1, 1, *IM_SIZE)
        assert m.is_complex()

    def test_gaussian_normalised(self) -> None:
        """Gaussian marker max is ~1.0."""
        m = _build_corner_gaussians(IM_SIZE, offset=0.8, sigma=0.03)
        assert abs(m.abs().max().item() - 1.0) < 1e-6

    def test_gaussian_four_corners(self) -> None:
        """Gaussian has 4 peaks at corner positions."""
        m = _build_corner_gaussians(IM_SIZE, offset=0.8, sigma=0.03)
        mag = m.abs().squeeze()
        # Check that corners have significant signal
        q = 8  # corner region size
        corners = [
            mag[:q, :q].max(),
            mag[:q, -q:].max(),
            mag[-q:, :q].max(),
            mag[-q:, -q:].max(),
        ]
        for v in corners:
            assert v > 0.3, f"Corner peak too weak: {v:.3f}"

    def test_cross_shape(self) -> None:
        """Cross builder returns [1, 1, H, W] complex."""
        m = _build_corner_crosses(IM_SIZE, offset=0.8)
        assert m.shape == (1, 1, *IM_SIZE)
        assert m.is_complex()

    def test_cross_normalised(self) -> None:
        """Cross marker max is ~1.0."""
        m = _build_corner_crosses(IM_SIZE, offset=0.8)
        assert abs(m.abs().max().item() - 1.0) < 1e-6

    def test_cross_has_arm_structure(self) -> None:
        """Cross markers have cruciform (non-circular) structure."""
        m = _build_corner_crosses(IM_SIZE, offset=0.8).abs().squeeze()
        # A cross should have more non-zero pixels along axes than diagonals
        assert m.sum() > 0, "Cross marker has no signal"
        # Cross should be sparser than gaussian
        g = _build_corner_gaussians(IM_SIZE, offset=0.8, sigma=0.03).abs().squeeze()
        cross_nnz = (m > 0.01).sum()
        gauss_nnz = (g > 0.01).sum()
        # Cross should typically be sparser
        assert cross_nnz > 0


# ──────────────────────────────────────────────────────────────────────
# CornerFiducialEmbedder Tests
# ──────────────────────────────────────────────────────────────────────


class TestCornerFiducialEmbedder:
    """Tests for the embedder module."""

    @pytest.mark.parametrize("marker_type", ["gaussian", "cross"])
    def test_forward_shape(self, complex_image: torch.Tensor, marker_type: str) -> None:
        """Embedder output shape matches input."""
        embedder = CornerFiducialEmbedder(im_size=IM_SIZE, marker_type=marker_type)
        joint, prior = embedder(complex_image)
        assert joint.shape == complex_image.shape
        assert prior.shape[2:] == complex_image.shape[2:]

    @pytest.mark.parametrize("marker_type", ["gaussian", "cross"])
    def test_markers_add_signal(self, complex_image: torch.Tensor, marker_type: str) -> None:
        """Markers add positive signal (total energy increases)."""
        embedder = CornerFiducialEmbedder(im_size=IM_SIZE, marker_type=marker_type)
        # Use an image with nonzero background so quantile scaling works
        bg_image = complex_image + 0.1 * torch.ones_like(complex_image)
        joint, _ = embedder(bg_image)
        energy_before = (bg_image.abs() ** 2).sum()
        energy_after = (joint.abs() ** 2).sum()
        assert energy_after > energy_before, "Markers should increase total energy"

    def test_invalid_marker_type(self) -> None:
        """Invalid marker_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown marker_type"):
            CornerFiducialEmbedder(im_size=IM_SIZE, marker_type="invalid")

    def test_single_coil_oversize_anatomy_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression (2026-06-22): a large *validation* batch tipped the
        single-coil anatomy flatten past ``torch.quantile``'s 2**24 cap →
        ``quantile() input tensor is too large`` on every validation batch
        (exp_vf_ib_infonce_v2 ``IBVFTrainingStrategy``). With the cap lowered
        the embedder must subsample, not raise.
        """
        import spectramr.infrastructure.physics.digital_twin_simulator as dts

        monkeypatch.setattr(dts, "_QUANTILE_MAX_ELEMS", 256)
        embedder = CornerFiducialEmbedder(im_size=(48, 48), marker_type="gaussian")
        # 48*48 = 2304 > 256 cap → exercises the subsample branch.
        anatomy = torch.randn(1, 1, 48, 48) + 0.2
        joint, _prior = embedder(anatomy)
        assert joint.shape == anatomy.shape
        assert torch.isfinite(joint).all()

    def test_multi_coil_branch_routes_through_robust_quantile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression (2026-06-25): the MULTI-COIL RSS branch still called raw
        ``torch.quantile(rss.float(), 0.75)`` while the single-coil / per-channel
        branches were already guarded — so a 3-D/5-D coil volume above the 2**24
        cap would raise ``quantile() input tensor is too large`` (the
        exp_vf_ib_infonce_v2 root cause, on whichever branch the arm hits). The
        branch must route through ``_robust_quantile`` so the cap guard applies.
        A monkeypatched ``_QUANTILE_MAX_ELEMS`` only affects ``_robust_quantile``
        (not raw ``torch.quantile``), so a small-tensor cap test can't catch this;
        we spy that the guarded helper is the one invoked.
        """
        import spectramr.infrastructure.physics.digital_twin_simulator as dts

        real = dts._robust_quantile
        calls: list[tuple[int, ...]] = []

        def _spy(x, q, dim=None):  # type: ignore[no-untyped-def]
            calls.append(tuple(x.shape))
            return real(x, q, dim=dim)

        monkeypatch.setattr(dts, "_robust_quantile", _spy)
        embedder = CornerFiducialEmbedder(im_size=(16, 16), marker_type="gaussian")
        anatomy = (torch.randn(1, 4, 16, 16) + 0.2).to(torch.complex64)  # 4-coil
        smaps = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
        joint, _prior = embedder(anatomy, coil_sensitivities=smaps)
        assert joint.shape == anatomy.shape
        assert calls, "multi-coil RSS branch must call _robust_quantile (cap guard)"


# ──────────────────────────────────────────────────────────────────────
# Robust-quantile guard (torch.quantile 2**24 cap)
# ──────────────────────────────────────────────────────────────────────


class TestRobustQuantile:
    """``_robust_quantile`` survives oversize reduced dims that crash
    ``torch.quantile`` outright (the exp_vf_ib_infonce_v2 root cause)."""

    def test_matches_torch_below_cap(self) -> None:
        """Below the cap it delegates verbatim (exact agreement)."""
        x = torch.randn(10_000)
        assert torch.allclose(_robust_quantile(x, 0.75), torch.quantile(x, 0.75))

    def test_matches_torch_below_cap_with_dim(self) -> None:
        x = torch.randn(3, 4, 5000)
        assert torch.allclose(_robust_quantile(x, 0.9, dim=-1), torch.quantile(x, 0.9, dim=-1))

    def test_subsamples_above_cap_close_estimate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a lowered cap the strided subsample still tracks the true
        quantile (uniform decimation is unbiased for a smooth quantile)."""
        import spectramr.infrastructure.physics.digital_twin_simulator as dts

        monkeypatch.setattr(dts, "_QUANTILE_MAX_ELEMS", 1000)
        x = torch.linspace(0.0, 1.0, 50_000)  # >cap, smooth
        out = dts._robust_quantile(x, 0.75)
        assert abs(float(out) - 0.75) < 0.02

    @pytest.mark.slow
    def test_survives_real_torch_cap(self) -> None:
        """The true regression: a flatten just over 2**24 elements makes bare
        ``torch.quantile`` raise, while ``_robust_quantile`` returns finite."""
        n = _QUANTILE_MAX_ELEMS + 1024  # ~16.78M float32 ≈ 67 MB
        x = torch.rand(n)
        with pytest.raises(RuntimeError, match="too large"):
            torch.quantile(x, 0.75)
        out = _robust_quantile(x, 0.75)
        assert torch.isfinite(out).all()
        assert 0.0 <= float(out) <= 1.0


# ──────────────────────────────────────────────────────────────────────
# Motion Model Tests
# ──────────────────────────────────────────────────────────────────────


class TestMotionModels:
    """Tests for motion simulation functions."""

    def test_rigid_shape(self, complex_kspace: torch.Tensor) -> None:
        """Rigid motion preserves shape."""
        out = simulate_rigid_motion_kspace(complex_kspace)
        assert out.shape == complex_kspace.shape

    def test_rigid_translation_only_preserves_energy(self, complex_kspace: torch.Tensor) -> None:
        """Pure-translation rigid motion (max_rotation=0) preserves energy.

        With rotation disabled the operator reduces to per-PE-line
        Fourier shift ramps, which preserve energy by Parseval. When
        ``max_rotation > 0`` the operator includes an image-space
        rotation step (grid_sample with bilinear interpolation) which
        loses ~1-2% energy; that is physically correct and is
        verified by the C1-regression test in test_digital_twin_fixes.
        """
        out = simulate_rigid_motion_kspace(complex_kspace, max_translation=3.0, max_rotation=0.0)
        energy_in = (complex_kspace.abs() ** 2).sum()
        energy_out = (out.abs() ** 2).sum()
        assert torch.allclose(energy_in, energy_out, rtol=1e-4)

    def test_periodic_shape(self, complex_kspace: torch.Tensor) -> None:
        """Periodic motion preserves shape."""
        out = simulate_periodic_motion(complex_kspace, amplitude=2.0, frequency=3.0)
        assert out.shape == complex_kspace.shape

    def test_periodic_energy_preserved(self, complex_kspace: torch.Tensor) -> None:
        """Periodic motion (phase-only per line) preserves energy."""
        out = simulate_periodic_motion(complex_kspace)
        energy_in = (complex_kspace.abs() ** 2).sum()
        energy_out = (out.abs() ** 2).sum()
        assert torch.allclose(energy_in, energy_out, rtol=1e-4)

    def test_random_shot_shape(self, complex_kspace: torch.Tensor) -> None:
        """Random shot motion preserves shape."""
        out = simulate_random_shot_motion(complex_kspace, max_shift=1.5)
        assert out.shape == complex_kspace.shape

    def test_random_shot_energy_preserved(self, complex_kspace: torch.Tensor) -> None:
        """Random shot motion (phase-only per line) preserves energy."""
        out = simulate_random_shot_motion(complex_kspace)
        energy_in = (complex_kspace.abs() ** 2).sum()
        energy_out = (out.abs() ** 2).sum()
        assert torch.allclose(energy_in, energy_out, rtol=1e-4)

    def test_elastic_shape(self, complex_kspace: torch.Tensor) -> None:
        """Elastic motion preserves shape."""
        out = simulate_elastic_motion(complex_kspace, im_size=IM_SIZE, num_modes=3, amplitude=1.0)
        assert out.shape == complex_kspace.shape


# ──────────────────────────────────────────────────────────────────────
# Corruption Function Tests
# ──────────────────────────────────────────────────────────────────────


class TestCorruptionFunctions:
    """Tests for scanner artifact simulation functions."""

    def test_b0_shape(self, complex_kspace: torch.Tensor) -> None:
        out = simulate_b0_inhomogeneity(complex_kspace, IM_SIZE, strength=0.3)
        assert out.shape == complex_kspace.shape

    def test_b1_shape(self, complex_image: torch.Tensor) -> None:
        out = simulate_b1_bias_field(complex_image, IM_SIZE, strength=0.2)
        assert out.shape == complex_image.shape

    def test_undersampling_reduces_energy(self, complex_kspace: torch.Tensor) -> None:
        """Undersampling zeros out k-space lines → less total energy."""
        out = simulate_undersampling(complex_kspace, acceleration=4.0)
        assert (out.abs() ** 2).sum() < (complex_kspace.abs() ** 2).sum()

    def test_gibbs_reduces_energy(self, complex_kspace: torch.Tensor) -> None:
        """Gibbs truncation removes outer k-space → less energy."""
        out = simulate_gibbs_ringing(complex_kspace, truncation_fraction=0.7)
        assert (out.abs() ** 2).sum() <= (complex_kspace.abs() ** 2).sum()

    def test_chemical_shift_shape(self, complex_kspace: torch.Tensor) -> None:
        out = simulate_chemical_shift(complex_kspace, IM_SIZE, shift_pixels=3.0)
        assert out.shape == complex_kspace.shape

    def test_chemical_shift_at_zero_shift_is_the_identity(
        self, complex_kspace: torch.Tensor
    ) -> None:
        """No shift means no chemical shift. This is the clean anchor of the D16 axis.

        It was not. The shifted fat was added on top of the FULL water signal, so the
        operator applied a ``(1 + fat_fraction)`` global gain even at ``shift_pixels=0``
        -- the "clean" end of the severity axis came back 1.35x brighter than its input.
        rel-L2 vs the reference then *decreased* as severity rose, and every L2-family
        metric (PSNR/NMSE/MSE) scored the axis backwards.
        """
        out = simulate_chemical_shift(complex_kspace, IM_SIZE, shift_pixels=0.0)
        assert torch.allclose(out, complex_kspace, atol=1e-5)

    def test_chemical_shift_partitions_signal_rather_than_amplifying_it(
        self, complex_kspace: torch.Tensor
    ) -> None:
        """Fat is DISPLACED, not conjured: the operator must not add energy.

        A fraction ``fat_fraction`` of the signal resonates at the fat frequency and is
        displaced along readout; the remaining water fraction is not. Total energy is
        redistributed, so it cannot exceed the input's (the two components partially
        cancel where they overlap).
        """
        out = simulate_chemical_shift(complex_kspace, IM_SIZE, shift_pixels=3.0)
        energy_in = (complex_kspace.abs() ** 2).sum()
        energy_out = (out.abs() ** 2).sum()
        assert energy_out <= energy_in * (1.0 + 1e-4), (
            "chemical shift amplified the signal; it must partition it "
            f"({energy_out:.4f} > {energy_in:.4f})"
        )
        # ...and it must actually DO something at a non-zero shift.
        assert not torch.allclose(out, complex_kspace, atol=1e-5)

    def test_eddy_current_shape(self, complex_kspace: torch.Tensor) -> None:
        out = simulate_eddy_current(complex_kspace, strength=0.1)
        assert out.shape == complex_kspace.shape

    def test_eddy_energy_preserved(self, complex_kspace: torch.Tensor) -> None:
        """Eddy current (phase-only per line) preserves energy."""
        out = simulate_eddy_current(complex_kspace, strength=0.1)
        energy_in = (complex_kspace.abs() ** 2).sum()
        energy_out = (out.abs() ** 2).sum()
        assert torch.allclose(energy_in, energy_out, rtol=1e-4)

    def test_spike_noise_shape(self, complex_kspace: torch.Tensor) -> None:
        out = simulate_spike_noise(complex_kspace, probability=0.01)
        assert out.shape == complex_kspace.shape

    def test_spike_noise_adds_energy(self, complex_kspace: torch.Tensor) -> None:
        """Spike noise adds random points → more total energy (statistically)."""
        torch.manual_seed(42)
        out = simulate_spike_noise(complex_kspace, probability=0.05, spike_amplitude=10.0)
        assert (out.abs() ** 2).sum() > (complex_kspace.abs() ** 2).sum()

    def test_rf_crosstalk_shape(self, complex_image: torch.Tensor) -> None:
        out = simulate_rf_crosstalk(complex_image, crosstalk_fraction=0.05)
        assert out.shape == complex_image.shape


# ──────────────────────────────────────────────────────────────────────
# Full Pipeline Tests
# ──────────────────────────────────────────────────────────────────────


class TestDigitalTwinSimulator:
    """Tests for the full Digital Twin pipeline."""

    @pytest.mark.parametrize("marker_type", ["gaussian", "cross"])
    @pytest.mark.parametrize("motion_type", ["rigid", "periodic", "random_shot", "elastic"])
    def test_full_pipeline_shape(
        self,
        complex_image: torch.Tensor,
        marker_type: str,
        motion_type: str,
    ) -> None:
        """Pipeline output shape matches input for all combinations."""
        sim = DigitalTwinSimulator(
            im_size=IM_SIZE,
            marker_type=marker_type,
            motion_type=motion_type,
            enable_chemical_shift=True,
            enable_eddy_current=True,
            enable_spike_noise=True,
            enable_rf_crosstalk=True,
        )
        corrupted, prior, joint_clean = sim(complex_image)
        assert corrupted.shape == complex_image.shape
        assert joint_clean.shape == complex_image.shape

    def test_invalid_motion_type(self) -> None:
        """Invalid motion_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown motion_type"):
            DigitalTwinSimulator(im_size=IM_SIZE, motion_type="invalid")

    def test_all_corruptions_disabled(self, complex_image: torch.Tensor) -> None:
        """With all corruptions disabled, output should still embed markers."""
        sim = DigitalTwinSimulator(
            im_size=IM_SIZE,
            enable_motion=False,
            enable_b0=False,
            enable_b1=False,
            enable_undersampling=False,
            enable_gibbs=False,
            enable_chemical_shift=False,
            enable_eddy_current=False,
            enable_spike_noise=False,
            enable_rf_crosstalk=False,
            snr_range=(100.0, 100.0),  # Very low noise
        )
        corrupted, prior, joint_clean = sim(complex_image)
        # Joint clean should have markers → differ from anatomy
        diff = (joint_clean.abs() - complex_image.abs()).sum()
        assert diff > 0, "Markers should be embedded"

    def test_marker_mask_shape(self) -> None:
        """marker_mask property returns correct shape."""
        sim = DigitalTwinSimulator(im_size=IM_SIZE)
        assert sim.marker_mask.shape == (1, 1, *IM_SIZE)


# ---------------------------------------------------------------------------
# Composite motion + amplitude curriculum (2026-05-26)
# ---------------------------------------------------------------------------
#
# Composite: multiple motion models applied sequentially in k-space to build
# mixed patterns. Curriculum: a severity multiplier annealed start→end over
# ramp_iters so training can start with high motion (off the identity
# solution) and relax to the nominal regime. Both are opt-in; the defaults
# preserve the single-motion_type, constant-amplitude behaviour.


def _img(h: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 1, h, h, dtype=torch.complex64)


def test_composite_motion_runs_and_differs_from_single() -> None:
    img = _img()
    single = DigitalTwinSimulator(
        im_size=(64, 64), motion_type="rigid", enable_b0=False, enable_b1=False
    )
    comp = DigitalTwinSimulator(
        im_size=(64, 64),
        motion_composite=["rigid", "periodic", "random_shot"],
        enable_b0=False,
        enable_b1=False,
    )
    torch.manual_seed(1)
    out_single = single(img.clone())[0]
    torch.manual_seed(1)
    out_comp = comp(img.clone())[0]
    assert out_comp.shape == out_single.shape
    # Stacking three motion models must change the corruption.
    assert not torch.allclose(out_single, out_comp, atol=1e-4)


def test_composite_motion_rejects_unknown_entry() -> None:
    with pytest.raises(ValueError, match="motion_composite"):
        DigitalTwinSimulator(im_size=(64, 64), motion_composite=["rigid", "banana"])


def test_empty_composite_falls_back_to_single_motion_type() -> None:
    """Default (no composite) must be identical to the legacy single path."""
    img = _img()
    a = DigitalTwinSimulator(
        im_size=(64, 64), motion_type="periodic", enable_b0=False, enable_b1=False
    )
    b = DigitalTwinSimulator(
        im_size=(64, 64),
        motion_type="periodic",
        motion_composite=[],
        enable_b0=False,
        enable_b1=False,
    )
    torch.manual_seed(2)
    out_a = a(img.clone())[0]
    torch.manual_seed(2)
    out_b = b(img.clone())[0]
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_motion_severity_is_a_constant_floor() -> None:
    """Severity is a sustained constant, not an annealing schedule.

    Annealing high→mild would let the near-clean input leak the target at the
    end of training, so the simulator exposes only a constant multiplier and
    has no step-driven curriculum that could lower it over training.
    """
    sim = DigitalTwinSimulator(im_size=(64, 64), motion_severity=2.5)
    assert sim._motion_severity == pytest.approx(2.5)
    assert not hasattr(sim, "set_iteration")


def test_default_motion_severity_is_nominal() -> None:
    sim = DigitalTwinSimulator(im_size=(64, 64))
    assert sim._motion_severity == pytest.approx(1.0)


def test_set_motion_severity_scales_translation() -> None:
    """Higher severity ⇒ larger motion ⇒ larger deviation from clean."""
    img = _img()
    sim = DigitalTwinSimulator(
        im_size=(64, 64), motion_type="rigid", enable_b0=False, enable_b1=False
    )
    clean_k = fft2c(img)
    sim.set_motion_severity(0.0)
    torch.manual_seed(3)
    low = (sim._apply_motion(clean_k) - clean_k).abs().mean()
    sim.set_motion_severity(4.0)
    torch.manual_seed(3)
    high = (sim._apply_motion(clean_k) - clean_k).abs().mean()
    assert high > low


# ---------------------------------------------------------------------------
# degradation_only + from_config (transversal data-transform support)
# ---------------------------------------------------------------------------


def test_degradation_only_skips_marker_embedding() -> None:
    """marker_prior must be an all-zero placeholder when markers are skipped."""
    img = _img()
    sim = DigitalTwinSimulator(
        im_size=(64, 64), motion_type="rigid", enable_b0=False, enable_b1=False
    )
    _corrupted, marker_prior, _clean = sim(img.clone(), degradation_only=True)
    assert torch.count_nonzero(marker_prior) == 0


def test_degradation_only_differs_from_marker_path() -> None:
    img = _img()
    sim = DigitalTwinSimulator(
        im_size=(64, 64), motion_type="rigid", enable_b0=False, enable_b1=False
    )
    torch.manual_seed(7)
    with_markers = sim(img.clone(), degradation_only=False)[0]
    torch.manual_seed(7)
    without = sim(img.clone(), degradation_only=True)[0]
    # The marker path stamps fiducials into the corner; degradation-only does not.
    assert not torch.allclose(with_markers, without, atol=1e-4)


def test_from_config_maps_motion_fields() -> None:
    from spectramr.config.schemas.physics import DigitalTwinConfig

    cfg = DigitalTwinConfig(motion_composite=["rigid", "periodic"], motion_severity=2.5)
    sim = DigitalTwinSimulator.from_config(cfg, (64, 64))
    assert sim.im_size == (64, 64)
    assert sim.motion_composite == ["rigid", "periodic"]
    assert sim._motion_severity == pytest.approx(2.5)


def test_motion_severity_saturates_not_amplifies_gradient() -> None:
    """Severity must not overwhelm the gradient.

    Motion is energy-preserving, so raising severity makes the task harder but
    the corrupted input's distance from the target is bounded by the signal
    energy — loss and gradient SATURATE rather than growing. Guards against a
    future change that makes severity scale signal *amplitude* (which would
    let the gradient blow up).
    """

    clean = _img() / _img().abs().max()
    distances = []
    for sev in [1.0, 4.0, 16.0, 32.0]:
        sim = DigitalTwinSimulator(
            im_size=(64, 64),
            motion_type="rigid",
            motion_severity=sev,
            enable_b0=False,
            enable_b1=False,
        )
        torch.manual_seed(1)
        corrupted, _, _ = sim(clean.clone(), degradation_only=True)
        distances.append((corrupted - clean).abs().norm().item())

    # Distance is bounded (no growth with severity): the max over high
    # severities must not exceed the severity=1 value by more than a small
    # margin — i.e. it has saturated, not amplified.
    base = distances[0]
    assert max(distances) <= base * 1.15, (
        f"corruption distance grew with severity (should saturate): {distances}"
    )


# ---------------------------------------------------------------------------
# Diffusion timestep coupling: corruption_factor must drive EVERY degradation
# (2026-05-26). Bug: motion ignored corruption_factor, so the cold-diffusion
# forward operator was discontinuous at t=0 (D(x,0) != x). b1 ignored its
# degradation_range. from_config dropped progressive_degradations/ranges.
# ---------------------------------------------------------------------------


def test_motion_honors_corruption_factor_continuity() -> None:
    """At corruption_factor=0 the operator must be (near) identity.

    Motion is the only enabled degradation; at cf=0 the corrupted image must
    collapse back to the clean anatomy. Pre-fix motion ignored cf and stayed
    at full severity, breaking the cold-diffusion D(x,0)=x assumption.
    """
    clean = _img() / _img().abs().max()
    sim = DigitalTwinSimulator(
        im_size=(64, 64),
        motion_type="rigid",
        motion_severity=4.0,
        enable_b0=False,
        enable_b1=False,
    )
    torch.manual_seed(5)
    corrupted0, _, _ = sim(clean.clone(), corruption_factor=0.0, degradation_only=True)
    assert torch.allclose(corrupted0, clean, atol=1e-2), (
        "D(x, t=0) must be ~identity but motion was applied at full severity"
    )
    torch.manual_seed(5)
    corrupted1, _, _ = sim(clean.clone(), corruption_factor=1.0, degradation_only=True)
    # At cf=1 motion is fully applied → far from clean.
    assert (corrupted1 - clean).abs().mean() > (corrupted0 - clean).abs().mean() * 5


def test_motion_corruption_ramps_with_cf() -> None:
    """Motion deviation must increase monotonically with corruption_factor."""
    clean = _img() / _img().abs().max()
    sim = DigitalTwinSimulator(
        im_size=(64, 64),
        motion_type="rigid",
        motion_severity=3.0,
        enable_b0=False,
        enable_b1=False,
    )
    devs = []
    for cf in (0.0, 0.25, 0.5, 1.0):
        torch.manual_seed(9)  # identical random draw; only cf scales it
        corrupted, _, _ = sim(clean.clone(), corruption_factor=cf, degradation_only=True)
        devs.append((corrupted - clean).abs().mean().item())
    assert devs[0] < devs[1] < devs[2] < devs[3], f"not monotone in cf: {devs}"


def test_b1_strength_respects_degradation_range() -> None:
    """b1 must honor its degradation_range, not the raw corruption_factor."""
    clean = _img() / _img().abs().max()
    common = dict(
        im_size=(64, 64),
        enable_motion=False,
        enable_b0=False,
        enable_b1=True,
        progressive_degradations=["b1"],
    )
    sim_off = DigitalTwinSimulator(degradation_ranges={"b1": (0.0, 0.0)}, **common)
    sim_on = DigitalTwinSimulator(degradation_ranges={"b1": (1.0, 1.0)}, **common)
    torch.manual_seed(11)
    out_off, _, _ = sim_off(clean.clone(), corruption_factor=1.0, degradation_only=True)
    torch.manual_seed(11)
    out_on, _, _ = sim_on(clean.clone(), corruption_factor=1.0, degradation_only=True)
    # range (0,0) ⇒ no B1; range (1,1) ⇒ full B1. Pre-fix both used raw cf=1
    # and were identical.
    assert not torch.allclose(out_off, out_on, atol=1e-3)


def test_from_config_forwards_progressive_degradations_and_ranges() -> None:
    from spectramr.config.schemas.physics import DigitalTwinConfig

    cfg = DigitalTwinConfig(
        progressive_degradations=["motion", "b0"],
        degradation_ranges={"motion": (0.1, 0.9), "b0": (0.0, 0.5)},
    )
    sim = DigitalTwinSimulator.from_config(cfg, (64, 64))
    assert sim.progressive_degradations == ["motion", "b0"]
    assert sim.degradation_ranges == {"motion": (0.1, 0.9), "b0": (0.0, 0.5)}


def test_degrade_at_maps_timestep_to_corruption() -> None:
    """degrade_at(t) is the timestep-indexed forward operator for diffusion."""
    clean = _img() / _img().abs().max()
    sim = DigitalTwinSimulator(
        im_size=(64, 64),
        motion_type="rigid",
        motion_severity=4.0,
        enable_b0=False,
        enable_b1=False,
    )
    torch.manual_seed(5)
    x0 = sim.degrade_at(clean.clone(), t=0, num_timesteps=10)
    torch.manual_seed(5)
    xT = sim.degrade_at(clean.clone(), t=10, num_timesteps=10)
    assert x0.shape == clean.shape
    assert torch.allclose(x0, clean, atol=1e-2)  # t=0 ⇒ identity
    assert (xT - clean).abs().mean() > (x0 - clean).abs().mean() * 5


# ──────────────────────────────────────────────────────────────────────
# Undersampling mask exposure (2026-06-02: feed DC layers the measurement)
# ──────────────────────────────────────────────────────────────────────


def test_simulate_undersampling_return_mask_roundtrip() -> None:
    """``return_mask=True`` yields the binary mask that produced the output."""
    torch.manual_seed(0)
    k = torch.randn(2, 1, 64, 64, dtype=torch.complex64)
    masked, mask = simulate_undersampling(k, acceleration=8.0, return_mask=True)
    assert mask.shape == (1, 1, 64, 64) or mask.shape == (1, 1, 64, 1)
    assert torch.equal(mask, (mask > 0).float())  # binary
    assert torch.allclose(masked, k * mask)  # the mask is the one applied
    # backward-compat: default call still returns a bare tensor
    assert isinstance(simulate_undersampling(k, acceleration=8.0), torch.Tensor)


def test_simulator_exposes_last_undersampling_mask() -> None:
    """A forward pass with undersampling on populates ``last_undersampling_mask``."""
    sim = DigitalTwinSimulator(
        im_size=(64, 64),
        enable_motion=False,
        enable_b0=False,
        enable_b1=False,
        enable_undersampling=True,
        acceleration=8.0,
    )
    assert sim.last_undersampling_mask is None  # nothing captured yet
    torch.manual_seed(1)
    clean = torch.randn(1, 1, 64, 64, dtype=torch.complex64)
    sim(clean)
    mask = sim.last_undersampling_mask
    assert mask is not None
    assert torch.equal(mask, (mask > 0).float())  # binary
    frac = mask.float().mean().item()
    assert 0.0 < frac < 1.0  # some lines sampled, some not


def test_simulator_mask_none_when_undersampling_disabled() -> None:
    """No undersampling ⇒ the mask stays None (never leak a stale mask)."""
    sim = DigitalTwinSimulator(
        im_size=(64, 64),
        enable_motion=False,
        enable_b0=False,
        enable_b1=False,
        enable_undersampling=False,
    )
    torch.manual_seed(2)
    clean = torch.randn(1, 1, 64, 64, dtype=torch.complex64)
    sim(clean)
    assert sim.last_undersampling_mask is None


# ══════════════════════════════════════════════════════════════════════
# Simulator correctness (issues #244, #246)
# ══════════════════════════════════════════════════════════════════════


class TestGibbsCleanAnchor:
    """#246 — ``truncation_fraction=1.0`` must be an exact identity.

    The window compared ``sqrt(kx^2 + ky^2)`` on a ``[-1, 1]^2`` grid — whose
    maximum is ``sqrt(2) ~ 1.414``, not 1.0 — against the retained fraction. So at
    the nominal "no Gibbs" setting the circular window still zeroed the k-space
    **corners**: 1 - pi/4 = 22.8% of the samples. The axis's declared clean anchor
    came back with a rel-L2 of 0.055 against its own input, a pedestal larger than
    the entire theta=1 distortion of several other axes.
    """

    def test_truncation_one_is_exact_identity(self) -> None:
        torch.manual_seed(0)
        k = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        assert torch.equal(simulate_gibbs_ringing(k, truncation_fraction=1.0), k)

    def test_corners_of_kspace_survive_at_truncation_one(self) -> None:
        # The corner is the sample at max radius — the one the un-normalised
        # window discarded. Its survival IS the bug's absence.
        k = torch.ones(1, 1, 16, 16, dtype=torch.complex64)
        out = simulate_gibbs_ringing(k, truncation_fraction=1.0)
        for i, j in [(0, 0), (0, 15), (15, 0), (15, 15)]:
            assert out[0, 0, i, j].abs() > 0, f"corner ({i}, {j}) was zeroed"

    def test_truncation_still_bites_below_one(self) -> None:
        k = torch.ones(1, 1, 32, 32, dtype=torch.complex64)
        kept = (simulate_gibbs_ringing(k, truncation_fraction=0.3).abs() > 0).float().mean()
        assert 0.0 < float(kept) < 0.5


class TestProgressiveDegradationRanges:
    """#244 — ``progressive_degradations`` / ``degradation_ranges`` were a no-op.

    ``_get_effective_cf`` returned ``base_cf`` for an unlisted feature, and a
    listed feature with the range ``(0.0, 1.0)`` returned ``0 + 1 * base_cf`` —
    the same thing. Since EVERY axis config declares exactly that range, the two
    branches were byte-identical and the whole mechanism was decorative, while the
    module docstring claimed it "ensures severity scales linearly with
    corruption_factor" (pitfall #15: an advertised knob nothing reads).

    Both knobs now have distinct, observable effects.
    """

    @staticmethod
    def _sim(**kw) -> DigitalTwinSimulator:
        base = dict(
            im_size=(32, 32),
            marker_type="none",
            enable_motion=False,
            enable_b0=False,
            enable_b1=False,
            snr_range=(100.0, 100.0),
        )
        base.update(kw)
        return DigitalTwinSimulator(**base)

    def test_empty_list_means_every_feature_is_progressive(self) -> None:
        """Backwards compatibility: the default every existing config relies on."""
        sim = self._sim(progressive_degradations=[], degradation_ranges={})
        assert sim._get_effective_cf("b1", 0.0) == pytest.approx(0.0)
        assert sim._get_effective_cf("b1", 0.5) == pytest.approx(0.5)
        assert sim._get_effective_cf("b1", 1.0) == pytest.approx(1.0)

    def test_unlisted_feature_is_static_when_a_list_is_declared(self) -> None:
        """The branch that used to be a no-op: unlisted != progressive."""
        sim = self._sim(progressive_degradations=["noise"])
        # Listed -> ramps with cf.
        assert sim._get_effective_cf("noise", 0.25) == pytest.approx(0.25)
        # Unlisted -> held at its configured strength, NOT ramped.
        assert sim._get_effective_cf("b1", 0.25) == pytest.approx(1.0)
        assert sim._get_effective_cf("b1", 0.0) == pytest.approx(1.0)

    def test_range_floors_the_severity(self) -> None:
        """vmin > 0 is a real floor: the degradation never fully vanishes."""
        sim = self._sim(
            progressive_degradations=["noise"],
            degradation_ranges={"noise": (0.3, 0.8)},
        )
        assert sim._get_effective_cf("noise", 0.0) == pytest.approx(0.3)
        assert sim._get_effective_cf("noise", 1.0) == pytest.approx(0.8)
        assert sim._get_effective_cf("noise", 0.5) == pytest.approx(0.55)

    def test_range_is_observable_end_to_end(self) -> None:
        """Not just arithmetic — a floored range must change the OUTPUT at cf=0."""
        clean = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        no_floor = self._sim(
            enable_b1=True,
            b1_strength=0.8,
            progressive_degradations=["b1"],
            degradation_ranges={"b1": (0.0, 1.0)},
        )
        floored = self._sim(
            enable_b1=True,
            b1_strength=0.8,
            progressive_degradations=["b1"],
            degradation_ranges={"b1": (0.9, 1.0)},
        )
        a, _, _ = no_floor(clean, corruption_factor=0.0, seed=0)
        b, _, _ = floored(clean, corruption_factor=0.0, seed=0)
        # cf=0 with no floor is (nearly) clean; with a 0.9 floor it is not.
        assert float((a - clean).norm() / clean.norm()) < 1e-4
        assert float((b - clean).norm() / clean.norm()) > 1e-2

    @pytest.mark.parametrize("bad", [(-0.1, 1.0), (0.0, 1.5), (0.8, 0.2)])
    def test_invalid_range_raises(self, bad: tuple[float, float]) -> None:
        """A knob that is read must also be validated (pitfall #15)."""
        with pytest.raises(ValueError, match="0 <= vmin <= vmax <= 1"):
            self._sim(degradation_ranges={"noise": bad})


class TestForwardSeed:
    """The ``seed`` kwarg the sweep needs to make a run reproducible.

    Every stochastic stage of the twin (motion poses, B0/B1 field draws, the
    undersampling mask, spike/zipper phases, the magic-angle fibre patches, the
    AWGN) reads the GLOBAL torch RNG, so without a seed the same
    ``(image, corruption_factor)`` gives a different answer on every call.
    """

    @staticmethod
    def _sim() -> DigitalTwinSimulator:
        return DigitalTwinSimulator(
            im_size=(32, 32),
            marker_type="none",
            motion_type="rigid",
            enable_motion=True,
            enable_b0=True,
            enable_b1=True,
            enable_undersampling=True,
            acceleration=4.0,
            enable_spike_noise=True,
            snr_range=(10.0, 10.0),
        )

    def test_same_seed_same_output(self) -> None:
        sim = self._sim()
        clean = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        a, _, _ = sim(clean, corruption_factor=0.7, seed=3)
        b, _, _ = sim(clean, corruption_factor=0.7, seed=3)
        assert torch.equal(a, b)

    def test_different_seed_different_output(self) -> None:
        sim = self._sim()
        clean = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        a, _, _ = sim(clean, corruption_factor=0.7, seed=3)
        b, _, _ = sim(clean, corruption_factor=0.7, seed=4)
        assert not torch.equal(a, b)

    def test_seed_none_is_unseeded(self) -> None:
        sim = self._sim()
        clean = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        a, _, _ = sim(clean, corruption_factor=0.7)
        b, _, _ = sim(clean, corruption_factor=0.7)
        assert not torch.equal(a, b)

    def test_seeding_does_not_leak_into_the_callers_rng(self) -> None:
        """A seeded twin must not perturb dataloader shuffling or dropout."""
        sim = self._sim()
        clean = torch.randn(1, 1, 32, 32, dtype=torch.complex64)

        torch.manual_seed(99)
        expected = torch.randn(4)

        torch.manual_seed(99)
        sim(clean, corruption_factor=0.5, seed=1234)
        after = torch.randn(4)

        assert torch.equal(expected, after), "forward(seed=...) leaked RNG state"

    def test_degrade_at_forwards_the_seed(self) -> None:
        sim = self._sim()
        clean = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
        a = sim.degrade_at(clean, t=5, num_timesteps=10, seed=7)
        b = sim.degrade_at(clean, t=5, num_timesteps=10, seed=7)
        assert torch.equal(a, b)


def test_extensions_taxonomy_cross_reference_is_accurate() -> None:
    """The module docstring cross-references the ``D1-D31`` taxonomy living in
    :mod:`digital_twin_extensions`. Verify that claim stays accurate (the
    ``D1-D20`` label it replaced was stale): the extensions registry is 31 ops."""
    from spectramr.infrastructure.physics.digital_twin_extensions import (
        DEGRADATION_REGISTRY,
    )

    assert len(DEGRADATION_REGISTRY) == 31


class TestRegistryDegradationsAreReachable:
    """The simulator used to validate ``progressive_degradations`` against a
    hardcoded 14-name frozenset, so **26 of the 31 registered physics
    degradations could not be named from any config** — only sim2rank's sweep
    could reach them. The legal set is now the union of the native axes and
    ``DEGRADATION_REGISTRY``.
    """

    @staticmethod
    def _registry_names() -> set[str]:
        from spectramr.infrastructure.physics.digital_twin_extensions import (
            DEGRADATION_REGISTRY,
        )

        return set(DEGRADATION_REGISTRY)

    def test_known_axes_is_the_union_of_both_banks(self) -> None:
        from spectramr.infrastructure.physics.digital_twin_simulator import (
            NATIVE_DEGRADATION_AXES,
            known_degradation_axes,
        )

        known = known_degradation_axes()
        assert known == NATIVE_DEGRADATION_AXES | self._registry_names()
        # The regression this guards: the old set was the native 14 alone.
        assert known > NATIVE_DEGRADATION_AXES

    def test_every_registry_degradation_is_nameable(self) -> None:
        """Each of the 31 registered ops now constructs without raising."""
        for name in sorted(self._registry_names()):
            DigitalTwinSimulator(im_size=IM_SIZE, progressive_degradations=[name])

    def test_unknown_name_still_raises(self) -> None:
        """Widening the vocabulary must not weaken the guard (pitfall #15)."""
        with pytest.raises(ValueError, match="Unknown progressive_degradations"):
            DigitalTwinSimulator(im_size=IM_SIZE, progressive_degradations=["b_0"])
        with pytest.raises(ValueError, match="Unknown degradation_ranges"):
            DigitalTwinSimulator(im_size=IM_SIZE, degradation_ranges={"nope": (0.0, 1.0)})

    def test_overlapping_names_are_handled_natively_not_twice(self) -> None:
        """Five names live in BOTH banks. The native stage is diffusion-coupled
        and already wired, so it wins; applying the registry copy as well would
        corrupt twice at one severity."""
        from spectramr.infrastructure.physics.digital_twin_simulator import (
            NATIVE_DEGRADATION_AXES,
        )

        overlap = sorted(NATIVE_DEGRADATION_AXES & self._registry_names())
        assert overlap, "expected a non-empty overlap; the guard is vacuous without it"

        sim = DigitalTwinSimulator(im_size=IM_SIZE, progressive_degradations=[*overlap, "rician"])
        assert sim._registry_degradations == ("rician",), (
            f"native axes {overlap} leaked into the registry pass and would be "
            "applied a second time"
        )

    def test_a_registry_only_axis_changes_the_output(self) -> None:
        """Reachable must mean *applied*, not merely accepted — the difference
        between wiring a knob and advertising one (pitfall #16)."""
        clean = torch.randn(1, 1, *IM_SIZE, dtype=torch.complex64)
        base = DigitalTwinSimulator(im_size=IM_SIZE)
        with_axis = DigitalTwinSimulator(im_size=IM_SIZE, progressive_degradations=["rician"])

        a, _, _ = base.forward(clean, corruption_factor=1.0, seed=11)
        b, _, _ = with_axis.forward(clean, corruption_factor=1.0, seed=11)
        assert not torch.allclose(a, b), (
            "progressive_degradations=['rician'] produced the same output as the "
            "unmodified pipeline, so the registry pass never ran"
        )

    def test_registry_pass_is_deterministic_under_a_seed(self) -> None:
        clean = torch.randn(1, 1, *IM_SIZE, dtype=torch.complex64)
        sim = DigitalTwinSimulator(im_size=IM_SIZE, progressive_degradations=["rician"])
        a, _, _ = sim.forward(clean, corruption_factor=1.0, seed=3)
        b, _, _ = sim.forward(clean, corruption_factor=1.0, seed=3)
        assert torch.equal(a, b)

    def test_registry_pass_preserves_the_complex_dtype_contract(self) -> None:
        """Regression: ``rician`` is intrinsically real, and returning it raw
        broke the pipeline's complex contract with ``ComplexFloat did not match
        Float``. The original phase is re-attached rather than zeroed."""
        clean = torch.randn(1, 1, *IM_SIZE, dtype=torch.complex64)
        for axis in ("rician", "complex_gaussian", "rigid_motion", "t2star_blur"):
            sim = DigitalTwinSimulator(im_size=IM_SIZE, progressive_degradations=[axis])
            out, _, _ = sim.forward(clean, corruption_factor=1.0, seed=5)
            assert out.is_complex(), f"{axis} broke the complex dtype contract"
            assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()


class TestSeveritySchedule:
    """The severity ramp reaches the simulator, and does not disturb the default."""

    @staticmethod
    def _sim(**kw):
        from spectramr.config.schemas.physics import DigitalTwinConfig
        from spectramr.infrastructure.physics.digital_twin_simulator import (
            DigitalTwinSimulator,
        )

        return DigitalTwinSimulator.from_config(DigitalTwinConfig(**kw), (32, 32))

    def test_the_default_effective_cf_is_bit_identical_to_the_linear_remap(self):
        """Guards every config written before the schedule existed."""
        sim = self._sim()
        for feat in ("motion", "noise", "b0", "b1", "rf_zipper"):
            vmin, vmax = sim.degradation_ranges.get(feat, (0.0, 1.0))
            for cf in (0.0, 0.17, 1 / 3, 0.5, 0.83, 1.0):
                eff = (
                    cf
                    if (not sim.progressive_degradations or feat in sim.progressive_degradations)
                    else 1.0
                )
                expected = vmin + (vmax - vmin) * eff
                assert sim._get_effective_cf(feat, cf).hex() == expected.hex()

    def test_the_schedule_survives_from_config(self):
        """The seam, not the schema. This funnel silently dropped
        progressive_degradations and degradation_ranges once already -- a schema
        test would pass while the simulator ran linear."""
        sim = self._sim(
            progressive_degradations=["motion", "noise"],
            degradation_ranges={"motion": (0.0, 1.0), "noise": (0.0, 1.0)},
            degradation_schedules={"motion": "power_law"},
            degradation_schedule_power=2.0,
        )
        assert sim._get_effective_cf("motion", 0.5) == pytest.approx(0.25)
        assert sim._get_effective_cf("noise", 0.5) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "schedule", ["linear", "polynomial", "power_law", "exponential", "step"]
    )
    def test_a_degenerate_range_still_pins_theta_under_every_schedule(self, schedule):
        """The invariant a fitted DegradationChain depends on (PR #787): the
        severities it emits are (theta, theta) pairs, and (vmax - vmin) is 0, so no
        ramp shape can move them. Parametrised because 'the maths says so' is
        exactly the kind of inference that stops holding after an edit."""
        sim = self._sim(
            progressive_degradations=["complex_gaussian"],
            degradation_ranges={"complex_gaussian": (0.53, 0.53)},
            degradation_schedule=schedule,
        )
        for cf in (0.0, 0.3, 0.7, 1.0):
            assert sim._get_effective_cf("complex_gaussian", cf) == pytest.approx(0.53)

    def test_severity_at_timestep_agrees_with_what_is_applied(self):
        """A thin wrapper on purpose: a second implementation of the ramp is how a
        scheduled severity and an applied one drift apart."""
        sim = self._sim(
            progressive_degradations=["motion"],
            degradation_ranges={"motion": (0.0, 1.0)},
            degradation_schedules={"motion": "power_law"},
        )
        assert sim.severity_at_timestep("motion", 500, 1001) == sim._get_effective_cf("motion", 0.5)

    def test_the_timestep_ladder_is_inspectable_without_a_forward_pass(self):
        """What a timestep-aware training will apply, before any data moves --
        the counterpart of KSpaceAccelerator.get_acceleration_factor(t)."""
        sim = self._sim(
            progressive_degradations=["motion"],
            degradation_ranges={"motion": (0.0, 1.0)},
            degradation_schedules={"motion": "power_law"},
        )
        ladder = [sim.severity_at_timestep("motion", t, 5) for t in range(5)]
        assert ladder == pytest.approx([0.0, 0.0625, 0.25, 0.5625, 1.0])

    def test_a_single_timestep_is_the_clean_end(self):
        sim = self._sim()
        assert sim.severity_at_timestep("motion", 0, 1) == pytest.approx(0.0)

    def test_zero_timesteps_raises(self):
        with pytest.raises(ValueError, match="num_timesteps must be positive"):
            self._sim().severity_at_timestep("motion", 0, 0)

    def test_direct_construction_rejects_an_unknown_scheduled_axis(self):
        """Half this class's callers bypass the schema (tests, the chain replay,
        the sweeps), and a key nothing looks up is silently inert."""
        from spectramr.infrastructure.physics.digital_twin_simulator import (
            DigitalTwinSimulator,
        )

        with pytest.raises(ValueError, match="Unknown degradation_schedules"):
            DigitalTwinSimulator(im_size=(32, 32), degradation_schedules={"not_an_axis": "linear"})


class TestAtAcceleration:
    """``DigitalTwinSimulator.at_acceleration`` (the OOD readout's seam, VF review 2026-09-03)."""

    @staticmethod
    def _sim(**kw) -> DigitalTwinSimulator:
        args = dict(
            im_size=IM_SIZE,
            enable_motion=False,
            snr_range=(100.0, 100.0),
            enable_undersampling=True,
            acceleration=4.0,
        )
        args.update(kw)
        return DigitalTwinSimulator(**args)

    @staticmethod
    def _sampled_fraction(sim: DigitalTwinSimulator) -> float:
        mask = sim.last_undersampling_mask
        assert mask is not None
        return float(mask.float().mean())

    def test_the_twin_undersamples_at_the_rung_inside_and_at_its_own_rate_outside(self) -> None:
        sim = self._sim()
        img = torch.complex(torch.randn(1, 1, *IM_SIZE), torch.randn(1, 1, *IM_SIZE))
        sim(img)
        in_distribution = self._sampled_fraction(sim)
        with sim.at_acceleration(16.0):
            assert sim.acceleration == 16.0 and sim.enable_undersampling is True
            sim(img)
            out_of_distribution = self._sampled_fraction(sim)
        # 16x keeps fewer lines than 4x; both keep the centre.
        assert out_of_distribution < in_distribution
        assert sim.acceleration == 4.0 and sim.enable_undersampling is True
        assert sim.last_undersampling_mask is None, "the OOD mask must not outlive the pass"

    def test_a_twin_that_never_undersampled_is_switched_on_only_inside(self) -> None:
        sim = self._sim(enable_undersampling=False, acceleration=1.0)
        img = torch.complex(torch.randn(1, 1, *IM_SIZE), torch.randn(1, 1, *IM_SIZE))
        with sim.at_acceleration(8.0):
            sim(img)
            assert sim.last_undersampling_mask is not None
        assert sim.enable_undersampling is False and sim.acceleration == 1.0

    def test_restored_when_the_pass_raises(self) -> None:
        """Planted violation: a failing rung must not leave the twin at 32x."""
        sim = self._sim()
        with pytest.raises(RuntimeError, match="boom"), sim.at_acceleration(32.0):
            raise RuntimeError("boom")
        assert sim.acceleration == 4.0 and sim.enable_undersampling is True

    def test_a_rate_at_or_below_one_is_refused(self) -> None:
        sim = self._sim()
        with pytest.raises(ValueError, match="must exceed 1.0"), sim.at_acceleration(1.0):
            pass
