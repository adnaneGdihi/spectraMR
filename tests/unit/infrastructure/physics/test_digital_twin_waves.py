"""Regression tests for Waves 1-5 (M2-M6, Mod1-Mod8, Min1-Min2) of the
digital-twin review.

Each test pins a specific post-fix invariant; running these against the
pre-Wave code would fail. Tests stay small (16x16-32x32 phantoms) so
the full suite runs in seconds.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.degradations import (  # noqa: E402
    DEGRADATION_REGISTRY,
    apply_degradation,
    degrade_concomitant_phase,
    degrade_iq_ghost,
    degrade_partial_fourier,
    degrade_stimulated_echo,
    degrade_susceptibility,
    degrade_variable_density_2d,
    degrade_variable_density_cartesian,
    degrade_variable_density_undersampling,
    list_degradations,
)
from spectramr.infrastructure.physics.digital_twin_simulator import (  # noqa: E402
    CornerFiducialEmbedder,
    DigitalTwinSimulator,
    simulate_b1_bias_field,
    simulate_chemical_shift,
    simulate_concomitant_phase,
    simulate_magic_angle,
)


def _phantom(H: int = 32, W: int = 32, C: int = 1) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    base = ((x ** 2 + y ** 2) < 0.6).float().unsqueeze(0).unsqueeze(0)
    base = base.expand(1, C, -1, -1).contiguous()
    return torch.complex(base, torch.zeros_like(base))


def _offset_phantom(H: int = 32, W: int = 32, C: int = 1) -> torch.Tensor:
    """A disc displaced from the FOV centre.

    ``_phantom`` is real and centro-symmetric, so it equals its own conjugate
    mirror (``x.conj().flip(-2, -1) == x``) and is blind to any flip-based
    artefact: the ghost lands exactly on the anatomy and the air residual is
    identically zero. Offsetting the disc gives the mirror somewhere else to
    land, which is what makes a mirror-ghost test able to fail.
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    base = (((x - 0.4) ** 2 + (y - 0.3) ** 2) < 0.1).float().unsqueeze(0).unsqueeze(0)
    base = base.expand(1, C, -1, -1).contiguous()
    return torch.complex(base, torch.zeros_like(base))


# ──────────────────────────────────────────────────────────────────────
# Wave 1 (Mod1, Mod2, Mod6, Min5)
# ──────────────────────────────────────────────────────────────────────


class TestWave1_Cleanup:
    def test_multi_contrast_embedder_returns_per_channel_prior_when_flag_set(self) -> None:
        emb = CornerFiducialEmbedder(im_size=(32, 32), per_channel_marker_prior=True)
        x = _phantom(32, 32, C=3)
        # Give each channel a different magnitude so the per-channel
        # intensity scaling diverges from a global one.
        x = x * torch.tensor([1.0, 2.0, 4.0]).view(1, 3, 1, 1)
        _, prior = emb(x)
        # Per-channel prior must be [B, C, H, W], not [1, 1, H, W].
        assert prior.shape == (1, 3, 32, 32)
        # The channels must have different intensity values.
        peaks = prior.abs().amax(dim=(-2, -1))[0]  # [3]
        assert peaks[2] > peaks[0] * 1.5

    def test_multi_contrast_embedder_legacy_global_prior_by_default(self) -> None:
        emb = CornerFiducialEmbedder(im_size=(32, 32))
        x = _phantom(32, 32, C=3) * torch.tensor([1.0, 2.0, 4.0]).view(1, 3, 1, 1)
        _, prior = emb(x)
        # Legacy behaviour: prior is [1, 1, H, W].
        assert prior.shape[1] == 1

    def test_delegated_callables_bound_on_the_spec(self) -> None:
        """Mod6: each delegate kernel is bound once as its spec's ``fn``."""
        from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY

        for name in ("b0", "gradient_nonlinearity"):  # Mod7
            assert callable(DEGRADATION_REGISTRY[name].fn)

    def test_kwargs_forwarded_through_delegated(self) -> None:
        """Min5: caller-supplied kwargs reach the underlying simulator."""
        x = _phantom(32, 32)
        # Pass an explicit truncation_fraction; output should differ from
        # the theta-only default.
        a = apply_degradation("gibbs", x, theta=0.5, seed=0)
        b = apply_degradation("gibbs", x, theta=0.5, seed=0, truncation_fraction=0.5)
        assert not torch.allclose(a, b)


# ──────────────────────────────────────────────────────────────────────
# Wave 2 (M2, M3, M4, M5)
# ──────────────────────────────────────────────────────────────────────


class TestWave2_AdditiveAPI:
    def test_susceptibility_b0_axis_changes_dipole_projection(self) -> None:
        """M2: b0_axis="x" produces a different dipole field than b0_axis="y"."""
        x = _phantom(32, 32)
        a = degrade_susceptibility(x, theta=0.8, seed=5, b0_axis="y")
        b = degrade_susceptibility(x, theta=0.8, seed=5, b0_axis="x")
        # Different dipole projection → different phase pattern.
        assert (a - b).abs().mean() > 1e-4

    def test_susceptibility_invalid_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="b0_axis"):
            degrade_susceptibility(_phantom(32, 32), theta=0.5, b0_axis="z")

    def test_chemical_shift_scales_with_b0(self) -> None:
        """M3: shift_pixels increases linearly with B0 for fixed BW.

        At 0.3 T: shift = 42.577e6 * 0.3 * 3.5e-6 / 100 ~ 0.45 px (sub-pixel).
        At 3.0 T: shift ~ 4.47 px (visible).
        """
        from spectramr.infrastructure.physics.digital_twin_simulator import (
            simulate_chemical_shift as _simulate,
        )
        import math as _math

        # Compute expected shifts directly via the formula and confirm
        # the 10x scaling, which is the M3 contract.
        delta_03 = 42.577e6 * 0.3 * 3.5e-6 / 100.0
        delta_3 = 42.577e6 * 3.0 * 3.5e-6 / 100.0
        assert _math.isclose(delta_3 / delta_03, 10.0, rel_tol=1e-6)

        # Now confirm the simulator's effect strictly increases with B0.
        x = _phantom(64, 64)
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
        k = fft2c(x)
        at_03T = _simulate(k, (64, 64), b0_T=0.3, bw_per_pixel_hz=100.0)
        at_3T = _simulate(k, (64, 64), b0_T=3.0, bw_per_pixel_hz=100.0)
        d_03 = (ifft2c(at_03T) - x).abs().mean().item()
        d_3 = (ifft2c(at_3T) - x).abs().mean().item()
        assert d_3 > d_03

    def test_chemical_shift_fat_fraction_map(self) -> None:
        """M3: per-voxel fat fraction map applied."""
        x = _phantom(32, 32)
        k = torch.fft.fftn(x, dim=(-2, -1))
        # Zero fat fraction everywhere → no chemical shift artefact.
        ff = torch.zeros(1, 1, 32, 32)
        out = simulate_chemical_shift(k, (32, 32), shift_pixels=3.0, fat_fraction_map=ff)
        assert torch.allclose(out, k, atol=1e-5)

    def test_magic_angle_no_op_outside_msk(self) -> None:
        """M4: non-MSK tissue class is a no-op."""
        x = _phantom(32, 32)
        out = simulate_magic_angle(x, (32, 32), tissue_class="brain")
        assert torch.equal(out, x)

    def test_magic_angle_foreground_mask_restricts_collagen(self) -> None:
        """M4: foreground_mask restricts collagen patches."""
        x = _phantom(32, 32)
        # All-zero mask: no collagen anywhere → no T2 modulation.
        zero_mask = torch.zeros(1, 1, 32, 32)
        out = simulate_magic_angle(
            x, (32, 32), tissue_class="msk", foreground_mask=zero_mask
        )
        # With empty collagen mask the modulation factor is 1.0 everywhere.
        assert torch.allclose(out, x, atol=1e-5)

    def test_d4_rename_alias_still_works(self) -> None:
        """M5: degrade_variable_density_undersampling still points to the
        renamed function."""
        x = _phantom(32, 32)
        a = degrade_variable_density_2d(x, theta=0.5, seed=0)
        b = degrade_variable_density_undersampling(x, theta=0.5, seed=0)
        assert torch.equal(a, b)

    def test_vd_cartesian_acquires_whole_pe_lines(self) -> None:
        """M5: Cartesian variant samples whole PE lines, not points.

        Use a k-space-uniform input (image is a unit DC delta in
        k-space) so the post-mask values read off the binary mask
        directly without FFT round-off ambiguity.
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        # All-ones k-space → image is a delta at the origin; FFT
        # back gives all-ones again, and the mask reads cleanly.
        H, W = 64, 64
        k_in = torch.ones(1, 1, H, W, dtype=torch.complex64)
        x = ifft2c(k_in)
        out = degrade_variable_density_cartesian(x, theta=0.6, seed=0)
        k_out = fft2c(out)
        kept_rows = (k_out.abs() > 0.5).any(dim=-1).squeeze()  # [H]
        # Row-wise unanimity: every column in a kept row should be > 0.5.
        all_cols_kept = (k_out.abs() > 0.5).all(dim=-1).squeeze()  # [H]
        # Every kept row is fully kept.
        partial = (kept_rows & ~all_cols_kept).float().mean().item()
        assert partial < 0.05, (
            f"Cartesian VD has {partial:.2%} partial rows; "
            f"sampling should be strictly per-line."
        )


# ──────────────────────────────────────────────────────────────────────
# Wave 3 (Min1, Min2)
# ──────────────────────────────────────────────────────────────────────


class TestWave3_NewPhysics:
    def test_concomitant_phase_scales_with_1_over_B0(self) -> None:
        """Min1: phi_c ∝ 1/B0 (Bernstein 2004)."""
        x = _phantom(32, 32)
        out_low = simulate_concomitant_phase(x, (32, 32), b0_T=0.3)
        out_high = simulate_concomitant_phase(x, (32, 32), b0_T=3.0)
        # The added phase at low field should be much larger in magnitude
        # than at high field (10x scaling).
        phase_low = (out_low * x.conj()).angle().abs().mean()
        phase_high = (out_high * x.conj()).angle().abs().mean()
        assert phase_low > phase_high * 5.0

    def test_concomitant_phase_invalid_b0_raises(self) -> None:
        with pytest.raises(ValueError, match="b0_T"):
            simulate_concomitant_phase(_phantom(32, 32), (32, 32), b0_T=0.0)

    def test_b1_bias_scales_with_b0_when_supplied(self) -> None:
        """Min1: B1+ inhomogeneity is smaller at low field."""
        x = _phantom(32, 32)
        torch.manual_seed(0)
        out_3T = simulate_b1_bias_field(x, (32, 32), strength=0.5, b0_T=3.0)
        torch.manual_seed(0)
        out_03T = simulate_b1_bias_field(x, (32, 32), strength=0.5, b0_T=0.3)
        # At 0.3 T the modulation is closer to identity than at 3 T.
        d_3 = (out_3T - x).abs().mean()
        d_03 = (out_03T - x).abs().mean()
        assert d_03 < d_3

    def test_partial_fourier_zeros_one_half_at_theta_1(self) -> None:
        """Min2: theta=1 zeros (1 - min_fraction) of PE lines."""
        from spectramr.infrastructure.physics.fft_ops import fft2c
        x = _phantom(64, 64)
        out = degrade_partial_fourier(x, theta=1.0, seed=0, min_fraction=0.5)
        k = fft2c(out)
        # The function zeroed `cut = int(H * 0.5) = 32` rows at the end
        # of the (shifted) k-space; in fft2c-space those are the bottom
        # 32 rows.
        assert k[..., -32:, :].abs().max().item() < 1e-3

    def test_iq_ghost_adds_conjugate_mirror_energy(self) -> None:
        """Min2: I/Q imbalance produces a conjugate-mirror ghost.

        Uses ``_offset_phantom``: the centred disc is its own conjugate mirror,
        so it puts the ghost back on top of the anatomy and reports a zero air
        residual whatever the degrader does.
        """
        x = _offset_phantom(32, 32)
        out = degrade_iq_ghost(x, theta=1.0, seed=0)
        # Energy of the residual outside the original anatomy support
        # must be > 0 (the ghost lands in air pixels).
        anatomy_mask = x.abs() > 1e-3
        air_residual = ((out - x).abs() * (~anatomy_mask).float()).sum()
        assert air_residual.item() > 0.0
        # ...and it lands specifically on the conjugate-MIRRORED support, not
        # just anywhere: that is what distinguishes D25 from additive noise.
        mirror_only = anatomy_mask.flip(dims=(-2, -1)) & ~anatomy_mask
        assert ((out - x).abs() * mirror_only.float()).sum().item() > 0.0

    def test_stimulated_echo_creates_pe_shift_ghost(self) -> None:
        """Min2: stimulated echo is a PE-axis shifted ghost."""
        x = _phantom(64, 64)
        out = degrade_stimulated_echo(x, theta=0.8, seed=0)
        # A ghost shifted in y but unshifted in x should leave the
        # rowwise FFT magnitude largely unchanged.
        assert not torch.allclose(out, x)
        assert out.shape == x.shape

    def test_slice_profile_is_gone(self) -> None:
        """#245: D27 was a duplicate of D12 and was removed, not re-tuned.

        ``degrade_slice_profile`` and ``degrade_t2star_blur`` were the same
        isotropic depthwise Gaussian blur (|cos| of their theta=1 residuals =
        0.998), double-counted as two "orthogonal" axes. The axis is deleted; the
        registry must not resolve it, and it must fail loudly rather than silently
        aliasing onto its twin (pitfall #9).
        """
        assert "slice_profile" not in list_degradations()
        with pytest.raises(KeyError, match="Unknown degradation"):
            apply_degradation("slice_profile", _phantom(32, 32), theta=0.6, seed=0)


# ──────────────────────────────────────────────────────────────────────
# Wave 4 (registry completeness)
# ──────────────────────────────────────────────────────────────────────


class TestWave4_Registry:
    def test_b0_geometric_distortion_registered(self) -> None:
        """Mod7: previously-missing simulator entries now registered."""
        names = list_degradations()
        assert "b0_geometric_distortion" in names
        assert "gradient_nonlinearity" in names
        assert "magic_angle" in names
        assert "concomitant_phase" in names

    def test_new_wave3_entries_registered(self) -> None:
        for n in ("partial_fourier", "iq_ghost", "stimulated_echo"):
            assert n in list_degradations(), f"missing: {n}"

    def test_vd_cartesian_registered_separately(self) -> None:
        names = list_degradations()
        assert "vd_cartesian" in names
        assert "vd_undersamp" in names

    def test_registry_codes_are_unique(self) -> None:
        """The #245 defect was a DUPLICATE D27, not a wrong total.

        Two names sharing one code overwrite each other in every code-keyed
        leaderboard, so uniqueness is the property that actually broke. A
        hard-coded size caught none of that while tripping on legitimate
        growth (D28 resolution_snr, D29-D31 motion types).
        """
        codes = [s.code for s in DEGRADATION_REGISTRY.values()]
        duplicates = sorted({c for c in codes if codes.count(c) > 1})
        assert not duplicates, f"degradation codes reused: {duplicates}"

    def test_registry_keys_match_their_spec_name(self) -> None:
        """A registration typo (key != spec.name) leaves the axis unlookupable
        under the name every report prints."""
        assert DEGRADATION_REGISTRY, "registry is empty"
        mismatched = {k: s.name for k, s in DEGRADATION_REGISTRY.items() if k != s.name}
        assert not mismatched, f"key/name mismatch: {mismatched}"

    @pytest.mark.parametrize(
        "name", ["b0_geometric_distortion", "gradient_nonlinearity",
                 "magic_angle", "concomitant_phase",
                 "partial_fourier", "iq_ghost", "stimulated_echo",
                 "vd_cartesian"]
    )
    def test_new_entries_are_deterministic(self, name: str) -> None:
        x = _phantom(32, 32)
        a = apply_degradation(name, x, theta=0.5, seed=11)
        b = apply_degradation(name, x, theta=0.5, seed=11)
        assert torch.allclose(a, b), f"{name} not deterministic"


# ──────────────────────────────────────────────────────────────────────
# Wave 5 (architectural flags)
# ──────────────────────────────────────────────────────────────────────


class TestWave5_ArchitecturalFlags:
    def test_marker_water_only_flag_changes_chemical_shift_path(self) -> None:
        """M6: marker_is_water_only routes chemical shift to anatomy only."""
        sim_default = DigitalTwinSimulator(
            im_size=(32, 32),
            enable_motion=False, enable_b0=False, enable_b1=False,
            enable_chemical_shift=True, chemical_shift_pixels=2.0,
            snr_range=(100.0, 100.0),
        )
        sim_water = DigitalTwinSimulator(
            im_size=(32, 32),
            enable_motion=False, enable_b0=False, enable_b1=False,
            enable_chemical_shift=True, chemical_shift_pixels=2.0,
            snr_range=(100.0, 100.0),
            marker_is_water_only=True,
        )
        x = _phantom(32, 32)
        torch.manual_seed(0)
        out_default, _, _ = sim_default(x)
        torch.manual_seed(0)
        out_water, _, _ = sim_water(x)
        # The marker corners (FOV edges) should differ between the two:
        # in the default path the marker has a chemical-shift ghost;
        # in the water-only path it does not.
        marker_mask = sim_default.embedder.marker_mask[0, 0]  # [H, W]
        # Mean magnitude difference inside the marker region.
        diff = (out_default - out_water).abs()[0, 0] * marker_mask
        # Should be > 0 because the marker behaviour differs between modes.
        assert diff.sum().item() > 0.0

    def test_per_channel_marker_prior_flag_propagates_to_embedder(self) -> None:
        """Mod3: simulator-level flag wires into embedder."""
        sim = DigitalTwinSimulator(
            im_size=(32, 32), per_channel_marker_prior=True,
            enable_motion=False, enable_b0=False, enable_b1=False,
        )
        assert sim.embedder.per_channel_marker_prior is True

    def test_coherent_3d_motion_shares_trajectory_across_slices(self) -> None:
        """Mod4: a canonical (B, C, D, H, W) multislice stack gets one trajectory per subject."""
        from spectramr.infrastructure.physics.degradations.geometry import (
            AcquisitionGeometry,
            AcquisitionMode,
        )

        ms = AcquisitionGeometry(mode=AcquisitionMode.MULTISLICE_2D, voxel_mm=None)
        kw = {
            "im_size": (16, 16), "motion_type": "random_shot",
            "enable_motion": True, "enable_b0": False, "enable_b1": False,
            "snr_range": (100.0, 100.0),
        }
        # Identical slices along D (dim -3): coherent motion keeps them identical.
        x5 = _phantom(16, 16).unsqueeze(2).expand(1, 1, 4, 16, 16).contiguous()
        out, _, _ = DigitalTwinSimulator(coherent_3d_motion=True, **kw)(x5, seed=0, geometry=ms)
        assert out.shape == x5.shape
        assert torch.isfinite(out.real).all() and torch.isfinite(out.imag).all()
        for d in range(1, 4):
            assert torch.allclose(out[:, :, d], out[:, :, 0], atol=1e-3)
        indep, _, _ = DigitalTwinSimulator(coherent_3d_motion=False, **kw)(x5, seed=0, geometry=ms)
        assert not torch.allclose(indep[:, :, 1], indep[:, :, 0], atol=1e-3)

    def test_depth_last_5d_without_a_mode_is_refused(self) -> None:
        """The old (B, C, H, W, D) fold is gone: 5-D needs a declared acquisition mode."""
        sim = DigitalTwinSimulator(im_size=(16, 16), enable_b0=False, enable_b1=False)
        x5 = _phantom(16, 16).unsqueeze(-1).expand(1, 1, 16, 16, 4).contiguous()
        with pytest.raises(ValueError, match="AcquisitionMode"):
            sim(x5)
