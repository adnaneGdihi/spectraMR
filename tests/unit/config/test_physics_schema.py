import typing

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.physics import (
    B0CorrectionConfig,
    B0SimulationConfig,
    BlochSimulationConfig,
    CoilSensitivityConfig,
    CompressedSensingConfig,
    DataConsistencyConfig,
    DigitalTwinConfig,
    IterativeRefinementConfig,
    KSpaceConfig,
    MotionCorrectionConfig,
    PhaseCorrectionConfig,
    PhaseCorrectionnConfig,
    PhysicsConfigSchema,
    PINNConfig,
    RegularizationConfig,
)


class TestPhysicsConfigSchema:
    def test_defaults(self):
        schema = PhysicsConfigSchema()
        assert schema.data_consistency.enabled is False
        assert schema.field_strength == 3.0

    def test_custom_values(self):
        schema = PhysicsConfigSchema(
            field_strength=1.5, data_consistency=DataConsistencyConfig(enabled=True)
        )
        assert schema.field_strength == 1.5
        assert schema.data_consistency.enabled is True

    def test_validation(self):
        with pytest.raises(ValidationError):
            PhysicsConfigSchema(field_strength=0)


class TestDataConsistencyConfig:
    def test_defaults(self):
        schema = DataConsistencyConfig()
        assert schema.enabled is False
        assert schema.weight == 1.0

    @pytest.mark.parametrize("key", ["acs_mask_center_fraction", "enable_acs_replacement"])
    def test_an_unread_acs_knob_is_refused_at_load(self, key):
        """#563: the block is extra="ignore", so deleting the field alone would drop it."""
        with pytest.raises(ValidationError, match=f"`physics.data_consistency.{key}` was removed"):
            DataConsistencyConfig(**{key: True if key.startswith("enable") else 0.08})

    def test_both_retired_knobs_are_named_in_one_error(self):
        both = r"(?s)acs_mask_center_fraction.*enable_acs_replacement"
        with pytest.raises(ValidationError, match=both):
            DataConsistencyConfig(acs_mask_center_fraction=0.08, enable_acs_replacement=True)

    def test_the_refusal_reaches_the_physics_block(self):
        with pytest.raises(ValidationError, match="enable_acs_replacement` was removed"):
            PhysicsConfigSchema(data_consistency={"enabled": True, "enable_acs_replacement": True})


class TestCoilSensitivityConfig:
    def test_defaults(self):
        schema = CoilSensitivityConfig()
        assert schema.enable_estimation is False
        assert "estimation_method" not in CoilSensitivityConfig.model_fields

    def test_the_unread_estimation_method_is_refused_naming_the_live_key(self):
        """#306: the estimator is chosen by physics.coil_processing.estimation.method."""
        with pytest.raises(ValidationError, match=r"coil_processing\.estimation\.method"):
            CoilSensitivityConfig(estimation_method="espirit")


class TestKSpaceConfig:
    def test_defaults(self):
        schema = KSpaceConfig()
        assert schema.enable_kspace_recon is False
        assert schema.enforce_hermitian_symmetry is True


class TestRegularizationConfig:
    def test_defaults(self):
        schema = RegularizationConfig()
        assert schema.enable_tv is False
        assert schema.tv_weight == 0.01


class TestIterativeRefinementConfig:
    def test_defaults(self):
        schema = IterativeRefinementConfig()
        assert schema.enabled is False
        assert schema.num_steps == 1


class TestPhaseCorrectionConfig:
    def test_defaults(self):
        schema = PhaseCorrectionConfig()
        assert schema.enable_phase_estimation is False

    def test_deprecated_alias_is_same_class(self):
        """The doubled-n spelling is a pure import-compat alias (remove 2026-08)."""
        assert PhaseCorrectionnConfig is PhaseCorrectionConfig


class TestB0CorrectionConfig:
    def test_defaults(self):
        schema = B0CorrectionConfig()
        assert schema.enable_b0_correction is False
        assert schema.num_shim_orders == 2


class TestB0SimulationConfig:
    def test_defaults(self):
        schema = B0SimulationConfig()
        assert schema.enabled is False
        assert schema.style == "smooth"


class TestMotionCorrectionConfig:
    def test_defaults(self):
        schema = MotionCorrectionConfig()
        assert schema.enable_motion_correction is False
        assert schema.motion_type == "translational"


class TestCompressedSensingConfig:
    def test_defaults(self):
        schema = CompressedSensingConfig()
        assert schema.enabled is False
        assert schema.sampling_pattern == "cartesian"


class TestPINNConfig:
    def test_defaults(self):
        schema = PINNConfig()
        assert schema.enabled is False
        assert schema.pde_type == "wave_equation"


class TestBlochSimulationConfig:
    def test_defaults(self):
        schema = BlochSimulationConfig()
        assert schema.enabled is False
        assert schema.target_tr == 3000.0


class TestDigitalTwinConfigMotion:
    """Composite-motion + motion-curriculum fields (2026-05-26)."""

    def test_defaults_preserve_single_motion(self):
        d = DigitalTwinConfig()
        assert d.motion_composite == []
        assert d.motion_severity == 1.0
        assert d.motion_type == "rigid"

    def test_composite_accepts_valid_models(self):
        d = DigitalTwinConfig(motion_composite=["rigid", "periodic", "random_shot"])
        assert d.motion_composite == ["rigid", "periodic", "random_shot"]

    def test_composite_rejects_unknown_model(self):
        with pytest.raises(ValidationError):
            DigitalTwinConfig(motion_composite=["rigid", "banana"])

    def test_motion_severity_default_and_custom(self):
        assert DigitalTwinConfig().motion_severity == 1.0
        assert DigitalTwinConfig(motion_severity=2.5).motion_severity == 2.5

    def test_motion_severity_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            DigitalTwinConfig(motion_severity=-1.0)

    def test_apply_as_transform_defaults_off(self):
        d = DigitalTwinConfig()
        assert d.apply_as_transform is False
        assert d.transform_degradation_only is True

    def test_apply_as_transform_opt_in(self):
        d = DigitalTwinConfig(apply_as_transform=True, transform_degradation_only=False)
        assert d.apply_as_transform is True
        assert d.transform_degradation_only is False


# ── MultiAcquisitionConfig (faithful AFI/DAM/dual-echo synthesis) ──────────


def test_multi_acquisition_config_defaults_and_validation() -> None:
    from spectramr.config.schemas.physics import (
        MultiAcquisitionConfig,
        PhysicsConfigSchema,
    )

    # default: disabled, mounted on PhysicsConfigSchema
    assert PhysicsConfigSchema().multi_acquisition.enabled is False
    assert MultiAcquisitionConfig(enabled=True, method="afi").method == "afi"

    # enabled without a method must raise (no silent no-op)
    with pytest.raises(ValidationError):
        MultiAcquisitionConfig(enabled=True)

    # bloch_siegert + mrf are supported methods
    assert (
        MultiAcquisitionConfig(enabled=True, method="bloch_siegert").method
        == "bloch_siegert"
    )
    mrf = MultiAcquisitionConfig(enabled=True, method="mrf")
    assert mrf.method == "mrf" and mrf.n_timepoints == 16

    # series methods (low-rank temporal / sub-voxel SR) + their knobs
    sr = MultiAcquisitionConfig(enabled=True, method="subvoxel_sr")
    assert sr.method == "subvoxel_sr" and sr.n_frames == 8 and sr.sr_scale == 2
    assert (
        MultiAcquisitionConfig(enabled=True, method="lowrank_temporal").method
        == "lowrank_temporal"
    )

    # bssfp_banding (phase-cycled B0 mapping) + its n_phase_cycles knob
    bssfp = MultiAcquisitionConfig(enabled=True, method="bssfp_banding")
    assert bssfp.method == "bssfp_banding"
    assert bssfp.n_phase_cycles == 4  # ESM identifiability floor
    assert (
        MultiAcquisitionConfig(
            enabled=True, method="bssfp_banding", n_phase_cycles=8
        ).n_phase_cycles
        == 8
    )
    # below the identifiability floor must be rejected at load
    with pytest.raises(ValidationError):
        MultiAcquisitionConfig(enabled=True, method="bssfp_banding", n_phase_cycles=2)

    # unknown / unimplemented method rejected at load (CLAUDE.md pitfall #9)
    with pytest.raises(ValidationError):
        MultiAcquisitionConfig(enabled=True, method="triple_angle")


def test_multi_acquisition_lambda_smooth_is_a_declared_knob() -> None:
    """The field-smoothness weight is config-driven, not hardcoded.

    Regression for the 2026-07 exp_vf_01 audit: the strategy hardcoded
    ``self._lambda_smooth = 0.01`` while the arm YAML declared a
    ``phase_smoothness_complex`` weight of 1.0. The declared weight was never
    read, so the YAML advertised a knob nothing consumed (CLAUDE.md #15).
    """
    from spectramr.config.schemas.physics import MultiAcquisitionConfig

    # default preserves the historical hardcoded value → existing arms unchanged
    assert (
        MultiAcquisitionConfig(enabled=True, method="subvoxel_sr").lambda_smooth == 0.01
    )
    assert (
        MultiAcquisitionConfig(
            enabled=True, method="subvoxel_sr", lambda_smooth=0.25
        ).lambda_smooth
        == 0.25
    )
    # zero disables the term; negative weights are rejected at load
    assert (
        MultiAcquisitionConfig(
            enabled=True, method="subvoxel_sr", lambda_smooth=0.0
        ).lambda_smooth
        == 0.0
    )
    with pytest.raises(ValidationError):
        MultiAcquisitionConfig(enabled=True, method="subvoxel_sr", lambda_smooth=-1.0)


def test_multi_acquisition_normalize_magnitude_is_a_declared_knob() -> None:
    """The M0-proxy normalisation is on by default and can be turned off.

    k-space targets keep their raw scale through the loader (clamping at a
    percentile would clip the DC peak), so the IFFT'd magnitude carries an
    arbitrary scanner scale that the objective inherits QUADRATICALLY — the
    2026-07 exp_vf_01 run hit a first-step gradient norm of 6477 against a clip
    of 1.0. Default-on fixes it; false reproduces a pre-2026-07 run.
    """
    from spectramr.config.schemas.physics import MultiAcquisitionConfig

    assert (
        MultiAcquisitionConfig(enabled=True, method="subvoxel_sr").normalize_magnitude
        is True
    )
    assert (
        MultiAcquisitionConfig(
            enabled=True, method="subvoxel_sr", normalize_magnitude=False
        ).normalize_magnitude
        is False
    )


# ---------------------------------------------------------------------------
# Sub-voxel registration ladder (2026-07-26)
# ---------------------------------------------------------------------------
from spectramr.config.schemas.physics import (  # noqa: E402
    MultiAcquisitionConfig,
)


def test_subvoxel_registration_defaults_to_blind() -> None:
    """Default must leave pre-2026-07-26 arms behaviourally unchanged."""
    cfg = MultiAcquisitionConfig(enabled=True, method="subvoxel_sr")
    assert cfg.subvoxel_registration.shift_source == "blind"


@pytest.mark.parametrize("source", ["blind", "recovered", "oracle"])
def test_all_three_ladder_rungs_are_accepted(source: str) -> None:
    cfg = MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        subvoxel_registration={"shift_source": source},
    )
    assert cfg.subvoxel_registration.shift_source == source


def test_unknown_shift_source_raises() -> None:
    """No silent fallback to blind (CLAUDE.md #9)."""
    with pytest.raises(ValidationError):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            subvoxel_registration={"shift_source": "psychic"},
        )


def test_recovered_rung_rejects_a_periodic_marker() -> None:
    """A perfectly periodic lattice has a comb spectrum and is ambiguous modulo
    one period, so it cannot be registered: 0.55 px error vs 0.003 px jittered."""
    with pytest.raises(ValidationError, match="marker_jitter > 0"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            subvoxel_registration={"shift_source": "recovered", "marker_jitter": 0.0},
        )


def test_registration_block_rejected_on_other_methods() -> None:
    with pytest.raises(ValidationError, match="only applies to"):
        MultiAcquisitionConfig(
            enabled=True,
            method="afi",
            subvoxel_registration={"shift_source": "oracle"},
        )


def test_single_frame_allowed_for_subvoxel_only() -> None:
    """One frame is the ablation floor for subvoxel_sr and meaningless for a
    series method; a blanket ge=2 blocked exp_vf_01's declared control."""
    assert MultiAcquisitionConfig(
        enabled=True, method="subvoxel_sr", n_frames=1
    ).n_frames == 1
    with pytest.raises(ValidationError, match="below the floor of 2"):
        MultiAcquisitionConfig(enabled=True, method="lowrank_temporal", n_frames=1)


# ---------------------------------------------------------------------------
# Physical-units fiducial geometry (PR-B)
# ---------------------------------------------------------------------------
def _reg(sr_scale: int = 2, **kw):
    return MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        sr_scale=sr_scale,
        subvoxel_registration=kw,
    ).subvoxel_registration


def test_physical_mode_is_off_by_default() -> None:
    """The synthetic M4Raw arm keeps pixel geometry, where the grid IS the
    resolution and voxels are isotropic."""
    assert _reg().effective_voxel_mm is None


def test_effective_voxel_selects_physical_mode() -> None:
    r = _reg(
        sr_scale=3,
        shift_source="recovered",
        voxel_mm=(0.49, 0.49, 1.0),
        effective_voxel_mm=(1.6, 1.6, 5.0),
    )
    assert r.effective_voxel_mm == (1.6, 1.6, 5.0)
    assert r.voxel_mm == (0.49, 0.49, 1.0)
    assert r.marker_kappa == 1.0


def test_millimetre_knobs_without_a_voxel_size_raise() -> None:
    """There would be nothing to convert them against, so accepting them would
    mean silently ignoring them."""
    with pytest.raises(ValidationError, match="effective_voxel_mm is unset"):
        _reg(marker_sigma_mm=(1.6, 1.6))


def test_kappa_below_one_is_rejected() -> None:
    """kappa < 1 puts the marker below the effective voxel, where it aliases and
    phase correlation acquires a bias no averaging removes."""
    with pytest.raises(ValidationError):
        _reg(effective_voxel_mm=(1.6, 1.6), marker_kappa=0.5)


@pytest.mark.parametrize(
    "bad", [(1.6,), (1.6, 0.0), (1.6, -1.0), (1.0, 1.0, 1.0, 1.0)]
)
def test_malformed_voxel_tuples_are_rejected(bad) -> None:
    with pytest.raises(ValidationError):
        _reg(effective_voxel_mm=bad)


def test_anisotropic_ulf_geometry_is_accepted_verbatim() -> None:
    """The measured 64mT protocol. Through-plane is 3.1x the in-plane size, so a
    scalar sigma would be wrong by that factor whatever value it took."""
    r = _reg(
        sr_scale=3,
        shift_source="recovered",
        voxel_mm=(0.49, 0.49, 1.0),
        effective_voxel_mm=(1.6, 1.6, 5.0),
        marker_sigma_mm=(1.6, 1.6, 5.0),
        marker_spacing_mm=(12.8, 12.8, 40.0),
    )
    assert r.marker_sigma_mm[2] / r.marker_sigma_mm[0] == pytest.approx(3.125)



# ---------------------------------------------------------------------------
# The stored grid is not the resolution (PR-1)
# ---------------------------------------------------------------------------
def test_effective_voxel_without_the_stored_grid_raises() -> None:
    """Defaulting voxel_mm to effective_voxel_mm yields sigma_px = kappa, i.e. a
    ONE-pixel marker on a grid 3-7x finer than the acquisition. That is the
    sub-resolution invisibility the physical mode exists to prevent, and it is
    exactly what the strategy shipped between #512 and #514."""
    with pytest.raises(ValidationError, match="voxel_mm \\(the STORED grid\\)"):
        _reg(shift_source="recovered", effective_voxel_mm=(1.6, 1.6))


def test_acquisition_finer_than_its_own_grid_is_rejected() -> None:
    """A scanner cannot resolve more than the grid the volume is stored on; the
    two arguments are almost certainly swapped."""
    with pytest.raises(ValidationError, match="FINER than"):
        _reg(
            shift_source="recovered",
            voxel_mm=(1.6, 1.6),
            effective_voxel_mm=(0.49, 0.49),
        )


def test_grid_and_effective_must_share_an_axis_count() -> None:
    with pytest.raises(ValidationError, match="same number of axes"):
        _reg(
            shift_source="recovered",
            voxel_mm=(0.49, 0.49),
            effective_voxel_mm=(1.6, 1.6, 5.0),
        )


@pytest.mark.parametrize(
    ("contrast", "grid", "effective", "sr_scale"),
    [
        ("T1w", (0.49, 0.49), (1.6, 1.6), 3),
        ("T2w", (0.22, 0.22), (1.6, 1.6), 7),
        ("FLAIR", (0.43, 0.43), (1.7, 1.7), 4),
    ],
)
def test_sr_scale_must_match_the_measured_resolution_gap(
    contrast: str, grid: tuple, effective: tuple, sr_scale: int
) -> None:
    """The gaps are read from the real ulf_paired manifest and reproduce the
    3.3x / 7.3x / 4.0x in-plane SR factors. subvoxel_sr pools by sr_scale and
    inverts THAT, so an arm pooling 2x while declaring a 3.3x gap would quote
    every super-Nyquist number against a Nyquist it never simulated."""
    cfg = MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        sr_scale=sr_scale,
        subvoxel_registration={
            "shift_source": "recovered",
            "voxel_mm": grid,
            "effective_voxel_mm": effective,
        },
    )
    assert cfg.sr_scale == sr_scale
    with pytest.raises(ValidationError, match="does not match the declared"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            sr_scale=sr_scale + 1,
            subvoxel_registration={
                "shift_source": "recovered",
                "voxel_mm": grid,
                "effective_voxel_mm": effective,
            },
        )


# ---------------------------------------------------------------------------
# Super-Nyquist band probe (PR-1)
# ---------------------------------------------------------------------------
def _probe(method: str = "subvoxel_sr", shift_source: str = "recovered", **kw):
    return MultiAcquisitionConfig(
        enabled=True,
        method=method,
        subvoxel_registration={"shift_source": shift_source},
        band_probe=kw,
    ).band_probe


def test_band_probe_is_off_by_default() -> None:
    """Pre-2026-07-26 arms must be behaviourally unchanged."""
    p = MultiAcquisitionConfig(enabled=True, method="afi").band_probe
    assert p.enabled is False and p.lambda_band == 0.0


def test_report_only_control_arm_is_valid() -> None:
    """enabled + lambda_band=0 is the control: same measurement, no constraint,
    so the two arms differ on exactly one knob (#17)."""
    p = _probe(enabled=True, lambda_band=0.0)
    assert p.enabled and p.lambda_band == 0.0


def test_weighted_but_disabled_probe_raises() -> None:
    """A weighted term that is never computed is a reported number with no
    mechanism behind it."""
    with pytest.raises(ValidationError, match=r"never be computed|never computed"):
        _probe(enabled=False, lambda_band=0.1)


def test_only_the_instrument_term_requires_a_marker() -> None:
    """The band PARTITION is defined by the decimation, not by the fiducial.

    Until 2026-07-26 ``enabled`` itself demanded ``shift_source='recovered'``,
    which would have denied the anatomy high-frequency term to the blind and
    oracle rungs of the shift ladder and confounded it with an objective
    difference. Only ``lambda_band`` — the term that literally pushes the
    fiducial through the network — needs an instrument.
    """
    for source in ("blind", "oracle"):
        # the partition alone is fine, and carries the anatomy term
        assert _probe(shift_source=source, enabled=True, lambda_anatomy=0.5).enabled
        with pytest.raises(ValidationError, match="shift_source='recovered'"):
            _probe(shift_source=source, enabled=True, lambda_band=0.1)
    # on the recovered rung both terms are available
    p = _probe(shift_source="recovered", enabled=True, lambda_band=0.1, lambda_anatomy=0.5)
    assert p.lambda_band == 0.1 and p.lambda_anatomy == 0.5


def test_band_probe_rejects_methods_with_no_decimation() -> None:
    with pytest.raises(ValidationError, match="requires method='subvoxel_sr'"):
        MultiAcquisitionConfig(
            enabled=True, method="afi", band_probe={"enabled": True}
        )


def test_band_probe_needs_a_band_on_each_side_of_nyquist() -> None:
    for bad in ({"n_sub_bands": 0}, {"n_super_bands": 0}, {"rho_max": 1.0}):
        with pytest.raises(ValidationError):
            _probe(enabled=True, **bad)


# ---------------------------------------------------------------------------
# Fiducial channels fed to the MODEL (PR-2)
# ---------------------------------------------------------------------------
def test_marker_channels_off_by_default() -> None:
    """Pre-2026-07-26 arms must be behaviourally unchanged."""
    assert MultiAcquisitionConfig(enabled=True, method="afi").marker_channels is False


def test_marker_channels_refuse_the_oracle_rung() -> None:
    """Under 'oracle' the true offsets are handed over, so the marker cannot be
    the registration source the arm would claim."""
    with pytest.raises(ValidationError, match="'recovered' or 'blind'"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            marker_channels=True,
            subvoxel_registration={"shift_source": "oracle"},
        )


def test_marker_channels_accepted_on_the_blind_rung() -> None:
    """VFU-08: blind + marker_channels makes the marker the only registration
    source, which is what the content-vs-marker keying pair has to test."""
    cfg = MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        marker_channels=True,
        subvoxel_registration={"shift_source": "blind"},
    )
    assert cfg.marker_channels is True


def test_blind_marker_channels_still_need_a_jittered_marker() -> None:
    with pytest.raises(ValidationError, match="marker_jitter > 0"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            marker_channels=True,
            subvoxel_registration={"shift_source": "blind", "marker_jitter": 0.0},
        )


def test_marker_channels_reject_methods_with_no_fiducial() -> None:
    with pytest.raises(ValidationError, match="requires method='subvoxel_sr'"):
        MultiAcquisitionConfig(enabled=True, method="afi", marker_channels=True)


def test_marker_channels_accepted_on_the_recovered_rung() -> None:
    cfg = MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        marker_channels=True,
        subvoxel_registration={"shift_source": "recovered"},
    )
    assert cfg.marker_channels is True


# ---------------------------------------------------------------------------
# Relaxometric fiducial calibration (PR-3)
# ---------------------------------------------------------------------------
def _relax(**kw):
    from spectramr.config.schemas.physics import RelaxometricCalibrationConfig

    return RelaxometricCalibrationConfig(**kw)


_ACQ = {"tr_ms": 500.0, "te_ms": 15.0, "flip_deg": 90.0}


def test_relaxometric_calibration_off_by_default() -> None:
    r = MultiAcquisitionConfig(enabled=True, method="afi").relaxometric_calibration
    assert r.enabled is False and r.factored is False


def test_enabled_requires_both_acquisitions() -> None:
    """kappa is a RATIO between two declared acquisitions; with one missing the
    factored form would silently become the identity."""
    with pytest.raises(ValidationError, match="requires BOTH"):
        _relax(enabled=True, source={"field_strength_t": 0.064, **_ACQ})


def test_same_field_on_both_sides_is_rejected() -> None:
    """Then T1 is identical on both sides and kappa collapses to a pure
    sequence ratio — not a field translation, and the mechanism is inert."""
    with pytest.raises(ValidationError, match="inert"):
        _relax(
            enabled=True,
            source={"field_strength_t": 3.0, **_ACQ},
            target={"field_strength_t": 3.0, **_ACQ},
        )


def test_factored_without_enabled_is_rejected() -> None:
    """The model would be scaled by a kappa the simulator never applied."""
    with pytest.raises(ValidationError, match="never applied"):
        _relax(factored=True)


def test_relaxometric_calibration_requires_a_fiducial() -> None:
    """The predicted gain is only checkable against a measured one on marker
    support, and there is no marker on the other rungs."""
    body = {
        "enabled": True,
        "source": {"field_strength_t": 0.064, **_ACQ},
        "target": {"field_strength_t": 3.0, **_ACQ},
    }
    with pytest.raises(ValidationError, match="shift_source='recovered'"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            subvoxel_registration={"shift_source": "blind"},
            relaxometric_calibration=body,
        )
    with pytest.raises(ValidationError, match="requires method='subvoxel_sr'"):
        MultiAcquisitionConfig(
            enabled=True, method="afi", relaxometric_calibration=body
        )


# ---------------------------------------------------------------------------
# Anchor-calibrated conformal prediction (PR-4)
# ---------------------------------------------------------------------------
def test_anchor_conformal_off_by_default() -> None:
    assert (
        MultiAcquisitionConfig(enabled=True, method="afi").anchor_conformal.enabled
        is False
    )


def test_anchor_conformal_requires_a_fiducial() -> None:
    """The calibration set IS the marker residual set; without a marker there
    are no residuals whose truth is known."""
    with pytest.raises(ValidationError, match="only the fiducial supplies"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            subvoxel_registration={"shift_source": "blind"},
            anchor_conformal={"enabled": True},
        )


def test_anchor_conformal_rejects_impossible_levels() -> None:
    from spectramr.config.schemas.physics import AnchorConformalConfig

    for bad in ({"alpha": 0.0}, {"alpha": 1.0}, {"n_strata": 0}, {"tolerance": -0.1}):
        with pytest.raises(ValidationError):
            AnchorConformalConfig(**bad)


# ---------------------------------------------------------------------------
# Fiducial-measured forward operator (PR-5)
# ---------------------------------------------------------------------------
def test_forward_psf_off_by_default() -> None:
    assert (
        MultiAcquisitionConfig(enabled=True, method="afi").forward_psf.enabled is False
    )


def test_forward_psf_requires_a_fiducial() -> None:
    """Solving for a kernel from a blurred image alone is a blind
    deconvolution; the fiducial is what makes it a linear solve."""
    with pytest.raises(ValidationError, match="blind deconvolution"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            subvoxel_registration={"shift_source": "blind"},
            forward_psf={"enabled": True},
        )


def test_measured_psf_without_a_model_error_is_rejected() -> None:
    """With true_sigma_px unset the simulator applies exactly the assumed blur,
    so measured and assumed agree by construction and the arm distinguishes
    nothing."""
    with pytest.raises(ValidationError, match="null control"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            subvoxel_registration={"shift_source": "recovered"},
            forward_psf={"enabled": True, "source": "measured"},
        )


def test_forward_psf_rejects_an_even_kernel_and_zero_regularisation() -> None:
    from spectramr.config.schemas.physics import ForwardPsfConfig

    with pytest.raises(ValidationError, match="odd"):
        ForwardPsfConfig(kernel_size=8)
    with pytest.raises(ValidationError):
        ForwardPsfConfig(mu=0.0)

class TestDegradationSchedules:
    """physics.digital_twin severity ramp -- undersampling's schedule, on severity."""

    def test_defaults_reproduce_the_pre_existing_linear_remap(self):
        from spectramr.config.schemas.enums import AccelerationSchedule
        from spectramr.config.schemas.physics import DigitalTwinConfig

        d = DigitalTwinConfig()
        assert d.degradation_schedule is AccelerationSchedule.LINEAR
        assert d.degradation_schedules == {}
        assert d.degradation_schedule_power == 2.0

    def test_per_axis_override_accepts_a_known_axis(self):
        from spectramr.config.schemas.physics import DigitalTwinConfig

        d = DigitalTwinConfig(
            progressive_degradations=["motion", "rigid_motion"],
            degradation_schedules={"motion": "power_law", "rigid_motion": "step"},
        )
        assert d.degradation_schedules["motion"].value == "power_law"

    def test_unknown_axis_raises(self):
        """Validated against known_degradation_axes() -- the 14 native plus the 32
        registry operators -- rather than a hand-written set, so a newly registered
        degradation is legal the day it lands."""
        from spectramr.config.schemas.physics import DigitalTwinConfig

        with pytest.raises(ValueError, match="unknown axes"):
            DigitalTwinConfig(degradation_schedules={"b_0": "linear"})

    def test_scheduling_a_static_axis_raises(self):
        """An axis absent from a non-empty progressive_degradations is held at
        effective cf 1.0, so its ramp shape is never consulted -- the schedule
        would be an unread knob by construction (pitfall #15)."""
        from spectramr.config.schemas.physics import DigitalTwinConfig

        with pytest.raises(ValueError, match="can never fire"):
            DigitalTwinConfig(
                progressive_degradations=["noise"],
                degradation_schedules={"motion": "power_law"},
            )

    def test_an_empty_progressive_list_means_every_axis_ramps(self):
        """Empty is the documented 'everything is progressive' case, so a schedule
        on any known axis is reachable and must not be rejected."""
        from spectramr.config.schemas.physics import DigitalTwinConfig

        assert DigitalTwinConfig(degradation_schedules={"motion": "step"})


class TestDigitalTwinAcquisition:
    """``physics.digital_twin.acquisition``: the scanner profile the twin simulates."""

    _CUSTOM: typing.ClassVar[dict] = {"readout_bw_hz_per_px": 200.0, "sequence": "spin_echo"}

    def test_absent_by_default(self):
        assert DigitalTwinConfig().acquisition is None

    def test_frozen_and_extra_forbid(self):
        from spectramr.config.schemas.physics import DigitalTwinAcquisitionConfig

        acq = DigitalTwinAcquisitionConfig(**self._CUSTOM)
        with pytest.raises(ValidationError):
            acq.readout_bw_hz_per_px = 400.0
        with pytest.raises(ValidationError, match="extra"):
            DigitalTwinAcquisitionConfig(**self._CUSTOM, slice_thicknes_mm=3.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("mode", "volume_4d"),
            ("slice_order", "random"),
            ("line_order", "spiral"),
            ("sequence", "flash"),
        ],
    )
    def test_unknown_enum_value_raises_at_load(self, field, value):
        with pytest.raises(ValidationError, match=field):
            DigitalTwinConfig(acquisition={**self._CUSTOM, field: value})

    def test_enum_vocabularies_equal_the_physics_enums(self):
        """The schema cannot import infrastructure, so its Literals are checked here."""
        from spectramr.config.schemas.physics import DigitalTwinAcquisitionConfig
        from spectramr.infrastructure.physics.degradations.geometry import (
            AcquisitionMode,
            LineOrder,
            SliceOrder,
        )
        from spectramr.infrastructure.physics.scanner_profile import Sequence

        fields = DigitalTwinAcquisitionConfig.model_fields
        for name, enum in [
            ("mode", AcquisitionMode),
            ("slice_order", SliceOrder),
            ("line_order", LineOrder),
            ("sequence", Sequence),
        ]:
            literal = next(a for a in typing.get_args(fields[name].annotation) if a is not None)
            assert set(typing.get_args(literal)) == {m.value for m in enum}, name

    @pytest.mark.parametrize("missing", ["readout_bw_hz_per_px", "sequence"])
    def test_without_a_preset_bandwidth_and_sequence_are_required(self, missing):
        block = {k: v for k, v in self._CUSTOM.items() if k != missing}
        with pytest.raises(ValidationError, match=missing):
            DigitalTwinConfig(acquisition=block)

    def test_a_preset_supplies_the_required_fields(self):
        dt = DigitalTwinConfig(acquisition={"scanner_profile": "fastmri_3T"})
        assert dt.acquisition.readout_bw_hz_per_px is None

    def test_unknown_preset_raises(self):
        with pytest.raises(ValidationError, match="scanner_profile"):
            DigitalTwinConfig(acquisition={"scanner_profile": "siemens_7T"})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("voxel_mm", (1.0, 0.0, 1.0)),
            ("slice_thickness_mm", 0.0),
            ("slice_gap_mm", -1.0),
            ("readout_bw_hz_per_px", 0.0),
        ],
    )
    def test_non_physical_numbers_raise(self, field, value):
        with pytest.raises(ValidationError, match=field):
            DigitalTwinConfig(acquisition={**self._CUSTOM, field: value})

    @pytest.mark.parametrize(
        ("field", "value", "reading_modes"),
        [
            ("voxel_mm", (3.0, 1.0, 1.0), {"slice_2d", "multislice_2d", "volume_3d"}),
            ("slice_thickness_mm", 1.0, {"multislice_2d"}),
            ("slice_order", "sequential", {"multislice_2d"}),
            ("line_order", "kz_inner", {"volume_3d"}),
        ],
    )
    @pytest.mark.parametrize("mode", [None, "slice_2d", "multislice_2d", "volume_3d"])
    def test_a_geometry_knob_loads_only_in_a_mode_that_reads_it(
        self, field, value, reading_modes, mode
    ):
        """Ruling 48: a knob no operator reads in the declared mode is refused, not inert."""
        block = {**self._CUSTOM, "mode": mode, "voxel_mm": (3.0, 1.0, 1.0), field: value}
        if (mode or "slice_2d") in reading_modes:
            assert getattr(DigitalTwinConfig(acquisition=block).acquisition, field) == value
        else:
            with pytest.raises(ValidationError, match=field):
                DigitalTwinConfig(acquisition=block)

    @pytest.mark.parametrize("mode", [None, "slice_2d", "multislice_2d", "volume_3d"])
    def test_slice_gap_is_refused_in_every_mode(self, mode):
        """No operator reads the gap: the stack spacing is voxel_mm[0]."""
        block = {**self._CUSTOM, "mode": mode, "voxel_mm": (4.0, 1.0, 1.0), "slice_gap_mm": 1.0}
        with pytest.raises(ValidationError, match="slice_gap_mm"):
            DigitalTwinConfig(acquisition=block)

    def test_slice_thickness_without_a_voxel_size_is_refused(self):
        """The through-slice factor reads thickness / voxel_mm[0], so thickness alone is inert."""
        block = {**self._CUSTOM, "mode": "multislice_2d", "slice_thickness_mm": 3.0}
        with pytest.raises(ValidationError, match="voxel_mm"):
            DigitalTwinConfig(acquisition=block)

    def test_slice_stack_fields_load_in_multislice_mode(self):
        block = {
            **self._CUSTOM,
            "mode": "multislice_2d",
            "voxel_mm": (4.0, 1.0, 1.0),
            "slice_thickness_mm": 3.0,
        }
        assert DigitalTwinConfig(acquisition=block).acquisition.slice_thickness_mm == 3.0

    def test_preset_field_strength_conflict_raises(self):
        """physics.field_strength is the one B0 owner; a preset at another field contradicts it."""
        with pytest.raises(ValidationError, match="field_strength"):
            PhysicsConfigSchema(
                field_strength=3.0,
                digital_twin={"acquisition": {"scanner_profile": "m4raw_0.3T"}},
            )

    def test_preset_at_the_declared_field_loads(self):
        physics = PhysicsConfigSchema(
            field_strength=0.3,
            digital_twin={"acquisition": {"scanner_profile": "m4raw_0.3T"}},
        )
        assert physics.digital_twin.acquisition.scanner_profile == "m4raw_0.3T"

    def test_preset_field_table_equals_the_physics_presets(self):
        from spectramr.config.schemas.physics import SCANNER_PRESET_FIELD_STRENGTH_T
        from spectramr.infrastructure.physics.scanner_presets import PRESETS

        assert {n: p.b0_T for n, p in PRESETS.items()} == SCANNER_PRESET_FIELD_STRENGTH_T


def test_subvoxel_input_channels_counts_every_block() -> None:
    """One owner of the subvoxel_sr input width (strategy setup + health check)."""
    from spectramr.config.schemas.physics import subvoxel_input_channels

    base = {"enabled": True, "method": "subvoxel_sr", "n_frames": 8}
    rec = {"shift_source": "recovered"}
    assert subvoxel_input_channels(MultiAcquisitionConfig(**base)) == 24
    marker = MultiAcquisitionConfig(**base, marker_channels=True, subvoxel_registration=rec)
    assert subvoxel_input_channels(marker) == 32
    psf = MultiAcquisitionConfig(
        **base, subvoxel_registration=rec, forward_psf={"enabled": True, "true_sigma_px": 2.4}
    )
    assert subvoxel_input_channels(psf) == 25


def test_tissue_map_factor_form_requires_a_factored_arm() -> None:
    """VFU-05: an unfactored arm applies no kappa, so a kappa shape is inert there."""
    acq = {"tr_ms": 500.0, "te_ms": 15.0, "flip_deg": 90.0}
    block = {
        "enabled": True,
        "source": {"field_strength_t": 0.064, **acq},
        "target": {"field_strength_t": 3.0, **acq},
        "factor_form": "tissue_map",
    }
    with pytest.raises(ValidationError, match="requires factored=True"):
        _relax(**block)
    assert _relax(**block, factored=True).factor_form == "tissue_map"
    with pytest.raises(ValidationError):
        _relax(**{**block, "factor_form": "per_voxel"}, factored=True)


def test_per_sample_markers_need_a_synthesised_fiducial() -> None:
    """VFU-02: with no marker there is nothing to realise per sample."""
    with pytest.raises(ValidationError, match="needs a synthesised fiducial"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            subvoxel_registration={"shift_source": "blind", "marker_realisation": "per_sample"},
        )
    cfg = MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        subvoxel_registration={"shift_source": "recovered", "marker_realisation": "per_sample"},
    )
    assert cfg.subvoxel_registration.marker_realisation == "per_sample"


def test_mismatched_probe_target_needs_the_instrument_term() -> None:
    """The wrong target only enters lambda_band; at 0 the control would be inert."""
    rec = {"shift_source": "recovered"}
    with pytest.raises(ValidationError, match="requires lambda_band > 0"):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            subvoxel_registration=rec,
            band_probe={"enabled": True, "target_realisation": "mismatched"},
        )
    MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        subvoxel_registration=rec,
        band_probe={"enabled": True, "lambda_band": 0.1, "target_realisation": "mismatched"},
    )


def test_heldout_subject_calibration_needs_no_fiducial() -> None:
    """VFU-09: anatomy residuals of held-out subjects calibrate; the marker is not read."""
    cfg = MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        anchor_conformal={"enabled": True, "calibration_source": "held_out_subjects"},
    )
    assert cfg.anchor_conformal.calibration_subject_count == 1
    with pytest.raises(ValidationError, match="shift_source='recovered'"):
        MultiAcquisitionConfig(
            enabled=True, method="subvoxel_sr", anchor_conformal={"enabled": True}
        )
    with pytest.raises(ValidationError):
        MultiAcquisitionConfig(
            enabled=True,
            method="subvoxel_sr",
            anchor_conformal={"enabled": True, "calibration_source": "validation"},
        )


def test_acquired_input_stream_is_one_blind_unmarked_frame() -> None:
    """VFU-07: the real 64 mT scan has one frame, no dither and no fiducial."""
    ok = MultiAcquisitionConfig(
        enabled=True, method="subvoxel_sr", n_frames=1, input_stream="acquired",
        fov_mask="acquired_support",
    )
    assert ok.input_stream == "acquired"
    for bad in (
        {"n_frames": 8},
        {"subvoxel_registration": {"shift_source": "recovered"}},
        {"fov_mask": "none"},
        {"band_probe": {"enabled": True, "lambda_band": 0.1}},
    ):
        kw = {"n_frames": 1, "fov_mask": "acquired_support", **bad}
        with pytest.raises(ValidationError, match="input_stream='acquired' is inconsistent"):
            MultiAcquisitionConfig(enabled=True, method="subvoxel_sr", input_stream="acquired", **kw)
