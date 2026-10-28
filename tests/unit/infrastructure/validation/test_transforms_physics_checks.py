"""Unit tests for Phase 8 — transforms + physics contracts.

Covers:
- check_sfc_scan_mode_matches_spatial_dims
- check_data_consistency_requires_kspace
- check_acceleration_consistency
- recommend_sense_loss_needs_sense_maps
- recommend_bloch_acquisition_params
- recommend_dc_without_acceleration
"""

from __future__ import annotations

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
from spectramr.infrastructure.validation.inference import (
    derive_tags,
    recommend_bloch_acquisition_params,
    recommend_dc_without_acceleration,
    recommend_sense_loss_needs_sense_maps,
)


class _NS:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _cfg(
    *,
    patch_size=(64, 64, 1),
    dataset_type="kspace",
    scan_mode=None,
    spatial_shape=None,
    dc_enabled=False,
    accel_base=None,
    accel_max=None,
    center_fraction=None,
    image_losses=(),
    kspace_losses=(),
    multi_contrast_enabled=False,
    n_contrasts=None,
    acquisition_params=None,
    model_type="cs_mno_operator",
    coil_processing_mode="rss_image",
    load_coil_sensitivity=False,
    coil_enable_estimation=False,
):
    mk = {}
    if scan_mode is not None:
        mk["scan_mode"] = scan_mode
    if spatial_shape is not None:
        mk["spatial_shape"] = spatial_shape
    mc = _NS(
        enabled=multi_contrast_enabled,
        n_contrasts=n_contrasts,
        acquisition_params=acquisition_params,
    )
    data = _NS(
        dataset_type=dataset_type,
        patch_size=patch_size,
        coil_processing_mode=coil_processing_mode,
        target_channels=1,
        in_channels=1,
        load_coil_sensitivity=load_coil_sensitivity,
        multi_contrast=mc,
    )
    model = _NS(
        model_type=model_type,
        in_channels=1,
        out_channels=1,
        spatial_dims=len([d for d in patch_size if d > 1]),
        input_type="image",
        model_kwargs=mk,
    )
    losses = _NS(
        output_domain="image",
        image_losses=[_NS(name=n, weight=1.0, enabled=True) for n in image_losses],
        kspace_losses=[_NS(name=n, weight=1.0, enabled=True) for n in kspace_losses],
        complex_losses=[],
    )
    physics = _NS(
        data_consistency=_NS(enabled=dc_enabled),
        kspace=_NS(enable_kspace_recon=False),
        coil_sensitivity=_NS(enable_estimation=coil_enable_estimation, precomputed_path=None),
    )
    accel = None
    if accel_base is not None or accel_max is not None or center_fraction is not None:
        accel = _NS(
            base_acceleration=accel_base,
            max_acceleration=accel_max,
            center_fraction=center_fraction,
        )
    training = _NS(
        training_mode="reconstruction",
        strategy_class="spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        task="reconstruction",
    )
    logging_cfg = _NS(save_validation_images=True, log_validation_images=True)
    return _NS(
        data=data,
        model=model,
        losses=losses,
        training=training,
        # Phase 11 renamed the top-level block `acceleration:` -> `undersampling:`.
        # The checks were migrated; this stand-in was not, so every one of them
        # reported "No acceleration section." and PASSED vacuously.
        logging=logging_cfg,
        physics=physics,
        undersampling=accel,
        adapters=None,
        metadata=None,
    )


# ---------------------------------------------------------------------------
# check_sfc_scan_mode_matches_spatial_dims
# ---------------------------------------------------------------------------


class TestSfcScanModeCheck:
    def test_hilbert_2d_with_2d_data_passes(self):
        cfg = _cfg(scan_mode="hilbert_2d", patch_size=(64, 64, 1))
        r = ConfigHealthChecker().check_sfc_scan_mode_matches_spatial_dims(cfg)
        assert r.passed and r.severity == "info"

    def test_hilbert_2d_with_1d_data_errors(self):
        cfg = _cfg(scan_mode="hilbert_2d", patch_size=(128, 1, 1))
        r = ConfigHealthChecker().check_sfc_scan_mode_matches_spatial_dims(cfg)
        assert not r.passed
        assert r.severity == "error"
        assert "rank 2" in r.message
        assert "rank 1" in r.message

    def test_raster_1d_with_1d_data_passes(self):
        cfg = _cfg(scan_mode="raster_1d", spatial_shape=[128])
        r = ConfigHealthChecker().check_sfc_scan_mode_matches_spatial_dims(cfg)
        assert r.passed

    def test_no_scan_mode_skipped(self):
        cfg = _cfg(scan_mode=None)
        r = ConfigHealthChecker().check_sfc_scan_mode_matches_spatial_dims(cfg)
        assert r.passed

    def test_spatial_shape_takes_precedence(self):
        cfg = _cfg(scan_mode="hilbert_3d", spatial_shape=[32, 32, 32], patch_size=(64, 64, 1))
        r = ConfigHealthChecker().check_sfc_scan_mode_matches_spatial_dims(cfg)
        assert r.passed


# ---------------------------------------------------------------------------
# check_data_consistency_requires_kspace
# ---------------------------------------------------------------------------


class TestDataConsistencyKspace:
    def test_dc_disabled_skipped(self):
        cfg = _cfg(dc_enabled=False)
        r = ConfigHealthChecker().check_data_consistency_requires_kspace(cfg)
        assert r.passed

    def test_dc_with_kspace_data_passes(self):
        cfg = _cfg(dc_enabled=True, dataset_type="kspace", coil_processing_mode="svd")
        r = ConfigHealthChecker().check_data_consistency_requires_kspace(cfg)
        assert r.passed

    def test_dc_with_image_data_passes(self):
        # image data is OK because DC can FFT it internally.
        cfg = _cfg(dc_enabled=True, dataset_type="nifti", coil_processing_mode="rss_image")
        r = ConfigHealthChecker().check_data_consistency_requires_kspace(cfg)
        assert r.passed

    def test_dc_with_pde_data_errors(self):
        cfg = _cfg(dc_enabled=True, dataset_type="pde_synthetic", coil_processing_mode="none")
        r = ConfigHealthChecker().check_data_consistency_requires_kspace(cfg)
        assert not r.passed and r.severity == "error"


# ---------------------------------------------------------------------------
# check_acceleration_consistency
# ---------------------------------------------------------------------------


class TestAccelerationConsistency:
    def test_no_acceleration_skipped(self):
        cfg = _cfg()
        r = ConfigHealthChecker().check_acceleration_consistency(cfg)
        assert r.passed

    def test_consistent_passes(self):
        cfg = _cfg(accel_base=2.0, accel_max=8.0, center_fraction=0.08)
        r = ConfigHealthChecker().check_acceleration_consistency(cfg)
        assert r.passed

    def test_base_greater_than_max_errors(self):
        cfg = _cfg(accel_base=8.0, accel_max=4.0, center_fraction=0.08)
        r = ConfigHealthChecker().check_acceleration_consistency(cfg)
        assert not r.passed and "empty schedule" in r.message

    def test_center_fraction_too_small_errors(self):
        cfg = _cfg(accel_base=2.0, accel_max=32.0, center_fraction=0.008)
        r = ConfigHealthChecker().check_acceleration_consistency(cfg)
        assert not r.passed and "center_fraction" in r.message

    def test_center_fraction_too_large_errors(self):
        cfg = _cfg(accel_base=2.0, accel_max=4.0, center_fraction=0.7)
        r = ConfigHealthChecker().check_acceleration_consistency(cfg)
        assert not r.passed


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestPhase8Tags:
    def test_scan_mode_tag(self):
        cfg = _cfg(scan_mode="hilbert_2d")
        names = {t.name: t.value for t in derive_tags(cfg)}
        assert names.get("scan_mode") == "hilbert_2d"

    def test_bloch_simulator_tag(self):
        cfg = _cfg(model_type="bloch_cycle_network")
        names = {t.name for t in derive_tags(cfg)}
        assert "has_bloch_simulator" in names

    def test_requires_sense_maps_tag(self):
        cfg = _cfg(image_losses=("sense_adjoint_l1",))
        names = {t.name for t in derive_tags(cfg)}
        assert "requires_sense_maps" in names

    def test_acceleration_tag_format(self):
        cfg = _cfg(accel_base=2.0, accel_max=8.0, center_fraction=0.08)
        names = {t.name: t.value for t in derive_tags(cfg)}
        assert names.get("acceleration") == "8x"


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestSenseLossRecommendation:
    def test_r020_fires_when_no_source(self):
        cfg = _cfg(image_losses=("sense_adjoint_l1",), load_coil_sensitivity=False)
        recs = recommend_sense_loss_needs_sense_maps(cfg)
        assert any(r.code == "AUDIT-R020" for r in recs)

    def test_r020_suggests_only_keys_it_names_and_the_schema_accepts(self):
        """#872: the pasted block taught `coil_sensitivity.estimation_method`, which
        nothing reads, while `yaml_keys` already named the live key."""
        import yaml

        from spectramr.config.schemas.physics import PhysicsConfigSchema

        cfg = _cfg(image_losses=("sense_adjoint_l1",), load_coil_sensitivity=False)
        (rec,) = [r for r in recommend_sense_loss_needs_sense_maps(cfg) if r.code == "AUDIT-R020"]
        doc = yaml.safe_load(rec.suggested_yaml)

        def leaves(node, prefix=""):
            for key, value in node.items():
                path = f"{prefix}{key}"
                yield from leaves(value, path + ".") if isinstance(value, dict) else (path,)

        suggested = set(leaves(doc))
        assert "physics.coil_sensitivity.estimation_method" not in suggested
        assert suggested <= set(rec.yaml_keys), suggested - set(rec.yaml_keys)
        physics = PhysicsConfigSchema(**doc["physics"])
        assert physics.coil_processing.estimation.method == "espirit"

    def test_r020_silent_when_source_declared(self):
        cfg = _cfg(image_losses=("sense_adjoint_l1",), load_coil_sensitivity=True)
        assert recommend_sense_loss_needs_sense_maps(cfg) == []

    def test_r020_silent_with_coil_estimation_enabled(self):
        # REGRESSION (2026-06-20): R020 checked the stale key
        # physics.coil_sensitivity.enabled, but the real field is enable_estimation,
        # so it false-fired on every cohort arm that enabled ESPIRiT/power-iter coil
        # estimation (46 arms). enable_estimation=True IS a coil-sensitivity source.
        cfg = _cfg(image_losses=("sense_adjoint_l1",), coil_enable_estimation=True)
        assert recommend_sense_loss_needs_sense_maps(cfg) == []


class TestBlochAcquisitionParams:
    def test_r021_fires_when_bloch_no_acq_params(self):
        cfg = _cfg(
            model_type="bloch_cycle_network",
            multi_contrast_enabled=True,
            n_contrasts=4,
            acquisition_params=None,
        )
        recs = recommend_bloch_acquisition_params(cfg)
        assert any(r.code == "AUDIT-R021" for r in recs)

    def test_r021_fires_when_partial_acq_params(self):
        cfg = _cfg(
            model_type="bloch_cycle_network",
            multi_contrast_enabled=True,
            n_contrasts=2,
            acquisition_params=[{"name": "T1w", "TR": 600.0}],  # missing TE
        )
        recs = recommend_bloch_acquisition_params(cfg)
        assert any(r.code == "AUDIT-R021" for r in recs)

    def test_r021_silent_when_complete(self):
        cfg = _cfg(
            model_type="bloch_cycle_network",
            multi_contrast_enabled=True,
            n_contrasts=2,
            acquisition_params=[
                {"name": "T1w", "TR": 600, "TE": 12, "FA": 70},
                {"name": "T2w", "TR": 4000, "TE": 80, "FA": 90},
            ],
        )
        assert recommend_bloch_acquisition_params(cfg) == []

    def test_r021_silent_for_non_bloch_model(self):
        cfg = _cfg(model_type="standard_unet", multi_contrast_enabled=True)
        assert recommend_bloch_acquisition_params(cfg) == []


class TestDcWithoutAcceleration:
    def test_r023_fires_when_dc_no_accel(self):
        cfg = _cfg(dc_enabled=True)  # no acceleration block
        recs = recommend_dc_without_acceleration(cfg)
        assert any(r.code == "AUDIT-R023" for r in recs)

    def test_r023_silent_when_accel_present(self):
        cfg = _cfg(dc_enabled=True, accel_base=2.0, accel_max=4.0, center_fraction=0.08)
        assert recommend_dc_without_acceleration(cfg) == []

    def test_r023_silent_when_the_digital_twin_undersamples(self):
        """VF review 2026-09-03: the twin masks inside the step and hands its mask
        to the DC layers, so a twin arm has no top-level block to declare."""
        cfg = _cfg(dc_enabled=True)  # no acceleration block
        cfg.physics.digital_twin = _NS(enabled=True, enable_undersampling=True, acceleration=4.0)
        assert recommend_dc_without_acceleration(cfg) == []

    def test_r023_fires_when_the_twin_is_present_but_does_not_undersample(self):
        """Planted violation: eleven VF arms declared DC with a twin that never masks."""
        cfg = _cfg(dc_enabled=True)
        cfg.physics.digital_twin = _NS(enabled=True, enable_undersampling=False, acceleration=4.0)
        recs = recommend_dc_without_acceleration(cfg)
        assert any(r.code == "AUDIT-R023" for r in recs)

    def test_r023_silent_when_dc_disabled(self):
        cfg = _cfg(dc_enabled=False)
        assert recommend_dc_without_acceleration(cfg) == []
