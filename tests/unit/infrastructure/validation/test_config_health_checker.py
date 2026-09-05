"""Unit tests for ConfigHealthChecker.check_domain_alignment.

These tests pin the channel-mapping table that lives in
``ConfigHealthChecker.check_domain_alignment``. The checker mirrors the
runtime behavior of ``torchio_subject_builder._apply_coil_processing``
— if those two ever drift, this validator silently emits the wrong
expected count and the pre-flight DomainMismatch protection becomes a
false sense of security.

Tests use ``SimpleNamespace`` rather than a real ``TrainingSettings``:
the checker only reads a handful of attributes, and a real settings
object would force us to satisfy many unrelated schema requirements,
making the tests fragile to v6.0 schema evolution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
    HealthCheckReport,
    HealthCheckResult,
    validate_config_health,
)
from tests.utils.config_block_stub import block_stub
from tests.utils.data_config_stub import DataConfigStub
from tests.utils.repo_scripts import require_repo_file


def _make_config(
    *,
    coil_processing_mode: str = "rss",
    dataset_type: str = "fastmri_kspace",
    num_virtual_coils: int = 4,
    in_channels: int = 2,
    out_channels: int | None = 2,
    target_channels: int | None = None,
    model_type: str = "unet",
    output_domain: str | None = None,
    strategy_class: str | None = None,
) -> Any:
    """Build a minimal config-shaped object for the checker.

    The kwargs stay FLAT (that is how every caller below reads), but the object
    is assembled in the schema's real NESTED shape via the shared
    ``DataConfigStub``: the checker walks ``data.coils.*`` / ``data.domain.*`` /
    ``data.split.*`` / ``data.sampling.*``, and a stand-in still carrying those
    flat is a shape no config produces -- it would be testing the wrong
    question. Using the shared stub means a sub-block added by a later phase
    arrives here for free instead of as thirty ``no attribute`` failures.
    """
    data_kwargs: dict[str, Any] = {
        "dataset_type": dataset_type,
        "coil_processing_mode": coil_processing_mode,
        "num_virtual_coils": num_virtual_coils,
    }
    if target_channels is not None:
        data_kwargs["target_channels"] = target_channels

    losses = SimpleNamespace(output_domain=output_domain) if output_domain is not None else None

    return SimpleNamespace(
        data=DataConfigStub(**data_kwargs),
        model=SimpleNamespace(
            in_channels=in_channels,
            out_channels=out_channels,
            model_type=model_type,
        ),
        losses=losses,
        training=SimpleNamespace(strategy_class=strategy_class),
    )


def test_the_stand_in_mirrors_the_real_data_schema() -> None:
    """``_make_config``'s ``data`` stand-in must have the shape a real one has.

    A SimpleNamespace agrees with the schema only until someone moves a field.
    Phase 9a moved the coil knobs to ``data.coils.*`` and this file's stand-in
    kept them flat, so 32 checks were being exercised against a shape nothing
    produces -- they read a legacy path, got a ``getattr`` default, and asserted
    on the answer to the wrong question.

    Asserts the ATTRIBUTE PATHS the checker actually walks, not the whole
    schema: a stand-in is allowed to be minimal, it is not allowed to be a
    different shape.
    """
    from spectramr.config.settings import TrainingSettings
    from tests.unit.config.test_settings import _minimal_config

    real = TrainingSettings(**_minimal_config()).data
    stand_in = _make_config().data

    for path in ("coils.processing_mode", "coils.num_virtual_coils", "dataset_type"):
        node_real: Any = real
        node_fake: Any = stand_in
        for part in path.split("."):
            assert hasattr(node_real, part), f"real schema lost {path!r}"
            assert hasattr(node_fake, part), (
                f"stand-in is missing {path!r} -- the real schema has it, so "
                "every check reading that path is being tested against a shape "
                "no config produces"
            )
            node_real = getattr(node_real, part)
            node_fake = getattr(node_fake, part)


def _errors(results: list[HealthCheckResult]) -> list[HealthCheckResult]:
    return [r for r in results if not r.passed and r.severity == "error"]


def _warnings(results: list[HealthCheckResult]) -> list[HealthCheckResult]:
    return [r for r in results if not r.passed and r.severity == "warning"]


class TestPassedResultSeverityLabelling:
    """``report.warnings`` counts only ``not passed`` results, so a
    ``passed=True`` result labelled 'warning' is an invisible non-warning. The
    registry-unavailable capability skips therefore use 'info', not 'warning'."""

    def test_passed_true_warning_is_not_surfaced(self) -> None:
        report = HealthCheckReport()
        report.results.append(
            HealthCheckResult(
                passed=True,
                check_name="skip",
                message="registry unavailable",
                severity="warning",
            )
        )
        # The mislabelled "warning" never appears in report.warnings.
        assert report.warnings == []

    def test_info_labelled_skip_is_honest(self) -> None:
        report = HealthCheckReport()
        report.results.append(
            HealthCheckResult(
                passed=True,
                check_name="skip",
                message="registry unavailable",
                severity="info",
            )
        )
        # An info-labelled skip is also not a warning — but honestly so.
        assert report.warnings == []
        assert report.results[0].severity == "info"


class TestRSSMode:
    def test_in_channels_2_passes(self) -> None:
        config = _make_config(coil_processing_mode="rss", in_channels=2, out_channels=2)
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _errors(results) == []

    def test_in_channels_8_fails(self) -> None:
        # Common bug: developer leaves in_channels=8 from multi-coil run
        # but switches coil mode to 'rss' which collapses to 2 channels.
        config = _make_config(coil_processing_mode="rss", in_channels=8, out_channels=8)
        results = ConfigHealthChecker().check_domain_alignment(config)
        errs = _errors(results)
        assert len(errs) == 2  # in_channels and out_channels both mismatch
        assert any("in_channels=8" in e.message for e in errs)
        assert any("out_channels=8" in e.message for e in errs)


class TestMagnitudeMode:
    def test_in_channels_1_passes(self) -> None:
        config = _make_config(coil_processing_mode="magnitude", in_channels=1, out_channels=1)
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _errors(results) == []

    def test_in_channels_2_fails(self) -> None:
        config = _make_config(coil_processing_mode="magnitude", in_channels=2, out_channels=2)
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert len(_errors(results)) >= 1


class TestVFComplexStackingDoubling:
    """VF-family strategies (virtual_fiducial / motion_meta / vf_admm / ib_vf)
    real-stack a 1-channel complex magnitude into 2 real channels before the
    generator. For rss_image/magnitude (1 complex channel) the model-visible
    count is therefore 2 — the audit must expect 2, not 1, or it converts a
    runtime DomainMismatch crash into a (worse) silent audit skip. Guarded on
    ``expected_channels == 1`` so svd arms (already 2*N real) are unaffected.
    Reproduces the 2026-05-24 VF dispatch channel-mismatch failures."""

    VF_STRATEGY = (
        "spectramr.infrastructure.training.strategies."
        "virtual_fiducial_strategy.ConcreteVirtualFiducialStrategy"
    )
    MOTION_META_STRATEGY = (
        "spectramr.infrastructure.training.strategies."
        "motion_meta_strategy.ConcreteMotionMetaTrainingStrategy"
    )

    def test_rss_image_in2_out2_passes_under_vf_strategy(self) -> None:
        config = _make_config(
            coil_processing_mode="rss_image",
            in_channels=2,
            out_channels=2,
            target_channels=1,
            strategy_class=self.VF_STRATEGY,
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _errors(results) == []
        assert _warnings(results) == []

    def test_rss_image_in1_fails_under_vf_strategy(self) -> None:
        # The original bug: in_channels=1 audits clean naively but crashes at
        # the first forward (model got 2 channels). The doubling makes the
        # audit reject in=1 up front.
        config = _make_config(
            coil_processing_mode="rss_image",
            in_channels=1,
            out_channels=1,
            target_channels=1,
            strategy_class=self.VF_STRATEGY,
        )
        errs = _errors(ConfigHealthChecker().check_domain_alignment(config))
        assert any("in_channels=1" in e.message for e in errs)

    def test_motion_meta_strategy_also_doubles(self) -> None:
        config = _make_config(
            coil_processing_mode="rss_image",
            in_channels=2,
            out_channels=2,
            target_channels=1,
            strategy_class=self.MOTION_META_STRATEGY,
        )
        assert _errors(ConfigHealthChecker().check_domain_alignment(config)) == []

    def test_svd_vf_arm_not_doubled(self) -> None:
        # svd delivers 2*num_virtual_coils real channels already; the strategy
        # round-trips them (no net doubling), so in=8 stays correct and in=16
        # would be wrong.
        ok = _make_config(
            coil_processing_mode="svd",
            num_virtual_coils=4,
            in_channels=8,
            out_channels=8,
            target_channels=1,
            strategy_class=self.VF_STRATEGY,
        )
        assert _errors(ConfigHealthChecker().check_domain_alignment(ok)) == []
        doubled = _make_config(
            coil_processing_mode="svd",
            num_virtual_coils=4,
            in_channels=16,
            out_channels=16,
            strategy_class=self.VF_STRATEGY,
        )
        assert len(_errors(ConfigHealthChecker().check_domain_alignment(doubled))) >= 1

    def test_rss_image_in1_passes_without_vf_strategy(self) -> None:
        # A non-VF strategy (e.g. TTO) operates on the 1-ch magnitude directly,
        # so in=1 must remain valid — the doubling is strategy-scoped.
        config = _make_config(
            coil_processing_mode="rss_image",
            in_channels=1,
            out_channels=1,
            target_channels=1,
            strategy_class="spectramr.infrastructure.training.strategies."
            "tto_strategy.ConcreteTTOStrategy",
        )
        assert _errors(ConfigHealthChecker().check_domain_alignment(config)) == []

    # ── distillation_strategy regression (cluster smoke 20260605, job 7095209) ──
    # ConcreteDistillationStrategy._compute_losses_impl real-stacks the
    # twin-corrupted complex image (cat([real, imag])), so an rss_image 1-ch
    # magnitude becomes 2 model-visible channels — exactly like the VF family.
    # It was missing from _COMPLEX_STACKING_STRATEGY_MARKERS, so the audit
    # expected in=1 and rejected the correct in=2 for eval_c2/c3/c7 + exp_c4.
    DISTILLATION_STRATEGY = (
        "spectramr.infrastructure.training.strategies."
        "distillation_strategy.ConcreteDistillationStrategy"
    )

    def test_rss_image_in2_out2_passes_under_distillation(self) -> None:
        config = _make_config(
            coil_processing_mode="rss_image",
            in_channels=2,
            out_channels=2,
            target_channels=1,
            strategy_class=self.DISTILLATION_STRATEGY,
        )
        assert _errors(ConfigHealthChecker().check_domain_alignment(config)) == []

    def test_rss_image_in1_fails_under_distillation(self) -> None:
        config = _make_config(
            coil_processing_mode="rss_image",
            in_channels=1,
            out_channels=1,
            target_channels=1,
            strategy_class=self.DISTILLATION_STRATEGY,
        )
        errs = _errors(ConfigHealthChecker().check_domain_alignment(config))
        assert any("in_channels=1" in e.message for e in errs)


class TestSVDMode:
    def test_matching_virtual_coils_passes(self) -> None:
        config = _make_config(
            coil_processing_mode="svd",
            num_virtual_coils=4,
            in_channels=8,  # 2 * 4
            out_channels=8,
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _errors(results) == []

    def test_mismatched_virtual_coils_fails(self) -> None:
        config = _make_config(
            coil_processing_mode="svd",
            num_virtual_coils=8,
            in_channels=4,  # would expect 16
            out_channels=4,
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        errs = _errors(results)
        assert len(errs) >= 1
        assert any("expected=16" in e.message for e in errs)


class TestFlattenMode:
    def test_even_in_channels_passes_as_info(self) -> None:
        # Cannot derive expected count without knowing physical coil count,
        # but parity check (even) must pass.
        config = _make_config(coil_processing_mode="flatten", in_channels=8, out_channels=8)
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _errors(results) == []
        assert any(r.severity == "info" for r in results)

    def test_odd_in_channels_fails(self) -> None:
        config = _make_config(coil_processing_mode="flatten", in_channels=7, out_channels=7)
        results = ConfigHealthChecker().check_domain_alignment(config)
        errs = _errors(results)
        assert len(errs) == 1
        assert "odd" in errs[0].message


class TestNoneMode:
    def test_kspace_with_too_few_channels_fails(self) -> None:
        config = _make_config(
            coil_processing_mode="none",
            dataset_type="fastmri_kspace",
            in_channels=1,
            out_channels=1,
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        errs = _errors(results)
        assert len(errs) == 1
        assert "minimum 2" in errs[0].message

    def test_kspace_with_enough_channels_passes_as_info(self) -> None:
        config = _make_config(
            coil_processing_mode="none",
            dataset_type="fastmri_kspace",
            in_channels=8,
            out_channels=8,
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _errors(results) == []
        assert any(r.severity == "info" for r in results)


class TestSyntheticDataset:
    def test_skips_strict_check(self) -> None:
        # synthetic generators have varied channel semantics — skip check.
        config = _make_config(
            coil_processing_mode="rss",  # would normally force in_channels=2
            dataset_type="synthetic",
            in_channels=42,  # bogus value should NOT trigger an error
            out_channels=99,
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _errors(results) == []
        assert any(r.severity == "info" for r in results)


class TestModelTypeAllowlist:
    def test_evidential_unet_skips_out_channels_check(self) -> None:
        # evidential_unet emits 4 outputs per pixel (mean, v, alpha, beta)
        # so out_channels intentionally differs from in_channels.
        config = _make_config(
            coil_processing_mode="rss",
            in_channels=2,
            out_channels=8,  # 4 * 2 — would normally fail
            model_type="evidential_unet",
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _errors(results) == []

    def test_unknown_model_still_enforces_out_channels_match(self) -> None:
        config = _make_config(
            coil_processing_mode="rss",
            in_channels=2,
            out_channels=8,
            model_type="unet",  # not on allowlist
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert any("out_channels=8" in e.message and not e.passed for e in results)


class TestTargetChannelsCarveOut:
    def test_image_domain_magnitude_target_acceptable(self) -> None:
        # rss + image-domain loss + target=1 (magnitude) is the canonical
        # "complex prediction reduced to magnitude for the loss" pattern.
        config = _make_config(
            coil_processing_mode="rss",
            in_channels=2,
            out_channels=2,
            target_channels=1,
            output_domain="image",
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert _warnings(results) == []  # target_channels mismatch suppressed

    def test_kspace_domain_target_one_is_rss_reference_convention(self) -> None:
        # 2026-05-05: the carve-out was extended from output_domain='image'
        # to ALL output_domain values. The standard fastMRI RSS-magnitude
        # reference convention is universal: strategies (kspace_cold_diffusion,
        # VarNet, ...) RSS-combine multi-coil model output to a 1-ch
        # magnitude before comparing to the 1-ch target, regardless of
        # the loss-side declared output_domain.
        config = _make_config(
            coil_processing_mode="rss",
            in_channels=2,
            out_channels=2,
            target_channels=1,
            output_domain="kspace",
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        warns = _warnings(results)
        assert not any("target_channels=1" in w.message for w in warns), (
            "target_channels=1 should be tolerated under RSS-reference convention"
        )


class TestEarlyReturns:
    def test_no_data_section_returns_empty(self) -> None:
        config: Any = SimpleNamespace(
            data=None,
            model=SimpleNamespace(in_channels=2, out_channels=2, model_type="unet"),
            losses=None,
        )
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert results == []

    def test_no_in_channels_returns_empty(self) -> None:
        config = _make_config(in_channels=None)  # type: ignore[arg-type]
        results = ConfigHealthChecker().check_domain_alignment(config)
        assert results == []


class TestValidateConfigHealthReturnsReport:
    """Pin the H1 refactor: validate_config_health must return the report.

    The pipeline relies on ``report.errors`` filtering by ``check_name``
    to fail fast on domain mismatches. If this contract regresses to a
    bool, the fail-fast in ``train.py`` is silently bypassed.
    """

    def test_returns_report_not_bool(self) -> None:
        # A known-failing config (rss + 8 channels): the caller must be able to
        # introspect domain_alignment errors off the returned report.
        #
        # Built from the REAL schema rather than a SimpleNamespace. This test
        # drives `run_all_checks`, which touches every top-level block, so a
        # hand-built stand-in has to grow a new attribute every time a phase
        # adds a sub-block -- it broke on `data.coils` (phase 9a) and then
        # immediately on `optimization.precision` (phase 8). A real
        # TrainingSettings tracks the schema by construction.
        from spectramr.config.settings import TrainingSettings
        from tests.unit.config.test_settings import _minimal_config

        raw = _minimal_config()
        raw.setdefault("data", {}).update(
            {
                "coil_processing_mode": "rss",  # folds to data.coils.processing_mode
                "dataset_type": "fastmri_kspace",
                "num_virtual_coils": 4,
            }
        )
        raw.setdefault("model", {}).update(
            {"model_type": "unet", "in_channels": 8, "out_channels": 8}
        )
        raw.setdefault("training", {}).setdefault("training_mode", "reconstruction")
        config = TrainingSettings(**raw)
        assert config.data.coils.processing_mode == "rss", "fixture drifted"

        report = validate_config_health(config)

        assert isinstance(report, HealthCheckReport)
        assert not report.passed
        domain_errors = [r for r in report.errors if r.check_name == "domain_alignment"]
        assert len(domain_errors) >= 1, (
            f"Expected at least one domain_alignment error, got: {report.errors}"
        )


class TestSVDCompressionPhaseSafety:
    """Pin the phase-loss-prevention audit (added 2026-05-05).

    The bug: ``coil_processing_mode: svd`` with a dataset that already
    real-stacks complex k-space causes the SVDCoilCompressionTransform
    to silently skip — a CLAUDE.md #9 silent fallback that loses phase
    information for downstream SENSE-adjoint losses.
    """

    def _config(self, **kwargs) -> Any:
        defaults = dict(
            coil_processing_mode="svd",
            num_virtual_coils=4,
            in_channels=8,  # = 2 * num_virtual_coils → suspicious
            out_channels=2,
            target_channels=1,
            output_domain="image",
        )
        defaults.update(kwargs)
        cfg = _make_config(**{k: v for k, v in defaults.items() if k != "transforms_pre_model"})
        if kwargs.get("transforms_pre_model"):
            cfg.data.transforms = SimpleNamespace(pre_model=kwargs["transforms_pre_model"])
        return cfg

    def test_svd_with_2nv_in_channels_fails(self) -> None:
        cfg = self._config()  # 4*2 == 8
        result = ConfigHealthChecker().check_svd_compression_phase_safety(cfg)
        assert not result.passed and result.severity == "error"
        assert "phase" in result.message.lower() or "silent" in result.message.lower()

    def test_explicit_complex_to_real_adapter_passes(self) -> None:
        cfg = self._config(
            transforms_pre_model=[
                SimpleNamespace(name="complex_to_real_imag_interleave"),
            ]
        )
        result = ConfigHealthChecker().check_svd_compression_phase_safety(cfg)
        assert result.passed

    def test_non_svd_mode_skipped(self) -> None:
        cfg = self._config(coil_processing_mode="rss", in_channels=2, num_virtual_coils=4)
        result = ConfigHealthChecker().check_svd_compression_phase_safety(cfg)
        assert result.passed and result.severity == "info"


class TestValidationImageDomainSafe:
    """Pin the experiment_11_kspace_cold_diffusion doubled-brain audit."""

    def _config(
        self,
        *,
        input_type: str = "kspace",
        save_validation_images: bool = True,
        compute_image_metrics: bool = False,
    ) -> Any:
        cfg = _make_config()
        cfg.model.input_type = input_type
        # Both keys moved: `logging.save_validation_images` ->
        # `logging.images.save_validation` (phase 10b) and the metrics flag was
        # never on `logging:` at all -- it is `validation.scoring.enable_image_metrics`
        # (#679). A stand-in carrying the flat pair models a config the schema
        # cannot produce, so it exercised neither branch of the real check.
        cfg.logging = SimpleNamespace(
            images=SimpleNamespace(save_validation=save_validation_images)
        )
        cfg.validation = SimpleNamespace(
            scoring=SimpleNamespace(enable_image_metrics=compute_image_metrics, compute=[])
        )
        return cfg

    def test_kspace_save_without_image_metrics_fails(self) -> None:
        cfg = self._config()
        result = ConfigHealthChecker().check_validation_image_domain_safe(cfg)
        assert not result.passed and result.severity == "error"
        assert "doubled-brain" in result.message or "compute_image_metrics" in result.message

    def test_compute_image_metrics_true_passes(self) -> None:
        cfg = self._config(compute_image_metrics=True)
        result = ConfigHealthChecker().check_validation_image_domain_safe(cfg)
        assert result.passed

    def test_image_input_type_skipped(self) -> None:
        cfg = self._config(input_type="image")
        result = ConfigHealthChecker().check_validation_image_domain_safe(cfg)
        assert result.passed and result.severity == "info"

    def test_save_disabled_passes(self) -> None:
        cfg = self._config(save_validation_images=False)
        result = ConfigHealthChecker().check_validation_image_domain_safe(cfg)
        assert result.passed


@pytest.mark.parametrize(
    ("coil_mode", "in_ch", "expected_passes"),
    [
        ("rss", 2, True),
        ("rss", 4, False),
        ("magnitude", 1, True),
        ("magnitude", 2, False),
        ("svd", 8, True),  # 2 * num_virtual_coils=4
        ("svd", 6, False),
        ("flatten", 8, True),  # parity check only
        ("flatten", 7, False),
    ],
)
def test_in_channels_branches(coil_mode: str, in_ch: int, expected_passes: bool) -> None:
    """Smoke-test the channel-derivation table across modes."""
    config = _make_config(coil_processing_mode=coil_mode, in_channels=in_ch, out_channels=in_ch)
    results = ConfigHealthChecker().check_domain_alignment(config)
    has_error = any(not r.passed and r.severity == "error" for r in results)
    assert has_error != expected_passes, (
        f"coil_mode={coil_mode}, in_ch={in_ch}: "
        f"expected_passes={expected_passes}, has_error={has_error}\n"
        f"results={[(r.passed, r.severity, r.message) for r in results]}"
    )


# ---------------------------------------------------------------------------
# VF campaign Phase 1: checkpoint_existence (CW-2) + marker_subspace_conditioning (I-5)
# ---------------------------------------------------------------------------


def test_checkpoint_existence_no_dependency_info_passes():
    """An arm declaring no checkpoint/basis info-passes (synthetic-injection arms)."""
    cfg = SimpleNamespace()
    res = ConfigHealthChecker().check_checkpoint_existence(cfg)
    assert res.passed and res.severity == "info"


def test_checkpoint_existence_present_passes(tmp_path):
    ckpt = tmp_path / "teacher.ckpt"
    ckpt.write_text("x")
    cfg = SimpleNamespace(model=SimpleNamespace(checkpoint_path=str(ckpt)))
    res = ConfigHealthChecker().check_checkpoint_existence(cfg)
    assert res.passed and res.severity == "info"


def test_checkpoint_existence_missing_warns(tmp_path):
    """A declared-but-absent artefact is a warning (--strict escalates at dispatch)."""
    missing = tmp_path / "nope.ckpt"
    cfg = SimpleNamespace(model=SimpleNamespace(checkpoint_path=str(missing)))
    res = ConfigHealthChecker().check_checkpoint_existence(cfg)
    assert not res.passed
    assert res.severity == "warning"
    assert "nope.ckpt" in res.message
    assert "model.checkpoint_path" in res.yaml_keys


# VF campaign Phase 1b: produced_by_arm deferred campaign dependency (Option A,
# 2026-06-16). A downstream eval/calibration/DPS arm whose checkpoint is built by
# an upstream campaign arm info-passes the standalone --strict pre-flight, while
# the real existence gate still fires at checkpoint-load time.


def test_checkpoint_existence_deferred_to_producer_info_passes():
    """Missing, campaign-artefact-rooted checkpoint + produced_by_arm → info-pass."""
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            checkpoint_path="experiments/active/reconstruction/m4raw_unet_4x.ckpt"
        ),
        checkpoint=SimpleNamespace(produced_by_arm="baseline_m4raw_unet_4x"),
    )
    res = ConfigHealthChecker().check_checkpoint_existence(cfg)
    assert res.passed and res.severity == "info"
    assert "baseline_m4raw_unet_4x" in res.message


def test_checkpoint_existence_produced_by_arm_requires_artefact_root(tmp_path):
    """produced_by_arm must NOT wave through an arbitrary non-campaign path."""
    rogue = tmp_path / "rogue.ckpt"  # absolute tmp path, not experiments/active|results
    cfg = SimpleNamespace(
        model=SimpleNamespace(checkpoint_path=str(rogue)),
        checkpoint=SimpleNamespace(produced_by_arm="whatever"),
    )
    res = ConfigHealthChecker().check_checkpoint_existence(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert res.fix_hint is not None


def test_checkpoint_existence_produced_by_arm_does_not_defer_marker_artefacts():
    """A precomputed marker artefact is NOT a training-arm output → still warns."""
    cfg = SimpleNamespace(
        certification=SimpleNamespace(
            conformal=SimpleNamespace(marker_basis_path="experiments/active/markers/ghost.pt")
        ),
        checkpoint=SimpleNamespace(produced_by_arm="baseline_x"),
    )
    res = ConfigHealthChecker().check_checkpoint_existence(cfg)
    assert not res.passed
    assert res.severity == "warning"


def test_checkpoint_existence_present_path_unaffected_by_produced_by_arm(tmp_path):
    """produced_by_arm is inert when the artefact already exists (normal pass)."""
    ckpt = tmp_path / "teacher.ckpt"
    ckpt.write_text("x")
    cfg = SimpleNamespace(
        model=SimpleNamespace(checkpoint_path=str(ckpt)),
        checkpoint=SimpleNamespace(produced_by_arm="baseline_x"),
    )
    res = ConfigHealthChecker().check_checkpoint_existence(cfg)
    assert res.passed and res.severity == "info"


def test_marker_conditioning_no_basis_info_skips():
    cfg = SimpleNamespace()
    res = ConfigHealthChecker().check_marker_subspace_conditioning(cfg)
    assert res.passed and res.severity == "info"


def test_marker_conditioning_orthonormal_passes(tmp_path):
    """An orthonormal (identity) marker basis has κ = 1 → pass."""
    torch = pytest.importorskip("torch")
    basis = tmp_path / "ortho.pt"
    torch.save(torch.eye(8), basis)
    cfg = SimpleNamespace(
        certification=SimpleNamespace(conformal=SimpleNamespace(marker_basis_path=str(basis)))
    )
    res = ConfigHealthChecker().check_marker_subspace_conditioning(cfg)
    assert res.passed and res.severity == "info"
    assert "κ(M)" in res.message


def test_marker_conditioning_ill_conditioned_errors(tmp_path):
    """A near-singular basis (κ ≫ kappa_max) → error with a fix hint."""
    torch = pytest.importorskip("torch")
    mat = torch.eye(4)
    mat[-1, -1] = 1e-9  # σ_min ≈ 1e-9 → κ ≈ 1e9 ≫ 1e4
    basis = tmp_path / "singular.pt"
    torch.save(mat, basis)
    cfg = SimpleNamespace(
        certification=SimpleNamespace(conformal=SimpleNamespace(marker_basis_path=str(basis)))
    )
    res = ConfigHealthChecker().check_marker_subspace_conditioning(cfg)
    assert not res.passed
    assert res.severity == "error"
    assert res.fix_hint is not None


def test_marker_conditioning_absent_file_info_skips(tmp_path):
    """A declared-but-absent basis is info-skipped here (gated by checkpoint_existence)."""
    cfg = SimpleNamespace(
        certification=SimpleNamespace(
            conformal=SimpleNamespace(marker_basis_path=str(tmp_path / "ghost.pt"))
        )
    )
    res = ConfigHealthChecker().check_marker_subspace_conditioning(cfg)
    assert res.passed and res.severity == "info"


# ---------------------------------------------------------------------------
# check_coil_processing_consistency — reject invalid physics.coil_processing
# combinations at audit time (Tier 0/1) instead of at runtime (pitfall #10).
# ---------------------------------------------------------------------------


def _coil_cfg(
    *,
    compression: dict[str, Any] | None = None,
    estimation: dict[str, Any] | None = None,
    combine: dict[str, Any] | None = None,
    output: dict[str, Any] | None = None,
    model_type: str = "unet",
    in_channels: int = 2,
) -> Any:
    return SimpleNamespace(
        physics=SimpleNamespace(
            coil_processing=SimpleNamespace(
                compression=SimpleNamespace(
                    **(compression or {"method": "none", "num_virtual_coils": 4})
                ),
                estimation=SimpleNamespace(
                    **(estimation or {"method": "power_iter", "enabled": True, "maps_path": None})
                ),
                combine=SimpleNamespace(**(combine or {"method": "rss"})),
                output=SimpleNamespace(
                    **(output or {"domain": "kspace", "channels": "real_interleaved"})
                ),
            )
        ),
        model=SimpleNamespace(model_type=model_type, in_channels=in_channels),
    )


def test_coil_audit_magnitude_requires_combine() -> None:
    cfg = _coil_cfg(
        combine={"method": "none"},
        output={"domain": "image", "channels": "magnitude"},
    )
    res = ConfigHealthChecker().check_coil_processing_consistency(cfg)
    assert not res.passed and res.severity == "error"


def test_coil_audit_complex_incompatible_with_rss() -> None:
    cfg = _coil_cfg(
        combine={"method": "rss"},
        output={"domain": "kspace", "channels": "complex"},
    )
    res = ConfigHealthChecker().check_coil_processing_consistency(cfg)
    assert not res.passed and res.severity == "error"


def test_coil_audit_complex_with_sense_ok() -> None:
    # sense is keep-coils at data-load, so complex output is fine.
    cfg = _coil_cfg(
        estimation={"method": "espirit", "enabled": True, "maps_path": None},
        combine={"method": "sense"},
        output={"domain": "kspace", "channels": "complex"},
    )
    assert ConfigHealthChecker().check_coil_processing_consistency(cfg).passed


class TestCoilProcessingConsistency:
    def _run(self, cfg):
        return ConfigHealthChecker().check_coil_processing_consistency(cfg)

    def test_no_block_is_info_pass(self):
        cfg = SimpleNamespace(physics=SimpleNamespace())
        res = self._run(cfg)
        assert res.passed and res.severity == "info"

    def test_sense_without_estimation_errors(self):
        cfg = _coil_cfg(
            estimation={"method": "none", "enabled": False, "maps_path": None},
            combine={"method": "sense"},
        )
        res = self._run(cfg)
        assert not res.passed and res.severity == "error"

    def test_sense_with_estimation_passes(self):
        cfg = _coil_cfg(
            estimation={"method": "espirit", "enabled": True, "maps_path": None},
            combine={"method": "sense"},
        )
        assert self._run(cfg).passed

    def test_gcc_is_reserved_error(self):
        cfg = _coil_cfg(compression={"method": "gcc", "num_virtual_coils": 4})
        res = self._run(cfg)
        assert not res.passed and res.severity == "error"

    def test_file_without_maps_path_errors(self):
        cfg = _coil_cfg(estimation={"method": "file", "enabled": True, "maps_path": None})
        res = self._run(cfg)
        assert not res.passed and res.severity == "error"

    def test_file_with_maps_path_passes(self):
        cfg = _coil_cfg(estimation={"method": "file", "enabled": True, "maps_path": "/tmp/m.pt"})
        assert self._run(cfg).passed

    def test_svd_in_channels_mismatch_errors(self):
        # nvc=4 ⇒ in_channels must be 8; here it's 2.
        cfg = _coil_cfg(
            compression={"method": "svd", "num_virtual_coils": 4},
            in_channels=2,
            model_type="unet",
        )
        res = self._run(cfg)
        assert not res.passed and res.severity == "error"

    def test_svd_in_channels_match_passes(self):
        cfg = _coil_cfg(
            compression={"method": "svd", "num_virtual_coils": 4},
            in_channels=8,
            model_type="unet",
        )
        assert self._run(cfg).passed

    def test_svd_in_channels_skipped_for_concat_model(self):
        # kspace_cold_diffusion concatenates smaps → in_channels check skipped.
        cfg = _coil_cfg(
            compression={"method": "svd", "num_virtual_coils": 4},
            in_channels=2,
            model_type="kspace_cold_diffusion",
        )
        assert self._run(cfg).passed

    def test_clean_svd_espirit_sense_passes(self):
        cfg = _coil_cfg(
            compression={"method": "svd", "num_virtual_coils": 4},
            estimation={"method": "espirit", "enabled": True, "maps_path": None},
            combine={"method": "sense"},
            in_channels=8,
            model_type="unet",
        )
        assert self._run(cfg).passed


class TestStrategyRegistry:
    """V-1: ``check_strategy_registry`` must reject an unregistered
    ``training_mode`` instead of silently passing (NN#3 — no silent
    fallbacks). A typo like ``'difusion'`` must surface at audit time
    rather than deferring a ``ConfigurationError`` to runtime.
    """

    @staticmethod
    def _cfg(*, training_mode: str | None, strategy_class: str | None = None) -> Any:
        return SimpleNamespace(
            training=SimpleNamespace(
                training_mode=training_mode,
                strategy_class=strategy_class,
            ),
        )

    def test_training_mode_valid_passes(self) -> None:
        checker = ConfigHealthChecker()
        # Resolve a genuinely-registered short name from the live registry so
        # the test does not pin a specific paradigm name.
        checker._lazy_load_registries()
        assert checker._strategy_registry, "strategy registry should populate"
        valid_mode = sorted(checker._strategy_registry)[0]
        result = checker.check_strategy_registry(
            self._cfg(training_mode=valid_mode, strategy_class=None)
        )
        assert result.passed is True
        assert result.severity == "info"

    def test_training_mode_typo_fails(self) -> None:
        checker = ConfigHealthChecker()
        checker._lazy_load_registries()
        # A mode guaranteed not to be registered.
        assert "difusion_typo_xyz" not in (checker._strategy_registry or set())
        result = checker.check_strategy_registry(
            self._cfg(training_mode="difusion_typo_xyz", strategy_class=None)
        )
        assert result.passed is False
        assert result.severity == "error"
        assert result.category == "strategy_registry"
        assert "training.training_mode" in result.yaml_keys

    def test_training_mode_with_empty_registry_passes(self) -> None:
        checker = ConfigHealthChecker()
        # Force the empty-registry branch. ``_lazy_load_registries`` short-
        # circuits when ``_model_registry is not None``, so pre-seeding both
        # caches keeps the strategy registry empty for this call.
        checker._model_registry = set()
        checker._strategy_registry = set()
        result = checker.check_strategy_registry(
            self._cfg(training_mode="anything", strategy_class=None)
        )
        assert result.passed is True
        assert result.severity == "info"


class TestScientificMetadata:
    """Pins ConfigHealthChecker.check_scientific_metadata (2026-06 validation campaign).

    Surfaces the *metric-mismatch* failure mode via metadata.primary_metric. It is
    deliberately ADVISORY (severity=info, passed=True) — primary_metric is a free
    human label that often differs harmlessly from the precise emitted key
    (e.g. 'psnr' vs 'val_robust_mri_psnr'), so a hard warning would false-positive
    on ~34/148 existing configs and break --strict smoke. These tests pin that it
    never emits an error or warning, normalises cascading suffixes, and skips prose
    values. ``metadata`` is the mounted ``ExperimentMetadataSchema`` (2026-09-02);
    the dict shape it used to carry is gone.
    """

    @staticmethod
    def _cfg(
        *,
        primary_metric: str | None = None,
        validation_metrics: list[str] | None = None,
        expected_outcome: str | None = None,
        baseline: str | None = None,
    ) -> Any:
        meta = SimpleNamespace(
            primary_metric=primary_metric, expected_outcome=expected_outcome, baseline=baseline
        )
        return SimpleNamespace(
            metadata=meta,
            validation=SimpleNamespace(scoring=SimpleNamespace(compute=validation_metrics or [])),
        )

    def test_inactive_when_primary_metric_unset(self) -> None:
        # Forward-looking: no field set => no results (existing configs unaffected).
        assert ConfigHealthChecker().check_scientific_metadata(self._cfg()) == []

    def test_advisory_never_errors_or_warns(self) -> None:
        # Even when the metric is NOT emitted, the check must stay advisory (info)
        # so it can never break the audit / smoke gate.
        cfg = self._cfg(primary_metric="val_bland_altman_bias", validation_metrics=["psnr", "ssim"])
        results = ConfigHealthChecker().check_scientific_metadata(cfg)
        assert _errors(results) == []
        assert _warnings(results) == []
        assert all(r.severity == "info" and r.passed for r in results)
        assert any("not in validation.metrics" in r.message for r in results)

    def test_a_floor_without_a_baseline_is_flagged(self) -> None:
        """Planted violation: a position relative to nothing."""
        results = ConfigHealthChecker().check_scientific_metadata(
            self._cfg(expected_outcome="floor")
        )
        assert [r.check_name for r in _warnings(results)] == [
            "scientific_metadata_expected_outcome"
        ]
        assert "metadata.baseline is not declared" in _warnings(results)[0].message

    def test_a_ceiling_against_a_baseline_passes(self) -> None:
        results = ConfigHealthChecker().check_scientific_metadata(
            self._cfg(expected_outcome="ceiling", baseline="control_arm")
        )
        assert _warnings(results) == [] and _errors(results) == []
        assert any("against baseline 'control_arm'" in r.message for r in results)

    def test_comparable_stands_on_its_own(self) -> None:
        results = ConfigHealthChecker().check_scientific_metadata(
            self._cfg(expected_outcome="comparable")
        )
        assert _warnings(results) == [] and _errors(results) == []
        assert [r.check_name for r in results] == ["scientific_metadata_expected_outcome"]

    def test_computed_metric_is_clean(self) -> None:
        cfg = self._cfg(primary_metric="val_psnr", validation_metrics=["psnr", "ssim"])
        results = ConfigHealthChecker().check_scientific_metadata(cfg)
        assert _warnings(results) == [] and _errors(results) == []
        assert any("declared" in r.message for r in results)

    def test_cascading_suffix_is_normalized(self) -> None:
        # val_psnr_2x (cascading key) should match base 'psnr' in validation.metrics.
        cfg = self._cfg(primary_metric="val_psnr_2x", validation_metrics=["psnr"])
        results = ConfigHealthChecker().check_scientific_metadata(cfg)
        assert any("declared" in r.message for r in results)

    def test_prose_primary_metric_is_skipped(self) -> None:
        # A descriptive value (with spaces/parens) must not be flagged as not-emitted.
        cfg = self._cfg(
            primary_metric="val_loss (masked-patch reconstruction MSE proxy)",
            validation_metrics=["psnr"],
        )
        results = ConfigHealthChecker().check_scientific_metadata(cfg)
        assert all("not in validation.metrics" not in r.message for r in results)


class TestAccelerationConsistencyCenterFraction:
    """The center_fraction band check must enforce the band its message claims.

    The 2026-06-12 audit found the code rejected ``cf < 0.01`` while the message
    and fix-hint advertised ``[0.04, 0.5]`` — so cf in [0.01, 0.04) (e.g. the 28
    high-R arms at 0.03) silently passed under a message that claimed otherwise.
    The honest, non-breaking fix made the message match the enforced 0.01 floor.
    """

    @staticmethod
    def _cfg(center_fraction: float):
        # Phase 11 renamed the top-level block `acceleration:` -> `undersampling:`;
        # both readers here take `getattr(config, "undersampling")`. Note the
        # vacuous half of this drift: `test_high_r_center_fraction_passes` asserts
        # `res.passed` and kept passing after the move because the check SKIPPED,
        # while the two negative cases failed loudly. Expect that split whenever a
        # block moves under a stand-in.
        return SimpleNamespace(
            undersampling=SimpleNamespace(
                center_fraction=center_fraction,
                base_acceleration=1.0,
                max_acceleration=8.0,
            )
        )

    def test_high_r_center_fraction_passes(self) -> None:
        # 0.03 is a legitimate high-acceleration ACS fraction used by ~28 arms.
        res = ConfigHealthChecker().check_acceleration_consistency(self._cfg(0.03))
        assert res.passed

    def test_below_floor_fails_with_consistent_message(self) -> None:
        res = ConfigHealthChecker().check_acceleration_consistency(self._cfg(0.005))
        assert not res.passed
        # Message must state the ACTUAL enforced band, not the old [0.04, 0.5].
        assert "[0.01, 0.5]" in res.message
        assert "0.04" not in res.message

    def test_above_ceiling_fails(self) -> None:
        res = ConfigHealthChecker().check_acceleration_consistency(self._cfg(0.6))
        assert not res.passed


# ---------------------------------------------------------------------------
# check_metric_backend_available: fail fast when a metric has no importable
# backend (LPIPS has a lpips-package fallback; ms_ssim/uqi/kid/fid do not).
# ---------------------------------------------------------------------------


def _cfg_with_metrics(metrics: list[str]) -> Any:
    # `validation.metrics` -> `validation.scoring.compute` (phase 10a).
    return SimpleNamespace(validation=SimpleNamespace(scoring=SimpleNamespace(compute=metrics)))


class TestMetricBackendAvailable:
    def test_no_metrics_is_info_pass(self) -> None:
        res = ConfigHealthChecker().check_metric_backend_available(_cfg_with_metrics([]))
        assert all(r.passed for r in res)

    def test_torchmetrics_available_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import spectramr.core.metrics.evaluation_metrics as em

        monkeypatch.setattr(em, "TORCHMETRICS_AVAILABLE", True)
        res = ConfigHealthChecker().check_metric_backend_available(
            _cfg_with_metrics(["ms_ssim", "psnr"])
        )
        assert all(r.passed for r in res)

    def test_unbacked_metric_errors_when_torchmetrics_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import spectramr.core.metrics.evaluation_metrics as em

        monkeypatch.setattr(em, "TORCHMETRICS_AVAILABLE", False)
        res = ConfigHealthChecker().check_metric_backend_available(_cfg_with_metrics(["ms_ssim"]))
        errs = _errors(res)
        assert len(errs) == 1
        assert "ms_ssim" in errs[0].message
        assert "validation.metrics" in errs[0].yaml_keys

    def test_lpips_fallback_avoids_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # torchmetrics missing but the lpips package present -> LPIPS is backed, no error.
        pytest.importorskip("lpips")
        import spectramr.core.metrics.evaluation_metrics as em

        monkeypatch.setattr(em, "TORCHMETRICS_AVAILABLE", False)
        res = ConfigHealthChecker().check_metric_backend_available(
            _cfg_with_metrics(["lpips", "psnr"])
        )
        assert _errors(res) == []


# ===========================================================================
# Cluster-failure pre-flight guards (2026-07-07): the two failure families the
# mrixfields run surfaced, moved to Tier-0/1 audit time + a general data-leak
# guard that compares the train/val splits for every split strategy.
# ===========================================================================
import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def _write_mrixfields_manifest(path: _Path, records: list) -> None:
    path.write_text(_json.dumps({"data_root": ".", "records": records}))


def _field_rec(subject: str, field: float, *, group: str, contrast: str = "T1w") -> dict:
    rel = f"{subject}/{contrast}/{field}/vol.nii.gz"
    return {
        "subject_id": subject,
        "contrast": contrast,
        "pairing_group": group,
        "field_strength": field,
        "relative_path": rel,
        "file_id": rel,
    }


# --- Check B: contrast conditioning must ride a threaded strategy -----------
def _contrast_cfg(*, enabled: bool, strategy_class: str | None) -> Any:
    return SimpleNamespace(
        model=SimpleNamespace(model_kwargs={"use_contrast_conditioning": enabled}),
        training=SimpleNamespace(strategy_class=strategy_class),
    )


class TestContrastStrategyThreaded:
    def test_enabled_on_threaded_strategy_passes(self) -> None:
        res = ConfigHealthChecker().check_contrast_conditioning_strategy_threaded(
            _contrast_cfg(enabled=True, strategy_class="field_bridge")
        )
        assert res.passed
        assert res.severity != "error"

    def test_enabled_on_unthreaded_strategy_errors(self) -> None:
        res = ConfigHealthChecker().check_contrast_conditioning_strategy_threaded(
            _contrast_cfg(enabled=True, strategy_class="cross_field_translation_gan")
        )
        assert not res.passed
        assert res.severity == "error"
        assert "contrast_id" in res.message

    def test_disabled_is_noop(self) -> None:
        res = ConfigHealthChecker().check_contrast_conditioning_strategy_threaded(
            _contrast_cfg(enabled=False, strategy_class="cross_field_translation_gan")
        )
        assert res.passed

    def test_model_kwargs_object_form(self) -> None:
        cfg = SimpleNamespace(
            model=SimpleNamespace(model_kwargs=SimpleNamespace(use_contrast_conditioning=True)),
            training=SimpleNamespace(strategy_class="ulf_dps"),
        )
        assert ConfigHealthChecker().check_contrast_conditioning_strategy_threaded(cfg).passed


# --- Check A: mrixfields pairing viability ----------------------------------
def _mrixfields_cfg(
    *,
    policy: str,
    target_field: float | None = None,
    validation_index_path: str | None = None,
    index_path: str | None = None,
) -> Any:
    # Nested shape: phase 9a moved the mrixfields knobs to `data.mrixfields.*`.
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_type="mrixfields",
            mrixfields=SimpleNamespace(
                pairing_policy=policy,
                target_field=target_field,
            ),
            validation_index_path=validation_index_path,
            index_path=index_path,
        )
    )


class TestMrixfieldsPairingViable:
    def test_non_mrixfields_dataset_skipped(self) -> None:
        cfg = SimpleNamespace(data=SimpleNamespace(dataset_type="fastmri_kspace"))
        res = ConfigHealthChecker().check_mrixfields_pairing_viable(cfg)
        assert _errors(res) == []

    def test_field_pinned_policy_requires_target_field(self) -> None:
        res = ConfigHealthChecker().check_mrixfields_pairing_viable(
            _mrixfields_cfg(policy="fixed_target", target_field=None)
        )
        assert len(_errors(res)) >= 1
        assert any("target_field" in r.message for r in _errors(res))

    def test_stale_singleton_manifest_errors_and_names_regeneration(self, tmp_path: _Path) -> None:
        val = tmp_path / "val.json"
        # fields present but every group is a singleton — the stale signature.
        recs = [
            _field_rec(f"vol{i}", f, group=f"vol{i}|T1w")
            for i, f in enumerate([0.1, 1.5, 3.0, 5.0, 7.0])
        ]
        _write_mrixfields_manifest(val, recs)
        res = ConfigHealthChecker().check_mrixfields_pairing_viable(
            _mrixfields_cfg(
                policy="fixed_target",
                target_field=7.0,
                validation_index_path=str(val),
            )
        )
        errs = _errors(res)
        assert len(errs) >= 1
        assert any("build_mrixfields2026_manifest" in r.message for r in errs)

    def test_paired_manifest_passes(self, tmp_path: _Path) -> None:
        val = tmp_path / "val.json"
        # one group carrying all fields — pairs are formable.
        recs = [_field_rec("vol0", f, group="vol0|T1w") for f in [0.1, 1.5, 3.0, 5.0, 7.0]]
        _write_mrixfields_manifest(val, recs)
        res = ConfigHealthChecker().check_mrixfields_pairing_viable(
            _mrixfields_cfg(
                policy="fixed_target",
                target_field=7.0,
                validation_index_path=str(val),
            )
        )
        assert _errors(res) == []

    def test_absent_manifest_skipped(self, tmp_path: _Path) -> None:
        res = ConfigHealthChecker().check_mrixfields_pairing_viable(
            _mrixfields_cfg(
                policy="fixed_target",
                target_field=7.0,
                validation_index_path=str(tmp_path / "gone.json"),
            )
        )
        assert _errors(res) == []  # gitignored locally → skip, not fail


# --- Check C: train/val split leakage ---------------------------------------
class TestTrainValSplitLeakage:
    def test_overlapping_manifests_error(self, tmp_path: _Path) -> None:
        train = tmp_path / "train.json"
        val = tmp_path / "val.json"
        _write_mrixfields_manifest(
            train,
            [_field_rec("A", 3.0, group="A|T1w"), _field_rec("B", 3.0, group="B|T1w")],
        )
        _write_mrixfields_manifest(
            val,
            [_field_rec("B", 7.0, group="B|T1w")],  # subject B leaks
        )
        cfg = SimpleNamespace(
            data=DataConfigStub(
                index_path=str(train),
                validation_index_path=str(val),
                validation_split=0.1,
                holdout_site=None,
                split_strategy="manifest",
            )
        )
        res = ConfigHealthChecker().check_train_val_split_leakage(cfg)
        assert not res.passed
        assert res.severity == "error"
        assert "B" in res.message or "B" in "".join(res.fix_hint or "")

    def test_disjoint_manifests_pass(self, tmp_path: _Path) -> None:
        train = tmp_path / "train.json"
        val = tmp_path / "val.json"
        _write_mrixfields_manifest(train, [_field_rec("A", 3.0, group="A|T1w")])
        _write_mrixfields_manifest(val, [_field_rec("Z", 7.0, group="Z|T1w")])
        cfg = SimpleNamespace(
            data=DataConfigStub(
                index_path=str(train),
                validation_index_path=str(val),
                validation_split=0.1,
                holdout_site=None,
                split_strategy="manifest",
            )
        )
        res = ConfigHealthChecker().check_train_val_split_leakage(cfg)
        assert res.passed
        assert res.severity != "error"

    def test_absent_manifests_skipped(self, tmp_path: _Path) -> None:
        cfg = SimpleNamespace(
            data=DataConfigStub(
                index_path=str(tmp_path / "t.json"),
                validation_index_path=str(tmp_path / "v.json"),
                validation_split=0.1,
                holdout_site=None,
                split_strategy="manifest",
            )
        )
        res = ConfigHealthChecker().check_train_val_split_leakage(cfg)
        assert res.passed  # skipped is non-fatal
        assert res.severity != "error"


def test_all_three_guards_wired_into_run_all_checks() -> None:
    names = {
        "contrast_conditioning_strategy_threaded",
        "mrixfields_pairing_viable",
        "train_val_split_leakage",
    }
    import inspect

    src = inspect.getsource(ConfigHealthChecker.run_all_checks)
    for n in names:
        assert f"check_{n}" in src, f"check_{n} not wired into run_all_checks"


class TestAcsWithinCenterBand:
    """Coil-map calibration region must support the kernel (fail-fast), and the
    ACS-vs-preserved-center-band relationship is surfaced (inference-time g-factor)."""

    def _cfg(
        self,
        *,
        dataset_type: str = "m4raw",
        method: str = "power_iter",
        enabled: bool = True,
        kernel_size: int = 12,
        acs_size: int = 24,
        center_fraction: float = 0.08,
        patch_w: int = 256,
    ) -> Any:
        return SimpleNamespace(
            data=DataConfigStub(dataset_type=dataset_type, patch_size=[patch_w, patch_w, 1]),
            undersampling=SimpleNamespace(center_fraction=center_fraction),
            physics=SimpleNamespace(
                coil_processing=SimpleNamespace(
                    estimation=SimpleNamespace(
                        method=method,
                        enabled=enabled,
                        kernel_size=kernel_size,
                        acs_size=acs_size,
                    )
                )
            ),
        )

    def test_acs_smaller_than_kernel_errors(self) -> None:
        r = ConfigHealthChecker().check_acs_within_center_band(
            self._cfg(acs_size=8, kernel_size=12)
        )
        assert not r.passed
        assert r.severity == "error"

    def test_acs_ge_kernel_passes(self) -> None:
        r = ConfigHealthChecker().check_acs_within_center_band(
            self._cfg(acs_size=24, kernel_size=12)
        )
        assert r.passed

    def test_non_kspace_not_applicable(self) -> None:
        r = ConfigHealthChecker().check_acs_within_center_band(
            self._cfg(dataset_type="nifti_paired")
        )
        assert r.passed

    def test_method_none_not_applicable(self) -> None:
        r = ConfigHealthChecker().check_acs_within_center_band(self._cfg(method="none"))
        assert r.passed

    def test_acs_exceeds_center_band_is_noted(self) -> None:
        # 24 > 0.08*256≈20 → passes (train/val calibrate from the dense target)
        # but the message flags the inference-time ACS-vs-center relationship.
        r = ConfigHealthChecker().check_acs_within_center_band(
            self._cfg(acs_size=24, center_fraction=0.08, patch_w=256)
        )
        assert r.passed
        assert "center" in r.message.lower()

    def test_registered_in_run_all_checks(self) -> None:
        import inspect

        src = inspect.getsource(ConfigHealthChecker.run_all_checks)
        assert "check_acs_within_center_band" in src


class TestVaePretrainAutoencodesSingleField:
    """A vae_pretrain arm on paired data must autoencode ONE field (hf_to_hf /
    ulf_to_ulf), not train a translation direction that corrupts the stage-2
    latent. The vae_pretrain tag spares the legitimate paired VAE translators."""

    @staticmethod
    def _cfg(*, mode="vae", ds="nifti_paired", bmode="hf_to_hf", tag_type="vae_pretrain"):
        return SimpleNamespace(
            training=SimpleNamespace(training_mode=mode),
            # `data.bidirectional_mode` -> `data.pairing.bidirectional_mode` (phase 9f).
            data=SimpleNamespace(
                dataset_type=ds,
                pairing=SimpleNamespace(bidirectional_mode=bmode),
            ),
            metadata={"tags": {"type": tag_type}},
        )

    def test_vae_pretrain_translation_direction_errors(self) -> None:
        r = ConfigHealthChecker().check_vae_pretrain_autoencodes_single_field(
            self._cfg(bmode="ulf_to_hf")
        )
        assert not r.passed and r.severity == "error"
        assert "autoencoder" in r.message.lower()

    def test_vae_pretrain_hf_to_hf_passes(self) -> None:
        r = ConfigHealthChecker().check_vae_pretrain_autoencodes_single_field(
            self._cfg(bmode="hf_to_hf")
        )
        assert r.passed

    def test_vae_pretrain_ulf_to_ulf_passes(self) -> None:
        r = ConfigHealthChecker().check_vae_pretrain_autoencodes_single_field(
            self._cfg(bmode="ulf_to_ulf")
        )
        assert r.passed

    def test_paired_translator_not_flagged(self) -> None:
        # complex_vae / disentangled_vae translators use ulf_to_hf on purpose;
        # the vae_pretrain tag is the discriminator that spares them.
        for tag in ("complex_vae", "disentangled_vae"):
            r = ConfigHealthChecker().check_vae_pretrain_autoencodes_single_field(
                self._cfg(bmode="ulf_to_hf", tag_type=tag)
            )
            assert r.passed, f"translator tagged {tag} must not be flagged"

    def test_non_paired_vae_not_applicable(self) -> None:
        r = ConfigHealthChecker().check_vae_pretrain_autoencodes_single_field(
            self._cfg(ds="fastmri_kspace", bmode="ulf_to_hf")
        )
        assert r.passed

    def test_non_vae_mode_not_applicable(self) -> None:
        r = ConfigHealthChecker().check_vae_pretrain_autoencodes_single_field(
            self._cfg(mode="reconstruction", bmode="ulf_to_hf")
        )
        assert r.passed

    def test_registered_in_run_all_checks(self) -> None:
        import inspect

        src = inspect.getsource(ConfigHealthChecker.run_all_checks)
        assert "check_vae_pretrain_autoencodes_single_field" in src


class TestPersistentWorkersOnShortEpoch:
    """The 2026-07-10 mrixfields GPU-starvation guard.

    ``persistent_workers=False`` re-forks every worker (each re-importing torch)
    at every epoch boundary. Over a 12-iteration epoch that starved ~70 arms to
    ~0% GPU util for days without ever crashing, so nothing detected it.
    """

    @staticmethod
    def _manifest(tmp_path: Any, n_records: int) -> str:
        import json

        p = tmp_path / "train.json"
        p.write_text(
            json.dumps({"records": [{"relative_path": f"v{i}.nii.gz"} for i in range(n_records)]})
        )
        return str(p)

    @staticmethod
    def _cfg(
        *,
        index_path: str | None = None,
        dataset_type: str = "mrixfields",
        num_workers: int = 4,
        persistent_workers: bool = False,
        batch_size: int = 4,
    ) -> Any:
        # Nested shape: phase 9a moved these three to `data.loader.*`.
        return SimpleNamespace(
            data=SimpleNamespace(
                dataset_type=dataset_type,
                index_path=index_path,
                loader=SimpleNamespace(
                    num_workers=num_workers,
                    persistent_workers=persistent_workers,
                    batch_size=batch_size,
                ),
            )
        )

    def test_short_epoch_without_persistent_workers_is_flagged(self, tmp_path: Any) -> None:
        # 45 records / batch 4 = a 12-iteration epoch: the real mrixfields shape.
        r = ConfigHealthChecker().check_persistent_workers_on_short_epoch(
            self._cfg(index_path=self._manifest(tmp_path, 45))
        )
        assert not r.passed
        assert r.severity == "warning"
        assert "persistent_workers" in r.fix_hint
        assert "12-iteration epoch" in r.message

    def test_persistent_workers_true_passes(self, tmp_path: Any) -> None:
        r = ConfigHealthChecker().check_persistent_workers_on_short_epoch(
            self._cfg(index_path=self._manifest(tmp_path, 45), persistent_workers=True)
        )
        assert r.passed

    def test_no_workers_passes(self, tmp_path: Any) -> None:
        # Without worker processes there is nothing to respawn.
        r = ConfigHealthChecker().check_persistent_workers_on_short_epoch(
            self._cfg(index_path=self._manifest(tmp_path, 45), num_workers=0)
        )
        assert r.passed

    def test_long_epoch_amortizes_respawn(self, tmp_path: Any) -> None:
        # 1939 records / batch 4 = ~485 iters/epoch — the real b22 shape, which
        # ran ~2 s/it while its 45-volume siblings crawled at ~3.5 s/it.
        r = ConfigHealthChecker().check_persistent_workers_on_short_epoch(
            self._cfg(index_path=self._manifest(tmp_path, 1939))
        )
        assert r.passed

    def test_slice_expanding_dataset_is_skipped_not_guessed(self, tmp_path: Any) -> None:
        # npy_slice explodes one record into many samples, so record count does
        # NOT bound the epoch. Guessing here would block a legitimate config.
        r = ConfigHealthChecker().check_persistent_workers_on_short_epoch(
            self._cfg(index_path=self._manifest(tmp_path, 45), dataset_type="npy_slice")
        )
        assert r.passed
        assert r.severity == "info"

    def test_absent_manifest_is_skipped(self) -> None:
        # Manifests are gitignored; the audit must not crash when one is missing.
        r = ConfigHealthChecker().check_persistent_workers_on_short_epoch(
            self._cfg(index_path="data/manifests/does_not_exist.json")
        )
        assert r.passed
        assert r.severity == "info"

    def test_nifti_paired_short_epoch_is_flagged(self, tmp_path: Any) -> None:
        # The ULF paired cohort: 32 records at batch_size 1 = a 32-iteration epoch.
        cfg = self._cfg(dataset_type="nifti_paired", batch_size=1)
        cfg.data.index_path = None
        cfg.data.paired_manifest_path = self._manifest(tmp_path, 32)
        r = ConfigHealthChecker().check_persistent_workers_on_short_epoch(cfg)
        assert not r.passed
        assert r.severity == "warning"

    def test_registered_in_run_all_checks(self) -> None:
        import inspect

        src = inspect.getsource(ConfigHealthChecker.run_all_checks)
        assert "check_persistent_workers_on_short_epoch" in src


class TestCurriculumTargetsResolvable:
    """Pre-flight guard for the ``loss_schedule`` base-weight resolution.

    ``mrixfields_field_cocycle_anyfield`` passed the dispatch audit 124/0/0 on
    2026-07-23 and then died 22 minutes into training: its ``ramp_cocycle`` rule
    targets ``cocycle_consistency``, whose weight lived only in
    ``training.field_cocycle.cocycle_weight``. ``LossScheduleController`` resolves a
    rule's base through the loss-weight SSOT, which sees only ``losses.*``, so
    ``resolve_loss_weight`` correctly refused to invent one and raised. The existing
    ``curriculum_targets_consumed`` check passed — the target IS consumed — so
    "consumed" and "resolvable" need separate guards.
    """

    @staticmethod
    def _cfg(targets: list[str], declared: list[tuple[str, float]]):
        return SimpleNamespace(
            loss_schedule=SimpleNamespace(
                enabled=True,
                rules=[SimpleNamespace(name=f"rule_{t}", target=t) for t in targets],
            ),
            losses=SimpleNamespace(
                image_losses=[{"name": n, "weight": w} for n, w in declared],
                kspace_losses=[],
                complex_losses=[],
            ),
        )

    def test_undeclared_target_is_a_hard_error(self) -> None:
        cfg = self._cfg(["cocycle_consistency"], [("l1", 1.0)])
        r = ConfigHealthChecker().check_curriculum_targets_resolvable(cfg)
        assert not r.passed
        assert r.severity == "error"
        assert "cocycle_consistency" in r.message

    def test_declaring_the_target_clears_the_error(self) -> None:
        cfg = self._cfg(["cocycle_consistency"], [("l1", 1.0), ("cocycle_consistency", 0.1)])
        r = ConfigHealthChecker().check_curriculum_targets_resolvable(cfg)
        assert r.passed, r.message

    def test_target_with_a_schema_lambda_default_resolves(self) -> None:
        # `adversarial` has a `lambda_adv` schema field, so it resolves without an
        # explicit image_losses entry — this must NOT be reported.
        cfg = self._cfg(["adversarial"], [("l1", 1.0)])
        assert ConfigHealthChecker().check_curriculum_targets_resolvable(cfg).passed

    def test_no_schedule_is_informational_not_a_failure(self) -> None:
        cfg = SimpleNamespace(loss_schedule=None, losses=None)
        r = ConfigHealthChecker().check_curriculum_targets_resolvable(cfg)
        assert r.passed and r.severity == "info"

    def test_disabled_schedule_is_not_checked(self) -> None:
        cfg = self._cfg(["cocycle_consistency"], [("l1", 1.0)])
        cfg.loss_schedule.enabled = False
        assert ConfigHealthChecker().check_curriculum_targets_resolvable(cfg).passed

    def test_reports_every_unresolvable_rule_not_just_the_first(self) -> None:
        cfg = self._cfg(["cocycle_consistency", "seg_consistency"], [("l1", 1.0)])
        r = ConfigHealthChecker().check_curriculum_targets_resolvable(cfg)
        assert not r.passed
        assert "cocycle_consistency" in r.message and "seg_consistency" in r.message


# ---------------------------------------------------------------------------
# Synthesised-stack strategies (2026-07-26)
# ---------------------------------------------------------------------------
def _subvoxel_cfg(in_channels: int = 24, n_frames: int = 8, marker_channels: bool = False):
    """A subvoxel_sr arm on an IMAGE dataset, where the coil arithmetic bites."""
    import pathlib
    import tempfile

    import yaml

    from spectramr.config.settings import TrainingSettings

    src = require_repo_file("experiments/inprogress/vf/exp_vf_01_subvoxel_superres_v2.yaml")
    raw = yaml.safe_load(src.read_text())
    raw["model"]["in_channels"] = in_channels
    macq = raw["physics"]["multi_acquisition"]
    macq["n_frames"] = n_frames
    if marker_channels:
        macq["marker_channels"] = True
        macq["subvoxel_registration"]["shift_source"] = "recovered"
    tmp = pathlib.Path(tempfile.mkdtemp()) / "arm.yaml"
    tmp.write_text(yaml.safe_dump(raw))
    return TrainingSettings.from_yaml(str(tmp))


def test_synthesised_stack_width_is_checked_not_skipped() -> None:
    """``ConcreteMultiAcquisitionStrategy`` never feeds the loaded batch to the
    generator, so the coil arithmetic describes a tensor the model never sees.

    Until 2026-07-26 these arms sat on ``coil_processing_mode: none`` +
    ``dataset_type: kspace``, where the count is file-dependent and the check
    skipped. Moving them to an image dataset made it derive 1 channel and reject
    a correct ``in_channels`` of 24. Checking the RIGHT quantity beats skipping.
    """
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    results = ConfigHealthChecker().check_domain_alignment(_subvoxel_cfg(24, 8))
    assert results and all(r.passed for r in results)
    assert all(r.severity == "info" for r in results)
    assert "24 channels" in results[0].message


def test_wrong_synthesised_stack_width_is_an_error() -> None:
    """The audit must catch at load time what the strategy raises on at setup."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    results = ConfigHealthChecker().check_domain_alignment(_subvoxel_cfg(8, 8))
    assert results and not results[0].passed
    assert results[0].severity == "error"


def test_fiducial_channels_widen_the_expected_stack() -> None:
    """With the marker fed to the model the contract is n*4, not n*3."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    ck = ConfigHealthChecker()
    assert ck.check_domain_alignment(_subvoxel_cfg(32, 8, marker_channels=True))[0].passed
    assert not ck.check_domain_alignment(_subvoxel_cfg(24, 8, marker_channels=True))[0].passed


# ---------------------------------------------------------------------------
# Marker relaxometry must bracket what it certifies (2026-07-26)
# ---------------------------------------------------------------------------
def _relax_cfg(t1_src: float, t1_tgt: float):
    from types import SimpleNamespace

    from spectramr.config.schemas.physics import MultiAcquisitionConfig

    acq = {"tr_ms": 500.0, "te_ms": 15.0, "flip_deg": 90.0}
    macq = MultiAcquisitionConfig(
        enabled=True,
        method="subvoxel_sr",
        subvoxel_registration={"shift_source": "recovered"},
        relaxometric_calibration={
            "enabled": True,
            "source": {"field_strength_t": 0.064, **acq},
            "target": {"field_strength_t": 3.0, **acq},
            "marker_t1_ms": t1_src,
            "marker_t1_target_ms": t1_tgt,
            "marker_t2_ms": 80.0,
        },
    )
    return SimpleNamespace(physics=SimpleNamespace(multi_acquisition=macq))


def test_marker_kappa_inside_the_tissue_range_passes() -> None:
    """A single-material marker pins ONE point on the transfer curve; the rest
    is the Bottomley model extrapolated. That is a limit either way, but it is
    only defensible while the measured point brackets the tissues."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    r = ConfigHealthChecker().check_marker_kappa_in_tissue_range(_relax_cfg(500.0, 900.0))
    assert r.passed and r.severity == "info"
    assert "pins ONE point" in r.message


def test_marker_kappa_outside_the_tissue_range_is_an_error() -> None:
    """A short-T1 marker gives kappa 0.9998 against a tissue range of
    [0.453, 0.865] — it calibrates by extrapolation, so a kappa verified on it
    says nothing about any tissue the network has to translate. Nothing
    rejected that before 2026-07-26."""
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    r = ConfigHealthChecker().check_marker_kappa_in_tissue_range(_relax_cfg(50.0, 60.0))
    assert not r.passed and r.severity == "error"
    assert "OUTSIDE the range" in r.message


def test_kappa_check_is_inert_when_calibration_is_off() -> None:
    from types import SimpleNamespace

    from spectramr.config.schemas.physics import MultiAcquisitionConfig
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    cfg = SimpleNamespace(
        physics=SimpleNamespace(
            multi_acquisition=MultiAcquisitionConfig(enabled=True, method="afi")
        )
    )
    assert ConfigHealthChecker().check_marker_kappa_in_tissue_range(cfg).passed


def test_marker_kappa_check_is_invoked_by_run_all_checks():
    """Assert the SEAM, not the unit.

    `check_marker_kappa_in_tissue_range` had three tests that called it directly,
    so they proved the check works while `run_all_checks` never invoked it. It had
    therefore never rejected anything. Calling the method is not evidence the
    pipeline calls it, so this test inspects the driver instead.

    Found by `meta.health_checker_no_orphan_checks`.
    """
    from spectramr.infrastructure.validation.witness.checks.meta_orphan_checks import (
        invoked_check_methods,
    )

    assert "check_marker_kappa_in_tissue_range" in invoked_check_methods(), (
        "the check is defined but run_all_checks does not call it, so it protects nothing"
    )


class TestLossDomainBlockMatch:
    """``check_loss_domain_block_match``, rerouted through the capability layer.

    It used to read ``LossRegistry._loss_domains`` raw and carry its own
    ``{"complex_losses": "complex"}`` map. That agreed with the registry's
    adapter only by coincidence -- both happened to spell it ``complex`` on the
    raw side -- so one fact had two sources that were merely coincident.

    Every case below asserts the check CHANGES its answer, not that it returns
    ``passed=True``: an absent block and a valid one both pass, so a test that
    only reads the boolean cannot tell a working check from a skipped one.
    """

    @staticmethod
    def _settings(losses: dict[str, Any]) -> Any:
        from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
        from spectramr.config.settings import TrainingSettings

        return TrainingSettings.settings_from_dict(
            {
                "config_version": CANONICAL_CONFIG_VERSION,
                "model": {"model_type": "unet"},
                "data": {"dataset_type": "image"},
                "optimization": {},
                "logging": {},
                "losses": losses,
            }
        )

    def _verdicts(self, losses: dict[str, Any]) -> list[tuple[bool, str]]:
        results = ConfigHealthChecker().check_loss_domain_block_match(self._settings(losses))
        return [(r.passed, r.message) for r in results]

    def test_a_latent_loss_under_image_losses_is_rejected(self) -> None:
        """Direction one. ``physics_equivariance`` is registered
        ``domain='latent'``."""
        verdicts = self._verdicts(
            {
                "output_domain": "image",
                "image_losses": [{"name": "gw_cross_field", "weight": 1.0}],
            }
        )
        assert any(not passed for passed, _ in verdicts), verdicts
        assert any("gw_cross_field" in msg for _, msg in verdicts)

    def test_the_same_loss_under_latent_losses_is_accepted(self) -> None:
        """Direction two. A check that only ever fires proves as little as one
        that never does."""
        verdicts = self._verdicts(
            {
                "output_domain": "latent",
                "latent_losses": [{"name": "gw_cross_field", "weight": 1.0}],
            }
        )
        assert all(passed for passed, _ in verdicts), verdicts

    def test_a_generic_loss_is_legal_in_every_block(self) -> None:
        """The corpus declares ``l1`` under ``image_losses`` 334 times and under
        ``kspace_losses`` 7 times. Both are correct -- ``l1`` is registered
        ``domain='agnostic'`` and imposes no placement constraint.

        This is the branch that was passing incidentally: "agnostic" failed to
        map onto the ``Domain`` literal and arrived as ``None``, so the generic
        losses were surviving on the unannotated escape hatch.
        """
        for block, domain in (
            ("image_losses", "image"),
            ("kspace_losses", "kspace"),
            ("complex_losses", "complex_image"),
        ):
            verdicts = self._verdicts(
                {"output_domain": domain, block: [{"name": "l1", "weight": 1.0}]}
            )
            assert all(passed for passed, _ in verdicts), (block, verdicts)

    def test_the_latent_list_is_actually_inspected(self) -> None:
        """``latent_losses`` is new; if the check's block map missed it, entries
        there would be graded against nothing."""
        from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS

        assert "latent_losses" in LOSS_LIST_DOMAINS
        # `hermitian_symmetry` is registered domain='kspace'. Deliberately not
        # `log_spectral`, which reads as k-space from its name but is one of the
        # 104 unannotated losses -- it soft-skips, so it would have made this
        # test pass while proving nothing.
        verdicts = self._verdicts(
            {
                "output_domain": "latent",
                "latent_losses": [{"name": "hermitian_symmetry", "weight": 1.0}],
            }
        )
        assert any(not passed for passed, _ in verdicts), (
            "a kspace loss under latent_losses was not rejected, so the new "
            f"list is not being inspected: {verdicts}"
        )


class TestMetricNamesAreRegistered:
    """``metrics.compute`` is only safe to prefer over the flags if the names
    in it are checked. A flag naming a nonexistent metric merely does nothing;
    a list entry naming one must be a startup error."""

    @staticmethod
    def _settings(metrics: dict[str, Any]) -> Any:
        from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
        from spectramr.config.settings import TrainingSettings

        return TrainingSettings.settings_from_dict(
            {
                "config_version": CANONICAL_CONFIG_VERSION,
                "model": {"model_type": "unet"},
                "data": {"dataset_type": "image"},
                "optimization": {},
                "logging": {},
                "metrics": metrics,
            }
        )

    def test_a_flagless_registered_metric_is_accepted(self) -> None:
        """``brisque`` and ``auroc`` are registered with no ``compute_*`` flag,
        so they were unreachable from any config before the list existed."""
        r = ConfigHealthChecker().check_metric_names_are_registered(
            self._settings({"compute": ["brisque", "auroc", "psnr"]})
        )
        assert r.passed, r.message

    def test_an_unregistered_name_is_rejected(self) -> None:
        """``advanced_metrics`` is the identity target of
        ``compute_advanced_metrics`` -- a flag 249 arms set, defaulting True,
        that no code reads. As a list entry it must not pass quietly."""
        r = ConfigHealthChecker().check_metric_names_are_registered(
            self._settings({"compute": ["psnr", "advanced_metrics"]})
        )
        assert not r.passed
        assert "advanced_metrics" in r.message
        assert r.severity == "error"

    def test_an_arm_on_the_flags_is_skipped(self) -> None:
        """~800 corpus arms have no list; the check must have no opinion on
        them rather than a favourable one."""
        r = ConfigHealthChecker().check_metric_names_are_registered(
            self._settings({"compute_psnr": True})
        )
        assert r.passed
        assert r.severity == "info"
        assert "not used" in r.message

    def test_it_is_wired_into_the_tier1_dispatch(self) -> None:
        """A check nothing calls is worth exactly nothing."""
        import inspect

        src = inspect.getsource(ConfigHealthChecker.run_all_checks)
        assert "check_metric_names_are_registered" in src


class TestDeclaredKeysAreNotDiscarded:
    """Issues #675 / #681: keys an ``extra="ignore"`` block silently drops.

    This class of defect is invisible to every other check in the file, because
    the resolved config is *correct* — the value simply is not in it. The YAML
    advertises a knob (419 arms set ``logging.project_name``, 417 set
    ``enable_wandb``; neither field exists) and the run never sees it.

    The check reads ``ExecutionLedger``, which is a **contextvar**, so the two
    cases that must never be confused are "measured, found none" and "nothing
    armed the ledger, so nobody looked". Both would otherwise render as a clean
    pass — the exact silence this check exists to break.
    """

    @staticmethod
    def _run(config):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        return ConfigHealthChecker().check_declared_keys_are_not_discarded(config)

    def test_no_ledger_reports_not_measured_not_clean(self) -> None:
        from spectramr.core.execution_ledger import ExecutionLedger

        ExecutionLedger.reset()
        assert ExecutionLedger.current() is None
        result = self._run(object())
        assert result.passed is True
        assert "NOT measured" in result.message, (
            "an unarmed ledger must not read as 'no discarded keys' — that is "
            "the silence-reads-as-success defect"
        )

    def test_a_clean_load_says_so_distinctly(self) -> None:
        from spectramr.core.execution_ledger import ExecutionLedger

        ExecutionLedger.begin_run(source="test")
        result = self._run(object())
        assert result.passed is True
        assert "every declared key reached" in result.message
        assert "NOT measured" not in result.message

    def test_a_dropped_key_is_reported_with_its_dotted_path(self) -> None:
        from spectramr.core.execution_ledger import (
            ExecutionLedger,
            SubstitutionClass,
        )

        ledger = ExecutionLedger.begin_run(source="test")
        ledger.record(
            class_id=SubstitutionClass.EXTRA_IGNORE_DROPPED,
            site="test",
            stage="config_finalize",
            path="logging.project_name",
            requested="spectramr_research",
            resolved=None,
            reason="test fixture",
            severity="error",
        )
        result = self._run(object())
        assert "logging.project_name" in result.message
        assert result.yaml_keys == ["logging.project_name"]
        assert result.fix_hint

    def test_it_is_advisory_by_design(self) -> None:
        """~1,279 corpus declarations. A warning would exit 2 under --strict
        (pitfall #10) and fail hundreds of arms for a no-behaviour-change
        finding. Same polarity as ``check_workflow_declared``: report, then
        ratchet once the corpus is drained. Pinned so the ratchet is a decision
        rather than an accident."""
        from spectramr.core.execution_ledger import (
            ExecutionLedger,
            SubstitutionClass,
        )

        ledger = ExecutionLedger.begin_run(source="test")
        ledger.record(
            class_id=SubstitutionClass.EXTRA_IGNORE_DROPPED,
            site="test",
            stage="config_finalize",
            path="logging.enable_wandb",
            requested=True,
            resolved=None,
            reason="test fixture",
            severity="error",
        )
        result = self._run(object())
        assert result.severity == "info"
        assert result.passed is True

    def test_other_substitution_classes_do_not_trigger_it(self) -> None:
        """Anti-vacuity. ``EXTRA_ALLOW_UNTYPED`` is a different finding (the key
        is carried, just unvalidated); counting it here would inflate the number
        and blur two distinct defects."""
        from spectramr.core.execution_ledger import (
            ExecutionLedger,
            SubstitutionClass,
        )

        ledger = ExecutionLedger.begin_run(source="test")
        ledger.record(
            class_id=SubstitutionClass.EXTRA_ALLOW_UNTYPED,
            site="test",
            stage="config_finalize",
            path="training.diffusion.canary",
            requested=1,
            resolved=1,
            reason="test fixture",
            severity="warning",
        )
        result = self._run(object())
        assert "every declared key reached" in result.message


class TestPinMemoryCheckActuallyRuns:
    """`check_pin_memory_no_cuda` short-circuited to a passing "n/a" for EVERY config.

    It read `getattr(config, "device", "")`, but phase 4b moved the key to
    `run.device`, so the string was always `""`, never `"cpu"`, and the guard
    returned before reaching the check. Line 6651 already read the CANONICAL
    `data.loader.pin_memory` -- half the check had been migrated. "Skipped" and
    "passed" are the same boolean here, which is why nothing noticed.
    """

    @staticmethod
    def _settings(device: str, pin_memory: bool):
        from spectramr.config.settings import TrainingSettings

        return TrainingSettings.settings_from_dict(
            {
                "data": {
                    "train_path": "/tmp/t",
                    "val_path": "/tmp/v",
                    "pin_memory": pin_memory,
                },
                "optimization": {"learning_rate": 1e-4},
                "logging": {},
                "model": {"model_type": "unet"},
                "run": {"device": device},
            }
        )

    def test_cpu_plus_pin_memory_now_fires(self):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        result = ConfigHealthChecker().check_pin_memory_no_cuda(self._settings("cpu", True))
        assert result.passed is False
        assert "pin_memory" in result.message

    def test_cuda_is_genuinely_not_applicable(self):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        result = ConfigHealthChecker().check_pin_memory_no_cuda(self._settings("cuda", True))
        assert result.passed is True  # n/a -- pin_memory is correct on CUDA

    def test_cpu_without_pin_memory_passes(self):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        result = ConfigHealthChecker().check_pin_memory_no_cuda(self._settings("cpu", False))
        assert result.passed is True


class TestLegacyConfigVersionIsRefusedByTheLoader:
    """The successor to ``check_config_version_is_canonical``, now deleted.

    That check read the ledger for a ``config_version``
    ``VALUE_CHANGED_ON_FINALIZE`` record emitted by the fold in
    ``_bind_config_version``. With legacy versions refused outright the fold is
    gone, so the record can never appear and the check could only ever return
    "canonical" -- a pass that measures nothing (pitfall #16).

    What replaced it is strictly stronger and one layer earlier: a legacy
    version cannot produce a ``TrainingSettings`` at all, so there is no
    resolved config left for any audit to run on. These pin that, and pin that
    the deleted check does not quietly come back.
    """

    @staticmethod
    def _write(tmp_path: Any, declared: str) -> Any:
        import yaml

        doc = {
            "config_version": declared,
            "data": {"train_path": "/tmp/t", "val_path": "/tmp/v"},
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
        }
        path = tmp_path / f"arm_{declared.replace('.', '_')}.yaml"
        path.write_text(yaml.dump(doc))
        return path

    @pytest.mark.parametrize("declared", ["6.0", "6.1"])
    def test_a_legacy_version_cannot_be_loaded_at_all(self, tmp_path, declared):
        from spectramr.config.settings import TrainingSettings

        with pytest.raises(ValueError, match="not supported"):
            TrainingSettings.from_yaml(self._write(tmp_path, declared))

    def test_the_canonical_version_still_loads(self, tmp_path):
        """Anti-vacuity. Without this, a loader broken for EVERY version would
        satisfy the refusal test above and still look like a working ratchet."""
        from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
        from spectramr.config.settings import TrainingSettings

        settings = TrainingSettings.from_yaml(self._write(tmp_path, CANONICAL_CONFIG_VERSION))
        assert settings.run.config_version == CANONICAL_CONFIG_VERSION

    def test_the_dead_check_is_gone(self):
        """Asserted via ``hasattr``, not by scanning ``run_all_checks`` source.

        The call site keeps a comment naming the deleted method, so a source
        scan would match the comment and pass whether or not the method exists.
        """
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        assert not hasattr(ConfigHealthChecker, "check_config_version_is_canonical")


class TestM4RawNexTargetModeDeclared:
    """An m4raw arm silent on `data.target_mode` grades against a degraded target.

    `complex_mean` is the schema default and is right for every dataset EXCEPT
    M4Raw, whose repetitions are phase-incoherent: complex-averaging them cancels
    signal, so the "high-SNR NEX reference" lands BELOW one repetition in SNR with
    scrambled phase.

    Measured 2026-08-03 over `git ls-files experiments` -- stating the roots,
    because a count whose scope is unstated is how this repo produced four
    different wrong migration numbers: 110 arms declare `dataset_type: m4raw`,
    107 resolve (3 unloadable), of which **63 pass and 44 are flagged**.
    """

    @staticmethod
    def _load(tmp_path, name: str, **data_overrides):
        import yaml

        from spectramr.config.settings import TrainingSettings

        data = {"train_path": "/tmp/t", "val_path": "/tmp/v"}
        data.update(data_overrides)
        path = tmp_path / f"{name}.yaml"
        path.write_text(
            yaml.dump(
                {
                    "config_version": "1.0",
                    "data": data,
                    "optimization": {"learning_rate": 1e-4},
                    "logging": {},
                    "model": {"model_type": "unet"},
                }
            )
        )
        return TrainingSettings.from_yaml(path)

    @staticmethod
    def _check(settings):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        return ConfigHealthChecker().check_m4raw_nex_target_mode_declared(settings)

    def test_m4raw_arm_silent_on_target_mode_is_flagged(self, tmp_path):
        settings = self._load(tmp_path, "silent", dataset_type="m4raw")
        # The tell: the RESOLVED value cannot distinguish this arm from one that
        # chose complex_mean deliberately. Only model_fields_set can.
        assert settings.data.target_mode == "complex_mean"
        assert "target_mode" not in settings.data.model_fields_set

        result = self._check(settings)
        assert result.passed is False
        assert result.severity == "info"  # advisory: 47 arms would newly fail
        assert "does not declare" in result.message
        assert result.yaml_keys == ["data.target_mode"]
        assert "phase_aligned_mean" in result.fix_hint

    def test_phase_aligned_mean_passes(self, tmp_path):
        settings = self._load(
            tmp_path, "aligned", dataset_type="m4raw", target_mode="phase_aligned_mean"
        )
        result = self._check(settings)
        assert result.passed is True
        assert "phase_aligned_mean" in result.message

    def test_explicit_complex_mean_reads_differently_from_silence(self, tmp_path):
        """A default nobody chose and an affirmative choice are two findings.

        Resolution collapses them -- both read `complex_mean`. Reporting them
        identically would tell an owner triaging the 47 silent arms nothing about
        which ones were a deliberate legacy comparison.
        """
        chosen = self._load(tmp_path, "chosen", dataset_type="m4raw", target_mode="complex_mean")
        silent = self._load(tmp_path, "silent2", dataset_type="m4raw")
        assert chosen.data.target_mode == silent.data.target_mode == "complex_mean"

        chosen_msg = self._check(chosen).message
        silent_msg = self._check(silent).message
        assert chosen_msg != silent_msg
        assert "explicitly" in chosen_msg
        assert "does not declare" in silent_msg

    def test_non_m4raw_arm_is_not_flagged(self, tmp_path):
        """complex_mean is the correct default everywhere else -- do not nag."""
        settings = self._load(tmp_path, "kspace", dataset_type="kspace")
        result = self._check(settings)
        assert result.passed is True
        assert "n/a" in result.message

    def test_the_check_is_wired_into_run_all_checks(self):
        import inspect

        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        src = inspect.getsource(ConfigHealthChecker.run_all_checks)
        assert "check_m4raw_nex_target_mode_declared" in src


class TestComponentKwargsReachConstructor:
    """Model kwargs that never reach a constructor must be visible, not fatal.

    Scope note: after the loss builder was made to RAISE, this check does NOT
    cover loss kwargs. It surfaces the two families that legitimately cannot
    raise -- model kwargs dropped by the signature filter, and keys accepted
    into a `**kwargs` constructor with nothing proving they are read (#878).

    Advisory because the model-kwargs corpus is UNMEASURED. Contrast the loss
    builder, which raises precisely because its violation count is a measured 0.
    A `warning` here would exit 2 under the strict smoke wrapper (pitfall #10)
    on a population nobody has counted.
    """

    @staticmethod
    def _run(config):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        return ConfigHealthChecker().check_component_kwargs_reach_constructor(config)

    def test_no_ledger_reports_not_measured_not_clean(self) -> None:
        from spectramr.core.execution_ledger import ExecutionLedger

        ExecutionLedger.reset()
        assert ExecutionLedger.current() is None
        result = self._run(object())
        assert result.passed is True
        assert "NOT measured" in result.message

    def test_a_clean_build_says_so_distinctly(self) -> None:
        from spectramr.core.execution_ledger import ExecutionLedger

        ExecutionLedger.begin_run(source="test")
        result = self._run(object())
        assert result.passed is True
        assert "NOT measured" not in result.message

    def test_a_dropped_kwarg_is_advisory_and_names_the_consumer(self) -> None:
        from spectramr.core.execution_ledger import ExecutionLedger, unconsumed_keys

        ExecutionLedger.begin_run(source="test")
        unconsumed_keys(
            {"nowhere": 1},
            {"alpha"},
            site="test",
            stage="model_build",
            consumer="SomeGenerator.__init__",
        )
        result = self._run(object())
        assert result.passed is True, "advisory until the corpus is measured"
        assert result.severity == "info"
        assert "nowhere" in result.message
        assert "SomeGenerator" in result.message

    def test_var_kwargs_acceptance_is_reported_too(self) -> None:
        """Accepted-into-**kwargs is not consumption; it must not read clean."""
        from spectramr.core.execution_ledger import (
            ExecutionLedger,
            SubstitutionClass,
            unconsumed_keys,
        )

        ExecutionLedger.begin_run(source="test")
        unconsumed_keys(
            {"unverified": 1},
            {"alpha"},
            site="test",
            stage="model_build",
            consumer="Swallower.__init__(**kwargs)",
            class_id=SubstitutionClass.EXTRA_ALLOW_UNTYPED,
        )
        result = self._run(object())
        assert result.passed is True
        assert "unverified" in result.message

    def test_unrelated_substitution_classes_are_ignored(self) -> None:
        """`EXTRA_IGNORE_DROPPED` belongs to the sibling check; counting it here
        would double-report every one of the corpus's ~1,279 hits."""
        from spectramr.core.execution_ledger import ExecutionLedger, SubstitutionClass

        ledger = ExecutionLedger.begin_run(source="test")
        ledger.record(
            class_id=SubstitutionClass.EXTRA_IGNORE_DROPPED,
            site="test",
            stage="parse",
            path="data.someone_elses_key",
            requested=1,
            resolved=None,
            reason="belongs to check_declared_keys_are_not_discarded",
        )
        result = self._run(object())
        assert "someone_elses_key" not in result.message


class TestAccelerationPresentAdvisesTheCanonicalBlock:
    """The check READS the canonical block; its remediation must NAME it too.

    ``check_acceleration_present`` has always read ``config.undersampling``,
    but its message, ``yaml_keys`` and ``fix_hint`` still named the
    pre-2026-08-02 ``acceleration:``. A user following that advice wrote a
    retired spelling -- which LOADS (the rename's posture is ``fold``), and
    that is precisely why nothing caught it: the guidance was wrong in the one
    direction that never produces a failure.

    ``yaml_keys`` is user-facing rather than an auto-fix key: ``cli/app.py``
    prints it as ``keys: ...`` and serialises it into the audit JSON.
    """

    @staticmethod
    def _kspace_arm_without_undersampling():
        return SimpleNamespace(
            data=SimpleNamespace(dataset_type="kspace"),
            undersampling=None,
        )

    def test_it_still_fires(self) -> None:
        """Guard the guard: the advice fix must not have silenced the check."""
        result = ConfigHealthChecker().check_acceleration_present(
            self._kspace_arm_without_undersampling()
        )
        assert result.passed is False
        assert result.severity == "error"

    def test_the_remediation_names_only_the_canonical_block(self) -> None:
        result = ConfigHealthChecker().check_acceleration_present(
            self._kspace_arm_without_undersampling()
        )
        advice = f"{result.message}\n{result.fix_hint}"
        assert list(result.yaml_keys) == ["undersampling"]
        assert "undersampling:" in advice
        # The retired spelling must not be suggested as a block to ADD. Bare
        # "acceleration" still appears legitimately (`base_acceleration`), so
        # key on the block form.
        assert "\nacceleration:" not in advice
        assert "top-level acceleration:" not in advice

    def test_a_declared_block_passes(self) -> None:
        result = ConfigHealthChecker().check_acceleration_present(
            SimpleNamespace(
                data=SimpleNamespace(dataset_type="kspace"),
                undersampling=SimpleNamespace(base_acceleration=4),
            )
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# #931 — the check granted a pass to any arm declaring the knob
# ---------------------------------------------------------------------------


class TestMetricDomainMatchesLossOutput:
    """`check_metric_domain_matches_loss_output` must ask the resolver.

    It used to ask `bool(metrics.transform)` — so declaring the knob asserted
    the domains were bridged, without checking that the named transform exists
    or that a bridge was even needed. A facade (pitfall #16) inside the audit.
    """

    @staticmethod
    def _config(*, loss_domain="kspace", metric_domain="image", **metrics_kwargs):
        metrics_kwargs.setdefault("transform", None)
        return SimpleNamespace(
            losses=SimpleNamespace(policy=SimpleNamespace(output_domain=loss_domain)),
            metrics=SimpleNamespace(domain=metric_domain, **metrics_kwargs),
            validation=block_stub("validation"),
        )

    def test_a_transform_the_dispatcher_cannot_execute_fails_the_audit(self):
        """The 146 arms declaring `ifft_mag_combine` / `ifft_mag`.

        No branch dispatches either, so before this change they silently graded
        untransformed tensors. Now the run raises, and the audit says so
        pre-flight instead of letting it die mid-epoch.
        """
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(
            self._config(transform="ifft_mag_combine")
        )
        assert not result.passed
        assert result.severity == "error"
        assert "not implemented" in result.message

    def test_the_fix_hint_refuses_to_guess_the_intent(self):
        """Aliasing to `ifft_magnitude` is the wrong repair for 112 of the 146.

        Those have `losses.output_domain` and `infer_output_domain` both
        `image`; an IFFT there gives a Fourier magnitude, not a coil combine.
        The hint must say so rather than proposing a one-line sed.
        """
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(
            self._config(transform="ifft_mag_combine")
        )
        assert "Do NOT assume" in (result.fix_hint or "")

    def test_a_losing_declaration_is_still_checked(self):
        """`metrics.transform` loses precedence on the validation path.

        But `_compute_training_metrics` dispatches from that very block, so an
        undispatchable name there is still a live defect and must be reported
        even when `output_transform` outranks it.
        """
        config = self._config(transform="ifft_mag_combine")
        config.validation = block_stub("validation", output_transform="ifft_magnitude")
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(config)
        assert not result.passed

    def test_a_transform_firing_where_domains_agree_is_a_finding(self):
        """The inverse facade.

        `bridged = name is not None` would certify an arm whose metric and loss
        domains ALREADY agree and which still runs an IFFT — no mismatch
        existed, and the transform moves the metric out of the shared domain.
        Exactly the 112-arm shape that made aliasing unsafe.
        """
        config = self._config(loss_domain="image", metric_domain="image")
        config.validation = block_stub("validation", output_transform="ifft_magnitude")
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(config)
        assert "already agree" in result.message

    def test_unbridged_mismatch_is_still_reported(self):
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(self._config())
        assert "may grade the wrong physical quantity" in result.message
        assert result.severity == "info"

    def test_the_fix_hint_names_the_knob_the_validation_path_reads(self):
        """It used to recommend `metrics.transform`.

        That knob is read by the training-metrics path only, so recommending it
        for a validation-domain mismatch is how 236 declarations accumulated
        against a key that could not fix the reported problem.
        """
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(self._config())
        assert "validation.scoring.output_transform" in (result.fix_hint or "")

    def test_the_none_sentinel_is_not_a_transform(self):
        """`metrics.transform: 'none'` bridges nothing — it IS the absence."""
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(
            self._config(transform="none")
        )
        assert result.passed
        assert "may grade the wrong physical quantity" in result.message

    def test_a_real_transform_still_counts_as_a_bridge(self):
        config = self._config(transform=None)
        config.validation = block_stub("validation", output_transform="ifft_sense_adjoint")
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(config)
        assert result.passed
        assert "bridged" in result.message

    def test_a_dispatchable_metrics_transform_still_bridges(self):
        """This check is about `metrics.domain`, and the path that reads it —
        `_compute_training_metrics` — is the one that dispatches
        `metrics.transform`. So the key DOES bridge here, even though the
        validation path ignores it. Only an *undispatchable* name may not.
        """
        result = ConfigHealthChecker().check_metric_domain_matches_loss_output(
            self._config(transform="ifft_sense_adjoint")
        )
        assert result.passed
        assert "bridged" in result.message


# ---------------------------------------------------------------------------
# #933 — check_physics_config asked `physics is None`, which the schema
# makes unsatisfiable (PhysicsConfigSchema is always constructed). Re-aimed
# at the answerable question: physics present but INERT (data consistency
# disabled AND no undersampling/sampling-mask block) on a k-space
# reconstruction/diffusion arm.
#
# Lands at severity="info" (advisory-first), NOT "error": a 2026-08-12
# blast-radius measurement found the predicate genuinely fires on only
# 2/120 applicable experiments/inprogress arms, but at least one of those
# two (workflow_baselines/b0_structural_denoise_m4raw.yaml) is a documented,
# deliberate no-physics denoising control, not a bug -- its own header says
# "No acceleration, no physics block, no DC, no coil maps, no adapters".
# A blanket workflow.task=="denoising" exemption would not be correct
# either (the second flagged arm shares the posture under
# workflow.task=="reconstruction"), so this stays advisory rather than
# guessing a task-vocabulary exemption. See config_health_checker.py's
# check_physics_config docstring for the full reasoning.
# ---------------------------------------------------------------------------


def _kspace_recon_settings(*, data_consistency_enabled: bool, sampling_mask: str | None) -> Any:
    """Minimal config-shaped object for check_physics_config.

    ``sampling_mask`` maps onto ``undersampling.sampling_pattern``
    (``config/schemas/acceleration.py``, the v6.1 alias for the dominant
    Cartesian sampling pattern) when given; ``None`` means no
    ``undersampling:`` block was declared at all — the same "block absence"
    signal ``check_acceleration_present`` reads.
    """
    return SimpleNamespace(
        training=SimpleNamespace(
            strategy_class=(
                "spectramr.infrastructure.training.strategies."
                "reconstruction.ReconstructionTrainingStrategy"
            )
        ),
        data=SimpleNamespace(
            dataset_type="kspace",
            coils=SimpleNamespace(processing_mode="rss"),
        ),
        physics=SimpleNamespace(data_consistency=SimpleNamespace(enabled=data_consistency_enabled)),
        undersampling=(
            None if sampling_mask is None else SimpleNamespace(sampling_pattern=sampling_mask)
        ),
    )


def test_physics_check_flags_an_inert_physics_block() -> None:
    """A k-space recon arm with DC disabled and no sampling mask has inert physics.

    The old gate asked `physics is None`, which the schema makes unsatisfiable. This
    asks the answerable question, so it must be capable of a non-passing result.
    """
    settings = _kspace_recon_settings(data_consistency_enabled=False, sampling_mask=None)
    results = ConfigHealthChecker().check_physics_config(settings)
    assert results, "check emitted nothing at all -- it is still structurally dead"
    assert any(not r.passed for r in results)


def test_physics_check_inert_finding_is_advisory_not_blocking() -> None:
    """The inert-physics finding must be severity="info" -- NOT "error"/"warning".

    A repointed error/warning-severity landing would false-positive on
    documented no-physics denoising/restoration controls (e.g.
    workflow_baselines/b0_structural_denoise_m4raw.yaml, whose own header
    says "No acceleration, no physics block, no DC, no coil maps, no
    adapters"), reproducing the exact failure mode check_legacy_schema_mixing
    was deleted for: firing on legitimate, documented, intentional use.
    `passed` stays False -- the structural finding (DC off, no undersampling
    block) is definite -- but `severity="info"` keeps it out of both
    HealthCheckReport.passed/.warnings (single-file audit) and the bulk-mode
    n_errors/n_warnings tally, so it never blocks --strict.
    """
    settings = _kspace_recon_settings(data_consistency_enabled=False, sampling_mask=None)
    results = ConfigHealthChecker().check_physics_config(settings)
    inert = [r for r in results if not r.passed]
    assert inert, "the inert case must still produce a non-passing result"
    assert all(r.severity == "info" for r in inert)


def test_physics_check_passes_and_says_so_when_physics_is_live() -> None:
    """The passed=True fallback: 'ran and found nothing' must be distinguishable."""
    settings = _kspace_recon_settings(data_consistency_enabled=True, sampling_mask="equispaced")
    results = ConfigHealthChecker().check_physics_config(settings)
    assert results and all(r.passed for r in results)


def test_physics_check_is_a_noop_off_the_recon_diffusion_kspace_path() -> None:
    """Anti-vacuity: the applicability gate must still exclude non-k-space arms."""
    settings = SimpleNamespace(
        training=SimpleNamespace(strategy_class="some_other_strategy"),
        data=SimpleNamespace(
            dataset_type="nifti_paired", coils=SimpleNamespace(processing_mode=None)
        ),
        physics=SimpleNamespace(data_consistency=SimpleNamespace(enabled=False)),
        undersampling=None,
    )
    assert ConfigHealthChecker().check_physics_config(settings) == []


class TestDiffusionStepChecksReadTheCanonicalField:
    """The three step-count checks resolve `timesteps`, not the retired name.

    Each used to be `getattr(diff_cfg, "timesteps", None) or getattr(diff_cfg,
    "num_timesteps", None)`. The fallback half was dead — no schema carries
    `num_timesteps` — but unlike the reads in `forward_probe` and
    `quality_matching_strategy`, these tried the CANONICAL name FIRST and were
    therefore correct all along. Removing dead weight should not change an
    answer, so pin the answers.
    """

    def test_no_schema_carries_the_retired_field(self) -> None:
        """The premise of the deletion, asserted rather than assumed."""
        import importlib
        import pkgutil

        import spectramr.config.schemas as schemas

        carriers = []
        for mod_info in pkgutil.walk_packages(schemas.__path__, schemas.__name__ + "."):
            try:
                module = importlib.import_module(mod_info.name)
            except Exception:
                continue
            for attr in dir(module):
                fields = getattr(getattr(module, attr, None), "model_fields", None)
                if isinstance(fields, dict) and "num_timesteps" in fields:
                    carriers.append(f"{mod_info.name}.{attr}")
        assert not carriers, f"a schema carries `num_timesteps` again: {carriers}"

    def test_the_acceleration_block_carries_schedule_steps(self) -> None:
        """Anti-vacuity for the sibling deletion: `accel` genuinely has this
        field, so reading it alone is not the same as reading nothing."""
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        assert "schedule_steps" in AccelerationConfigSchema.model_fields


class TestFatalHealthChecks:
    """``FATAL_HEALTH_CHECKS`` is the checker's declaration of which failures
    are terminal, so the pipeline no longer hardcodes a check name.

    The set is load-bearing in one direction only: adding a name makes a run
    that used to warn-and-continue abort instead. Both current members are
    checks the builder raises on regardless, so membership changes WHERE the
    run dies, never WHETHER -- which is what makes them safe to promote.
    """

    def test_the_deepspeed_extra_check_is_fatal(self):
        """It was not, and its own message described the resulting waste."""
        from spectramr.infrastructure.validation.config_health_checker import (
            FATAL_HEALTH_CHECKS,
        )

        assert "deepspeed_extra_installed" in FATAL_HEALTH_CHECKS

    def test_the_set_is_small_and_deliberate(self):
        """~150 checks return severity='error'; promoting them wholesale would
        turn the gate into a blanket refusal of configs that genuinely train.

        This is a tripwire, not a style rule: if the set grows past a handful,
        someone is using it as a severity synonym rather than a certainty claim.
        """
        from spectramr.infrastructure.validation.config_health_checker import (
            FATAL_HEALTH_CHECKS,
        )

        assert len(FATAL_HEALTH_CHECKS) <= 5, (
            "FATAL_HEALTH_CHECKS is for checks whose failure makes the run "
            "IMPOSSIBLE, not for every error-severity finding"
        )

    def test_a_missing_deepspeed_extra_is_reported_as_an_error(self, monkeypatch):
        """The check's own polarity, independent of the pipeline wiring.

        A stub config rather than a real ``TrainingSettings``: the check reads
        only ``config.parallel.strategy`` through ``getattr``, and building a
        schema-valid deepspeed config means satisfying a cross-field validator
        (``strategy: deepspeed`` requires ``deepspeed.enabled: true``) that has
        nothing to do with whether the package is importable.
        """
        import importlib.util
        from types import SimpleNamespace

        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        cfg = SimpleNamespace(parallel=SimpleNamespace(strategy="deepspeed"))
        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
        result = ConfigHealthChecker().check_deepspeed_extra_installed(cfg)

        assert result.passed is False
        assert result.severity == "error"
        assert result.fix_hint is not None and "deepspeed" in result.fix_hint

    def test_the_check_passes_when_the_package_is_importable(self, monkeypatch):
        """CONTROL: proves the test above is driven by the probe, not by the
        stub config always failing."""
        import importlib.util
        from types import SimpleNamespace

        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        cfg = SimpleNamespace(parallel=SimpleNamespace(strategy="deepspeed"))
        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
        result = ConfigHealthChecker().check_deepspeed_extra_installed(cfg)

        assert result.passed is True


class TestZeroStageHasRanksToShard:
    """ZeRO partitions across ranks; at world size 1 it partitions nothing.

    The check exists because single-rank ZeRO is not neutral -- the engine still
    builds 500 MB reduction buckets, installs the gradient hooks, and logs
    ``zero_stage=2`` as though it were sharding. But its POLARITY is the design
    decision worth pinning, and it is the opposite of what the defect suggests:
    it must never gate. ``parallel.num_devices`` defaults to 1 and is
    OVERWRITTEN from ``LOCAL_WORLD_SIZE`` by ``pipelines/distributed.py`` on any
    ``train-distributed`` launch, so a warning would fire on every
    launcher-driven DeepSpeed arm in the corpus -- and no config edit could
    silence it, because the knob it pointed at is the one the launcher discards.
    """

    @staticmethod
    def _cfg(strategy="deepspeed", stage=2, num_devices=1, num_nodes=1):
        from types import SimpleNamespace

        return SimpleNamespace(
            parallel=SimpleNamespace(
                strategy=strategy,
                num_devices=num_devices,
                num_nodes=num_nodes,
                deepspeed=SimpleNamespace(zero_stage=stage),
            )
        )

    @staticmethod
    def _run(cfg):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        return ConfigHealthChecker().check_deepspeed_zero_stage_has_ranks_to_shard(cfg)

    def test_a_non_deepspeed_arm_is_not_applicable(self):
        result = self._run(self._cfg(strategy="ddp"))
        assert result.passed is True
        assert "n/a" in result.message

    def test_stage_zero_is_not_applicable(self):
        """``zero_stage: 0`` is the deliberate single-rank answer, so flagging it
        would flag the fix this check recommends."""
        result = self._run(self._cfg(stage=0))
        assert result.passed is True
        assert "n/a" in result.message

    @pytest.mark.parametrize(
        ("num_devices", "num_nodes", "world"),
        [(2, 1, 2), (4, 1, 4), (1, 2, 2), (4, 2, 8)],
    )
    def test_a_multi_rank_topology_has_something_to_shard(self, num_devices, num_nodes, world):
        """World size is the PRODUCT; a 1x2 arm shards as surely as a 2x1 one."""
        result = self._run(self._cfg(num_devices=num_devices, num_nodes=num_nodes))
        assert result.passed is True
        assert str(world) in result.message

    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_every_stage_is_flagged_at_one_rank(self, stage):
        """Stage 1/2/3 all divide by world size, so all three are inert at 1."""
        result = self._run(self._cfg(stage=stage))
        assert result.category == "deepspeed_zero_stage_inert_at_single_rank"
        assert "partitions nothing" in result.message

    def test_the_advisory_never_gates(self):
        """THE assertion. ``HealthCheckReport.passed`` counts only non-passing
        errors and ``.warnings`` only non-passing warnings, so an info result
        that passes contributes to neither -- which is what keeps a
        launcher-driven arm's audit exit code at 0."""
        from spectramr.infrastructure.validation.config_health_checker import (
            HealthCheckReport,
        )

        result = self._run(self._cfg())
        assert result.passed is True
        assert result.severity == "info"

        report = HealthCheckReport(results=[result])
        assert report.passed is True
        assert report.warnings == []
        assert report.errors == []

    def test_the_advisory_still_says_what_to_do(self):
        """``__rich__`` renders ``fix_hint`` only when the result does NOT pass,
        so on a passing advisory the guidance must be in the message or it
        reaches --json and nothing else."""
        result = self._run(self._cfg())
        assert "zero_stage: 0" in result.message
        assert "train-distributed" in result.message
        assert result.fix_hint is not None

    def test_only_the_finding_branch_asks_to_be_reported(self):
        """The flag is opt-in per BRANCH, not per check.

        Three of this check's four returns are confirmations -- "not deepspeed",
        "zero_stage=0", "N ranks to partition across". Emitting those is how an
        advisory channel becomes a wall nobody reads: on a real arm 140 of 141
        results are passing ``info`` results, so anything broader than a
        per-branch opt-in buries the finding among its own siblings.
        """
        assert self._run(self._cfg()).always_report is True
        for cfg in (
            self._cfg(strategy="ddp"),
            self._cfg(stage=0),
            self._cfg(num_devices=4),
        ):
            assert self._run(cfg).always_report is False

    def test_the_message_names_the_launch_not_the_allocation(self):
        """``LOCAL_WORLD_SIZE`` is what ``--nproc_per_node`` was given.

        The message used to say ``train-distributed`` overwrites ``num_devices``
        "from the allocation". It does not -- ``pipelines/distributed.py`` reads
        ``LOCAL_WORLD_SIZE``, a fact about the LAUNCH. Conflating the two is
        what let a ``--gpus=4`` allocation run a single rank unremarked (#1274),
        so the correction is load-bearing rather than cosmetic.
        """
        message = self._run(self._cfg()).message
        assert "LOCAL_WORLD_SIZE" in message
        assert "from the allocation" not in message

    def test_the_advisory_survives_log_summary(self, caplog):
        """End to end: the operator on the TRAIN path sees the diagnosis.

        The incident run printed ``Config Health: 141/141 checks passed`` and
        not one character of this message, because ``log_summary`` rendered only
        non-passing results. This pins that it no longer can.
        """
        import logging

        from spectramr.infrastructure.validation.config_health_checker import (
            HealthCheckReport,
        )

        report = HealthCheckReport(results=[self._run(self._cfg())])
        with caplog.at_level(
            logging.INFO,
            logger="spectramr.infrastructure.validation.config_health_checker",
        ):
            report.log_summary()

        assert any("partitions nothing" in r.getMessage() for r in caplog.records)
        # ...and it is still not a finding anything gates on.
        assert report.passed is True
        assert report.warnings == []

    def test_the_check_is_wired_into_run_all_checks(self):
        """A check nobody invokes is an unwired knob (non-negotiable #8). The
        meta-orphan witness also guards this; asserting it here means the paired
        test fails at the same commit rather than in a different suite."""
        import ast
        import inspect
        import textwrap

        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        source = textwrap.dedent(inspect.getsource(ConfigHealthChecker.run_all_checks))
        invoked = {
            node.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute) and node.attr.startswith("check_")
        }
        assert "check_deepspeed_zero_stage_has_ranks_to_shard" in invoked


class TestAPassingAdvisoryReachesTheLog:
    """``info`` must mean "logged without gating", never "unloggable" (#1275).

    ``log_summary`` rendered a result only when it did NOT pass. A check that
    returns ``passed=True`` at ``info`` severity -- the polarity that keeps it
    from gating the audit -- was therefore counted in the denominator and
    silently discarded, while ``spectramr audit`` printed the identical result in
    full (``cli/app.py`` prints every result and ``__rich__`` gives a passing
    one a green icon). Legible on one surface, structurally unreachable on the
    other, and neither surface said so.

    Advisory and invisible are different things. ``always_report`` is what lets
    the code express the first without the second.
    """

    @staticmethod
    def _result(**kwargs):
        from spectramr.infrastructure.validation.config_health_checker import (
            HealthCheckResult,
        )

        base = {
            "passed": True,
            "check_name": "demo",
            "message": "THE-ADVISORY-TEXT",
            "severity": "info",
        }
        base.update(kwargs)
        return HealthCheckResult(**base)

    @staticmethod
    def _report(*results):
        from spectramr.infrastructure.validation.config_health_checker import (
            HealthCheckReport,
        )

        return HealthCheckReport(results=list(results))

    @staticmethod
    def _emit(report, caplog):
        import logging

        with caplog.at_level(
            logging.INFO,
            logger="spectramr.infrastructure.validation.config_health_checker",
        ):
            report.log_summary()
        # getMessage(), not .message: the latter is set by a Formatter, so it
        # exists only if something formatted the record first. That holds when
        # this file runs alone and not always in a full-suite ordering, which is
        # how two sibling tests in test_legacy_adapters.py pass in isolation and
        # AttributeError in a wide run.
        return " | ".join(r.getMessage() for r in caplog.records)

    def test_a_flagged_passing_result_is_emitted(self, caplog):
        report = self._report(self._result(always_report=True))
        assert "THE-ADVISORY-TEXT" in self._emit(report, caplog)

    def test_an_unflagged_passing_result_stays_silent(self, caplog):
        """Anti-noise, and the reason the fix is not ``severity == "info"``.

        140 of the 141 results on the arm that prompted this are passing
        ``info`` results, and 16 of those carry a ``category`` -- nearly all of
        them "not applicable" or "check skipped". Either of those gates would
        have traded one invisible finding for sixteen visible non-findings.
        """
        report = self._report(self._result(category="looks_structured"))
        assert "THE-ADVISORY-TEXT" not in self._emit(report, caplog)

    def test_a_failure_is_still_rendered_at_its_severity(self, caplog):
        """The pre-existing branch is untouched: this adds a case, it does not
        change a polarity."""
        report = self._report(
            self._result(passed=False, severity="error", message="THE-ERROR-TEXT")
        )
        text = self._emit(report, caplog)
        assert "THE-ERROR-TEXT" in text
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_reporting_it_moves_no_verdict(self):
        """THE assertion, and the reason this is safe.

        ``passed`` / ``warnings`` / ``errors`` count only NON-passing results,
        so a flagged passing advisory contributes to none of them. Making it
        visible cannot change any surface's exit code, which is what keeps
        ``test_the_advisory_never_gates`` true.
        """
        report = self._report(self._result(always_report=True), self._result(always_report=False))
        assert report.passed is True
        assert report.warnings == []
        assert report.errors == []

    def test_the_flag_reaches_the_json_view(self):
        """``to_dict`` is ``asdict``, so ``--json`` and the ledger see the key.

        Additive and defaulted, so a consumer reading the documented fields is
        unaffected -- but it IS a schema change, and pinning it means a future
        rename fails loudly instead of silently dropping a field the artifact
        already carried.
        """
        assert self._result(always_report=True).to_dict()["always_report"] is True
        assert self._result().to_dict()["always_report"] is False


class TestDeclaredModelKwargsAreRead:
    """``check_declared_model_kwargs_are_read`` -- declared knobs the model never reads.

    Pairs with the detector's own suite in ``test_inert_knobs.py``; this covers
    only the check wrapper: config plumbing, severity, and wiring.
    """

    @staticmethod
    def _cfg(model_type="kspace_cold_diffusion", **kwargs):
        return SimpleNamespace(
            model=SimpleNamespace(model_type=model_type, model_kwargs=dict(kwargs))
        )

    @staticmethod
    def _run(cfg):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        return ConfigHealthChecker().check_declared_model_kwargs_are_read(cfg)

    def test_flags_a_declared_knob_the_model_never_reads(self):
        result = self._run(self._cfg(activation="complex", base_channels=64))
        assert "activation" in result.message
        assert result.yaml_keys == ["model.model_kwargs.activation"]

    def test_does_not_flag_a_knob_the_model_reads(self):
        result = self._run(self._cfg(base_channels=64))
        assert "activation" not in result.message
        assert result.yaml_keys == []

    def test_no_model_kwargs_is_a_clean_pass(self):
        result = self._run(self._cfg())
        assert result.passed is True
        assert result.yaml_keys == []

    def test_unresolvable_model_type_says_not_measured(self):
        """Absence of evidence must not read as evidence of absence."""
        result = self._run(self._cfg(model_type="no_such_model", activation="x"))
        assert result.passed is True
        assert "NOT measured" in result.message

    def test_the_advisory_never_gates(self):
        """90/647 inprogress arms are affected, so this must not exit 2 (pitfall #10)."""
        from spectramr.infrastructure.validation.config_health_checker import (
            HealthCheckReport,
        )

        result = self._run(self._cfg(activation="complex"))
        assert result.passed is True
        assert result.severity == "info"

        report = HealthCheckReport(results=[result])
        assert report.passed is True
        assert report.warnings == []
        assert report.errors == []

    def test_the_advisory_still_says_what_to_do(self):
        result = self._run(self._cfg(activation="complex"))
        assert result.fix_hint is not None
        assert "DELIBERATELY_UNREAD" in result.fix_hint

    def test_the_check_is_wired_into_run_all_checks(self):
        """A check nobody invokes is itself an unwired knob (non-negotiable #8)."""
        import ast
        import inspect
        import textwrap

        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        source = textwrap.dedent(inspect.getsource(ConfigHealthChecker.run_all_checks))
        invoked = {
            node.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute) and node.attr.startswith("check_")
        }
        assert "check_declared_model_kwargs_are_read" in invoked


class TestRepetitionCountIsAchievable:
    """``model_kwargs.num_repetitions`` must match what the data can supply.

    M4Raw ships 3 repetitions for T1/T2 and 2 for FLAIR, so the ``4`` that four
    arms declared is satisfiable by no contrast at all (#1173). Built from the
    REAL schema rather than a SimpleNamespace: the tautology this check had to
    dodge (a materialised ``contrast_map`` default) only exists on the real
    model, so a stub would test a different object than the one that ships.
    """

    @staticmethod
    def _settings(
        *,
        model_kwargs: dict[str, Any] | None = None,
        dataset_type: str = "m4raw",
        contrast_map: dict[str, int] | None = None,
    ) -> Any:
        from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
        from spectramr.config.settings import TrainingSettings

        data: dict[str, Any] = {"dataset_type": dataset_type}
        if contrast_map is not None:
            data["multi_contrast"] = {"contrast_map": contrast_map}
        return TrainingSettings.settings_from_dict(
            {
                "config_version": CANONICAL_CONFIG_VERSION,
                "model": {"model_type": "unet", "model_kwargs": model_kwargs or {}},
                "data": data,
                "optimization": {},
                "logging": {},
            }
        )

    def test_undeclared_is_skipped(self) -> None:
        """Applicability is gated on the same opt-in that gates building the
        layer, so the red count equals the set of arms that actually asked."""
        r = ConfigHealthChecker().check_repetition_count_is_achievable(self._settings())
        assert r.passed and r.severity == "info"

    def test_declared_on_a_dataset_that_serves_no_repetitions_is_rejected(self) -> None:
        """PLANT (shape 1): ``dataset_type: kspace`` is UniversalMRIDataset,
        which has no repetition handling at all -- the knob sizes a fusion layer
        this data can never feed."""
        r = ConfigHealthChecker().check_repetition_count_is_achievable(
            self._settings(model_kwargs={"num_repetitions": 4}, dataset_type="kspace")
        )
        assert not r.passed and r.severity == "error"
        assert "serves no repetitions" in r.message

    def test_a_single_literal_across_heterogeneous_contrasts_is_rejected(self) -> None:
        """PLANT (shape 2): the sharpest outcome. An arm that does not restrict
        its contrasts spans counts {3, 3, 2}, so NO literal is correct -- which
        is why the four arms cannot be fixed by find-and-replace."""
        r = ConfigHealthChecker().check_repetition_count_is_achievable(
            self._settings(model_kwargs={"num_repetitions": 4})
        )
        assert not r.passed and r.severity == "error"
        assert "DIFFER" in r.message

    def test_a_homogeneous_contrast_subset_with_the_right_value_passes(self) -> None:
        """PLANT (shape 3) and the NON-VACUITY guard for contrast resolution.

        If the contrast lookup were vacuous -- as it was when it read the
        nonexistent ``data.contrasts``, or as it would be if it read the
        materialised ``contrast_map`` default by value -- this arm would fall
        into the heterogeneous branch above and fail. Passing here is the proof
        that restricting the contrasts actually reaches the check.
        """
        r = ConfigHealthChecker().check_repetition_count_is_achievable(
            self._settings(
                model_kwargs={"num_repetitions": 3},
                contrast_map={"T1": 0, "T2": 1},
            )
        )
        assert r.passed, r.message

    def test_a_homogeneous_contrast_subset_with_the_wrong_value_is_rejected(
        self,
    ) -> None:
        """PLANT (shape 4): same restriction, wrong count."""
        r = ConfigHealthChecker().check_repetition_count_is_achievable(
            self._settings(
                model_kwargs={"num_repetitions": 4},
                contrast_map={"T1": 0, "T2": 1},
            )
        )
        assert not r.passed and r.severity == "error"
        assert "supplies 3" in r.message

    def test_flair_alone_accepts_two(self) -> None:
        """PLANT (shape 5): the low count is reachable too, so the check is not
        merely 'anything but 3 is wrong'."""
        r = ConfigHealthChecker().check_repetition_count_is_achievable(
            self._settings(model_kwargs={"num_repetitions": 2}, contrast_map={"FLAIR": 0})
        )
        assert r.passed, r.message

    def test_an_unknown_contrast_is_skipped_not_defaulted(self) -> None:
        """PLANT (shape 6): PD's count is documented as "variable", so it has no
        entry in the producer map. A missing key must read as "cannot validate"
        -- never as zero, and never as a default (non-negotiable 3)."""
        r = ConfigHealthChecker().check_repetition_count_is_achievable(
            self._settings(model_kwargs={"num_repetitions": 4}, contrast_map={"PD": 0})
        )
        assert r.passed and r.severity == "info"
        assert "cannot validate" in r.message

    def test_contrast_declaration_is_read_through_model_fields_set(self) -> None:
        """The guard that a BEHAVIOURAL test cannot currently distinguish.

        ``multi_contrast.contrast_map`` defaults to a fully-populated
        ``{T1, T2, FLAIR, PD}`` on every arm, so reading it by value would treat
        every config as having declared all four contrasts. Today that happens
        to yield the same achievable set as "unrestricted" ({3, 2}), because the
        default IS the whole corpus -- so a by-value implementation passes every
        behavioural test in this class. Verified by planting it: all 8 stayed
        green.

        That coincidence is exactly what makes the wrong implementation look
        right, and it evaporates the moment the default changes: a default of
        ``{"T1": 0}`` would make a by-value read report a homogeneous ``{3}`` for
        every arm that never mentioned contrasts, silently converting the
        "no literal is correct" finding into a confident wrong number.

        So this pins the MECHANISM, and says plainly that it is doing so.
        """
        import ast
        import inspect
        import textwrap

        # Read the AST with the docstring stripped, NOT the raw source: the
        # explanatory comment above the code and this test's own docstring both
        # contain the string "model_fields_set", so a substring check on
        # getsource() matches the prose and passes even when the code no longer
        # does it. Verified by planting exactly that -- the first version of this
        # gate stayed green against a by-value implementation.
        fn = ast.parse(
            textwrap.dedent(
                inspect.getsource(ConfigHealthChecker.check_repetition_count_is_achievable)
            )
        ).body[0]
        body = fn.body[1:] if ast.get_docstring(fn) else fn.body
        code = "\n".join(ast.unparse(n) for n in body)
        assert "model_fields_set" in code, (
            "contrast declaration must be resolved through model_fields_set in "
            "CODE, not by reading the materialised contrast_map default"
        )

    def test_the_check_is_registered_in_the_report(self) -> None:
        """A check nobody calls is not a gate (non-negotiable 16)."""
        import inspect

        src = inspect.getsource(ConfigHealthChecker)
        assert "self.check_repetition_count_is_achievable(config)" in src


# ---------------------------------------------------------------------------
# check_dc_knobs_inert_by_method (#1525)
#
# ``inert_knobs.py`` coins "inert by method" using dc_weight-under-hard as its
# worked example and records that nothing reports it. This is the report, so it
# ships with a planted violation per shape AND the negatives that prove it
# discriminates (non-negotiable 15).
# ---------------------------------------------------------------------------


class TestDCKnobsInertByMethod:
    @staticmethod
    def _config(model_kwargs=None, with_physics=True, **dc_overrides):
        from types import SimpleNamespace

        dc = {
            "enabled": True,
            "method": "hard",
            "weight": 1.0,
            "train_noise_level": 0.01,
            "eval_noise_level": 0.005,
            "noise_type": "gaussian",
        }
        dc.update(dc_overrides)
        return SimpleNamespace(
            model=SimpleNamespace(model_kwargs=dict(model_kwargs or {})),
            physics=SimpleNamespace(data_consistency=SimpleNamespace(**dc))
            if with_physics
            else None,
        )

    @staticmethod
    def _run(config):
        from spectramr.infrastructure.validation.config_health_checker import (
            ConfigHealthChecker,
        )

        return ConfigHealthChecker().check_dc_knobs_inert_by_method(config)

    # --- planted violations, one per shape --------------------------------
    def test_reports_dc_weight_declared_in_the_physics_block(self) -> None:
        result = self._run(self._config(method="hard", weight=0.5))
        assert "dc_weight" in result.message
        assert "0.5" in result.message
        assert result.yaml_keys == ["physics.data_consistency.dc_weight"]

    def test_reports_dc_weight_declared_in_model_kwargs(self) -> None:
        """Both declaration sites exist in the corpus; a physics-only check is half blind."""
        result = self._run(
            self._config(with_physics=False, model_kwargs={"dc_method": "hard", "dc_weight": 0.5})
        )
        assert "dc_weight" in result.message

    def test_reports_a_noise_level_under_a_method_that_cannot_read_it(self) -> None:
        result = self._run(self._config(method="soft", train_noise_level=0.09))
        assert "train_noise_level" in result.message

    # --- negatives: it must stay silent ----------------------------------
    # ``yaml_keys`` is the discriminator, not a substring of the message: the
    # CLEAN message is "no declared DC knob is inert ...", so asserting on the
    # word "inert" would pass on both branches and prove nothing.
    def test_silent_when_hard_declares_the_default_weight(self) -> None:
        assert not self._run(self._config(method="hard", weight=1.0)).yaml_keys

    def test_silent_when_soft_declares_a_weight(self) -> None:
        """soft READS dc_weight as lambda_init."""
        assert not self._run(self._config(method="soft", weight=0.5)).yaml_keys

    def test_silent_when_dc_is_disabled(self) -> None:
        result = self._run(self._config(enabled=False, method="hard", weight=0.5))
        assert not result.yaml_keys

    def test_clean_and_dirty_messages_actually_differ(self) -> None:
        """Guards the assertion above from becoming vacuous."""
        clean = self._run(self._config(method="hard", weight=1.0))
        dirty = self._run(self._config(method="hard", weight=0.5))
        assert clean.message != dirty.message
        assert not clean.yaml_keys and dirty.yaml_keys

    # --- severity discipline ---------------------------------------------
    def test_is_advisory_while_the_corpus_is_non_zero(self) -> None:
        """A warning exits 2 under the strict wrapper and would take down 54 arms."""
        result = self._run(self._config(method="hard", weight=0.5))
        assert result.passed is True
        assert result.severity == "info"

    def test_fix_hint_says_delete_not_honour(self) -> None:
        """Honouring dc_weight under hard would convert hard DC into soft DC."""
        hint = self._run(self._config(method="hard", weight=0.5)).fix_hint
        assert "Delete the key" in hint
        assert "replaces" in hint


class TestAccelerationPresentIsScopedToReconstructionFamilies:
    """Cohort review 2026-09-02: an undersampling block on a VAE / GAN / calibration
    arm is the inert declaration ``undersampling_block_is_applied`` refuses, so this
    check must not demand one there. One rule for both checks."""

    @staticmethod
    def _arm():
        return SimpleNamespace(
            data=SimpleNamespace(dataset_type="kspace"),
            undersampling=None,
            workflow=None,
        )

    def test_reconstruction_family_still_fails_without_a_block(self, monkeypatch):
        """The planted violation this check has always caught."""
        from spectramr.infrastructure.training import strategy_factory as sf
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        monkeypatch.setattr(
            sf.TrainingStrategyFactory,
            "get_strategy_class",
            lambda self, c: ReconstructionTrainingStrategy,
        )
        result = ConfigHealthChecker().check_acceleration_present(self._arm())
        assert result.passed is False

    def test_generative_family_passes_without_a_block(self, monkeypatch):
        from spectramr.infrastructure.training import strategy_factory as sf
        from spectramr.infrastructure.training.strategies.vae import VAETrainingStrategy

        monkeypatch.setattr(
            sf.TrainingStrategyFactory, "get_strategy_class", lambda self, c: VAETrainingStrategy
        )
        result = ConfigHealthChecker().check_acceleration_present(self._arm())
        assert result.passed is True and "does not reconstruct" in result.message

    def test_unresolvable_strategy_keeps_the_old_reach(self, monkeypatch):
        from spectramr.infrastructure.training import strategy_factory as sf

        def _boom(self, c):
            raise ValueError("unknown training_mode")

        monkeypatch.setattr(sf.TrainingStrategyFactory, "get_strategy_class", _boom)
        assert ConfigHealthChecker().check_acceleration_present(self._arm()).passed is False


class TestScheduleStepsOnAFullySampledArm:
    """`acceleration_schedule_steps_match` has no schedule to match on a fully
    sampled arm; a base of 1.0 alone is still a ladder to the default max."""

    @staticmethod
    def _cfg(**accel):
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        diffusion = SimpleNamespace(timesteps=1000, model_fields_set={"timesteps"})
        return SimpleNamespace(
            undersampling=AccelerationConfigSchema(**accel),
            training=SimpleNamespace(diffusion=diffusion),
        )

    def test_fully_sampled_declaration_is_skipped(self) -> None:
        result = ConfigHealthChecker().check_acceleration_schedule_steps_match_diffusion(
            self._cfg(base_acceleration=1.0, max_acceleration=1.0)
        )
        assert result.passed is True and "no mask schedule" in result.message

    def test_base_one_alone_still_needs_a_matching_schedule(self) -> None:
        """Planted violation: timesteps 1000 against the default 10000 steps."""
        result = ConfigHealthChecker().check_acceleration_schedule_steps_match_diffusion(
            self._cfg(base_acceleration=1.0)
        )
        assert result.passed is False and result.severity == "error"


class TestConcomitantAtClinicalFieldIsAnAdvisory:
    """A passed result with severity "warning" was invisible to the single-arm
    audit and an ERROR(strict) in the bulk run; it is an info that is reported."""

    @staticmethod
    def _cfg(b0: float):
        return SimpleNamespace(
            physics=SimpleNamespace(field_strength=b0, concomitant=SimpleNamespace(enabled=True))
        )

    def test_three_tesla_is_reported_info(self) -> None:
        result = ConfigHealthChecker().check_concomitant_requires_low_field(self._cfg(3.0))
        assert result.passed is True and result.severity == "info"
        assert getattr(result, "always_report", False) is True
        assert "no-op" in result.message

    def test_ulf_is_appropriate(self) -> None:
        result = ConfigHealthChecker().check_concomitant_requires_low_field(self._cfg(0.064))
        assert result.passed is True and result.severity == "info"


class TestConditionOnInputDoublesTheInput:
    """`condition_on_input: true` concatenates the measurement onto the noised
    target, so the model reads twice the loaded width and emits the target width."""

    @staticmethod
    def _cfg(
        in_channels: int, out_channels: int, *, condition: bool, model_type: str = "hdsf_hld_mamba"
    ):
        return SimpleNamespace(
            model=SimpleNamespace(
                model_type=model_type,
                in_channels=in_channels,
                out_channels=out_channels,
                model_kwargs={},
            ),
            data=SimpleNamespace(
                dataset_type="nifti_paired",
                coils=SimpleNamespace(processing_mode="rss", num_virtual_coils=None),
                pairing=SimpleNamespace(single_contrast=True),
                multi_contrast=SimpleNamespace(enabled=False),
                domain=SimpleNamespace(target_channels=None),
            ),
            training=SimpleNamespace(diffusion=SimpleNamespace(condition_on_input=condition)),
            physics=None,
        )

    def test_two_input_channels_pass_when_the_measurement_is_concatenated(self) -> None:
        results = ConfigHealthChecker().check_domain_alignment(self._cfg(2, 1, condition=True))
        assert _errors(results) == []
        assert any("condition_on_input" in r.message for r in results)

    def test_the_same_widths_fail_without_the_concat(self) -> None:
        """Planted violation: in_channels=2 on a 1-channel loader with no conditioning."""
        results = ConfigHealthChecker().check_domain_alignment(self._cfg(2, 1, condition=False))
        assert [r.check_name for r in _errors(results)] == ["domain_alignment"]

    def test_cold_and_latent_arms_do_not_concatenate(self) -> None:
        """The strategy's gate reads model.model_type, so the predicate does too.

        Tested on the predicate: a ``kspace_cold_diffusion`` arm reaches the
        relaxed input check first (the S-map concat rule), which hides it from
        ``check_domain_alignment``'s verdict.
        """
        predicate = ConfigHealthChecker._conditions_on_input
        assert predicate(self._cfg(2, 1, condition=True)) is True
        assert (
            predicate(self._cfg(2, 1, condition=True, model_type="kspace_cold_diffusion")) is False
        )
        assert (
            predicate(self._cfg(2, 1, condition=True, model_type="latent_gaussian_diffusion"))
            is False
        )
        assert predicate(self._cfg(2, 1, condition=True, model_type="ldm")) is False
        assert predicate(self._cfg(2, 1, condition=False)) is False


class TestNexTargetModeAcceptsThePair:
    """``rep_pair`` is phase-aligned by construction, so it is a coherent NEX target."""

    @staticmethod
    def _cfg(mode: str):
        from spectramr.config.schemas.data import DataConfigSchema

        data = DataConfigSchema(
            dataset_type="m4raw",
            target_mode=mode,
            nex_target_exclude_input=(mode == "rep_pair"),
        )
        return SimpleNamespace(data=data)

    def test_rep_pair_is_coherent(self) -> None:
        result = ConfigHealthChecker().check_m4raw_nex_target_mode_declared(self._cfg("rep_pair"))
        assert result.passed is True and "rep_pair" in result.message

    def test_complex_mean_is_still_the_finding(self) -> None:
        """Planted violation."""
        result = ConfigHealthChecker().check_m4raw_nex_target_mode_declared(
            self._cfg("complex_mean")
        )
        assert result.passed is False


class TestSliceLevelRecordsQueueShape:
    """``check_slice_level_records_queue_shape`` (#1757): a slice record is a
    depth-1 subject that serves one sample, so the patch depth must be 1 and the
    train queue must draw one sample per record, or be bypassed."""

    @staticmethod
    def _cfg(
        *,
        records: bool = True,
        patch=(256, 256, 1),
        spv: int = 1,
        dataset_type: str = "m4raw",
        train_sampler: dict | None = None,
    ):
        from spectramr.config.schemas.data import DataConfigSchema

        kwargs: dict = {
            "dataset_type": dataset_type,
            "slice_level_records": records,
            "sampling": {"patch_size": patch, "samples_per_volume": spv},
        }
        if train_sampler is not None:
            kwargs["modes"] = {"train": {"sampler": train_sampler}}
        return SimpleNamespace(data=DataConfigSchema(**kwargs))

    @staticmethod
    def _check(cfg):
        return ConfigHealthChecker().check_slice_level_records_queue_shape(cfg)

    def test_depth_one_and_one_sample_per_record_pass(self) -> None:
        result = self._check(self._cfg())
        assert result.passed is True
        assert "once per epoch" in result.message

    def test_a_depth_three_patch_is_refused(self) -> None:
        """Planted violation: the depth-3 arm."""
        result = self._check(self._cfg(patch=(256, 256, 3)))
        assert result.passed is False and result.severity == "error"
        assert "depth 3" in result.message
        assert "data.sampling.patch_size" in result.yaml_keys

    def test_samples_per_volume_above_one_is_refused_via_the_legacy_field(self) -> None:
        """Planted violation, legacy spelling: ``sampling.samples_per_volume``
        is what ``derive_modes_from_legacy`` hands the train queue."""
        result = self._check(self._cfg(spv=4))
        assert result.passed is False and result.severity == "error"
        assert "samples_per_volume=4" in result.message
        assert "data.modes.train.sampler.samples_per_volume" in result.yaml_keys

    def test_samples_per_volume_above_one_is_refused_via_the_mode_block(self) -> None:
        """Planted violation, mode-block spelling."""
        result = self._check(self._cfg(train_sampler={"type": "uniform", "samples_per_volume": 8}))
        assert result.passed is False
        assert "samples_per_volume=8" in result.message

    def test_the_resolved_value_is_the_one_the_build_uses(self) -> None:
        """A declared train block wins over the legacy field, exactly as
        ``TorchIOQueueConfig.from_training_config`` resolves it at build time."""
        result = self._check(
            self._cfg(spv=4, train_sampler={"type": "uniform", "samples_per_volume": 1})
        )
        assert result.passed is True

    def test_the_full_sampler_bypasses_the_queue(self) -> None:
        result = self._check(self._cfg(train_sampler={"type": "full", "samples_per_volume": 4}))
        assert result.passed is True
        assert "bypasses" in result.message

    def test_both_findings_are_reported_together(self) -> None:
        result = self._check(self._cfg(patch=(128, 128, 3), spv=4))
        assert result.passed is False
        assert "depth 3" in result.message and "samples_per_volume=4" in result.message

    def test_a_two_element_patch_is_depth_one(self) -> None:
        assert self._check(self._cfg(patch=(256, 256))).passed is True

    def test_knob_off_is_not_applicable(self) -> None:
        result = self._check(self._cfg(records=False, patch=(256, 256, 3), spv=4))
        assert result.passed is True and "n/a" in result.message

    def test_a_non_m4raw_arm_is_not_applicable(self) -> None:
        result = self._check(self._cfg(records=False, dataset_type="kspace"))
        assert result.passed is True and "n/a" in result.message

    def test_the_check_is_wired_into_run_all_checks(self) -> None:
        import inspect

        src = inspect.getsource(ConfigHealthChecker.run_all_checks)
        assert "self.check_slice_level_records_queue_shape(config)" in src
