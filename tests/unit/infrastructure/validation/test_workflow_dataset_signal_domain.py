"""Tests for check_workflow_dataset_signal_domain.

The dataset-level companion to check_workflow_signal_domain. The regime-level
check compares a model against everything the *regime* can be
(``mri_structural`` spans ``{image, kspace, complex_image}``); a k-space model on
magnitude-only mrixfields data passes it because ``kspace`` is a legal structural
domain in the abstract. This check compares the model against what THIS dataset
actually materialises — the mismatch the regime-level check cannot see.
"""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.config.schemas.enums import Regime
from spectramr.config.schemas.workflow import WorkflowConfigSchema
from spectramr.data.datasets.axis_exposure import DATASET_TYPE_SIGNAL_DOMAINS
from spectramr.infrastructure.validation.config_health_checker import (
    ConfigHealthChecker,
    HealthCheckResult,
)
from spectramr.models.capabilities import ModelCapabilities


def _check(
    regime: Regime | None,
    input_domain,
    dataset_type: str | None,
    *,
    adapters=None,
    model_type: str = "_ds_probe",
):
    """Run the check with a stubbed model registry entry.

    The registry is patched rather than a real model chosen, so the test states
    the (regime, input_domain, dataset_type) triple directly.
    """
    checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
    workflow = WorkflowConfigSchema(regime=regime) if regime is not None else None
    cfg = SimpleNamespace(
        workflow=workflow,
        adapters=adapters,
        data=SimpleNamespace(dataset_type=dataset_type),
        model=SimpleNamespace(
            model_type=model_type,
            in_channels=1,
            out_channels=1,
            spatial_dims=2,
            input_type=None,
        ),
    )
    from spectramr.models.registry import MODEL_REGISTRY

    caps = (
        ModelCapabilities(input_domain=input_domain)
        if input_domain is not None
        else ModelCapabilities()
    )
    MODEL_REGISTRY[model_type] = {"capabilities": caps, "mode": "reconstruction"}
    try:
        return checker.check_workflow_dataset_signal_domain(cfg)
    finally:
        MODEL_REGISTRY.pop(model_type, None)


class TestCatchesRealMismatches:
    def test_a_kspace_model_on_magnitude_only_mrixfields_is_rejected(self) -> None:
        """THE hole this check closes, invisible to the regime-level check.

        mrixfields is magnitude-only image data, but ``kspace`` is a legal
        ``mri_structural`` domain — so check_workflow_signal_domain PASSES a
        k-space model here. This dataset-level check rejects it: the loader never
        produces k-space, so the model is dead on arrival.
        """
        r = _check(Regime.STRUCTURAL, "kspace", "mrixfields")
        assert not r.passed
        assert r.severity == "error"
        assert "disjoint" in r.message
        assert "mrixfields" in r.message

    def test_an_image_model_on_raw_kspace_data_is_rejected(self) -> None:
        r = _check(Regime.STRUCTURAL, "image", "m4raw")
        assert not r.passed
        assert r.severity == "error"

    def test_the_failure_names_the_dataset_and_offers_a_fix(self) -> None:
        r = _check(Regime.STRUCTURAL, "kspace", "mrixfields")
        assert r.yaml_keys == ["workflow.regime", "data.dataset_type", "model.model_type"]
        assert "adapter" in (r.fix_hint or "")


class TestDoesNotRejectLegitimateArms:
    def test_an_image_model_on_mrixfields_passes(self) -> None:
        assert _check(Regime.STRUCTURAL, "image", "mrixfields").passed

    def test_a_kspace_model_on_raw_kspace_passes(self) -> None:
        assert _check(Regime.STRUCTURAL, "kspace", "m4raw").passed

    def test_a_multi_domain_model_passes_if_any_domain_fits(self) -> None:
        # input_domain may be a tuple; one overlap with the dataset is enough.
        assert _check(Regime.STRUCTURAL, ("pde_grid", "image"), "mrixfields").passed

    def test_an_enabled_pre_model_adapter_skips_the_check(self) -> None:
        # A pre_model adapter (e.g. fft_image_to_kspace) may bridge the domain, so a
        # k-space model on image data is NOT a mismatch when one is declared. The
        # adapter must actually BE a pre_model step: this previously passed a bare
        # SimpleNamespace(), which declared no adapter at all and still skipped —
        # the test encoded the over-broad escape hatch rather than the rule.
        adapters = SimpleNamespace(
            pre_model=[SimpleNamespace(name="fft_image_to_kspace", enabled=True)]
        )
        r = _check(Regime.STRUCTURAL, "kspace", "mrixfields", adapters=adapters)
        assert r.passed
        assert "adapter" in r.message

    def test_an_unannotated_dataset_is_skipped(self) -> None:
        r = _check(Regime.STRUCTURAL, "kspace", "synthetic")
        assert r.passed
        assert "skipped" in r.message

    def test_a_model_without_input_domain_is_skipped(self) -> None:
        r = _check(Regime.STRUCTURAL, None, "mrixfields")
        assert r.passed
        assert "skipped" in r.message

    def test_no_workflow_declared_is_skipped(self) -> None:
        assert _check(None, "kspace", "mrixfields").passed


def test_the_check_is_wired_into_the_audit_run(monkeypatch) -> None:
    """An unwired check is the facade this repo polices (severity_calibration #279).

    Asserted BEHAVIOURALLY — by monkeypatching the check and driving the real
    ``run_all_checks`` — not by grepping ``inspect.getsource`` for the call site.
    The source-text form was green against a COMMENTED-OUT call, which is the exact
    escape ``tests/unit/pipelines/test_maturity_gate_wiring.py`` documents having
    observed. A wiring test that survives its own mutation is not a wiring test.
    """
    called: list[bool] = []

    def _spy(self, config):
        called.append(True)
        return HealthCheckResult(
            passed=True,
            check_name="workflow_dataset_signal_domain",
            message="spy",
            severity="info",
        )

    monkeypatch.setattr(
        ConfigHealthChecker, "check_workflow_dataset_signal_domain", _spy
    )
    checker = ConfigHealthChecker()
    report = checker.run_all_checks(_runnable_config())

    assert called, "run_all_checks never invoked check_workflow_dataset_signal_domain"
    assert any(r.check_name == "workflow_dataset_signal_domain" for r in report.results), (
        "the check ran but its result never reached the report"
    )


def test_image_domain_set_agrees_with_the_signal_domain_ssot() -> None:
    """Cross-table invariant: the two magnitude-vs-kspace tables must not disagree.

    ``_IMAGE_DOMAIN_DATASET_TYPES`` (channel derivation, config_health_checker) and
    ``DATASET_TYPE_SIGNAL_DOMAINS`` (workflow signal domain, axis_exposure) both
    classify a dataset as image-vs-kspace. Every dataset the SSOT calls image-only
    MUST be in the image-domain set, or ``_derive_expected_channels`` and the
    workflow check would disagree about the same dataset — the drift that made
    mrixfields return 2 channels (real+imag) for magnitude data.
    """
    image_only = {
        dt for dt, doms in DATASET_TYPE_SIGNAL_DOMAINS.items() if doms == frozenset({"image"})
    }
    missing = image_only - ConfigHealthChecker._IMAGE_DOMAIN_DATASET_TYPES
    assert not missing, (
        f"{sorted(missing)} are image-only per the signal-domain SSOT but missing "
        "from _IMAGE_DOMAIN_DATASET_TYPES — the two tables disagree."
    )
    # And no kspace/complex dataset may be mislabelled image-domain.
    for dt, doms in DATASET_TYPE_SIGNAL_DOMAINS.items():
        if "kspace" in doms:
            assert dt not in ConfigHealthChecker._IMAGE_DOMAIN_DATASET_TYPES, dt


def test_mrixfields_is_now_image_domain_for_channel_derivation() -> None:
    """The concrete bug this fixes: mrixfields must count as image-domain (1 channel).

    Omitting it made _derive_expected_channels take the kspace path and expect 2
    channels (real+imag RSS) for magnitude-only [0,1] data.
    """
    assert ConfigHealthChecker._is_image_domain_dataset("mrixfields")


def _runnable_config():
    """A minimal real ``TrainingSettings`` that ``run_all_checks`` can walk."""
    from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.settings_from_dict(
        {
            "config_version": CANONICAL_CONFIG_VERSION,
            "data": {
                "train_path": "/tmp/train",
                "val_path": "/tmp/val",
                "batch_size": 2,
                "dataset_type": "mrixfields",
            },
            "optimization": {"learning_rate": 1e-4},
            "logging": {},
            "model": {"model_type": "unet"},
            "workflow": {"regime": "mri_structural", "task": "synthesis"},
        }
    )


class TestAdapterEscapeHatchIsNarrow:
    """The skip is for a pre_model adapter that actually runs — nothing wider.

    ``adapters is not None`` disarmed the check on ANY adapters object, so a chain
    that bridges nothing (``pre_metric`` only, every step disabled, or an empty
    ``adapters: {}``) silently turned off a hard error. Only ``pre_model`` sits
    between dataset and model, and only an enabled step runs.
    """

    def test_an_enabled_pre_model_adapter_still_skips(self) -> None:
        adapters = SimpleNamespace(
            pre_model=[SimpleNamespace(name="fft_image_to_kspace", enabled=True)]
        )
        result = _check(Regime.STRUCTURAL, "kspace", "mrixfields", adapters=adapters)
        assert result.passed
        assert "pre_model adapter" in result.message

    def test_an_empty_adapters_object_no_longer_disarms_the_check(self) -> None:
        result = _check(
            Regime.STRUCTURAL, "kspace", "mrixfields", adapters=SimpleNamespace()
        )
        assert not result.passed

    def test_a_non_pre_model_hook_no_longer_disarms_the_check(self) -> None:
        adapters = SimpleNamespace(
            pre_metric=[SimpleNamespace(name="rss", enabled=True)], pre_model=[]
        )
        result = _check(Regime.STRUCTURAL, "kspace", "mrixfields", adapters=adapters)
        assert not result.passed

    def test_an_all_disabled_pre_model_chain_no_longer_disarms_the_check(self) -> None:
        adapters = SimpleNamespace(
            pre_model=[SimpleNamespace(name="fft_image_to_kspace", enabled=False)]
        )
        result = _check(Regime.STRUCTURAL, "kspace", "mrixfields", adapters=adapters)
        assert not result.passed


class TestTheReconstructionStrategyBridgesKspaceToImage:
    """#2296: an image network over a k-space loader is served ``A^H y`` by the
    reconstruction strategy, so the check must compare the bridged domain -- and
    must still fire where no strategy applies that adjoint."""

    @staticmethod
    def _ei_arm(strategy_class: str | None = None):
        from pathlib import Path

        from spectramr.config.settings import TrainingSettings
        from spectramr.models.init_registry import populate_model_registry

        populate_model_registry()
        root = Path(__file__).resolve().parents[4]
        config = TrainingSettings.from_yaml(
            root / "experiments/inprogress/self_supervised/ei_unet_m4raw_r4.yaml"
        )
        if strategy_class is not None:
            training = config.training.model_copy(
                update={"strategy_class": strategy_class, "training_mode": "diffusion"}
            )
            config = config.model_copy(update={"training": training})
        return config

    def test_the_ei_arm_passes_through_the_strategy_bridge(self) -> None:
        checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
        result = checker.check_workflow_dataset_signal_domain(self._ei_arm())
        assert result.passed, result.message
        assert "adjoint" in result.message

    def test_a_strategy_without_the_bridge_is_still_rejected(self) -> None:
        """The planted violation: same data and model, a strategy that feeds k-space."""
        checker = ConfigHealthChecker.__new__(ConfigHealthChecker)
        config = self._ei_arm(
            "spectramr.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy"
        )
        result = checker.check_workflow_dataset_signal_domain(config)
        assert not result.passed and result.severity == "error"
