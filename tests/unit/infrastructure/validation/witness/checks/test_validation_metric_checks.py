"""``validation_metric_names_resolve`` (cohort review 2026-09-02, T0.5). Planted violation first."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.core.metrics.validation_names import UNIVERSAL_VALIDATION_KEYS
from spectramr.infrastructure.validation.witness.checks import validation_metric_checks as vmc
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject

_REGISTERED = {"psnr", "ssim", "hfen"}


def _registered(name: str) -> bool:
    return name in _REGISTERED


def _settings(compute=(), best=None, early=None):
    return SimpleNamespace(
        validation=SimpleNamespace(scoring=SimpleNamespace(compute=list(compute))),
        metrics=SimpleNamespace(best_metric_name=best),
        early_stopping=SimpleNamespace(metric=early),
    )


_U = UNIVERSAL_VALIDATION_KEYS


def _subject(
    settings, emitted: frozenset[str] | None, monkeypatch, universal: frozenset[str] = _U
) -> WitnessSubject:
    name = "DeclaringStrategy" if emitted else "SilentStrategy"
    monkeypatch.setattr(
        vmc, "validation_owners_for", lambda _s: (emitted or frozenset(), universal, name)
    )
    import spectramr.core.metrics.registry as reg

    monkeypatch.setattr(reg.MetricsRegistry, "is_registered", staticmethod(_registered))
    return WitnessSubject.for_audit(None, settings)


# --- the predicate --------------------------------------------------------------


def test_registered_and_emitted_names_resolve() -> None:
    names = ["psnr", "field_mse", "val_field_abs_bias", "val_psnr_mean", "val_psnr_2x", "val_loss"]
    assert (
        vmc.unresolved_metric_names(
            names, _registered, frozenset({"val_field_mse", "val_field_abs_bias"}), universal=_U
        )
        == []
    )


def test_a_selector_no_path_produces_is_unresolved() -> None:
    """The planted violation: exp_vf_22 once selected on val_nmse and never early-stopped."""
    assert vmc.unresolved_metric_names(["val_nmse"], _registered, frozenset(), universal=_U) == [
        "val_nmse"
    ]


def test_cascade_and_heldout_suffixes_are_stripped() -> None:
    assert (
        vmc.unresolved_metric_names(
            ["val_hfen_heldout_64x", "val_ssim_median"], _registered, frozenset(), universal=_U
        )
        == []
    )


# --- the witness ----------------------------------------------------------------


def test_declaring_strategy_with_an_ownerless_name_is_an_error(monkeypatch) -> None:
    subject = _subject(
        _settings(best="val_psnr", early="val_field_typo"), frozenset({"val_field_mse"}), monkeypatch
    )
    verdict = vmc.validation_metric_names_resolve(subject)
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert "val_field_typo" in verdict.message


def test_declaring_strategy_with_resolved_names_passes(monkeypatch) -> None:
    subject = _subject(
        _settings(compute=["field_bias", "field_mse"], best="val_field_abs_bias"),
        frozenset({"val_field_bias", "val_field_mse", "val_field_abs_bias"}),
        monkeypatch,
    )
    assert vmc.validation_metric_names_resolve(subject).passed is True


def test_silent_strategy_is_reported_unverified_not_failed(monkeypatch) -> None:
    """The census step of the ratchet."""
    subject = _subject(_settings(best="val_crlb_objective"), None, monkeypatch)
    verdict = vmc.validation_metric_names_resolve(subject)
    assert verdict.passed is True and verdict.severity is Severity.INFO
    assert verdict.message.startswith("UNVERIFIED")


def test_compute_entries_are_left_to_the_health_check(monkeypatch) -> None:
    """#2819: a suffixed compute entry is not resolved here by stripping the suffix.

    The computer receives ``psnr_4x`` verbatim and the runtime rejects it, so this
    witness must not report it as resolving; ``check_metric_names_are_registered``
    owns the compute list through the runtime's own predicate.
    """
    subject = _subject(_settings(compute=["psnr_4x", "field_typo"]), frozenset({"x"}), monkeypatch)
    verdict = vmc.validation_metric_names_resolve(subject)
    assert verdict.passed is True
    assert verdict.message == "no checkpoint selector declared"


def test_a_suffixed_selector_still_resolves(monkeypatch) -> None:
    subject = _subject(_settings(best="val_psnr_4x", early="val_loss"), frozenset({"x"}), monkeypatch)
    assert vmc.validation_metric_names_resolve(subject).passed is True


def test_val_loss_selector_errors_on_the_metrics_mixin_path(monkeypatch) -> None:
    """PLANTED (#2838): ``MetricsMixin.validation_step`` never writes ``val_loss``."""
    subject = _subject(
        _settings(best="val_psnr", early="val_loss"),
        frozenset({"x"}),
        monkeypatch,
        universal=frozenset(),
    )
    verdict = vmc.validation_metric_names_resolve(subject)
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert "val_loss" in verdict.message


def test_owners_come_from_the_resolved_strategy_class(monkeypatch) -> None:
    """The witness reads the same per-path rule the runtime gate applies."""
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    from spectramr.models.capabilities import StrategyCapabilities

    class MetricsPathStrategy(MetricsMixin):
        capabilities = StrategyCapabilities(emitted_metrics=frozenset({"x"}))

    monkeypatch.setattr(
        TrainingStrategyFactory, "get_strategy_class", lambda _f, _s: MetricsPathStrategy
    )
    assert vmc.validation_owners_for(object()) == (
        frozenset({"x"}),
        frozenset(),
        "MetricsPathStrategy",
    )


def test_an_unresolvable_strategy_keeps_the_exemption(monkeypatch) -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    def _boom(_f, _s):
        raise KeyError("no such training_mode")

    monkeypatch.setattr(TrainingStrategyFactory, "get_strategy_class", _boom)
    assert vmc.validation_owners_for(object()) == (frozenset(), _U, "unresolved")


def test_no_names_declared_is_a_pass(monkeypatch) -> None:
    assert (
        vmc.validation_metric_names_resolve(_subject(_settings(), None, monkeypatch)).passed is True
    )


def test_the_qmri_strategy_declares_the_keys_its_arms_name() -> None:
    """Pins the declaration the 10 vf qMRI arms rely on, at its source."""
    from spectramr.infrastructure.training.strategies.multi_acquisition_strategy import (
        ConcreteMultiAcquisitionStrategy,
    )

    emitted = ConcreteMultiAcquisitionStrategy.capabilities.emitted_metrics
    assert {"val_field_mse", "val_field_bias", "val_field_abs_bias", "val_field_mae"} <= emitted


def test_witness_is_registered_after_discovery() -> None:
    import spectramr.infrastructure.validation.witness  # noqa: F401

    assert get_witness_registry().get("validation_metric_names_resolve") is not None
