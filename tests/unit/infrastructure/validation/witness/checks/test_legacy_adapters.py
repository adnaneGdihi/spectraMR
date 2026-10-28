"""The three legacy stacks, republished through one gate.

These assert the *union*, which is the whole point. Before this, `bootstrap` ran
`ValidatorRegistry` and had never run `ConfigHealthChecker`; `_audit_one` ran
`ConfigHealthChecker` + `compatibility_matrix` and had never run
`ValidatorRegistry`. Both directions were real, so neither surface was a superset
of the other and the union was what nobody checked.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.validation.witness import (
    Tier,
    WitnessSubject,
    get_witness_registry,
    run_witnesses,
)
from spectramr.infrastructure.validation.witness.checks.legacy_adapters import (
    verdict_to_health_result,
)
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    WitnessVerdict,
)

BASE = {
    "model": {"model_type": "unet"},
    "data": {"batch_size": 2},
    "training": {"max_iterations": 10, "training_mode": "reconstruction"},
    "optimization": {"learning_rate": 1e-4},
    "logging": {},
}

T01 = frozenset({Tier.T0, Tier.T1})


def _settings(extra=None):
    import copy

    from spectramr.config.settings import TrainingSettings

    data = copy.deepcopy(BASE)
    if extra:
        data.update(extra)
    return TrainingSettings.settings_from_dict(data)


def test_all_three_legacy_stacks_are_registered():
    """A stack that is not registered is a stack one surface will still miss."""
    names = {w.name for w in get_witness_registry().all()}
    assert {
        "legacy.config_health_checker",
        "legacy.compatibility_matrix",
        "legacy.validator_registry",
    } <= names


def test_one_gate_run_yields_verdicts_from_every_stack():
    subject = WitnessSubject.for_audit(config_path=None, settings=_settings())
    verdicts = run_witnesses(subject, tiers=T01)

    sources = {v.witness_name.split(":")[0] for v in verdicts}
    assert "health" in sources, "ConfigHealthChecker did not run"
    assert "legacy.validator_registry" in sources, "ValidatorRegistry did not run"
    # meta witnesses need no config at all and must run on every surface
    assert any(v.witness_name.startswith("meta.") for v in verdicts)


def test_no_witness_crashed_on_a_valid_config():
    """A crashed detector is reported as a finding, so it must not be silent.

    If an adapter's field mapping is wrong (CompatMessage has no `passed`, for
    instance) it surfaces here rather than as a quietly-clean audit.
    """
    subject = WitnessSubject.for_audit(config_path=None, settings=_settings())
    crashed = [
        v
        for v in run_witnesses(subject, tiers=T01)
        if v.category in {"witness_crash", "witness_scheduling"}
    ]
    assert crashed == [], [v.message for v in crashed]


def test_validator_registry_adapter_reports_a_missing_training_mode():
    """DEFECT: the check the audit could never see before.

    `training_mode_specified` lives in ValidatorRegistry, which `_audit_one` had
    zero references to — so this config passed audit and then failed at
    `build_container`.
    """
    import copy

    from spectramr.config.settings import TrainingSettings

    data = copy.deepcopy(BASE)
    data["training"] = {"max_iterations": 10}  # no training_mode / strategy_class
    subject = WitnessSubject.for_audit(
        config_path=None, settings=TrainingSettings.settings_from_dict(data)
    )
    verdicts = run_witnesses(subject, tiers=T01)

    registry_fails = [
        v for v in verdicts if v.witness_name == "legacy.validator_registry" and not v.passed
    ]
    assert registry_fails, "the audit still cannot see ValidatorRegistry findings"
    assert "training_mode" in registry_fails[0].message


def test_validator_registry_findings_are_reported_but_never_blocking():
    """CONTROL + the ratchet decision, asserted.

    The adapter reports at WARNING until the rule set has been shown satisfiable
    corpus-wide. This test pins that decision, so promoting it to ERROR is a
    deliberate change rather than a side effect.
    """
    subject = WitnessSubject.for_audit(config_path=None, settings=_settings())
    registry = [
        v
        for v in run_witnesses(subject, tiers=T01)
        if v.witness_name == "legacy.validator_registry"
    ]
    assert registry, "the audit still cannot see ValidatorRegistry findings"
    assert all(str(v.severity) != "error" for v in registry), (
        "ValidatorRegistry findings must not block until the ratchet is taken deliberately"
    )


def test_ci_subject_does_not_schedule_settings_only_witnesses():
    """A corpus walk with no resolved settings must not crash or silently pass."""
    verdicts = run_witnesses(WitnessSubject.for_ci(None, {}), tiers=T01)
    names = {v.witness_name for v in verdicts}
    assert not any(n.startswith("health:") for n in names)
    # ...but the config-free meta witness still runs.
    assert any(n.startswith("meta.") for n in names)


class TestVerdictBridge:
    """Verdicts must round-trip into the shape the CLI already consumes."""

    def test_fields_survive(self):
        v = WitnessVerdict(
            witness_name="health:foo",
            passed=False,
            message="bad",
            severity=Severity.ERROR,
            category="silent_fallback",
            yaml_keys=("a.b",),
            fix_hint="do x",
        )
        r = verdict_to_health_result(v)
        assert r.check_name == "health:foo"
        assert r.passed is False
        assert r.severity == "error"
        assert r.category == "silent_fallback"
        assert r.yaml_keys == ["a.b"]
        assert r.fix_hint == "do x"

    @pytest.mark.parametrize("sev", [Severity.ERROR, Severity.WARNING, Severity.INFO])
    def test_severity_is_a_plain_string_not_an_enum_repr(self, sev):
        """`str(StrEnum)` gives the value; an enum repr would break exit codes."""
        r = verdict_to_health_result(WitnessVerdict("w", True, "m", severity=sev))
        assert r.severity in {"error", "warning", "info"}
        assert "Severity." not in r.severity


def test_health_adapter_does_not_narrate_the_summary_twice(caplog):
    """Issue #619: one process reached ``validate_config_health`` twice.

    ``pipelines/train.py`` runs it as a fail-fast gate, then ``bootstrap`` runs
    the witness registry, whose ``legacy.config_health_checker`` adapter ran the
    same 124 checks again. Both unconditionally called ``report.log_summary()``,
    so every job log carried a duplicated "Config Health: n/m checks passed".
    The adapter republishes each result as a verdict, so its own narration is
    redundant -- and silence here must not cost any verdict.
    """
    import logging

    from spectramr.infrastructure.validation.witness.checks.legacy_adapters import (
        config_health_checker,
    )

    subject = WitnessSubject.for_audit(config_path=None, settings=_settings())
    logger_name = "spectramr.infrastructure.validation.config_health_checker"
    with caplog.at_level(logging.INFO, logger=logger_name):
        verdicts = config_health_checker(subject)

    summaries = [r for r in caplog.records if "Config Health:" in r.message]
    assert summaries == [], f"adapter narrated the summary: {summaries}"
    assert verdicts, "silencing the log must not cost the verdicts"


def test_validate_config_health_still_logs_for_its_own_callers(caplog):
    """The pipeline's fail-fast gate keeps its summary line."""
    import logging

    from spectramr.infrastructure.validation.config_health_checker import (
        validate_config_health,
    )

    logger_name = "spectramr.infrastructure.validation.config_health_checker"
    with caplog.at_level(logging.INFO, logger=logger_name):
        validate_config_health(_settings())

    assert any("Config Health:" in r.message for r in caplog.records)
