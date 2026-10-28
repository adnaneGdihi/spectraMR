"""``check_selection_rule_metrics_declared``: a rule without its metrics is an error, with them it passes."""

from __future__ import annotations

from types import SimpleNamespace

from spectramr.config.schemas.enums import SelectionRule
from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

_REQUIRED = [
    "ring_nrmse_mid1",
    "ring_nrmse_mid2",
    "ring_power_ratio_mid1",
    "ring_power_ratio_mid2",
    "ring_power_ratio_high",
    "ring_target_snr_mid1",
    "ring_target_snr_mid2",
    "ring_target_snr_high",
    "fabrication_excess",
    "nse_hall",
]


def _cfg(rule: SelectionRule, compute: list[str] | None) -> SimpleNamespace:
    return SimpleNamespace(
        validation=SimpleNamespace(
            selection=SimpleNamespace(rule=rule), scoring=SimpleNamespace(compute=compute)
        )
    )


def test_rule_none_passes_without_metrics() -> None:
    result = ConfigHealthChecker().check_selection_rule_metrics_declared(
        _cfg(SelectionRule.NONE, None)
    )
    assert result.passed


def test_guarded_rule_with_every_metric_passes() -> None:
    result = ConfigHealthChecker().check_selection_rule_metrics_declared(
        _cfg(SelectionRule.GUARDED_RING_FIDELITY, list(_REQUIRED))
    )
    assert result.passed


def test_planted_missing_ring_metric_is_an_error() -> None:
    compute = [m for m in _REQUIRED if m != "ring_target_snr_mid2"]
    result = ConfigHealthChecker().check_selection_rule_metrics_declared(
        _cfg(SelectionRule.GUARDED_RING_FIDELITY, compute)
    )
    assert not result.passed
    assert result.severity == "error"
    assert "ring_target_snr_mid2" in result.message


def test_undeclared_compute_list_is_an_error_under_a_rule() -> None:
    result = ConfigHealthChecker().check_selection_rule_metrics_declared(
        _cfg(SelectionRule.GUARDED_RING_FIDELITY, None)
    )
    assert not result.passed
