"""``guarded_ring_fidelity``: one planted violation per guard, the partial suffix, and loud refusals."""

from __future__ import annotations

import pytest

from spectramr.config.schemas.enums import SelectionRule
from spectramr.config.schemas.validation import ValidationSelectionConfigSchema
from spectramr.infrastructure.training.strategies.selection_rules import (
    apply_selection_rule,
    required_metrics,
)

CFG = ValidationSelectionConfigSchema(rule=SelectionRule.GUARDED_RING_FIDELITY)


def _clean() -> dict[str, float]:
    metrics = {
        "val_ring_nrmse_mid1_vs_zf_mean": 0.8,
        "val_ring_nrmse_mid2_vs_zf_mean": 0.6,
        "val_fabrication_excess_8x": 0.2,
        "val_nse_hall_8x": 0.8,
    }
    for ring in ("mid1", "mid2", "high"):
        metrics[f"val_ring_power_ratio_{ring}_8x"] = 0.9
        metrics[f"val_ring_target_snr_{ring}_8x"] = 10.0
    return metrics


def _run(metrics: dict[str, float], *, complete: bool = True) -> dict[str, float]:
    return apply_selection_rule(
        SelectionRule.GUARDED_RING_FIDELITY, metrics, CFG, complete=complete
    )


def test_clean_cascade_scores_the_mean_ring_error_and_does_not_trip() -> None:
    out = _run(_clean())
    assert out["val_guarded_ring_fidelity_mean"] == pytest.approx(0.7)
    assert out["val_selection_guard_tripped"] == 0.0


def test_fabrication_violation_trips_and_is_penalised() -> None:
    metrics = _clean() | {"val_fabrication_excess_8x": 0.5}
    out = _run(metrics)
    assert out["val_selection_guard_tripped"] == 1.0
    assert out["val_guarded_ring_fidelity_mean"] == pytest.approx(0.7 + CFG.penalty * 0.2)


def test_hallucination_violation_trips() -> None:
    out = _run(_clean() | {"val_nse_hall_8x": 0.95})
    assert out["val_selection_guard_tripped"] == 1.0


def test_power_violation_trips_only_on_a_well_measured_ring() -> None:
    loud = _clean() | {"val_ring_power_ratio_mid2_8x": 1.2}
    assert _run(loud)["val_selection_guard_tripped"] == 1.0
    weak = _clean() | {"val_ring_power_ratio_mid2_8x": 1.2, "val_ring_target_snr_mid2_8x": 2.0}
    assert _run(weak)["val_selection_guard_tripped"] == 0.0


def test_partial_cascade_uses_partial_inputs_and_names_the_partial_composite() -> None:
    metrics = _clean()
    for ring in ("mid1", "mid2"):
        metrics[f"val_ring_nrmse_{ring}_vs_zf_mean_partial"] = metrics.pop(
            f"val_ring_nrmse_{ring}_vs_zf_mean"
        )
    out = _run(metrics, complete=False)
    assert "val_guarded_ring_fidelity_mean_partial" in out
    assert "val_guarded_ring_fidelity_mean" not in out


def test_missing_ring_inputs_raise_instead_of_degrading() -> None:
    metrics = _clean()
    del metrics["val_ring_nrmse_mid1_vs_zf_mean"]
    del metrics["val_ring_nrmse_mid2_vs_zf_mean"]
    with pytest.raises(ValueError, match="ring_nrmse"):
        _run(metrics)


def test_missing_guard_input_raises() -> None:
    metrics = _clean()
    del metrics["val_fabrication_excess_8x"]
    with pytest.raises(ValueError, match="fabrication_excess"):
        _run(metrics)


def test_rule_none_emits_nothing_and_needs_no_inputs() -> None:
    assert apply_selection_rule(SelectionRule.NONE, {}, CFG, complete=True) == {}
    assert required_metrics("none") == ()


def test_unknown_rule_raises() -> None:
    with pytest.raises(ValueError):
        apply_selection_rule("best_guess", _clean(), CFG, complete=True)
    with pytest.raises(ValueError):
        required_metrics("best_guess")


def test_required_metrics_name_the_ring_and_guard_inputs() -> None:
    names = required_metrics(SelectionRule.GUARDED_RING_FIDELITY)
    assert {"ring_nrmse_mid2", "ring_power_ratio_high", "fabrication_excess", "nse_hall"} <= set(
        names
    )
