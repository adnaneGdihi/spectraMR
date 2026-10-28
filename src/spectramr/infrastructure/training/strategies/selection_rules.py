"""Checkpoint-selection rules over the cascade's stamped metrics (``validation.selection``).

A rule turns the zero-fill-normalised ring errors and the 8x guard metrics into one
selectable composite, ``val_guarded_ring_fidelity_mean`` (``_partial`` when the cascade is
incomplete, per #1303), plus ``val_selection_guard_tripped``. The rule set is keyed on
``SelectionRule``, the same enum the schema validates against, so an unknown rule name is
refused at config load and cannot reach :func:`apply_selection_rule`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from spectramr.config.schemas.enums import SelectionRule
from spectramr.config.schemas.validation import ValidationSelectionConfigSchema

#: Rings whose target SNR (per-coil power over sigma^2) reaches this are power-guarded.
SNR_GUARD_MIN = 4.0
_RINGS = ("mid1", "mid2", "high")
_GUARD_RUNG = 8


@dataclass(frozen=True)
class RuleSpec:
    """A rule's required inputs (for the config health check) and its evaluator."""

    required_metrics: tuple[str, ...]
    apply: Callable[[Mapping[str, float], ValidationSelectionConfigSchema, bool], dict[str, float]]


def _excess(value: float | None, limit: float, key: str) -> float:
    if value is None:
        raise ValueError(
            f"guarded_ring_fidelity needs {key!r} in the cascade, but it was not stamped"
        )
    return max(0.0, value - limit)


def _guarded_ring_fidelity(
    metrics: Mapping[str, float], cfg: ValidationSelectionConfigSchema, complete: bool
) -> dict[str, float]:
    suffix = "_mean" if complete else "_mean_partial"
    ring_errors = [
        metrics[f"val_ring_nrmse_{ring}_vs_zf{suffix}"]
        for ring in ("mid1", "mid2")
        if f"val_ring_nrmse_{ring}_vs_zf{suffix}" in metrics
    ]
    if not ring_errors:
        raise ValueError(
            f"guarded_ring_fidelity needs val_ring_nrmse_mid{{1,2}}_vs_zf{suffix}, "
            "but the cascade stamped neither; add ring_nrmse_mid1/mid2 to validation.scoring.compute"
        )
    guard = _GUARD_RUNG
    fab = metrics.get(f"val_fabrication_excess_{guard}x")
    nse = metrics.get(f"val_nse_hall_{guard}x")
    excess = _excess(fab, cfg.fabrication_max, f"val_fabrication_excess_{guard}x")
    excess += _excess(nse, cfg.nse_hall_max, f"val_nse_hall_{guard}x")
    for ring in _RINGS:
        snr = metrics.get(f"val_ring_target_snr_{ring}_{guard}x")
        ratio = metrics.get(f"val_ring_power_ratio_{ring}_{guard}x")
        if snr is None or ratio is None or not math.isfinite(ratio):
            continue
        if snr >= SNR_GUARD_MIN:
            excess += max(0.0, ratio - cfg.power_ratio_max)
    mean_error = sum(ring_errors) / len(ring_errors)
    return {
        f"val_guarded_ring_fidelity{suffix}": mean_error + cfg.penalty * excess,
        "val_selection_guard_tripped": 1.0 if excess > 0.0 else 0.0,
    }


_REQUIRED_RING_METRICS = (
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
)

#: The registry. Every ``SelectionRule`` member except ``NONE`` must be a key here.
RULES: dict[SelectionRule, RuleSpec] = {
    SelectionRule.GUARDED_RING_FIDELITY: RuleSpec(
        required_metrics=_REQUIRED_RING_METRICS, apply=_guarded_ring_fidelity
    ),
}


def required_metrics(rule: str | SelectionRule) -> tuple[str, ...]:
    """Metric names a rule reads; empty for ``none``. An unknown name raises (rule 3)."""
    resolved = SelectionRule(rule)
    if resolved is SelectionRule.NONE:
        return ()
    return RULES[resolved].required_metrics


def apply_selection_rule(
    rule: str | SelectionRule,
    metrics: Mapping[str, float],
    cfg: ValidationSelectionConfigSchema,
    *,
    complete: bool,
) -> dict[str, float]:
    """The rule's emitted metrics, or ``{}`` for ``none``. Unknown names raise ``ValueError``."""
    resolved = SelectionRule(rule)
    if resolved is SelectionRule.NONE:
        return {}
    return RULES[resolved].apply(metrics, cfg, complete)


__all__ = ["RULES", "SNR_GUARD_MIN", "apply_selection_rule", "required_metrics"]
