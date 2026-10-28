"""Tests for the metrics-config dataclasses.

Targets ``spectramr.core.metrics.types``. Defines:

- ``MetricMode`` enum (MIN / MAX)
- ``DEFAULT_METRIC_DIRECTIONS`` mapping (per-name MIN/MAX)
- ``MetricSpec`` (single metric: name, direction, weight, enabled)
- ``ValidationMetricsConfig`` (collection + primary)

Categories:

- Default-direction lookup correctly returns MIN for ``mse``, MAX for ``psnr``
- ``MetricSpec.from_name`` falls back to MAX for unknown names
- ``MetricSpec.from_dict`` parses ``"min"`` / ``"max"`` strings
- ``ValidationMetricsConfig.from_dict``: list / dict / None inputs
- ``ValidationMetricsConfig.from_list`` short-circuit
"""

from __future__ import annotations

import pytest

from spectramr.core.metrics.types import (
    DEFAULT_METRIC_DIRECTIONS,
    MetricMode,
    MetricSpec,
    ValidationMetricsConfig,
)

# ---------------------------------------------------------------------------
# DEFAULT_METRIC_DIRECTIONS
# ---------------------------------------------------------------------------


def test_psnr_is_max() -> None:
    """``psnr`` is registered as higher-is-better."""
    assert DEFAULT_METRIC_DIRECTIONS["psnr"] == MetricMode.MAX


def test_mse_is_min() -> None:
    """``mse`` is registered as lower-is-better."""
    assert DEFAULT_METRIC_DIRECTIONS["mse"] == MetricMode.MIN


def test_lpips_is_min() -> None:
    """LPIPS is a perceptual *distance* — lower is better."""
    assert DEFAULT_METRIC_DIRECTIONS["lpips"] == MetricMode.MIN


# ---------------------------------------------------------------------------
# MetricMode enum
# ---------------------------------------------------------------------------


def test_metric_mode_values() -> None:
    """``MetricMode`` has documented string values."""
    assert MetricMode.MIN.value == "min"
    assert MetricMode.MAX.value == "max"


# ---------------------------------------------------------------------------
# MetricSpec.from_name
# ---------------------------------------------------------------------------


def test_from_name_uses_default_direction() -> None:
    """``from_name('mse')`` picks MIN from the registry."""
    spec = MetricSpec.from_name("mse")
    assert spec.name == "mse"
    assert spec.direction == MetricMode.MIN


def test_from_name_unknown_raises_instead_of_defaulting_to_max() -> None:
    """Unknown name → raises. It used to default to MAX.

    ``from_name`` fed ``DEFAULT_METRIC_DIRECTIONS.get(name, MetricMode.MAX)``, so
    any lower-is-better metric absent from that legacy table (``lpips``, ``fid``,
    ``val_hfen``, …) became a MAX spec and best-checkpoint selection retained the
    WORST weights (#208). An unresolvable name is a config error, not a default.
    """
    from spectramr.core.metrics.metric_directions import UnknownMetricDirectionError

    with pytest.raises(UnknownMetricDirectionError):
        MetricSpec.from_name("custom_metric")


def test_from_name_resolves_lower_is_better_from_the_ssot() -> None:
    """A lower-is-better metric must produce a MIN spec, not a MAX one."""
    assert MetricSpec.from_name("lpips").direction == MetricMode.MIN
    assert MetricSpec.from_name("psnr").direction == MetricMode.MAX


def test_from_name_is_case_insensitive() -> None:
    """``PSNR`` matches ``psnr`` via lowercasing."""
    spec = MetricSpec.from_name("PSNR")
    assert spec.direction == MetricMode.MAX


# ---------------------------------------------------------------------------
# MetricSpec.from_dict
# ---------------------------------------------------------------------------


def test_from_dict_explicit_direction_max() -> None:
    """Explicit ``direction='max'`` is honoured."""
    spec = MetricSpec.from_dict({"name": "x", "direction": "max"})
    assert spec.direction == MetricMode.MAX


def test_from_dict_direction_min() -> None:
    """``direction='min'`` parses to MIN."""
    spec = MetricSpec.from_dict({"name": "x", "direction": "min"})
    assert spec.direction == MetricMode.MIN


def test_from_dict_default_weight_and_enabled() -> None:
    """Missing keys default to ``weight=1.0`` and ``enabled=True``."""
    spec = MetricSpec.from_dict({"name": "x"})
    assert spec.weight == 1.0
    assert spec.enabled is True


def test_from_dict_custom_weight() -> None:
    """Custom weight propagates."""
    spec = MetricSpec.from_dict({"name": "x", "weight": 2.5})
    assert spec.weight == 2.5


# ---------------------------------------------------------------------------
# ValidationMetricsConfig
# ---------------------------------------------------------------------------


def test_from_dict_none_returns_defaults() -> None:
    """``from_dict(None)`` returns the documented psnr+ssim+mse default set."""
    cfg = ValidationMetricsConfig.from_dict(None)
    names = {m.name for m in cfg.metrics}
    assert names == {"psnr", "ssim", "mse"}
    assert cfg.primary_metric == "psnr"


def test_from_dict_list_of_strings() -> None:
    """List of metric-name strings → list of ``MetricSpec``."""
    cfg = ValidationMetricsConfig.from_dict({"metrics": ["psnr", "lpips"]})
    names = [m.name for m in cfg.metrics]
    assert names == ["psnr", "lpips"]
    # lpips picked up MIN direction from the registry.
    lpips = next(m for m in cfg.metrics if m.name == "lpips")
    assert lpips.direction == MetricMode.MIN


def test_from_dict_dict_of_booleans() -> None:
    """Dict ``{name: enabled}`` keeps only the enabled metrics."""
    cfg = ValidationMetricsConfig.from_dict({"metrics": {"psnr": True, "ssim": False}})
    names = [m.name for m in cfg.metrics]
    assert names == ["psnr"]


def test_from_dict_full_specs() -> None:
    """Dict items can be full ``MetricSpec``-like dicts."""
    cfg = ValidationMetricsConfig.from_dict(
        {"metrics": [{"name": "psnr", "weight": 2.0}], "primary_metric": "psnr"}
    )
    assert cfg.metrics[0].weight == 2.0


@pytest.mark.parametrize("declared", [[], {"psnr": False, "ssim": False}])
def test_from_dict_declared_empty_metrics_builds_no_metrics(declared) -> None:
    """A declared empty set is honoured, never replaced by PSNR/SSIM/MSE (#2839)."""
    cfg = ValidationMetricsConfig.from_dict({"metrics": declared, "primary_metric": "val_loss"})
    assert cfg.metrics == []
    assert cfg.primary_metric == "val_loss"


@pytest.mark.parametrize("d", [{}, {"metrics": None}, {"primary_metric": "ssim"}])
def test_from_dict_undeclared_metrics_keeps_defaults(d) -> None:
    """No ``metrics`` value at all is the undeclared case: the documented default applies."""
    cfg = ValidationMetricsConfig.from_dict(d)
    assert {m.name for m in cfg.metrics} == {"psnr", "ssim", "mse"}
    assert cfg.primary_metric == d.get("primary_metric", "psnr")


def test_from_list_short_circuit() -> None:
    """``from_list`` constructs config + sets primary to first name."""
    cfg = ValidationMetricsConfig.from_list(["lpips", "psnr"])
    assert cfg.primary_metric == "lpips"
    assert {m.name for m in cfg.metrics} == {"lpips", "psnr"}


def test_from_list_empty_uses_psnr_primary() -> None:
    """Empty list → empty metrics, primary defaults to ``psnr``."""
    cfg = ValidationMetricsConfig.from_list([])
    assert cfg.metrics == []
    assert cfg.primary_metric == "psnr"
