"""Metric Types and Configuration.

Defines the data structures and configuration classes for the metrics system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricMode(Enum):
    """Optimization direction for a metric."""

    MIN = "min"
    MAX = "max"


# DEPRECATED as a resolution path — kept only so existing importers keep working.
#
# This table is NO LONGER consulted when resolving a metric's direction. It was
# the third of three disagreeing direction tables, and because every lookup
# ended in ``.get(name, MetricMode.MAX)`` it silently maximized the 37+
# lower-is-better metrics it does not list (``lpips``, ``fid``,
# ``nll_bits_per_dim``, ``val_hfen``, …) — inverting best-checkpoint selection
# (#208).
#
# The single resolver is now
# ``spectramr.core.metrics.metric_directions.metric_higher_is_better``, which reads
# the registry (where ``@register_metric`` already injects the direction) and
# raises on a key it cannot resolve. ``tests/unit/core/metrics/test_metric_directions.py``
# asserts this table never contradicts it, so it cannot drift back apart.
DEFAULT_METRIC_DIRECTIONS: dict[str, MetricMode] = {
    # Higher is better
    "psnr": MetricMode.MAX,
    "ssim": MetricMode.MAX,
    "vif": MetricMode.MAX,
    "fsim": MetricMode.MAX,
    "uqi": MetricMode.MAX,
    "snr": MetricMode.MAX,
    "pearson": MetricMode.MAX,
    "correlation": MetricMode.MAX,
    "cosine_similarity": MetricMode.MAX,
    # Corrected: gradient entropy is an autofocus criterion — it is MINIMIZED
    # (a sharp image has a sparse, low-entropy gradient distribution). This entry
    # said MAX while the registry/METRIC_SPECS SSOT says lower-is-better; it was
    # the one outright contradiction between the two tables.
    "gradient_entropy": MetricMode.MIN,
    "ms_ssim": MetricMode.MAX,
    "fber": MetricMode.MAX,
    "wm2max": MetricMode.MAX,
    "cnr": MetricMode.MAX,
    "dice": MetricMode.MAX,
    "iou": MetricMode.MAX,
    "ndc": MetricMode.MAX,
    "tsnr": MetricMode.MAX,
    "dietrich_snr": MetricMode.MAX,
    "blur": MetricMode.MAX,
    "volume_similarity": MetricMode.MAX,
    "sfnr": MetricMode.MAX,
    "pe_cross_corr": MetricMode.MAX,
    # Lower is better
    "mse": MetricMode.MIN,
    "mae": MetricMode.MIN,
    "rmse": MetricMode.MIN,
    "nmse": MetricMode.MIN,
    "nrmse": MetricMode.MIN,
    "hfen": MetricMode.MIN,
    "focal_frequency": MetricMode.MIN,
    "gmsd": MetricMode.MIN,
    "lpips": MetricMode.MIN,
    "fid": MetricMode.MIN,
    "kid": MetricMode.MIN,
    "l1": MetricMode.MIN,
    "l1_loss": MetricMode.MIN,
    "loss": MetricMode.MIN,
    "val_loss": MetricMode.MIN,
    "gradient_error": MetricMode.MIN,
    "phase_mse": MetricMode.MIN,
    "kspace_error": MetricMode.MIN,
    "ipen": MetricMode.MIN,
    "complex_hfen": MetricMode.MIN,
    "efc": MetricMode.MIN,
    "qi1": MetricMode.MIN,
    "cjv": MetricMode.MIN,
    "hd95": MetricMode.MIN,
    "fwhm": MetricMode.MIN,
    "spike_detection": MetricMode.MIN,
    "dvars": MetricMode.MIN,
    "gcor": MetricMode.MIN,
    "dists": MetricMode.MIN,
    "gsr": MetricMode.MIN,
    "spike_percent": MetricMode.MIN,
    "neg_voxels": MetricMode.MIN,
    "rase": MetricMode.MIN,
    "sam": MetricMode.MIN,
    "fd": MetricMode.MIN,
    "aor": MetricMode.MIN,
    "piesno": MetricMode.MIN,
    "medicalnet_distance": MetricMode.MIN,
}


@dataclass
class MetricSpec:
    """Specification for a single validation metric."""

    name: str
    direction: MetricMode = MetricMode.MAX
    weight: float = 1.0
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetricSpec":
        """Create from dictionary config."""
        direction = d.get("direction", "max")
        if isinstance(direction, str):
            direction = MetricMode.MAX if direction.lower() == "max" else MetricMode.MIN
        return cls(
            name=d["name"],
            direction=direction,
            weight=d.get("weight", 1.0),
            enabled=d.get("enabled", True),
        )

    @classmethod
    def from_name(cls, name: str) -> "MetricSpec":
        """Create from a metric name, resolving direction from the SSOT.

        Resolves via
        :func:`~spectramr.core.metrics.metric_directions.metric_higher_is_better`
        (registry-backed, raises on an undeclared key) rather than the legacy
        ``DEFAULT_METRIC_DIRECTIONS.get(name, MAX)`` default, which silently
        maximized every lower-is-better metric missing from that table. See #208.
        """
        # Local import: metric_directions -> registry -> (lazily) this module.
        from spectramr.core.metrics.metric_directions import metric_higher_is_better

        higher = metric_higher_is_better(name)
        return cls(name=name, direction=MetricMode.MAX if higher else MetricMode.MIN)


@dataclass
class ValidationMetricsConfig:
    """Configuration for validation metrics computation."""

    metrics: list[MetricSpec] = field(default_factory=list)
    primary_metric: str = "psnr"
    compute_all_available: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ValidationMetricsConfig":
        """Create from dictionary config.

        Only an undeclared metric set gets the PSNR/SSIM/MSE default: ``d`` is
        ``None`` or carries no ``metrics`` value. A declared set is taken as
        written, so an empty one builds a computer with no metrics (#2839).
        """
        if d is None or d.get("metrics") is None:
            return cls(
                metrics=[
                    MetricSpec.from_name("psnr"),
                    MetricSpec.from_name("ssim"),
                    MetricSpec.from_name("mse"),
                ],
                primary_metric=(d or {}).get("primary_metric", "psnr"),
                compute_all_available=(d or {}).get("compute_all_available", False),
            )

        metrics = []
        metrics_config = d["metrics"]

        if isinstance(metrics_config, dict):
            # Handler for dict of booleans: {"psnr": True, "ssim": False}
            for metric_name, is_enabled in metrics_config.items():
                if is_enabled:
                    metrics.append(MetricSpec.from_name(metric_name))
        else:
            # Handler for list of metrics
            for m in metrics_config:
                if isinstance(m, str):
                    # Simple string metric name
                    metrics.append(MetricSpec.from_name(m))
                elif isinstance(m, dict):
                    # Full metric spec: {"name": "psnr", "weight": 2.0}
                    metrics.append(MetricSpec.from_dict(m))

        return cls(
            metrics=metrics,
            primary_metric=d.get("primary_metric", "psnr"),
            compute_all_available=d.get("compute_all_available", False),
        )

    @classmethod
    def from_list(cls, metric_names: list[str]) -> "ValidationMetricsConfig":
        """Create from simple list of metric names."""
        metrics = [MetricSpec.from_name(name) for name in metric_names]
        primary = metric_names[0] if metric_names else "psnr"
        return cls(metrics=metrics, primary_metric=primary)
