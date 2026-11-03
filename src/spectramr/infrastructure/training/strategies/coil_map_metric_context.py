"""Coil-map ``MetricContext`` for the twin-driven validation metrics (``g_factor``).

The analytic SENSE g-factor is a function of the coil maps and the acceleration
alone. ``vf_admm`` scored its configured metrics on the magnitude pair with no
context, so ``g_factor`` never saw a map. This module estimates the maps from
the clean multi-coil target through the coil-map owner
(``estimate_smaps_calibrated`` with ``physics.coil_processing.estimation``) and
hands them over with the twin's current acceleration. It runs only when the arm
configures a coil-map metric, and it raises when that metric cannot be measured.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.infrastructure.physics.coil_sensitivity import (
    estimate_smaps_calibrated,
    resolve_estimation_settings,
)
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.utils.kspace_view import log_scaling_enabled

__all__ = ["COIL_MAP_METRICS", "coil_map_metric_context", "configured_coil_map_metrics"]

#: Registered metrics that read ``MetricContext.coil_maps`` from this seam.
COIL_MAP_METRICS: tuple[str, ...] = ("g_factor",)

#: Decision 12 of the inprogress fix plan: ESPIRiT on the virtual coils when the
#: arm declares no ``physics.coil_processing.estimation`` method.
DEFAULT_ESTIMATION_METHOD = "espirit"


def configured_coil_map_metrics(computer: Any) -> tuple[str, ...]:
    """The enabled specs of ``computer`` that need coil maps (aliases resolved)."""
    if computer is None:
        return ()
    wanted = {name: {name, *MetricsRegistry.get_aliases(name)} for name in COIL_MAP_METRICS}
    found: list[str] = []
    for spec in computer.config.metrics:
        if not spec.enabled:
            continue
        lookup = spec.name.lower()
        found.extend(name for name, names in wanted.items() if lookup in names)
    return tuple(found)


def coil_map_metric_context(
    computer: Any, config: Any, target_image: torch.Tensor, acceleration: float
) -> MetricContext | None:
    """Build the context for the configured coil-map metrics, or ``None`` if none are.

    Args:
        computer: The strategy's validation metrics computer.
        config: The run config (``physics.coil_processing.estimation`` is read).
        target_image: Clean complex coil images ``(B, C, H, W)``.
        acceleration: The undersampling rate the twin applied for this pass.

    Raises:
        ValueError: A coil-map metric is configured but the target carries a
            single coil, or ``acceleration`` exceeds the coil count; SENSE is
            then underdetermined and the g-factor is unbounded, not measurable.
    """
    metrics = configured_coil_map_metrics(computer)
    if not metrics:
        return None
    if not torch.is_complex(target_image) or target_image.dim() != 4:
        raise ValueError(
            f"{metrics} need complex coil images (B, C, H, W); got "
            f"{tuple(target_image.shape)} {target_image.dtype}."
        )
    num_coils = int(target_image.shape[1])
    rate = round(float(acceleration))
    if num_coils < 2 or rate > num_coils:
        raise ValueError(
            f"{metrics} configured at R={acceleration:g} on {num_coils} coil(s): the SENSE "
            "unfolding is underdetermined (R > coils), so the g-factor is unbounded and "
            "cannot be measured. Drop the metric or declare more (virtual) coils."
        )
    method, est_kwargs = resolve_estimation_settings(config, default=DEFAULT_ESTIMATION_METHOD)
    maps = estimate_smaps_calibrated(
        fft2c(target_image),
        log_scaled=log_scaling_enabled(config),
        method=method,
        **est_kwargs,
    )
    return MetricContext(coil_maps=maps, acceleration=float(acceleration))
