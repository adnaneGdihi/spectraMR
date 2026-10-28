"""Measured-map counterparts of the oracle-map cold-diffusion validation metrics.

``kspace_cold_diffusion`` validation calibrates its coil maps from the fully
sampled reference, and every SENSE-combined number inherits them: the
``ifft_sense_adjoint`` image metrics, the zero-filled baseline that normalises
the selector, and the measurement-aware trust functionals. At R=32 the rung
acquires a handful of phase-encode lines, so those numbers are graded with maps
no deployed reconstruction could have. ``validation.scoring.measurement_maps:
metrics`` scores the SAME prediction a second time with maps calibrated from
the rung's masked measurement alone, through the estimator predict uses
(:func:`~spectramr.infrastructure.physics.coil_sensitivity.estimate_smaps_from_measurement`),
and publishes the result beside the oracle numbers under a ``_measmaps`` suffix.
The oracle numbers, the selector and every loss are untouched.

The prediction itself stays oracle-conditioned: the generator receives the
oracle maps (concatenated on the single-step path, passed to ``sample()`` on the
multi-step one). Re-sampling with measured conditioning would double the
reverse-sampler cost per rung and is not offered.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.physics.coil_sensitivity import estimate_smaps_from_measurement
from spectramr.models.diffusion.reverse_path import takes_coil_maps

__all__ = [
    "APPLIED_KEY",
    "MEASURED_IMAGE_METRICS",
    "MEASURED_MAPS_SUFFIX",
    "MEASURED_MAP_MODES",
    "SINGLE_COIL_REASON",
    "MeasuredMapStash",
    "declared_measured_keys",
    "measured_map_stash",
    "measured_maps_declared",
    "measured_mean_bases",
    "measurement_maps_defect",
    "score_with_measured_maps",
]

MEASURED_MAPS_SUFFIX = "_measmaps"
MEASURED_MAP_MODES: tuple[str, ...] = ("off", "metrics")

#: Per rung: the fraction of slices scored with non-degenerate measured maps; 0.0
#: on a skipped (single-coil) rung or one with no acquired line in the ACS window.
APPLIED_KEY = "val_measmaps_applied"

#: One coil has no sensitivity *profile* to estimate: RSS-normalised maps are a
#: unit-modulus phase, so the magnitude metrics would equal the oracle ones.
SINGLE_COIL_REASON = "single_coil"

#: Image metrics re-scored under the measured maps, intersected with the arm's
#: configured set. A fixed list keeps the extra cost bounded (no second LPIPS)
#: and makes the emitted key set declarable.
MEASURED_IMAGE_METRICS: tuple[str, ...] = (
    "psnr",
    "robust_mri_psnr",
    "hfen",
    "ssim",
    "power_spectrum_consistency",
)

#: The transform whose output depends on the coil maps. Under any other the
#: image metrics are map-free and re-scoring them would only duplicate them.
_MAP_READING_TRANSFORM = "ifft_sense_adjoint"


@dataclass(frozen=True)
class MeasuredMapStash:
    """One rung's measured maps, or the declared reason there are none."""

    maps: torch.Tensor | None
    skip_reason: str | None = None


def measured_maps_declared(config: Any) -> bool:
    """Whether ``validation.scoring.measurement_maps`` asks for the family.

    The value is validated elsewhere, once each: the schema's ``Literal`` refuses
    an unknown mode at load, and the ``measurement_maps_need_coil_maps`` witness
    refuses the knob on a model validated without coil maps. Read with
    ``getattr`` because hand-built test hosts carry partial validation blocks; an
    absent field reads as the schema default, ``"off"``.
    """
    scoring = getattr(getattr(config, "validation", None), "scoring", None)
    return getattr(scoring, "measurement_maps", "off") == "metrics"


def measurement_maps_defect(mode: object, model_type: object) -> str | None:
    """The finding for a declared mode the model cannot honour, or None."""
    if mode in (None, "off"):
        return None
    if mode not in MEASURED_MAP_MODES:
        return (
            f"validation.scoring.measurement_maps={mode!r}; expected one of {MEASURED_MAP_MODES}."
        )
    if not takes_coil_maps(model_type):
        return (
            f"validation.scoring.measurement_maps={mode!r} needs a model validated with "
            f"coil maps (kspace_cold_diffusion); model_type={model_type!r} takes none, so "
            "the measured-map family would never be scored."
        )
    return None


def measured_map_stash(
    masked_measurement: torch.Tensor, *, config: Any, log_scaled: bool
) -> MeasuredMapStash:
    """Maps calibrated from the rung's masked measurement, the model's own input."""
    maps = estimate_smaps_from_measurement(masked_measurement, config=config, log_scaled=log_scaled)
    if maps is None:
        raise ConfigurationError(
            "physics.coil_processing.estimation resolved to no maps; the measured-map "
            "family cannot be scored."
        )
    if maps.shape[1] < 2:
        return MeasuredMapStash(maps=None, skip_reason=SINGLE_COIL_REASON)
    return MeasuredMapStash(maps=maps)


def measured_mean_bases(configured: Iterable[str], measurement_aware: Iterable[str]) -> list[str]:
    """Base names whose ``_<R>x`` columns the cascade averages into ``_mean``."""
    rescored = set(MEASURED_IMAGE_METRICS) | set(measurement_aware)
    return [f"{name}{MEASURED_MAPS_SUFFIX}" for name in configured if name in rescored]


def declared_measured_keys(measurement_aware: Iterable[str]) -> frozenset[str]:
    """Every key the family can write, for ``capabilities.emitted_metrics``."""
    bases = [*MEASURED_IMAGE_METRICS, *measurement_aware]
    keys = {f"val_{b}{MEASURED_MAPS_SUFFIX}" for b in bases}
    keys |= {f"val_zf_{b}{MEASURED_MAPS_SUFFIX}" for b in ("psnr", "hfen")}
    keys |= {
        f"val_hfen{MEASURED_MAPS_SUFFIX}_vs_zf_mean",
        f"val_hfen{MEASURED_MAPS_SUFFIX}_vs_zf_mean_partial",
        APPLIED_KEY,
    }
    return frozenset(keys)


def measured_map_family(
    host: Any,
    stash: object,
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_filled: torch.Tensor | None,
    computer: Any,
    mask: torch.Tensor | None,
    transform: str,
    data_range: float,
) -> dict[str, float]:
    """The family for one rung of ``host`` (a ``DiffusionTrainingStrategy``).

    Module-level so a ``MagicMock`` host, whose stash is never a
    :class:`MeasuredMapStash`, gets ``{}`` rather than an auto-stubbed method's
    return value. ``{}`` also when the arm did not declare the family; the
    applied flag at ``0.0`` with a logged reason when it declared it and the
    rung has one coil.
    """
    if not isinstance(stash, MeasuredMapStash):
        return {}
    if stash.maps is None:
        if not getattr(host, "_measured_maps_skip_logged", False):
            host.logging_service.log_info(
                f"[Validation Metrics] measurement_maps declared but skipped "
                f"({stash.skip_reason}): one coil has no sensitivity profile, so the "
                f"_measmaps family would repeat the oracle magnitudes; {APPLIED_KEY}=0."
            )
            host._measured_maps_skip_logged = True
        return {APPLIED_KEY: 0.0}

    def _views(p: torch.Tensor, t: torch.Tensor, maps: torch.Tensor) -> Any:
        return host._metric_transform_views(p, t, host.config.validation, smaps=maps)

    def _aware(p: torch.Tensor, maps: torch.Tensor) -> dict[str, float]:
        if zero_filled is None:
            return {}
        return host._measurement_aware_metrics(p, target, zero_filled, computer, mask, smaps=maps)

    return score_with_measured_maps(
        maps=stash.maps,
        transform=transform,
        views=_views,
        measurement_aware=_aware,
        computer=computer,
        pred=pred,
        target=target,
        zero_filled=zero_filled,
        data_range=data_range,
        zf_metrics=host._ZF_BASELINE_METRICS,
    )


def score_with_measured_maps(
    *,
    maps: torch.Tensor,
    transform: str,
    views: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any],
    measurement_aware: Callable[[torch.Tensor, torch.Tensor], Mapping[str, float]],
    computer: Any,
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_filled: torch.Tensor | None,
    data_range: float,
    zf_metrics: tuple[str, ...],
) -> dict[str, float]:
    """The ``val_*_measmaps`` family for one rung.

    ``views(pred, target, maps)`` is the strategy's metric transform with the
    maps overridden; ``measurement_aware(pred, maps)`` is its trust-functional
    seam. ``data_range`` is the oracle family's, reused so the two families sit
    on one axis and differ only through the maps (and no host read is added).
    """
    out: dict[str, float] = {}
    if transform == _MAP_READING_TRANSFORM:
        scored = views(pred, target, maps).compute(
            computer, data_range=data_range, only=MEASURED_IMAGE_METRICS
        )
        out.update({f"val_{k}{MEASURED_MAPS_SUFFIX}": v for k, v in scored.items()})
        if zero_filled is not None:
            zf = views(zero_filled, target, maps).compute(
                computer, data_range=data_range, only=zf_metrics
            )
            out.update({f"val_zf_{k}{MEASURED_MAPS_SUFFIX}": v for k, v in zf.items()})
    aware = measurement_aware(pred, maps)
    out.update({f"val_{k}{MEASURED_MAPS_SUFFIX}": v for k, v in aware.items()})
    # Fraction of slices whose maps carry any energy: a rung with no line inside
    # the ACS window calibrates all-zero maps, and its family is then degenerate
    # (zero images). A device tensor, folded into the one fused host transfer of
    # ``_convert_metrics_to_floats``, so the key set stays rank-independent.
    out[APPLIED_KEY] = (maps.abs().flatten(1).amax(dim=1) > 0).float().mean()
    return out
