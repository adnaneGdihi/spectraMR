"""What the Blackwell ranker reads from a run: clean 2-D images (slices of volumes, cropped
to one shape), the metrics it can call on magnitude displays, and their contexts."""

from __future__ import annotations

import torch

from ..types import MetricEvaluationDataset, MetricSet

#: Slices taken from each 3-D volume, evenly spaced over its central 60%.
SLICES_PER_VOLUME = 4


def _slices(clean: torch.Tensor) -> list[torch.Tensor]:
    if clean.dim() == 2:
        return [clean]
    if clean.dim() == 3:
        return [clean[0]]
    if clean.dim() == 4:
        vol = clean[0]
        depth = vol.shape[0]
        picks = torch.linspace(0.2 * (depth - 1), 0.8 * (depth - 1), min(depth, SLICES_PER_VOLUME))
        return [vol[int(i)] for i in torch.unique(picks.round().long())]
    raise ValueError(f"blackwell reads (H, W), (C, H, W) or (C, D, H, W); got {tuple(clean.shape)}")


def _centre_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    top, left = (x.shape[-2] - h) // 2, (x.shape[-1] - w) // 2
    return x[..., top : top + h, left : left + w]


def read_contents(
    dataset: MetricEvaluationDataset, limit: int
) -> tuple[list[torch.Tensor], dict[str, object]]:
    """Up to ``limit`` clean 2-D images, one shape, and how they were obtained.

    A 3-D volume contributes :data:`SLICES_PER_VOLUME` central slices. Images of
    different sizes are centre-cropped to the smallest common shape: a clean reference
    only needs to be an image, and the crop is stamped. A zero image is skipped, because
    its noise scale (its RMS) is zero.
    """
    images: list[torch.Tensor] = []
    seen: set[str] = set()
    skipped = 0
    for s in dataset.samples:
        if s.content_id in seen:
            continue
        seen.add(s.content_id)
        for img in _slices(s.clean):
            if float(img.abs().max()) == 0.0:
                skipped += 1
                continue
            images.append(img)
            if len(images) == limit:
                break
        if len(images) == limit:
            break
    if len(images) < 2:
        raise ValueError(f"blackwell needs at least two non-zero 2-D contents, found {len(images)}")
    shapes = sorted({(int(x.shape[-2]), int(x.shape[-1])) for x in images})
    h, w = min(sh[0] for sh in shapes), min(sh[1] for sh in shapes)
    info: dict[str, object] = {
        "n_contents": len(images),
        "source_contents": len(seen),
        "source_shapes": [list(sh) for sh in shapes],
        "shape": [h, w],
        "cropped": len(shapes) > 1,
        "skipped_zero": skipped,
    }
    return [_centre_crop(x, h, w) for x in images], info


def split_metrics(metric_set: MetricSet) -> tuple[list[str], list[str]]:
    """Metrics to call, and ``requires_complex`` ones skipped: every display is magnitude."""
    from spectramr.core.metrics.registry import MetricsRegistry

    complex_only = [m for m in metric_set.names() if MetricsRegistry.requires_complex(m)]
    return [m for m in metric_set.names() if m not in complex_only], complex_only


def metric_contexts(names: list[str], contents: list[torch.Tensor]) -> dict[str, list[object]]:
    from spectramr.core.metrics.registry import MetricsRegistry

    from ..simulator import _build_sample_context

    out: dict[str, list[object]] = {}
    for name in names:
        if MetricsRegistry.needs_context(name):
            needs = MetricsRegistry.needs(name)
            out[name] = [_build_sample_context(c[None], needs) for c in contents]
    return out


def no_signal(status: str, finite: float | None) -> dict[str, object]:
    keys = (
        "concordance_sharp",
        "bvr_raw",
        "bvr_raw_certified",
        "bvr_sharp",
        "bvr_sharp_certified",
        "equivalence_display_rate",
        "equivalence_display_certified",
        "probe_violation_rate",
        "probe_violations_certified",
        "coherent",
        "coherence_status",
        "n_preferences",
        "preferences_set_aside",
        "incoherence_certificate",
        "kappa",
    )
    return {"status": status, "finite_fraction": finite, **dict.fromkeys(keys)}


__all__ = [
    "SLICES_PER_VOLUME",
    "metric_contexts",
    "no_signal",
    "read_contents",
    "split_metrics",
]
