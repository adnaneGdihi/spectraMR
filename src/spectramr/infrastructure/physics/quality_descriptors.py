"""Measured quality attributes of a volume — the target vocabulary for chain fitting.

Attribute names ARE registered metric names. There is deliberately no parallel
attribute-to-metric table: a hand-maintained mapping is what drifted out of sync in
``scripts/sim2rank/simulator_calibration.py`` (issue #301), leaving 23 of 27 axes on a
metadata-only fallback while the headline curves silently never rendered.

Geometry is separated from quality on purpose. Voxel spacing is a *header fact* read
from the ISMRMRD ``fieldOfView_mm / matrixSize`` pair; it is imposed on a synthesis
target, never fitted against a sharpness proxy, because a blur term would otherwise
absorb a geometry error and report a good residual on a wrong voxel size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.core.metrics.registry import MetricsRegistry, get_metric
from spectramr.data.nifti_export import parse_ismrmrd_geometry

__all__ = [
    "QualityDescriptor",
    "UnmeasurableAttributeError",
    "measure",
    "measure_attributes",
    "read_spacing_mm",
    "resample_to_spacing",
    "validate_attributes",
]


class UnmeasurableAttributeError(ValueError):
    """A requested target attribute cannot be measured from a single volume."""


@dataclass(frozen=True, slots=True)
class QualityDescriptor:
    """What a volume's quality IS: its geometry plus its measured attributes."""

    spacing_mm: tuple[float, float, float]  # (slice, row, col)
    attributes: Mapping[str, float]

    def vector(self, keys: Sequence[str]) -> tuple[float, ...]:
        """Attribute values in the caller's order — the fitter's residual basis."""
        return tuple(float(self.attributes[k]) for k in keys)


def validate_attributes(attributes: Sequence[str]) -> None:
    """Raise unless every attribute is a registered no-reference metric.

    Called at config-load time so an unmeasurable target fails before a GPU is
    touched, rather than degrading to a silently skipped attribute (pitfall #15).
    """
    for name in attributes:
        if not MetricsRegistry.is_registered(name):
            raise UnmeasurableAttributeError(
                f"{name!r} is not a registered metric, so it cannot be measured or "
                "matched. List the options with "
                "spectramr.core.metrics.registry.list_available()."
            )
        if MetricsRegistry.requires_reference(name):
            raise UnmeasurableAttributeError(
                f"{name!r} requires a reference image. A quality target is measured "
                "from one volume with no ground truth, so only no-reference metrics "
                "can be matched."
            )


def _as_slice_stack(volume: torch.Tensor) -> torch.Tensor:
    """Coerce ``[S, H, W]`` or ``[H, W]`` to ``[S, H, W]``."""
    vol = volume if volume.dim() == 3 else volume.unsqueeze(0)
    if vol.dim() != 3:
        raise ValueError(f"volume must be [S, H, W] or [H, W]; got shape {tuple(volume.shape)}")
    return vol


def measure_attributes(
    volume: torch.Tensor,
    *,
    attributes: Sequence[str],
) -> dict[str, float]:
    """Attribute values only, skipping geometry -- the fitter's inner-loop path.

    Degradation does not change voxel spacing, so re-parsing the ISMRMRD XML on every
    objective evaluation would be hundreds of redundant XML parses per fit.
    """
    validate_attributes(attributes)
    # Metrics take [B, C, H, W]; the slice axis is the batch.
    batch = _as_slice_stack(volume).unsqueeze(1).float()
    # Called with prediction only: `validate_attributes` has already established
    # every name is `requires_reference=False`, and such metrics declare
    # `target: torch.Tensor | None = None` (the convention used at every existing
    # no-reference call site). The `IMetric` Protocol models only the
    # full-reference contract, so the one-argument form needs a local ignore.
    return {
        name: float(get_metric(name)(batch))  # type: ignore[call-arg]
        for name in attributes
    }


def read_spacing_mm(header: Any, volume: torch.Tensor) -> tuple[float, float, float]:
    """``(slice, row, col)`` mm for ``volume``, from its ISMRMRD header.

    Spacing comes from the header's own ``fieldOfView_mm / matrixSize`` pair, never
    from the array size: ``reconstruction_rss`` is a centre crop, so dividing FOV by
    the array size inflates the voxel.

    Raises rather than assuming: an unparseable header must stop a quality match,
    because a fabricated 1 mm isotropic default would silently resample every volume
    onto the wrong grid.
    """
    vol = _as_slice_stack(volume)
    geom = parse_ismrmrd_geometry(
        header,
        n_slices=int(vol.shape[0]),
        rows=int(vol.shape[1]),
        cols=int(vol.shape[2]),
    )
    return (float(geom.slice_mm), float(geom.row_mm), float(geom.col_mm))


def resample_to_spacing(
    volume: torch.Tensor,
    src_spacing_mm: tuple[float, float, float],
    dst_spacing_mm: tuple[float, float, float],
    *,
    return_to_source_grid: bool = True,
) -> torch.Tensor:
    """Impose ``dst_spacing_mm`` on an ``[S, H, W]`` volume, all three axes.

    This is how the GEOMETRIC half of a quality match is applied. Resolution is a
    header fact, so it is imposed here rather than fitted -- fitting it against a
    sharpness proxy would let a blur term absorb a geometry error and still report a
    good residual on a wrong voxel size.

    Coarser target spacing means fewer samples across the same FOV, so the volume is
    **area**-resampled down to the target grid. With ``return_to_source_grid``
    (default) it is then interpolated back to the original array shape: a paired
    restoration set needs input and target on the SAME grid, and the resolution loss
    survives the round trip as a genuinely broadened PSF, which is the point.

    **The slice axis is included, and that is deliberate.** Averaging *down* to a
    thicker slice fabricates nothing -- it is exactly what a thicker slice physically
    measures, since partial volume IS signal averaging across the slab. Only the
    opposite direction invents data.

    Raises:
        ValueError: any target spacing is FINER than the source. Interpolating up
            would invent resolution that was never acquired, in-plane or
            through-plane, and a synthetic "low-quality" volume carrying invented
            detail is worse than useless.
    """
    import torch.nn.functional as F

    vol = _as_slice_stack(volume)
    n_slices, rows, cols = vol.shape
    src = tuple(float(v) for v in src_spacing_mm)
    dst = tuple(float(v) for v in dst_spacing_mm)
    if min(src) <= 0.0 or min(dst) <= 0.0:
        raise ValueError(f"spacings must be positive; got src={src}, dst={dst}")

    finer = [
        (axis, s, d)
        for axis, s, d in zip(("slice", "row", "col"), src, dst, strict=True)
        if d < s - 1e-9
    ]
    if finer:
        detail = ", ".join(f"{a}: {s:g} -> {d:g} mm" for a, s, d in finer)
        raise ValueError(
            f"target spacing is FINER than the source on {detail}. Resampling up "
            "would invent resolution that was never acquired; a synthetic "
            "low-quality volume carrying invented detail is worse than useless. "
            "Check which cohort is the source."
        )

    # Same FOV, coarser voxel -> fewer samples. At least one sample per axis.
    new_shape = (
        max(1, round(n_slices * src[0] / dst[0])),
        max(1, round(rows * src[1] / dst[1])),
        max(1, round(cols * src[2] / dst[2])),
    )
    if new_shape == (n_slices, rows, cols):
        return vol

    # 5-D [N, C, S, H, W] so one call handles all three axes uniformly. `area` is
    # the correct downsampler: it AVERAGES, which is what both a larger in-plane
    # voxel and a thicker slice do to the signal.
    x5 = vol.float()[None, None]
    coarse = F.interpolate(x5, size=new_shape, mode="area")
    if not return_to_source_grid:
        return coarse[0, 0]
    restored = F.interpolate(
        coarse, size=(n_slices, rows, cols), mode="trilinear", align_corners=False
    )
    return restored[0, 0]


def measure(
    volume: torch.Tensor,
    header: Any,
    *,
    attributes: Sequence[str],
) -> QualityDescriptor:
    """Measure ``volume``'s descriptor. ``volume`` is ``[S, H, W]`` or ``[H, W]``.

    Spacing comes from the header's own ``fieldOfView_mm / matrixSize`` pair, never
    from the array size: ``reconstruction_rss`` is a centre crop, so dividing by the
    array size inflates the voxel.
    """
    vol = _as_slice_stack(volume)
    n_slices, rows, cols = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
    geom = parse_ismrmrd_geometry(header, n_slices=n_slices, rows=rows, cols=cols)

    return QualityDescriptor(
        spacing_mm=(float(geom.slice_mm), float(geom.row_mm), float(geom.col_mm)),
        attributes=measure_attributes(vol, attributes=attributes),
    )
