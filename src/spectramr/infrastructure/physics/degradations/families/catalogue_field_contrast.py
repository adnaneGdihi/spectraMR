"""Catalogue family: field-strength contrast change from quantitative maps."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.catalogue.field_contrast import (
    degrade_field_contrast,
    field_contrast_condition,
    field_contrast_volume,
)
from spectramr.infrastructure.physics.degradations.declarations import MAG
from spectramr.infrastructure.physics.degradations.families import MS, VOL
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.degradations.spec import (
    Realisation,
    Requirement,
    SliceKernel,
    entry,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

__all__ = ["ENTRIES"]

_NATIVE = Realisation.NATIVE

ENTRIES = (
    entry(
        code="D70",
        name="field_contrast",
        family="field",
        description="Tissue contrast re-evaluated at a lower field strength from q-maps",
        severity=SeveritySpec(
            (PhysicalParam("target_field", "T", 3.0, 0.05),),
            notes="B0_target = B0_src x (0.05 / B0_src) ** theta, log-linear, B0_src = b0_T "
            "(default 3 T; the endpoint 3 T is that default). The output is x x S(B0_target) / "
            "S(B0_src) voxel-wise, S the stage-1 signal of DifferentiableULFForwardOperator "
            "(power-law T1/T2 transport, beta 0.35 / gamma 0.05, then the Ernst equation) with "
            "PD = 1, so PD cancels and theta = 0 is the exact identity. The result is then rescaled per sample so its mean |.| over the object support (|x| > 0.1 of its per-sample max) equals x's: the overall brightness is a separate gain / SNR axis, so only the tissue contrast changes. x is the image acquired "
            "at b0_T: it supplies shape, device, dtype and the intensity scale, and its phase is "
            "kept. Requires the kwarg qmaps = {'t1', 't2' (ms, shaped like x with C or 1 "
            "channels), optional 'pd' (shape-checked, cancels)}. TR / TE (ms) default to the "
            "operator's 500 / 14 when not given; sequence in {spoiled_gre (default, fa_deg "
            "default 90), spin_echo (90 deg)}, any other raises; ti_ms raises. Voxels whose T1 or "
            "T2 is not positive (background) get ratio 1 before the per-sample intensity match. "
            "Operator noise, GNL and Maxwell "
            "stages are off.",
        ),
        affects=MAG,
        origin=ArtifactOrigin.SCANNER,
        implementations={
            AcquisitionMode.SLICE_2D: SliceKernel(degrade_field_contrast),
            MS: field_contrast_volume,
            VOL: field_contrast_volume,
        },
        realisation={AcquisitionMode.SLICE_2D: _NATIVE, MS: _NATIVE, VOL: _NATIVE},
        requires=frozenset({Requirement.QMAPS}),
        condition=field_contrast_condition,
        active_when=frozenset(),
    ),
)
