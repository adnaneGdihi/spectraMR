"""The degradation registry: every family's ``ENTRIES`` in canonical order, and its entry point.

The read-only queries over it live in :mod:`registry_queries`.

The deterministic-seed contract is the key requirement for the ranking pipeline
downstream (Sobol indices, mutual information, Bradley-Terry pairs): the same
``(theta, seed)`` MUST produce the same ``hat{x}`` so the same forward model can be
evaluated by many metrics (Saltelli et al., CPC 181:259-270, 2010).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch

from spectramr.infrastructure.physics.degradations._common import derive_axis_seed
from spectramr.infrastructure.physics.degradations.families import (
    catalogue_field_contrast,
    catalogue_flow,
    catalogue_hardware,
    catalogue_metal_chemistry,
    catalogue_multislice,
    catalogue_receive,
    catalogue_reconstruction,
    catalogue_sequence,
    catalogue_transmit,
    field,
    field_kspace,
    geometric,
    motion,
    noise,
    sampling,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    resolve_geometry,
)
from spectramr.infrastructure.physics.degradations.spec import (
    DegradationSpec,
    MissingInput,
    Requirement,
    UnsupportedAcquisitionMode,
)

if TYPE_CHECKING:
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

__all__ = [
    "DEGRADATION_REGISTRY",
    "apply_degradation",
    "check_requirements",
    "derive_axis_seed",
]

# The canonical order: list_degradations() and every sweep iterate the registry in it.
_ORDER: tuple[str, ...] = (
    "complex_gaussian",
    "rician",
    "emi",
    "cartesian_undersamp",
    "vd_undersamp",
    "vd_cartesian",
    "radial_undersamp",
    "rigid_motion",
    "pulsatile_motion",
    "translation_drift",
    "sudden_motion",
    "random_motion",
    "nyquist_ghost",
    "gibbs",
    "b0",
    "b0_geometric_distortion",
    "bias",
    "t2star_blur",
    "slice_pve",
    "eddy",
    "spike",
    "chemical_shift",
    "susceptibility",
    "coil_miscal",
    "geometric_warp",
    "gradient_nonlinearity",
    "magic_angle",
    "concomitant_phase",
    "partial_fourier",
    "iq_ghost",
    "stimulated_echo",
    "resolution_snr",
    # Catalogue axes are appended so existing sweep indices do not move.
    "respiratory_motion",
    "local_intermittent_motion",
    "metal_artifact",
    "chemical_shift_phase",
    "flow_ghosts",
    "flow_voids",
    "inflow_enhancement",
    "b0_drift",
    "gradient_delay",
    "gradient_scaling_drift",
    "mechanical_vibration",
    "correlated_coil_noise",
    "emi_burst",
    "receiver_gain_drift",
    "adc_clipping",
    "adc_quantization",
    "antialias_rolloff",
    "coil_dropout",
    "b1_transmit",
    "dielectric_b1",
    "imperfect_fat_sat",
    "imperfect_inversion",
    "sar_limited_flip",
    "fov_foldover",
    "tse_t2_blur",
    "t1_shot_blur",
    "offres_blur_noncartesian",
    "slice_crosstalk",
    "through_plane_pve",
    "spin_history",
    "off_resonance_slice_shift",
    "gfactor_noise",
    "pi_residual_aliasing",
    "cs_recon_artifacts",
    "recon_apodization",
    "zero_fill_interp",
    "gradwarp_residual",
    "field_contrast",
    # D19 (ULF→HF domain gap) is data-paired: handled outside the simulator.
)


def _build(entries: tuple[DegradationSpec, ...]) -> dict[str, DegradationSpec]:
    by_name: dict[str, DegradationSpec] = {}
    for spec in entries:
        if spec.name in by_name:
            raise ValueError(f"degradation {spec.name!r} is declared by more than one family")
        by_name[spec.name] = spec
    if set(by_name) != set(_ORDER) or len(_ORDER) != len(set(_ORDER)):
        raise ValueError(
            "registry order and family ENTRIES disagree: "
            f"unordered {sorted(set(by_name) - set(_ORDER))}, "
            f"undeclared {sorted(set(_ORDER) - set(by_name))}"
        )
    return {name: by_name[name] for name in _ORDER}


DEGRADATION_REGISTRY: Mapping[str, DegradationSpec] = _build(
    noise.ENTRIES
    + sampling.ENTRIES
    + motion.ENTRIES
    + field_kspace.ENTRIES
    + field.ENTRIES
    + geometric.ENTRIES
    + catalogue_metal_chemistry.ENTRIES
    + catalogue_flow.ENTRIES
    + catalogue_hardware.ENTRIES
    + catalogue_receive.ENTRIES
    + catalogue_transmit.ENTRIES
    + catalogue_sequence.ENTRIES
    + catalogue_multislice.ENTRIES
    + catalogue_reconstruction.ENTRIES
    + catalogue_field_contrast.ENTRIES
)


_Check = Callable[[torch.Tensor, Mapping[str, Any], "ScannerProfile | None"], bool]


def _has_kwarg(key: str) -> _Check:
    return lambda x, kwargs, profile: key in kwargs


def _is_epi(x: torch.Tensor, kwargs: Mapping[str, Any], profile: ScannerProfile | None) -> bool:
    # Imported at call time: scanner_profile imports this package.
    from spectramr.infrastructure.physics.scanner_profile import Sequence

    return profile is not None and profile.sequence is Sequence.EPI


_REQUIREMENT_MET: dict[Requirement, _Check] = {
    Requirement.MULTICOIL: lambda x, kwargs, profile: x.shape[1] > 1,
    Requirement.QMAPS: _has_kwarg("qmaps"),
    Requirement.FAT_MAP: _has_kwarg("fat_map"),
    Requirement.VESSEL_MAP: _has_kwarg("vessel_map"),
    Requirement.TRAJECTORY_RADIAL: _has_kwarg("trajectory"),
    Requirement.SEQUENCE_EPI: _is_epi,
}
if set(_REQUIREMENT_MET) != set(Requirement):
    raise RuntimeError(f"no check for requirements {sorted(set(Requirement) - set(_REQUIREMENT_MET))}")


def check_requirements(
    spec: DegradationSpec,
    x: torch.Tensor,
    kwargs: Mapping[str, Any],
    profile: ScannerProfile | None = None,
) -> None:
    """Raise :class:`MissingInput` for the first declared requirement the call does not meet.

    ``SEQUENCE_EPI`` is a fact of the scanner profile, so it is unmet without one.
    """
    for req in sorted(spec.requires):
        if not _REQUIREMENT_MET[req](x, kwargs, profile):
            raise MissingInput(spec.name, req)


def apply_degradation(
    name: str,
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    *,
    geometry: AcquisitionGeometry | None = None,
    profile: ScannerProfile | None = None,
    **kwargs,
) -> torch.Tensor:
    """Single entry point: apply ``name`` degradation at severity ``theta``.

    Args:
        name:     Registry key from :data:`DEGRADATION_REGISTRY`.
        x:        Input image ``(B, C, H, W)`` or volume ``(B, C, D, H, W)`` (real or complex).
        theta:    Severity ``in [0, 1]`` (clamped).
        seed:     Per-call deterministic seed.
        geometry: How ``x`` was acquired; required for a volume, ``None`` means a 2-D slice.
        profile:  Scanner the call simulates. It supplies the geometry, and a spec's
                  ``condition`` turns it into theta-scaled kwargs.

    Returns:
        Degraded input, same shape. Complex-valued unless the specific degradation is
        intrinsically real (D2 / D12 with real input).

    Raises:
        UnsupportedAcquisitionMode: ``name`` has no implementation for the geometry's mode.
        MissingInput: a declared requirement of ``name`` was not supplied.
        ValueError: both ``geometry`` and ``profile`` given, or a caller kwarg overlaps the
            profile's conditioned kwargs.
    """
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(
            f"Unknown degradation '{name}'. Valid: {sorted(DEGRADATION_REGISTRY.keys())}"
        )
    if geometry is not None and profile is not None:
        raise ValueError(
            f"degradation {name!r}: pass geometry or profile, not both; the profile carries "
            "its own geometry"
        )
    spec = DEGRADATION_REGISTRY[name]
    geom = resolve_geometry(x, profile.geometry if profile is not None else geometry)
    impl = spec.implementations.get(geom.mode)
    if impl is None:
        raise UnsupportedAcquisitionMode(
            name,
            geom.mode,
            spec.supported_modes,
            detail=spec.unsupported_detail.get(geom.mode, ""),
        )
    check_requirements(spec, x, kwargs, profile)
    theta = float(max(0.0, min(1.0, theta)))
    if profile is not None and spec.condition is not None:
        cond = spec.condition(profile, theta, geom, tuple(x.shape))
        overlap = sorted(set(cond) & set(kwargs))
        if overlap:
            raise ValueError(
                f"degradation {name!r}: kwargs {overlap} are set by scanner profile "
                f"{profile.name!r}; drop them or drop the profile"
            )
        kwargs = {**kwargs, **cond}
    return impl(x, theta, seed, geom, **kwargs)
