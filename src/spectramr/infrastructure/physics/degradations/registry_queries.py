"""Read-only queries over :data:`DEGRADATION_REGISTRY`: names by mode, origin and component."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.declarations import (
    DECLARED_THETA_ZERO_PEDESTAL,
    PHASE,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.degradations.registry import DEGRADATION_REGISTRY

__all__ = [
    "affected_components",
    "get_degradation_origin",
    "is_identity_at_theta_zero",
    "is_phase_only",
    "list_degradations",
    "phase_only_degradations",
    "requirement_free_degradations",
    "slice_capable_degradations",
]


def slice_capable_degradations() -> tuple[str, ...]:
    """The registered axes that run on a 2-D slice, in canonical order (all but volume-only ones)."""
    return tuple(
        n for n, s in DEGRADATION_REGISTRY.items() if AcquisitionMode.SLICE_2D in s.implementations
    )


def requirement_free_degradations() -> tuple[str, ...]:
    """Canonical-order names whose ``requires`` is empty: runnable on a bare single-coil input."""
    return tuple(n for n, spec in DEGRADATION_REGISTRY.items() if not spec.requires)


def list_degradations(origin: ArtifactOrigin | None = None) -> list[str]:
    """The canonical ordered degradation names, optionally filtered by origin.

    Args:
        origin: if given, keep only degradations arising at that point in the
            imaging chain (scanner / acquisition / patient). ``None`` = all.
    """
    if origin is None:
        return list(DEGRADATION_REGISTRY.keys())
    return [n for n, spec in DEGRADATION_REGISTRY.items() if spec.origin is origin]


def get_degradation_origin(name: str) -> ArtifactOrigin:
    """Where in the imaging chain ``name`` arises. Raises on an unknown name."""
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(f"Unknown degradation '{name}'. Valid: {sorted(DEGRADATION_REGISTRY)}")
    return DEGRADATION_REGISTRY[name].origin


def affected_components(name: str) -> frozenset[str]:
    """Which image components ``name`` perturbs — a subset of {magnitude, phase}."""
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(f"Unknown degradation '{name}'. Valid: {sorted(DEGRADATION_REGISTRY)}")
    return DEGRADATION_REGISTRY[name].affects


def is_phase_only(name: str) -> bool:
    """True when ``name`` leaves ``|x|`` untouched (issue #223).

    A caller that scores the sweep on ``.abs()`` MUST route a phase-only axis to
    the phase metrics against the phase-bearing complex tensor instead — scoring
    it on the magnitude image scores a constant.
    """
    return affected_components(name) == frozenset({PHASE})


def phase_only_degradations() -> list[str]:
    """The registered axes that move phase and nothing else."""
    return [n for n, s in DEGRADATION_REGISTRY.items() if s.phase_only]


def is_identity_at_theta_zero(name: str) -> bool:
    """True when ``D(x, 0) == x`` is required of ``name`` (all but the declared)."""
    if name not in DEGRADATION_REGISTRY:
        raise KeyError(f"Unknown degradation '{name}'. Valid: {sorted(DEGRADATION_REGISTRY)}")
    return name not in DECLARED_THETA_ZERO_PEDESTAL
