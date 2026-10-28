"""ArtifactOrigin taxonomy over the degradation registry."""

from __future__ import annotations

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    get_degradation_origin,
    list_degradations,
)


def test_every_degradation_has_an_origin() -> None:
    # The registry rebuild indexes the origins table with [name], so an
    # unclassified degradation fails at import — this pins the invariant.
    assert all(spec.origin is not None for spec in DEGRADATION_REGISTRY.values())


def test_origin_partition_is_complete() -> None:
    total = len(list_degradations())
    by_origin = sum(
        len(list_degradations(origin=o)) for o in ArtifactOrigin
    )
    assert by_origin == total


def test_list_by_origin_filters() -> None:
    scanner = set(list_degradations(origin=ArtifactOrigin.SCANNER))
    patient = set(list_degradations(origin=ArtifactOrigin.PATIENT))
    assert scanner.isdisjoint(patient)
    assert "complex_gaussian" in scanner
    assert "rigid_motion" in patient


def test_get_degradation_origin() -> None:
    assert get_degradation_origin("cartesian_undersamp") is ArtifactOrigin.ACQUISITION
