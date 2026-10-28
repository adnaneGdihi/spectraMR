"""Measurement exemptions in ``declarations`` stay explicit: each entry needs evidence."""

from __future__ import annotations

from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY
from spectramr.infrastructure.physics.degradations.declarations import DECLARED_PHYSICAL_FOV_MM


def test_the_physical_fov_declaration_is_pinned() -> None:
    # A new entry exempts an axis from the unit-voxel suites; add it here with its evidence.
    assert dict(DECLARED_PHYSICAL_FOV_MM) == {"dielectric_b1": 192.0}


def test_every_physical_fov_entry_is_a_registry_axis() -> None:
    assert set(DECLARED_PHYSICAL_FOV_MM) <= set(DEGRADATION_REGISTRY)
