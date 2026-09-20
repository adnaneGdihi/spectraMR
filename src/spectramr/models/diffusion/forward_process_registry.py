"""Registry for cold-diffusion forward processes.

The generator used to name ``KSpaceUndersamplingProcess`` at its construction
site, which made the acquisition model a property of the class rather than of
the config. A second acquisition -- golden-angle spokes -- turns that into a
choice, and non-negotiable 6 makes the choice a registry lookup rather than a
branch on a string.

Unknown names raise (non-negotiable 3): an acquisition model that silently fell
back to Cartesian masking would train, converge and report success while
degrading the data by a process the arm did not ask for.

**This module is the one owner of the name-to-class mapping** and states it
below, rather than collecting decorator side effects. A decorator registry is
recoverable only while its modules are unimported: once Python has cached them,
a registry that lost an entry cannot rebuild it, because re-importing re-runs no
decorator. Naming the members here makes population a pure function of this
file.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FORWARD_PROCESS_REGISTRY",
    "build_forward_process",
    "list_forward_processes",
]

#: Config-facing name -> the module path and class that implements it. Imports
#: are deferred to keep this module free of a cycle with the two it names.
_MEMBERS: dict[str, tuple[str, str]] = {
    "cartesian_mask": (
        "spectramr.models.diffusion.kspace_process",
        "KSpaceUndersamplingProcess",
    ),
    "golden_angle_spokes": (
        "spectramr.models.diffusion.noncartesian_spoke_process",
        "NonCartesianSpokeProcess",
    ),
}

FORWARD_PROCESS_REGISTRY: dict[str, type] = {}


def _populate() -> None:
    """Resolve every member, filling any entry that is absent.

    Idempotent and recoverable: it repairs a partially-filled registry instead
    of short-circuiting on the first entry, which is the failure
    ``.claude/rules/registries.md`` records for the model registry.
    """
    import importlib

    for name, (module_path, class_name) in _MEMBERS.items():
        if name in FORWARD_PROCESS_REGISTRY:
            continue
        FORWARD_PROCESS_REGISTRY[name] = getattr(importlib.import_module(module_path), class_name)


def list_forward_processes() -> list[str]:
    """Names an arm may declare, for error messages and the audit."""
    _populate()
    return sorted(FORWARD_PROCESS_REGISTRY)


def build_forward_process(name: str, **kwargs: Any) -> Any:
    """Resolve and construct a forward process, raising on an unknown name."""
    _populate()
    try:
        cls = FORWARD_PROCESS_REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown kspace_process_type {name!r}. Registered: "
            f"{', '.join(sorted(FORWARD_PROCESS_REGISTRY))}."
        ) from exc
    return cls(**kwargs)
