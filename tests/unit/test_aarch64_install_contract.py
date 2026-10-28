"""``pyproject.toml`` keeps the extras installable on linux/aarch64.

No CI lane runs on ARM64 any more (the self-hosted ``thor`` runner was retired
on 2026-10-28), but users install on Linux arm64 hosts, and two distributions
cannot be fetched prebuilt there. Their ``platform_machine != 'aarch64'``
markers are what keep ``pip install -e ".[dev]"`` from failing during
resolution, and these legs keep the markers scoped rather than deleted.

gudhi publishes no aarch64 wheel and no sdist for any release >= 3.12 (PyPI JSON
API, 2026-09-04). POT is deliberately NOT in the set -- it ships
manylinux_2_27_aarch64 wheels plus an sdist.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_REPO = Path(__file__).resolve().parents[2]

# Distributions that must NOT survive extras resolution on linux/aarch64.
#
# Derived by sweep, not by anecdote. The first entry was added from the one
# failure that happened to be observed, which is how a set like this silently
# goes stale -- so it was re-derived exhaustively on 2026-09-04 by two passes:
#   (i)  every one of the 257 packages in the tracked uv.lock, bucketed by
#        whether it offers a pure wheel, an aarch64 wheel, or only an sdist;
#   (ii) the PyPI simple API for each survivor, across ALL published versions,
#        because CI installs with pip and is not bound to the uv.lock pins.
# That pass found scalene, which pass (i) alone would have rated "buildable".
#
# The two entries fail for DIFFERENT reasons and the distinction is load-bearing:
#   gudhi   -- 332 wheels, 0 aarch64, 0 sdists: pip has nothing to install.
#   scalene -- 1231 wheels, 0 linux-aarch64 (its 17 arm wheels are macOS, which
#              reports platform_machine 'arm64'), sdist present, 4 compiled
#              objects: installable only by compiling C++ on the runner.
# Re-run both passes before trusting this set; membership moves with upstream.
UNAVAILABLE_ON_AARCH64 = {"gudhi", "scalene"}


def _resolve(extra: str, machine: str, seen: set[str] | None = None) -> set[str]:
    """Distribution names an extra pulls in on ``machine``, following self-refs."""
    seen = seen if seen is not None else set()
    if extra in seen:
        return set()
    seen.add(extra)
    with (_REPO / "pyproject.toml").open("rb") as fh:
        table = tomllib.load(fh)["project"]["optional-dependencies"]
    names: set[str] = set()
    for spec in table.get(extra, []):
        req = Requirement(spec)
        if req.marker and not req.marker.evaluate({"platform_machine": machine}):
            continue
        # `spectramr[a,b]` self-references compose the extras by reference.
        if req.name.replace("_", "-") == "spectramr":
            for nested in req.extras:
                names |= _resolve(nested, machine, seen)
        else:
            names.add(req.name.lower())
    return names


def _base_dependencies(machine: str) -> set[str]:
    """Distribution names in ``project.dependencies`` on ``machine``.

    A second scan root, for the same reason the overlay is one above. Every
    other leg in this module resolves ``optional-dependencies`` -- so the base
    requirement list, which is installed unconditionally and is where torch
    lives, was invisible to all of them. A package that becomes aarch64-hostile
    *there* would break the runner with every extras-based leg still green,
    because the blind spot and any planted violation live in different tables.
    """
    with (_REPO / "pyproject.toml").open("rb") as fh:
        specs = tomllib.load(fh)["project"]["dependencies"]
    names: set[str] = set()
    for spec in specs:
        req = Requirement(spec)
        if req.marker and not req.marker.evaluate({"platform_machine": machine}):
            continue
        names.add(req.name.lower())
    return names


def test_base_dependencies_are_installable_on_aarch64() -> None:
    """The unconditional requirement list must not carry an aarch64 blocker.

    Benign when written (the 2026-09-04 sweep put every blocker in an extra);
    this leg is what keeps it that way, since nothing else here reads the table.
    """
    base = _base_dependencies("aarch64")
    assert base, "project.dependencies resolved to nothing -- lookup is broken"
    offenders = base & UNAVAILABLE_ON_AARCH64
    assert not offenders, (
        f"{sorted(offenders)} are unconditional dependencies but cannot be "
        f"installed on linux/aarch64, so arm64 hosts cannot install "
        f"this package at all. Move them to an extra with a platform_machine "
        f"marker, as topology/gudhi and profile/scalene already are."
    )


@pytest.mark.parametrize("name", sorted(UNAVAILABLE_ON_AARCH64))
def test_marker_is_arch_scoped_not_a_deletion(name: str) -> None:
    """Each excluded distribution must still install on x86_64.

    Deleting the dependency outright would also make the test above pass, and
    would silently disable the cubical_ph_w2 arms (gudhi) or `spectramr profile`
    (scalene) on every machine that can run them. The marker is an architecture
    scope, not a removal, and this is the leg that tells the two apart.
    """
    assert name in _resolve("dev", "x86_64"), f"{name} was deleted, not scoped"
    assert name not in _resolve("dev", "aarch64")


@pytest.mark.parametrize("extra", ["topology", "all", "dev"])
def test_pot_stays_available_on_aarch64(extra: str) -> None:
    """POT ships aarch64 wheels; marking it off would be an over-correction."""
    assert "pot" in _resolve(extra, "aarch64")
