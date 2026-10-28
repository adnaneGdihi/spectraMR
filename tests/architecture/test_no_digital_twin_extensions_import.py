"""The retired ``digital_twin_extensions`` module is not imported or named anywhere.

Its contents live in ``spectramr.infrastructure.physics.degradations``; a surviving reference is
either a stale import or a path that would resurrect a second owner.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("src", "scripts", "tests")
# A bare module token: ``test_digital_twin_extensions.py`` (a test file name) is not the module.
_NAME = re.compile(r"(?<![A-Za-z0-9_])digital_twin_extensions(?![A-Za-z0-9_])")


def references(source: str) -> list[int]:
    """Line numbers on which ``source`` names the retired module."""
    return [i for i, line in enumerate(source.splitlines(), 1) if _NAME.search(line)]


def test_no_python_file_names_digital_twin_extensions() -> None:
    offenders = []
    for top in SCANNED:
        for path in sorted((REPO_ROOT / top).rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            offenders += [
                f"{path.relative_to(REPO_ROOT)}:{n}"
                for n in references(path.read_text(encoding="utf-8", errors="replace"))
            ]
    assert offenders == [], offenders


@pytest.mark.parametrize(
    "planted",
    [
        "from spectramr.infrastructure.physics.digital_twin_extensions import apply_degradation",
        "from spectramr.infrastructure.physics import digital_twin_extensions as ext",
        "import spectramr.infrastructure.physics.digital_twin_extensions",
        "mock.patch('spectramr.infrastructure.physics.digital_twin_extensions.apply_degradation')",
        "    from . import digital_twin_extensions  # function-local",
        "# see digital_twin_extensions.py",
    ],
)
def test_planted_reference_is_flagged(planted: str) -> None:
    assert references(planted) == [1]


@pytest.mark.parametrize(
    "clean",
    [
        "tests/unit/infrastructure/physics/test_digital_twin_extensions.py",
        "from spectramr.infrastructure.physics.degradations import apply_degradation",
    ],
)
def test_unrelated_text_is_not_flagged(clean: str) -> None:
    assert references(clean) == []
