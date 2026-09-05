"""The Makefile targets CI invokes must not assume a local .venv.

`make test-nightly` is the entry point of `manual-full-suite.yml` (and of the cluster
array). A hosted runner has no .venv, so an unguarded `. .venv/bin/activate` kills the
recipe on its first line -- the suite never runs, and the failure reads as a shell error
rather than a test failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAKEFILE = _REPO_ROOT / "Makefile"

# A target definition -- `name:` -- and not a variable assignment (`VENV :=`,
# `FUZZ_RUNS ?=`) nor a special target (`.PHONY:`).
_TARGET_RE = re.compile(r"^(?!\.)([A-Za-z0-9_][A-Za-z0-9_.\-]*)[ \t]*:(?![=])", re.MULTILINE)

# Lane targets a CI workflow invokes directly. Kept ONLY as a non-vacuity floor for
# the discovery below -- never as the scanned set. This list is what made this gate
# blind: `test-mutation` was absent from it, and `test-release`, which was present,
# has the one-line recipe `$(MAKE) test-mutation`, so the offending recipe was never
# read. `Test (release)` run 33938180442 died on exactly that line
# (`/bin/sh: 1: .: cannot open .venv/bin/activate`, `Makefile:369 ... Error 2`) with
# this gate green. Enumerating the targets to check is the same second-owner defect
# the rule warns about, so the scan now derives them.
_KNOWN_LANE_TARGETS = frozenset(
    {"test-precommit", "test-pr", "test-nightly", "test-release", "cov-unit", "cov-full"}
)


def _targets() -> tuple[str, ...]:
    """Every target defined in the Makefile, in file order."""
    return tuple(dict.fromkeys(_TARGET_RE.findall(_MAKEFILE.read_text())))


def _recipe(target: str) -> str:
    """The tab-indented body of a Makefile target."""
    text = _MAKEFILE.read_text()
    match = re.search(rf"^{re.escape(target)}[ \t]*:.*?\n((?:\t.*\n|\n)*)", text, re.MULTILINE)
    assert match is not None, f"target {target!r} not found in Makefile"
    return match.group(1)


def test_target_discovery_is_not_vacuous() -> None:
    """A scanner that finds nothing passes every assertion below it.

    The failure this gate exists for is invisible to an empty scan, so the scan
    itself is asserted before anything is derived from it.
    """
    found = set(_targets())
    missing = _KNOWN_LANE_TARGETS - found
    assert not missing, (
        f"target discovery missed known lane target(s) {sorted(missing)} -- "
        f"_TARGET_RE is broken, and every check below it would pass vacuously. "
        f"Found {len(found)} target(s)."
    )


@pytest.mark.parametrize("target", _targets())
def test_target_does_not_activate_the_venv_unconditionally(target: str) -> None:
    """EVERY target, not a hand-listed subset.

    `.` is a POSIX *special builtin*: when it cannot open its argument the shell
    exits outright, so `if . .venv/bin/activate && python -c "import mutmut"` does
    not fall through to its `else` -- the "skips gracefully when not installed"
    branch is unreachable on a runner with no .venv, and `make` reports Error 2.
    """
    recipe = _recipe(target)
    assert ". .venv/bin/activate" not in recipe, (
        f"`make {target}` activates .venv unconditionally, so it dies on a hosted "
        f"runner where no .venv exists. Use the $(VENV) guard."
    )


def test_venv_guard_is_defined_and_conditional() -> None:
    text = _MAKEFILE.read_text()
    match = re.search(r"^VENV\s*:?=(.*)$", text, re.MULTILINE)
    assert match is not None, "Makefile defines no VENV guard"
    assert "-f .venv/bin/activate" in match.group(1)


def test_format_target_uses_ruff_not_black() -> None:
    """One formatter. black is not a declared dependency and fights ruff-format."""
    recipe = _recipe("format")
    assert "ruff format" in recipe
    assert "black" not in recipe


def test_black_is_not_a_declared_dependency() -> None:
    """`make format` ran black while no extra declared it -- it was never installable."""
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    assert "black" not in pyproject
