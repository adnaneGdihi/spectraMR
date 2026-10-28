"""Tests for ``scripts/ci/report_discarded_config_keys.py``.

The census answers one question -- which declared keys does the schema drop --
and it can be wrong in two directions. Under-reporting is the obvious one. The
one that actually bit is the other: a config that **never loaded** was scored
the same as a config that loaded and was clean, because both paths returned
``[]``. The roll-up then printed ``scanned N`` over both populations together.

That is not hypothetical, and it is not confined to the corpus. Run against
``src/spectramr/config/presets`` -- three YAMLs that ship *inside the package* --
the pre-fix tool printed:

    scanned 3 config(s) under src/spectramr/config/presets
    0 declaration(s) SILENTLY DISCARDED across 0 key(s), in 0 arm(s)
      (none)

A clean bill of health over a directory where nothing loaded at all. The three
presets declare ``config_version: '6.0'``, which the loader rejects outright.

So the contract pinned here is a **tri-state**, not a count: ``None`` for
did-not-load, ``[]`` for loaded-and-clean, a non-empty list for loaded-and-dirty.
A test that only asserted "no discarded keys reported" would have passed against
the broken version -- so each test below is written to fail if the two silent
states are collapsed back together.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "ci" / "report_discarded_config_keys.py"
# Globbed, not spelled: the package directory is `spectramr` today and `spectramr`
# after the rename lands, and this module's fixtures RAISE on a missing file --
# so a hardcoded path would turn the rename merge red here rather than at the
# rename.
_REFERENCE = next(_REPO.glob("src/*/config/schemas/templates/v1.0_reference.yaml"))


@pytest.fixture(scope="module")
def census():
    """The census module. Failure RAISES -- a skip would make a deleted script
    read as a green test file."""
    spec = importlib.util.spec_from_file_location("_discarded_census", _SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _reference_plus_phantom_key(tmp_path: Path, name: str = "dirty.yaml") -> Path:
    """The reference template with one key no schema field backs.

    Built by a YAML round-trip rather than a string splice: splicing after the
    first ``checkpoint:`` match lands at whatever indentation that line happens
    to have, which produced a file that did not parse -- i.e. a fixture that
    exercised the did-not-load path while claiming to exercise the dirty one.
    `checkpoint` is `extra="ignore"`, so the key is dropped rather than refused.
    """
    import yaml

    doc = yaml.safe_load(_REFERENCE.read_text(encoding="utf-8"))
    doc.setdefault("checkpoint", {})["no_such_key_at_all"] = 1
    return _write(tmp_path, name, yaml.safe_dump(doc, sort_keys=False))


class TestTheThreeStatesAreDistinguishable:
    """``None`` / ``[]`` / non-empty must not collapse into each other."""

    def test_a_config_that_does_not_load_yields_none(self, census, tmp_path) -> None:
        """The state the old code spelled ``[]`` -- indistinguishable from clean."""
        arm = _write(tmp_path, "unsupported.yaml", "config_version: '6.0'\nrun:\n  name: x\n")
        assert census.discarded_keys(arm) is None

    def test_unparseable_yaml_also_yields_none(self, census, tmp_path) -> None:
        """A different load failure shape, same state. One `except` covers both,
        but only if the shapes are actually exercised."""
        arm = _write(tmp_path, "broken.yaml", "run: [unclosed\n")
        assert census.discarded_keys(arm) is None

    def test_a_clean_config_yields_an_empty_list_not_none(self, census) -> None:
        """The distinction has to hold from the other side too, or `None` is
        just a rename of `[]`. Sourced from the reference template the schema
        tests already keep honest, not from a hand-written fixture."""
        assert _REFERENCE.exists(), f"missing producer fixture: {_REFERENCE}"
        result = census.discarded_keys(_REFERENCE)
        assert result is not None, "the reference template must load"
        assert isinstance(result, list)

    def test_a_discarded_key_is_reported_with_its_block(self, census, tmp_path) -> None:
        """A key `CheckpointConfigSchema` does not declare is dropped: the block is
        `extra="ignore"`, so the declaration never reaches the run."""
        arm = _reference_plus_phantom_key(tmp_path)
        result = census.discarded_keys(arm)
        assert result is not None, "fixture must still load; only the key is bogus"
        assert ("checkpoint", "no_such_key_at_all") in result


class TestTheRollUpNamesTheUnreadFiles:
    """A count alone cannot carry this: the failure was a *reassuring* number."""

    def test_a_root_where_nothing_loads_does_not_read_as_clean(self, tmp_path) -> None:
        _write(tmp_path, "a.yaml", "config_version: '6.0'\nrun:\n  name: a\n")
        _write(tmp_path, "b.yaml", "config_version: '6.0'\nrun:\n  name: b\n")
        out = subprocess.run(
            [sys.executable, str(_SCRIPT), str(tmp_path)],
            capture_output=True, text=True, cwd=_REPO,
        ).stdout
        assert "0 read, 2 did not load" in out, out
        assert "DID NOT LOAD" in out, out
        # the specific regression: the old output said only this, and stopped.
        assert "a.yaml" in out and "b.yaml" in out, (
            "the unread files must be NAMED. A count tells the reader something "
            "was skipped; only the list tells them which coverage they lack.\n" + out
        )

    def test_the_scanned_line_splits_read_from_unread(self, tmp_path) -> None:
        _write(tmp_path, "ok.yaml", _REFERENCE.read_text(encoding="utf-8"))
        _write(tmp_path, "bad.yaml", "config_version: '6.0'\nrun:\n  name: b\n")
        out = subprocess.run(
            [sys.executable, str(_SCRIPT), str(tmp_path)],
            capture_output=True, text=True, cwd=_REPO,
        ).stdout
        assert "scanned 2 config(s)" in out, out
        assert "1 read, 1 did not load" in out, out


class TestTheGateStillGatesOnDiscardedKeysOnly:
    """`--max` counts discarded declarations. A load failure is a *different*
    gate's finding (`check_experiment_configs_load`), and folding it in here
    would give one defect two owners -- the thing non-negotiable 17 forbids."""

    def test_unreadable_configs_alone_do_not_trip_the_gate(self, tmp_path) -> None:
        _write(tmp_path, "bad.yaml", "config_version: '6.0'\nrun:\n  name: b\n")
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), str(tmp_path), "--max", "0"],
            capture_output=True, text=True, cwd=_REPO,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_a_discarded_key_does_trip_the_gate(self, tmp_path) -> None:
        _reference_plus_phantom_key(tmp_path)
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), str(tmp_path), "--max", "0"],
            capture_output=True, text=True, cwd=_REPO,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
