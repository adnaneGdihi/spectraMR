"""Tests for :mod:`tests.utils.dropped_key_baseline` (issue #1053).

The baseline's whole value is that it stays a RATCHET rather than becoming a
waiver. So the tests that matter are the ones that fail if it quietly turns into
"everything is allowed": an unrecorded pair must still be reportable, and the
recorded set must keep describing real arms.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.corpus import repo_root
from tests.utils.dropped_key_baseline import (
    baseline_path,
    recorded_dropped_keys,
    recorded_pairs,
)

# The baseline records debt in ``experiments/inprogress/`` arms. Neither the
# baseline nor the corpus ships in the public export, and the two absences do
# NOT cancel: with the file missing, ``recorded_pairs()`` returns empty, so
# ``test_every_recorded_arm_exists_on_disk`` and
# ``test_recorded_keys_are_dotted_config_paths`` iterate nothing and report
# green -- the exact vacuity the first test's docstring names. Skip visibly
# instead. Shipping the baseline is NOT the alternative fix: it names 337 arms
# that do not ship, so the existence test would go red rather than vacuous.
if not baseline_path().exists():
    pytest.skip(
        "dropped-key baseline absent: it records experiments/inprogress debt, "
        "and neither it nor the corpus ships in the public export",
        allow_module_level=True,
    )


def test_the_baseline_file_exists_and_is_populated():
    """A missing or empty file would make every sweep vacuously green."""
    assert baseline_path().exists(), "the recorded debt file is missing"
    assert recorded_pairs(), "the baseline parsed to nothing — the sweep is now toothless"


def test_relative_and_absolute_paths_agree():
    """``tracked_yamls`` yields absolute paths; older call sites carry relative
    strings. Disagreement between the two silently empties the exemption for
    half the callers."""
    arm = next(iter(recorded_pairs()))

    assert recorded_dropped_keys(arm)
    assert recorded_dropped_keys(repo_root() / arm) == recorded_dropped_keys(arm)


def test_an_unrecorded_arm_gets_no_exemption():
    """The ratchet's teeth: absence from the file must mean 'not allowed'."""
    assert recorded_dropped_keys("experiments/inprogress/does_not_exist.yaml") == frozenset()


def test_a_path_outside_the_repo_is_not_exempted():
    """A stray absolute path must not silently resolve to a blanket exemption."""
    assert recorded_dropped_keys(Path("/etc/passwd")) == frozenset()


def test_an_unrecorded_key_on_a_recorded_arm_still_fails():
    """Per-``(arm, key)``, not per arm.

    Recording ``data.volume_format`` for an arm must not also excuse a NEW
    dropped key on that same arm — otherwise one recorded pair silently
    exempts the whole file.
    """
    arm, keys = next(iter(recorded_pairs().items()))

    assert "definitely_not_a_real_key" not in recorded_dropped_keys(arm)
    assert keys, f"{arm} recorded with no keys"


def test_every_recorded_arm_exists_on_disk():
    """A pair naming a deleted arm can never be ratcheted down by a sweep that
    no longer visits it, so it would sit in the file forever."""
    missing = [arm for arm in recorded_pairs() if not (repo_root() / arm).exists()]

    assert not missing, (
        f"{len(missing)} recorded arm(s) no longer exist — rerun "
        "`python -m tests.utils.dropped_key_baseline --update`:\n  "
        + "\n  ".join(sorted(missing)[:20])
    )


def test_recorded_keys_are_dotted_config_paths():
    """Guards the parser: the file is ``<arm> <dotted.key>``, and an arm path
    containing a space would split wrong and record a nonsense key."""
    for arm, keys in recorded_pairs().items():
        for key in keys:
            assert key and not key.startswith("."), f"{arm} recorded a malformed key {key!r}"
            assert " " not in key, f"{arm} recorded a key containing a space: {key!r}"


@pytest.mark.parametrize("known_defect", ["data.volume_format", "checkpoint.save_best"])
def test_the_canonical_defects_are_recorded(known_defect):
    """Anti-vacuity, and a live cross-check against the debt the baseline records.

    ``data.volume_format`` is part of #1012; ``checkpoint.save_best`` took the
    place of ``checkpoint.save_dir`` (#1061) when that key was retired and
    drained (#2842). If either stops appearing, either the debt was genuinely paid — in
    which case the baseline must be regenerated — or the resolver broke and the
    sweep has gone blind, which looks identical from a green suite.
    """
    all_keys = {key for keys in recorded_pairs().values() for key in keys}

    assert known_defect in all_keys
