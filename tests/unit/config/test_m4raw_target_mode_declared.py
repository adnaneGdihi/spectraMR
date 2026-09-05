"""No m4raw arm may silently take the destructive NEX default (issue #694).

M4Raw stores phase-INCOHERENT repetitions -- separate acquisitions with
independent global phase drift. The shared schema default ``complex_mean``
plain-averages them in complex k-space, which CANCELS signal instead of
averaging it: the "high-SNR NEX reference" comes out with SNR *below* a single
repetition and with scrambled phase. ``.claude/rules/data.md`` records the
downstream damage -- it drove k-space cold diffusion to predict near-real
k-space, producing a 180-degree centro-symmetric doubled brain.

The default cannot be flipped, because ``complex_mean`` is correct for every
non-M4Raw dataset and the field is shared. So the guard has to live on the
corpus: an m4raw arm must SAY which mode it wants.

This is the ratchet for the 43-arm + 1-arm sweep that drained the silent
population to zero under ``inprogress/``. It is deliberately scoped to
``inprogress/`` -- the lifecycle directory CLAUDE.md names as where new arms
go, and the one this repo's rules let a task edit. ``active/``, ``validated/``,
``ablation/`` and ``hpo/`` are separate owner decisions, so a stale arm there
must not fail an unrelated PR.

Declaring ``complex_mean`` explicitly PASSES. That is the point: the finding is
"nobody chose this", not "this value is banned". A deliberate legacy comparison
is legitimate and is exactly what the audit check's two distinct messages
separate (resolution collapses them -- ``config.data.target_mode`` reads
``complex_mean`` either way -- so only ``model_fields_set`` can tell them
apart).
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
import yaml

from tests.utils.repo_scripts import skip_if_public_export

REPO = pathlib.Path(__file__).resolve().parents[3]
SCOPE = "experiments/inprogress"


def _tracked_inprogress_yamls() -> list[pathlib.Path]:
    """Tracked files only -- ``rglob`` would sweep up gitignored scratch.

    ``git ls-files`` and a filesystem glob disagree in this repo (the on-disk
    v5.0 count is ~613 vs 39 committed), and corpus claims are made about the
    committed tree.
    """
    out = subprocess.run(
        ["git", "ls-files", SCOPE],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    ).stdout.split()
    return [REPO / f for f in out if f.endswith((".yaml", ".yml"))]


def is_silent_m4raw(doc: object) -> bool:
    """The predicate, on a parsed document. Exported so tests can exercise it.

    Keys on the RAW YAML rather than on a resolved ``TrainingSettings``, and
    that is deliberate: three tracked m4raw arms do not load at all
    (`experiments/roadmap/exp_nihs_m4raw.yaml` and two under
    `experiments/training/`), so a resolution-based predicate would go quiet on
    exactly the arms nobody has looked at. All three are outside `inprogress/`
    today -- `test_an_unloadable_arm_is_still_visible_to_this_guard` pins that
    the guard would catch one if it moved in, rather than letting the fix's
    editor (which refuses unloadable files) and the guard disagree silently.
    """
    if not isinstance(doc, dict):
        return False
    data = doc.get("data")
    if not isinstance(data, dict) or data.get("dataset_type") != "m4raw":
        return False
    return "target_mode" not in data


def _m4raw_arms() -> list[tuple[pathlib.Path, dict]]:
    arms = []
    for p in _tracked_inprogress_yamls():
        try:
            doc = yaml.safe_load(p.read_text())
        except Exception:
            continue  # unparseable is a different guard's problem
        if not isinstance(doc, dict):
            continue
        data = doc.get("data")
        if isinstance(data, dict) and data.get("dataset_type") == "m4raw":
            arms.append((p, data))
    return arms


def test_the_scan_finds_arms_at_all() -> None:
    """Anti-vacuity. Without this, a broken enumerator makes the pin below
    trivially green -- which is how a corpus guard rots into decoration."""
    skip_if_public_export("experiments/ does not ship; no m4raw arm is present here")
    arms = _m4raw_arms()
    assert len(arms) >= 50, f"only {len(arms)} m4raw arms found under {SCOPE}"


def test_no_inprogress_m4raw_arm_leaves_target_mode_to_the_default() -> None:
    silent = [
        str(p.relative_to(REPO))
        for p, data in _m4raw_arms()
        if "target_mode" not in data
    ]
    assert not silent, (
        f"{len(silent)} m4raw arm(s) declare no `data.target_mode` and so take "
        "the shared default `complex_mean`, which cancels signal on "
        "phase-incoherent M4Raw repetitions. Declare "
        "`phase_aligned_mean` (or `complex_mean` explicitly, with a reason in "
        "`metadata`, if the legacy comparison is deliberate):\n  "
        + "\n  ".join(sorted(silent))
    )


@pytest.mark.parametrize("mode", ["phase_aligned_mean", "complex_mean"])
def test_both_explicit_modes_pass_the_predicate(mode: str) -> None:
    """Pins the polarity: the guard is about SILENCE, not about the value.

    A future tightening to "phase_aligned_mean only" is a separate owner
    decision and would have to change this test deliberately.
    """
    assert not is_silent_m4raw({"data": {"dataset_type": "m4raw", "target_mode": mode}})


def test_the_predicate_fires_on_a_silent_m4raw_arm() -> None:
    """Anti-vacuity for the predicate itself, not just for the enumerator."""
    assert is_silent_m4raw({"data": {"dataset_type": "m4raw"}})


def test_a_non_m4raw_arm_is_not_flagged() -> None:
    """The default is correct for every other dataset; only M4Raw is at risk."""
    assert not is_silent_m4raw({"data": {"dataset_type": "fastmri"}})


def test_an_unloadable_arm_is_still_visible_to_this_guard() -> None:
    """The seam between this guard and the sweep that drained the corpus.

    The #694 editor RESOLVES each file and refuses one that does not load, so
    three tracked m4raw arms were never fixed. This guard parses instead, so it
    would flag such an arm rather than skipping it -- the safer polarity, since
    an arm nobody can load is not an arm anyone has checked.

    Today all three sit outside `inprogress/` and the intersection is empty.
    Were one to move in, this predicate catches it and the message says what to
    declare, instead of the two halves silently disagreeing.
    """
    unloadable_shape = {"data": {"dataset_type": "m4raw", "some_key_that_breaks": 1}}
    assert is_silent_m4raw(unloadable_shape)
