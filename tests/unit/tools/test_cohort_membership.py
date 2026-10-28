"""Tests for the cohort-membership owner (scripts/cohort_membership.py).

The table moved here from three places that disagreed: the batch audit/probe
globbed a hardcoded four-cohort tuple (silently excluding mamba / geomamba /
mno), ``cohort_forensics_review.py`` held the alias table and the stem matcher,
and ``compile_diagnostics.py`` groups arms under a *display* name that is not
the directory name. These pin the properties the single owner has to hold for
``--cohort`` to mean one thing in every pass.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from tests.utils.repo_scripts import require_repo_file

_SCRIPT_REL = "scripts/cohort_membership.py"


def _load():
    script = require_repo_file(_SCRIPT_REL)
    spec = importlib.util.spec_from_file_location("cohort_membership", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalize_collapses_baseline_drift():
    mod = _load()
    assert mod.normalize("Arm_baseline_baseline_v2") == "arm_baseline_v2"
    assert mod.normalize("arm_baseline_v2") == "arm_baseline_v2"


def test_aliases_resolve_to_directory_names():
    mod = _load()
    assert mod.resolve_names(["mamba"]) == ["hilbert_mamba"]
    assert mod.resolve_names(["mrixfields"]) == ["mrixfields2026"]
    # A directory name passes through untouched, so both spellings work.
    assert mod.resolve_names(["kspace_filling"]) == ["kspace_filling"]


def test_no_cohort_means_every_cohort_the_bundle_covers():
    mod = _load()
    assert mod.resolve_names(None) == mod.DEFAULT_COHORTS
    assert mod.resolve_names([]) == mod.DEFAULT_COHORTS
    # The regression this module exists to stop: the mamba / neural-operator
    # cohorts were absent from the audit/probe default while the bundle
    # reported on their arms.
    for cohort in ("hilbert_mamba", "geomamba_ulf", "cs_mno"):
        assert cohort in mod.DEFAULT_COHORTS


def test_every_default_cohort_maps_to_a_display_group():
    """compile_diagnostics groups by a display name; a selectable cohort with no
    mapping would filter the compile down to nothing without erroring."""
    mod = _load()
    for cohort in mod.DEFAULT_COHORTS:
        assert cohort in mod.DISPLAY_COHORT, cohort
    assert mod.display_cohorts(["mamba"]) == ["mamba"]
    assert mod.display_cohorts(["hilbert_mamba"]) == ["mamba"]
    assert mod.display_cohorts(["mrixfields"]) == ["mrixfields"]


def test_display_cohorts_deduplicates():
    mod = _load()
    assert mod.display_cohorts(["mamba", "hilbert_mamba"]) == ["mamba"]


def test_cohort_yamls_recurses_into_sub_directories(tmp_path):
    mod = _load()
    cdir = tmp_path / "mycohort" / "nested"
    cdir.mkdir(parents=True)
    (cdir / "arm_a.yaml").write_text("{}")
    (tmp_path / "mycohort" / "arm_b.yaml").write_text("{}")
    stems = mod.cohort_stems(tmp_path, "mycohort")
    assert stems == ["arm_a", "arm_b"]


def test_cohort_yamls_is_empty_for_an_absent_directory(tmp_path):
    mod = _load()
    assert mod.cohort_yamls("does_not_exist", tmp_path) == []


def _make_tree(
    tmp_path: pathlib.Path, cohort: str, stems: list[str], arm_dirs: list[str]
) -> tuple[pathlib.Path, pathlib.Path]:
    inprogress = tmp_path / "inprogress" / cohort
    inprogress.mkdir(parents=True)
    for s in stems:
        (inprogress / f"{s}.yaml").write_text("{}")
    root = tmp_path / "results"
    for a in arm_dirs:
        (root / a / "debug_snapshots").mkdir(parents=True)
    return tmp_path / "inprogress", root


def test_assign_cohorts_matches_baseline_drift_and_flags_unassigned(tmp_path):
    mod = _load()
    inprogress, root = _make_tree(
        tmp_path,
        "c1",
        ["arm_baseline_v2", "arm_two"],
        ["arm_baseline_baseline_v2", "arm_two", "stranger"],
    )
    assigned, unassigned = mod.assign_cohorts(["c1"], inprogress, mod.reviewable_arms(root))
    assert sorted(assigned["c1"]) == ["arm_baseline_baseline_v2", "arm_two"]
    assert unassigned == ["stranger"]


def test_assign_cohorts_matches_the_mrixfields_output_prefix(tmp_path):
    """mrixfields YAMLs write output_dir mrixfields_<stem>, so the run dir never
    equals the stem."""
    mod = _load()
    inprogress, root = _make_tree(
        tmp_path, "mrixfields2026", ["b16_field_fno"], ["mrixfields_b16_field_fno"]
    )
    assigned, unassigned = mod.assign_cohorts(
        ["mrixfields2026"], inprogress, mod.reviewable_arms(root)
    )
    assert assigned["mrixfields2026"] == ["mrixfields_b16_field_fno"]
    assert unassigned == []


def test_reviewable_arms_skips_the_bundle_directories(tmp_path):
    """diagnostics/, dispatch/, mosaics/ and review/ sit beside the run dirs; a
    tree scan that treats them as arms reports bundle folders as experiments."""
    mod = _load()
    root = tmp_path / "results"
    for name in ("real_arm", *mod.NON_ARM_DIRS):
        (root / name / "debug_snapshots").mkdir(parents=True)
    assert mod.reviewable_arms(root) == ["real_arm"]


def test_reviewable_arms_on_a_missing_tree_is_empty_not_an_error(tmp_path):
    mod = _load()
    assert mod.reviewable_arms(tmp_path / "nope") == []


def test_arms_for_returns_a_flat_sorted_set(tmp_path):
    mod = _load()
    inprogress, root = _make_tree(tmp_path, "c1", ["b", "a"], ["a", "b", "other"])
    assert mod.arms_for(["c1"], root, inprogress) == ["a", "b"]


@pytest.mark.parametrize("flag", ["--list", "--arms", "--yamls"])
def test_cli_modes_exit_zero(tmp_path, flag, capsys):
    mod = _load()
    argv = [flag] if flag == "--list" else [flag, "kspace_filling"]
    assert mod.main([*argv, "--root", str(tmp_path)]) == 0
