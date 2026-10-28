"""Tests for ``spectramr profile``'s artifact layout and manifest.

The invariant under test is that every artifact lands under
``experiments/results/<experiment>/`` — the same location
``config_health_checker`` enforces on ``training.output_dir``. The four
``resolve_experiment_name`` cases are separated because they are the only place
the ``<experiment>`` segment is decided, and the fallback case is a derivation
that must stay visible in the manifest rather than pass for a declared value.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from spectramr.cli.profile_paths import (
    CONVENTION_PARTS,
    RESULTS_ROOT,
    build_profile_manifest,
    resolve_experiment_name,
    resolve_profile_paths,
)

#: Deliberately a stem that matches NO expected experiment name below. When it
#: was ``exp_11.yaml``, three tests expecting ``exp_11`` were tautologies: the
#: config-stem FALLBACK returns the same string as a correct match, so they
#: stayed green with the matching logic broken.
CFG = Path("experiments/inprogress/kspace_filling/some_arm_file.yaml")


# ------------------------------------------------- resolve_experiment_name


def test_explicit_experiment_wins_over_config():
    assert (
        resolve_experiment_name(
            "experiments/results/from_config", explicit="from_flag", config_path=CFG
        )
        == "from_flag"
    )


def test_name_comes_from_training_output_dir():
    assert (
        resolve_experiment_name(
            "experiments/results/exp_11_attention_none", explicit=None, config_path=CFG
        )
        == "exp_11_attention_none"
    )


def test_a_nested_output_dir_keeps_every_segment_so_the_profile_sits_in_the_arm():
    """``<cohort>/<arm>`` must not collapse to ``<cohort>``: arms would share one profile dir."""
    name = resolve_experiment_name(
        "experiments/results/campaigns/cohort_a/arm_7", explicit=None, config_path=CFG
    )
    assert name == "campaigns/cohort_a/arm_7"
    paths = resolve_profile_paths(name, "run-1")
    assert paths.profile_dir == Path("experiments/results/campaigns/cohort_a/arm_7/profile/run-1")


def test_absolute_output_dir_resolves_the_same_as_the_relative_spelling():
    """A cluster arm may carry an absolute output_dir; the segment is the same."""
    assert (
        resolve_experiment_name(
            "/scratch/u/runs/experiments/results/exp_11", explicit=None, config_path=CFG
        )
        == "exp_11"  # not "some_arm_file" -> the absolute path really matched
    )


@pytest.mark.parametrize(
    "declared",
    [None, "", "./training_output", "/tmp/somewhere", "experiments/results"],
)
def test_off_convention_output_dir_falls_back_to_the_config_stem(declared):
    """Including the bare root with no segment after it, which must not index."""
    assert resolve_experiment_name(declared, explicit=None, config_path=CFG) == "some_arm_file"


def test_redirecting_the_write_root_does_not_change_name_resolution(monkeypatch):
    """Regression: name matching must anchor on CONVENTION_PARTS, not RESULTS_ROOT.

    These are two concepts — where artifacts are WRITTEN (redirected in tests)
    and the layout convention the health checker enforces (fixed). When
    ``resolve_experiment_name`` matched against ``RESULTS_ROOT.parts``, pointing
    the write root at ``tmp_path`` made every declared ``experiments/results/X``
    stop matching, and every redirected run silently took the config-stem
    fallback instead.
    """
    import spectramr.cli.profile_paths as pp

    monkeypatch.setattr(pp, "RESULTS_ROOT", Path("/tmp/elsewhere/experiments/results"))
    assert (
        pp.resolve_experiment_name("experiments/results/exp_11", explicit=None, config_path=CFG)
        == "exp_11"  # not "some_arm_file" -> it did not take the stem fallback
    ), "name resolution followed the write root instead of the convention"


def test_results_root_matches_the_convention_the_health_checker_enforces():
    """If the enforced location ever moves, this constant must move with it."""
    assert RESULTS_ROOT.parts == CONVENTION_PARTS == ("experiments", "results")
    checker = Path("src/spectramr/infrastructure/validation/config_health_checker.py")
    if checker.is_file():
        assert "experiments/results/" in checker.read_text(encoding="utf-8")


# ----------------------------------------------------- resolve_profile_paths


def test_every_artifact_sits_under_the_experiment_results_dir(tmp_path):
    paths = resolve_profile_paths("exp_11", "run-1", results_root=tmp_path / "results")
    run_dir = tmp_path / "results" / "exp_11"
    assert paths.run_dir == run_dir
    assert paths.profile_dir == run_dir / "profile" / "run-1"
    for artifact in (paths.outfile, paths.manifest, paths.log):
        assert artifact.parent == paths.profile_dir


def test_production_paths_are_anchored_at_experiments_results():
    paths = resolve_profile_paths("exp_11", "run-1")
    assert paths.outfile.parts[:3] == ("experiments", "results", "exp_11")


def test_mkdirs_creates_the_profile_dir_scalene_will_not(tmp_path):
    """Scalene writes --outfile itself and does not create parents; without
    this the child dies at exit, after the whole run has been paid for."""
    paths = resolve_profile_paths("exp_11", "run-1", results_root=tmp_path)
    assert not paths.profile_dir.exists()
    paths.mkdirs()
    assert paths.profile_dir.is_dir()
    paths.mkdirs()  # idempotent


# -------------------------------------------------------------- the manifest


def _manifest(tmp_path, **over):
    paths = resolve_profile_paths("exp_11", "run-1", results_root=tmp_path)
    kwargs = {
        "paths": paths,
        "run_id": "run-1",
        "started_at": datetime(2026, 8, 23, 12, 0, 0),
        "target": "train",
        "mode": "cpu-only",
        "focus": "train-loop",
        "mode_flags": ("--cpu-only",),
        "focus_filters": ("pipelines/training_loop.py",),
        "config_path": CFG,
        "declared_output_dir": "experiments/results/exp_11",
        "device": "cuda",
        "iterations": 100,
        "argv": ["python", "-m", "scalene", "run"],
        "scalene_version": "2.3.0",
    }
    kwargs.update(over)
    return build_profile_manifest(**kwargs)


def test_manifest_records_every_mode(tmp_path):
    """'Which modes produced this profile' must be answerable from the artifact
    alone — cpu-only and full give legitimately different numbers."""
    m = _manifest(tmp_path)
    assert m["modes"] == {
        "target": "train",
        "mode": "cpu-only",
        "focus": "train-loop",
        "device": "cuda",
        "iterations": 100,
    }
    assert m["scalene"]["mode_flags"] == ["--cpu-only"]
    assert m["scalene"]["profile_only"] == ["pipelines/training_loop.py"]
    assert m["scalene"]["version"] == "2.3.0"


def test_manifest_records_declared_beside_applied(tmp_path):
    """The fallback derivation must be auditable, not inferred: the declared
    output_dir is recorded next to the run_dir actually used."""
    m = _manifest(tmp_path, declared_output_dir="./training_output")
    assert m["config"]["declared_training_output_dir"] == "./training_output"
    assert m["paths"]["run_dir"].endswith("exp_11")


def test_manifest_carries_the_literal_argv(tmp_path):
    """So a profile can be re-run without reconstructing it from prose."""
    assert _manifest(tmp_path)["command"] == ["python", "-m", "scalene", "run"]


def test_manifest_is_json_serialisable(tmp_path):
    import json

    json.loads(json.dumps(_manifest(tmp_path), default=str))


def test_manifest_provenance_is_fail_open(tmp_path, monkeypatch):
    """A missing git binary degrades one field, never the manifest."""
    monkeypatch.setattr(
        "spectramr.infrastructure.logging.provenance._git",
        lambda *a, **k: None,
    )
    m = _manifest(tmp_path)
    assert m["git"] == {"available": False}
    assert m["modes"]["mode"] == "cpu-only"


# --------------------------------------------------------------------------
# Output isolation: a profiling run must not overwrite the arm it profiles.
# --------------------------------------------------------------------------


def test_the_child_is_never_pointed_at_the_arms_real_results_dir(tmp_path):
    """The defect this fixed: `child_run_dir` used to BE `run_dir`.

    A 300-iteration profiling run of experiment_11 wrote `checkpoints/`
    (its best checkpoint among them), metrics CSVs and TensorBoard events straight over the
    arm's real results — and reported itself healthy while doing it.
    """
    paths = resolve_profile_paths("exp_11", "run-1", results_root=tmp_path / "results")

    assert paths.child_run_dir != paths.run_dir
    assert paths.profile_dir in paths.child_run_dir.parents
    # ...and specifically NOT a sibling of the profile artifacts, so a run's own
    # `phases/` or `logs/` cannot collide with the ones this verb writes.
    assert paths.child_run_dir.name == "run"


def test_mkdirs_creates_the_child_run_dir_too(tmp_path):
    paths = resolve_profile_paths("exp_11", "run-1", results_root=tmp_path / "results")
    paths.mkdirs()
    assert paths.profile_dir.is_dir()
    assert paths.child_run_dir.is_dir()


def test_the_injected_path_still_satisfies_the_health_checker():
    """Asserted against the REAL checker, not against a reading of its source.

    The claim that a deeper path is legal rests on `check_output_dir_convention`
    being a PREFIX test. If it ever becomes an exact match, every profiling run
    would fail its own config-health gate — so the claim is pinned here rather
    than in a comment.
    """
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.config_health_checker import (
        ConfigHealthChecker,
    )

    paths = resolve_profile_paths("exp_11", "run-1")
    cfg = TrainingSettings(
        model={},
        data={},
        optimization={},
        logging={},
        training={"output_dir": str(paths.child_run_dir)},
    )
    assert ConfigHealthChecker().check_output_dir_convention(cfg).passed


def test_manifest_records_where_the_child_actually_wrote(tmp_path):
    """Declared, real, and redirected — all three readable at once (NN14)."""
    paths = resolve_profile_paths("exp_11", "run-1", results_root=tmp_path / "results")
    m = build_profile_manifest(
        paths=paths,
        run_id="run-1",
        started_at=datetime(2026, 8, 24, 9, 0),
        target="train",
        mode="cpu-only",
        focus="all",
        mode_flags=("--cpu-only",),
        focus_filters=(),
        config_path=Path("a.yaml"),
        declared_output_dir="experiments/results/exp_11",
        device="cpu",
        iterations=100,
        argv=["scalene"],
        scalene_version="2.3.0",
    )
    assert m["paths"]["child_run_dir"].endswith("profile/run-1/run")
    assert m["paths"]["child_run_dir"] != m["paths"]["run_dir"]
    assert m["config"]["declared_training_output_dir"] == "experiments/results/exp_11"
