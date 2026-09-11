"""Planted collisions for the identity-path gate (non-negotiable 15).

A gate is only a gate for the violation shape you have watched it fail on, so
every clause here ships with a corpus that turns it red, and every documented
exemption ships with a corpus that must leave it green. The plants are built by
mutating a REAL corpus arm rather than by hand-writing a dict: a synthetic config
that does not load produces zero findings, which is indistinguishable from a
clean run and is exactly how a detector test passes vacuously. ``_place`` asserts
the mutation actually took, and ``test_every_planted_corpus_loads`` asserts no
plant was silently skipped as schema-invalid.

The shapes deliberately include two the obvious implementation misses:

* **same-stem duplicates** -- two arms with the same FILENAME in different
  cohorts. A census keyed on the stem collapses them into one entry and can never
  report them; that bug shipped in the first census written for #1930 and made it
  report 2 arms where the issue correctly said 5.
* **a collision spelled across the alias** -- one arm declaring ``save_dir`` and
  another ``checkpoint_dir``, the same field under its two names. A gate that
  parses YAML sees two unrelated keys and never compares them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "ci"))

import check_identity_paths_unique as gate  # noqa: E402

#: A real, loadable corpus arm. Every plant is this arm with its identity paths
#: rewritten, so a plant exercises the same schema the run does.
_REAL_ARM = (
    _REPO_ROOT / "experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml"
)

#: Where each in-scope identity path lives in the raw document.
_BLOCKS: dict[str, tuple[str, ...]] = {
    "training.output_dir": ("training", "output_dir"),
    "checkpoint.checkpoint_dir": ("checkpoint", "checkpoint_dir"),
    "checkpoint.save_dir": ("checkpoint", "save_dir"),
    "logging.sinks.dir": ("logging", "sinks", "dir"),
    "loss_logging.output_dir": ("loss_logging", "output_dir"),
    "loss_logging.csv_path": ("loss_logging", "csv_path"),
    "metrics.output_dir": ("metrics", "output_dir"),
    "logging.identity.experiment": ("logging", "identity", "experiment"),
}


def _strip_identity(doc: dict) -> None:
    """Remove every identity path, so a plant asserts on the keys it sets alone."""
    for segments in _BLOCKS.values():
        cursor = doc
        for segment in segments[:-1]:
            cursor = cursor.get(segment) if isinstance(cursor, dict) else None
            if not isinstance(cursor, dict):
                break
        else:
            cursor.pop(segments[-1], None)


def _place(destination: Path, **values: str) -> Path:
    """Write ``_REAL_ARM`` to ``destination`` with exactly ``values`` declared.

    Keys are the dotted identity paths. The write is verified: a plant that does
    not change the document would leave the gate green for the wrong reason.
    """
    doc = yaml.safe_load(_REAL_ARM.read_text())
    _strip_identity(doc)
    for dotted, value in values.items():
        segments = _BLOCKS[dotted.replace("__", ".")]
        cursor = doc
        for segment in segments[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[segments[-1]] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(doc, sort_keys=False))
    written = yaml.safe_load(destination.read_text())
    for dotted, value in values.items():
        cursor = written
        for segment in _BLOCKS[dotted.replace("__", ".")]:
            cursor = cursor[segment]
        assert cursor == value, f"plant did not take for {dotted}"
    return destination


def _findings(root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], int]:
    resolved, stems, unloadable = gate.collect(root)
    return gate.find_duplicates(resolved), gate.find_hijacks(resolved, stems), unloadable


# --------------------------------------------------------------------------
# Plants that must turn the gate RED -- one per shape the rule can take.
# --------------------------------------------------------------------------


def test_cross_cohort_duplicate_is_found(tmp_path: Path) -> None:
    """The #1930 headline shape: two cohorts writing to one results tree."""
    _place(
        tmp_path / "cohort_a" / "arm_one.yaml", training__output_dir="experiments/results/shared"
    )
    _place(
        tmp_path / "cohort_b" / "arm_two.yaml", training__output_dir="experiments/results/shared"
    )
    duplicates, _, unloadable = _findings(tmp_path)
    assert unloadable == 0
    assert duplicates == {
        "duplicate::training.output_dir::experiments/results/shared": [
            "cohort_a/arm_one.yaml",
            "cohort_b/arm_two.yaml",
        ]
    }


def test_same_cohort_duplicate_is_found(tmp_path: Path) -> None:
    """Two siblings in ONE directory collide just as destructively."""
    _place(tmp_path / "c" / "one.yaml", training__output_dir="experiments/results/shared")
    _place(tmp_path / "c" / "two.yaml", training__output_dir="experiments/results/shared")
    duplicates, _, _ = _findings(tmp_path)
    assert len(duplicates) == 1


def test_same_stem_duplicate_is_found(tmp_path: Path) -> None:
    """Same FILENAME in two cohorts.

    A census keyed on the stem adds the same string twice and can never see this
    shape -- the defect that made the first #1930 census under-report.
    """
    _place(tmp_path / "cohort_a" / "twin.yaml", training__output_dir="experiments/results/shared")
    _place(tmp_path / "cohort_b" / "twin.yaml", training__output_dir="experiments/results/shared")
    duplicates, _, _ = _findings(tmp_path)
    assert duplicates, "a stem-keyed census is blind to same-stem pairs; this must not be"
    arms = next(iter(duplicates.values()))
    assert arms == ["cohort_a/twin.yaml", "cohort_b/twin.yaml"]


def test_a_collision_spelled_across_the_alias_is_found(tmp_path: Path) -> None:
    """``save_dir`` and ``checkpoint_dir`` are ONE field under two names.

    This is the plant that justifies resolving instead of parsing: a gate reading
    the raw document sees two unrelated keys here and reports nothing.
    """
    _place(
        tmp_path / "a" / "one.yaml", checkpoint__save_dir="experiments/results/shared/checkpoints"
    )
    _place(
        tmp_path / "b" / "two.yaml",
        checkpoint__checkpoint_dir="experiments/results/shared/checkpoints",
    )
    duplicates, _, unloadable = _findings(tmp_path)
    assert unloadable == 0
    assert duplicates == {
        "duplicate::checkpoint.checkpoint_dir::experiments/results/shared/checkpoints": [
            "a/one.yaml",
            "b/two.yaml",
        ]
    }


def test_hijack_without_a_duplicate_is_found(tmp_path: Path) -> None:
    """Clause 1's blind spot: a pointer at an arm that spells its own path differently.

    ``follower`` names ``leader``; ``leader`` writes somewhere else entirely. No
    value repeats, so duplicate detection alone reports nothing.
    """
    _place(tmp_path / "c" / "leader.yaml", training__output_dir="experiments/results/leader_v2")
    _place(tmp_path / "c" / "follower.yaml", training__output_dir="experiments/results/leader")
    duplicates, hijacks, _ = _findings(tmp_path)
    assert duplicates == {}, "no value repeats, so clause 1 is silent by construction"
    assert hijacks == {
        "names-another-arm::c/follower.yaml::training.output_dir::experiments/results/leader": [
            "c/leader.yaml"
        ]
    }


def test_a_hijack_is_seen_at_any_depth(tmp_path: Path) -> None:
    """The named arm sits in a MIDDLE segment, not the last one.

    A first-or-last-segment rule misses ``.../<name>/checkpoints`` entirely, which
    scored the real offending arm at 1 key instead of 6.
    """
    _place(tmp_path / "c" / "leader.yaml", training__output_dir="experiments/results/leader_v2")
    _place(
        tmp_path / "c" / "follower.yaml",
        checkpoint__checkpoint_dir="experiments/results/leader/checkpoints",
    )
    _, hijacks, _ = _findings(tmp_path)
    assert len(hijacks) == 1


# --------------------------------------------------------------------------
# Negative controls -- documented exemptions that must stay GREEN.
# --------------------------------------------------------------------------


def test_group_directory_siblings_do_not_fire(tmp_path: Path) -> None:
    """``<group>/<stem>`` is unique per arm, so a shared group dir is not a collision.

    This is what lets clause 1 skip a group-directory heuristic entirely.
    """
    _place(
        tmp_path / "conformal" / "alpha.yaml",
        training__output_dir="experiments/results/conformal/alpha",
    )
    _place(
        tmp_path / "conformal" / "beta.yaml",
        training__output_dir="experiments/results/conformal/beta",
    )
    duplicates, hijacks, unloadable = _findings(tmp_path)
    assert unloadable == 0
    assert duplicates == {}
    assert hijacks == {}


def test_an_arm_naming_itself_is_not_a_hijack(tmp_path: Path) -> None:
    _place(
        tmp_path / "c" / "solo.yaml",
        training__output_dir="experiments/results/solo",
        checkpoint__checkpoint_dir="experiments/results/solo/checkpoints",
    )
    duplicates, hijacks, _ = _findings(tmp_path)
    assert duplicates == {}
    assert hijacks == {}


def test_relative_paths_are_out_of_scope(tmp_path: Path) -> None:
    """Both arms resolve ``./checkpoints`` and still produce no finding.

    Not an endorsement: relative values collide in the launch CWD and are a
    corpus-wide defect owned by the schema default, tracked separately. This test
    pins the SCOPE so the exclusion cannot be lost by accident.
    """
    one = _place(tmp_path / "c" / "one.yaml", checkpoint__save_dir="./checkpoints")
    _place(tmp_path / "c" / "two.yaml", checkpoint__save_dir="./checkpoints")
    duplicates, _, unloadable = _findings(tmp_path)
    assert unloadable == 0, "an unloadable plant would make this pass for the wrong reason"
    assert yaml.safe_load(one.read_text())["checkpoint"]["save_dir"] == "./checkpoints"
    assert duplicates == {}


def test_a_shared_identity_label_is_not_a_collision(tmp_path: Path) -> None:
    """Paired control/treatment arms share ``logging.identity.experiment`` on purpose."""
    _place(
        tmp_path / "c" / "control.yaml",
        training__output_dir="experiments/results/control",
        logging__identity__experiment="scas_8x_brain",
    )
    _place(
        tmp_path / "c" / "treatment.yaml",
        training__output_dir="experiments/results/treatment",
        logging__identity__experiment="scas_8x_brain",
    )
    duplicates, _, _ = _findings(tmp_path)
    assert duplicates == {}


def test_every_planted_corpus_loads(tmp_path: Path) -> None:
    """A plant that fails to resolve yields zero findings, faking a clean gate."""
    _place(tmp_path / "c" / "one.yaml", training__output_dir="experiments/results/x")
    resolved, _, unloadable = gate.collect(tmp_path)
    assert unloadable == 0
    assert resolved["c/one.yaml"] == {"training.output_dir": "experiments/results/x"}


# --------------------------------------------------------------------------
# The three-way ratchet, and NN18 on a missing baseline.
# --------------------------------------------------------------------------


def test_a_new_finding_fails(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("# empty\n")
    _place(tmp_path / "c" / "one.yaml", training__output_dir="experiments/results/shared")
    _place(tmp_path / "c" / "two.yaml", training__output_dir="experiments/results/shared")
    assert gate.main([str(tmp_path), "--baseline", str(baseline)]) == 1


def test_a_baselined_finding_passes(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.txt"
    _place(tmp_path / "c" / "one.yaml", training__output_dir="experiments/results/shared")
    _place(tmp_path / "c" / "two.yaml", training__output_dir="experiments/results/shared")
    assert gate.main([str(tmp_path), "--baseline", str(baseline), "--update-baseline"]) == 0
    assert gate.main([str(tmp_path), "--baseline", str(baseline)]) == 0


def test_a_cleaned_baseline_entry_is_reported_and_still_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("duplicate::training.output_dir::experiments/results/long_gone\n")
    _place(tmp_path / "c" / "one.yaml", training__output_dir="experiments/results/unique")
    assert gate.main([str(tmp_path), "--baseline", str(baseline)]) == 0
    assert "now clean" in capsys.readouterr().out


def test_strict_fails_on_a_baselined_finding(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.txt"
    _place(tmp_path / "c" / "one.yaml", training__output_dir="experiments/results/shared")
    _place(tmp_path / "c" / "two.yaml", training__output_dir="experiments/results/shared")
    gate.main([str(tmp_path), "--baseline", str(baseline), "--update-baseline"])
    assert gate.main([str(tmp_path), "--baseline", str(baseline), "--strict"]) == 1


def test_a_missing_baseline_fails_rather_than_inferring_an_empty_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Non-negotiable 18: absent is a state to report, never to infer.

    ``check_witness_corpus.read_baseline`` returns ``set()`` here, which grades
    every known finding as a new regression on one run and blesses the whole
    corpus on the next ``--update-baseline``.
    """
    _place(tmp_path / "c" / "one.yaml", training__output_dir="experiments/results/unique")
    assert gate.main([str(tmp_path), "--baseline", str(tmp_path / "absent.txt")]) == 1
    assert "baseline file is absent" in capsys.readouterr().out
    with pytest.raises(gate.MissingBaselineError):
        gate.read_baseline(tmp_path / "absent.txt")


def test_an_absent_corpus_declines_rather_than_passing_silently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert gate.main([str(tmp_path / "no_such_tree")]) == 0
    assert "declining" in capsys.readouterr().out


def test_the_gate_and_the_health_checker_share_one_identity_path_list() -> None:
    """One owner for the list, asserted by object identity (non-negotiable 17).

    Two checks apply these six paths: this gate compares them ACROSS the corpus,
    and ``ConfigHealthChecker.check_identity_paths_agree`` compares them WITHIN
    one config. A second copy of the tuple would let the two disagree about what
    an identity path even is, and the divergence would surface as a wrong count
    rather than as an error -- which is why this asserts ``is``, not ``==``: an
    equal-but-separate literal is exactly the state the rule forbids.
    """
    from spectramr.infrastructure.validation.config_health_checker import IDENTITY_PATHS

    assert gate.IDENTITY_PATHS is IDENTITY_PATHS


def test_the_label_the_gate_excludes_is_still_unread() -> None:
    """``logging.identity.experiment`` is excluded because NOTHING reads it.

    The exclusion is a measurement, not a taste, so it needs a tripwire: if the
    knob is ever wired, duplicate names across arms stop being harmless and this
    gate's scope has to be revisited.

    Matched on the **AST**, never on text. A text scan reports every comment and
    docstring that merely mentions the key -- the first version of this test
    failed on the explanatory comment above ``IDENTITY_PATHS``, which is prose,
    not a read. Walking attribute nodes also makes the two known non-readers
    disappear for the right reason rather than by an exemption list: both
    ``renames.py`` and ``paired_arms_diff_paths.py`` hold the path as a STRING,
    and a string is not an attribute access.

    The schema-key consumption ledger cannot stand in for this: it matches a
    key's LEAF token, and ``experiment`` is spelled all over ``src/`` (#1925),
    so it reports this key as consumed.
    """
    import ast

    src_root = _REPO_ROOT / "src" / "spectramr"
    readers = []
    for path in sorted(src_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:  # pragma: no cover - a broken file is a different test
            continue
        for node in ast.walk(tree):
            # `<anything>.identity.experiment` -- the only shape that READS it.
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "experiment"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "identity"
            ):
                rel = path.relative_to(_REPO_ROOT).as_posix()
                readers.append(f"{rel}:{node.lineno}")
    assert not readers, (
        "logging.identity.experiment now has a reader in src/, so it is no longer "
        "a write-only label. Revisit this gate's scope: 7 names are shared by 14 "
        "arms, 3 of them a control arm carrying its treatment arm's name.\n  "
        + "\n  ".join(readers)
    )


def test_the_unread_label_tripwire_can_actually_fire() -> None:
    """Anti-vacuity for the test above (non-negotiable 15).

    A scan that matched nothing -- a wrong node shape, a wrong root -- would pass
    the exclusion test forever and silently stop watching the knob. So run the
    same matcher over a source file that DOES read the key and require a hit.
    """
    import ast

    planted = ast.parse("run_name = config.logging.identity.experiment\n")
    hits = [
        node
        for node in ast.walk(planted)
        if isinstance(node, ast.Attribute)
        and node.attr == "experiment"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "identity"
    ]
    assert len(hits) == 1
