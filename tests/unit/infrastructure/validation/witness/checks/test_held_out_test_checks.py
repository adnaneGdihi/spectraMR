"""Held-out test-set witnesses (cohort review 2026-09-02, T0.3). Planted violations first."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.validation.witness.checks.held_out_test_checks import (
    REPORTING_ROLES,
    declares_held_out_test,
    held_out_test_declared,
    held_out_test_split_disjoint,
    paired_route_holdout_site,
)
from spectramr.infrastructure.validation.witness.registry import Severity, get_witness_registry
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


def _raw(role: str | None, data: dict | None) -> WitnessSubject:
    raw = {"metadata": {"tags": {"role": role}} if role else {}, "data": data or {}}
    return WitnessSubject.for_ci(None, raw)


# --- held_out_test_declared (advisory) ------------------------------------------


def test_baseline_without_a_test_set_is_flagged_as_advisory() -> None:
    """The planted violation: the corpus state on 2026-09-02 (0 of 647 arms)."""
    verdict = held_out_test_declared(_raw("baseline", {"source": {"index_path": "train.json"}}))
    assert verdict.passed is False
    assert verdict.severity is Severity.INFO  # advisory: never fails --strict yet
    assert "selection set" in verdict.message or "selects its checkpoint" in verdict.message


@pytest.mark.parametrize(
    "data",
    [
        {"source": {"test_index_path": "test.json"}},
        {"source": {"paired_manifest_path": "m.json"}, "enable_test_split": True},
        {"source": {"paired_manifest_path": "m.json"}, "split": {"test_subjects": ["07"]}},
    ],
)
def test_baseline_with_a_test_set_passes(data) -> None:
    assert held_out_test_declared(_raw("baseline", data)).passed is True


def test_non_reporting_roles_are_not_nagged() -> None:
    for role in ("ablation", "variant", "comparison", None):
        assert held_out_test_declared(_raw(role, {})).passed is True


def test_reporting_roles_is_the_advertised_set() -> None:
    assert frozenset({"baseline", "headline", "reference", "ssot"}) == REPORTING_ROLES


def test_role_is_read_from_metadata_top_level_too() -> None:
    subject = WitnessSubject.for_ci(None, {"metadata": {"role": "Baseline"}, "data": {}})
    assert held_out_test_declared(subject).passed is False


def test_declares_held_out_test_predicate() -> None:
    assert declares_held_out_test({"source": {"test_index_path": "t.json"}})
    # enable_test_split with no route has no test loader (the schema refuses it at load).
    assert not declares_held_out_test({"enable_test_split": True})
    assert not declares_held_out_test({"source": {"index_path": "a.json"}})
    assert not declares_held_out_test("not a dict")


def _typed(data: dict) -> SimpleNamespace:
    source = SimpleNamespace(**{"test_index_path": None, "paired_manifest_path": None, **data.get("source", {})})
    split = SimpleNamespace(**{"test_subjects": [], **data.get("split", {})})
    return SimpleNamespace(
        data=SimpleNamespace(
            source=source, split=split, enable_test_split=data.get("enable_test_split", False)
        )
    )


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"enable_test_split": True},
        {"source": {"test_index_path": "t.json"}},
        {"source": {"paired_manifest_path": "m.json"}},
        {"source": {"paired_manifest_path": "m.json"}, "enable_test_split": True},
        {"source": {"paired_manifest_path": "m.json"}, "split": {"test_subjects": ["07"]}},
    ],
)
def test_witness_and_loader_agree_on_what_declares_a_test_set(data) -> None:
    """One predicate: the audit's raw-dict view and the loaders' typed view never disagree."""
    from spectramr.infrastructure.builders.directors.test_split import declares_test_set

    assert declares_held_out_test(data) is declares_test_set(_typed(data))


# --- held_out_test_split_disjoint (error) ----------------------------------------


def _manifest(path: Path, subjects: list[str]) -> str:
    path.write_text(
        json.dumps(
            {
                "data_root": ".",
                "records": [
                    {"subject_id": s, "relative_path": f"{s}/vol.nii.gz", "file_id": s}
                    for s in subjects
                ],
            }
        )
    )
    return str(path)


def _settings(train: str | None, val: str | None, test: str | None) -> WitnessSubject:
    settings = SimpleNamespace(
        data=SimpleNamespace(
            source=SimpleNamespace(
                index_path=train, validation_index_path=val, test_index_path=test
            )
        )
    )
    return WitnessSubject.for_audit(None, settings)


def test_overlapping_test_manifest_is_an_error(tmp_path: Path) -> None:
    """The planted violation: a reported subject that also trained."""
    subject = _settings(
        _manifest(tmp_path / "train.json", ["sub-01", "sub-02"]),
        _manifest(tmp_path / "val.json", ["sub-03"]),
        _manifest(tmp_path / "test.json", ["sub-02"]),
    )
    verdict = held_out_test_split_disjoint(subject)
    assert verdict.passed is False and verdict.severity is Severity.ERROR
    assert "sub-02" in verdict.message


def test_disjoint_test_manifest_passes(tmp_path: Path) -> None:
    subject = _settings(
        _manifest(tmp_path / "train.json", ["sub-01"]),
        _manifest(tmp_path / "val.json", ["sub-02"]),
        _manifest(tmp_path / "test.json", ["sub-03"]),
    )
    assert held_out_test_split_disjoint(subject).passed is True


def test_undeclared_test_manifest_is_a_skip_not_a_pass_claim() -> None:
    verdict = held_out_test_split_disjoint(_settings("train.json", None, None))
    assert verdict.passed is True and verdict.message.startswith("skipped:")


def test_both_witnesses_are_registered_after_discovery() -> None:
    import spectramr.infrastructure.validation.witness  # noqa: F401

    registry = get_witness_registry()
    assert registry.get("held_out_test_declared") is not None
    assert registry.get("held_out_test_split_disjoint") is not None


# --- paired_route_holdout_site (error) -------------------------------------------


@pytest.mark.parametrize(
    "split",
    [
        {"type": "loso", "holdout_site": ["sub-0001", "sub-0002"]},  # the 46-arm corpus shape
        {"type": "loso", "holdout_site": ["0046", "0048"]},  # bare manifest-form ids
        {"type": "loso", "holdout_site": "sub-0046"},  # scalar string
        {"type": "random", "validation_fraction": 0.2, "holdout_site": ["0001"]},
        {"type": "auto", "holdout_site": "siteA"},  # a real site id raises on this route too
    ],
)
def test_holdout_site_on_a_paired_manifest_is_an_error(split) -> None:
    data = {"source": {"paired_manifest_path": "m.json"}, "split": split}
    verdict = paired_route_holdout_site(_raw(None, data))
    assert verdict.passed is False
    assert verdict.severity is Severity.ERROR
    assert "holdout_site" in verdict.message


@pytest.mark.parametrize(
    "data",
    [
        {
            "source": {"index_path": "i.json"},
            "split": {"type": "loso", "holdout_site": "site_1.5T"},
        },
        {"source": {"paired_manifest_path": "m.json"}, "split": {"type": "random"}},
        {"source": {"paired_manifest_path": "m.json"}, "split": {"holdout_site": None}},
        {},
    ],
)
def test_site_route_and_config_owned_paired_splits_pass(data) -> None:
    assert paired_route_holdout_site(_raw(None, data)).passed is True


def test_paired_route_holdout_site_is_registered_as_an_error() -> None:
    spec = get_witness_registry().get("paired_route_holdout_site")
    assert spec is not None
    assert spec.severity is Severity.ERROR
