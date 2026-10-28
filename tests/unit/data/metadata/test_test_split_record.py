"""A config-owned test partition is recorded once and re-derived identically afterwards."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from spectramr.data.metadata.test_split_record import (
    TEST_SPLIT_RECORD,
    derived_test_split_record,
    write_or_verify_test_split,
)

pytestmark = pytest.mark.unit


def _data():
    return SimpleNamespace(
        split=SimpleNamespace(
            validation_fraction=0.1,
            holdout_subject=None,
            loso_fold=None,
            test_subjects=None,
        ),
        test_split=0.1,
        enable_test_split=True,
        source=SimpleNamespace(paired_manifest_path="m.json"),
    )


def _stamp(test=("s9", "s8"), route="config_owned", sha="abc"):
    return {
        "route": route,
        "strategy": "random",
        "manifest_sha256": sha,
        "subjects": {"train": ["s1"], "val": ["s2"], "test": list(test)},
    }


def test_record_names_the_recipe_and_the_subjects() -> None:
    rec = derived_test_split_record(_data(), _stamp())
    assert rec["strategy"] == "random" and rec["seed"] is None
    assert rec["test_fraction"] == 0.1 and rec["test_subjects"] == ["s8", "s9"]
    assert rec["manifest_sha256"] == "abc"


@pytest.mark.parametrize("stamp", [None, {}, _stamp(route="manifest"), _stamp(test=())])
def test_no_record_without_a_config_owned_test_set(stamp) -> None:
    assert derived_test_split_record(_data(), stamp) is None


def test_first_run_writes_and_a_matching_rerun_passes(tmp_path) -> None:
    rec = derived_test_split_record(_data(), _stamp())
    write_or_verify_test_split(tmp_path, rec)
    assert json.loads((tmp_path / TEST_SPLIT_RECORD).read_text()) == rec
    write_or_verify_test_split(tmp_path, rec)


@pytest.mark.parametrize(
    "changed", [_stamp(test=("s9", "s7")), _stamp(sha="other")], ids=["subjects", "manifest"]
)
def test_planted_drift_in_the_test_set_raises(tmp_path, changed) -> None:
    write_or_verify_test_split(tmp_path, derived_test_split_record(_data(), _stamp()))
    with pytest.raises(ValueError, match="differs"):
        write_or_verify_test_split(tmp_path, derived_test_split_record(_data(), changed))


def test_an_inference_run_is_checked_against_the_training_record(tmp_path) -> None:
    write_or_verify_test_split(tmp_path, derived_test_split_record(_data(), _stamp()))
    inference = tmp_path / "inference"
    inference.mkdir()
    write_or_verify_test_split(inference, derived_test_split_record(_data(), _stamp()))
    with pytest.raises(ValueError, match="differs"):
        write_or_verify_test_split(
            inference, derived_test_split_record(_data(), _stamp(test=("s9", "s7")))
        )


def test_an_inference_run_without_a_training_record_raises(tmp_path) -> None:
    inference = tmp_path / "inference"
    inference.mkdir()
    with pytest.raises(ValueError, match="recorded no test partition"):
        write_or_verify_test_split(inference, derived_test_split_record(_data(), _stamp()))
