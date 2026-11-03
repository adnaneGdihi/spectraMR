"""One cadence owner with per-phase budgets; views follow the config's domain facts."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.training.snapshot.policy import SnapshotSchedule, ViewPolicy

pytestmark = pytest.mark.unit


def test_train_and_val_budgets_are_separate() -> None:
    s = SnapshotSchedule(max_calls=2)
    assert [s.due("train", i) for i in range(4)] == [True, True, False, False]
    assert s.due("val", 100) and s.due("val", 200) and not s.due("val", 300)


def test_interval_is_checked_before_the_budget_is_spent() -> None:
    s = SnapshotSchedule(interval_steps=10, max_calls=1)
    assert not s.due("train", 5)
    assert s.due("train", 10)
    assert not s.due("train", 20)


def test_disabled_and_validation_off() -> None:
    assert not SnapshotSchedule(enabled=False).due("train", 0)
    assert not SnapshotSchedule(validation=False).due("val", 0)


def test_an_unknown_phase_raises() -> None:
    with pytest.raises(ValueError, match="phase"):
        SnapshotSchedule().due("test", 0)


def test_image_loader_uses_the_name_heuristic() -> None:
    v = ViewPolicy().views(["noisy_kspace", "target"])
    assert v["noisy_kspace"].kspace and not v["noisy_kspace"].authoritative
    assert not v["target"].kspace


def test_kspace_loader_marks_canonical_names_authoritative() -> None:
    v = ViewPolicy(targets_kspace=True, log_scaled=True).views(
        ["target", "model_output", "smaps"], log_scaled_keys={"target"}
    )
    assert v["target"].kspace and v["target"].authoritative and v["target"].log_scaled
    assert not v["model_output"].kspace  # an image-emitting model on a k-space loader
    assert v["smaps"].kspace and not v["smaps"].authoritative and not v["smaps"].log_scaled


def test_domain_exact_makes_the_caller_set_the_whole_answer() -> None:
    v = ViewPolicy(targets_kspace=True).views(
        ["model_input"], in_kspace_keys=set(), domain_exact=True
    )
    assert not v["model_input"].kspace


def test_segments_ride_on_their_tensor() -> None:
    segs = [("x_t", 2, True), ("smaps", 2, True, False)]
    v = ViewPolicy(targets_kspace=True).views(["model_input"], segments={"model_input": segs})
    assert v["model_input"].segments == tuple(segs)
