"""Subject-level split-conformal intervals for a held-out contrast's synthesis."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.training.strategies.qmap_conformal import (
    SubjectResiduals,
    subject_half,
)


def _feed(acc: SubjectResiduals, subject: str, value: float, k: int = 1, n: int = 400) -> None:
    resid = torch.full((1, 1, 20, 20), value)
    acc.add([subject], k, resid, torch.ones(1, 1, 20, 20, dtype=torch.bool))


def test_the_split_is_stable_across_processes() -> None:
    # sha1, not hash(): Python salts str hashes per process, so the halves would move.
    assert [subject_half("sub_0042", s) for s in range(4)] == [
        subject_half("sub_0042", s) for s in range(4)
    ]
    halves = {subject_half(f"sub_{i:04d}", 0) for i in range(50)}
    assert halves == {0, 1}


def test_the_threshold_is_the_conformal_order_statistic_of_subject_scores() -> None:
    acc = SubjectResiduals()
    for i in range(40):  # subject i has every residual equal to i
        _feed(acc, f"s{i:02d}", float(i))
    out = acc.summary(alpha=0.1, beta=0.1, n_splits=1, names=("t1", "t2", "flair"))
    cal = sorted(i for i in range(40) if subject_half(f"s{i:02d}", 0) == 0)
    rank = math.ceil((len(cal) + 1) * 0.9)
    expected = float(cal[rank - 1])
    assert abs(out["val_conformal_q_t2"] - expected) <= 0.25  # one histogram bin
    evaluation = [i for i in range(40) if subject_half(f"s{i:02d}", 0) == 1]
    covered = sum(i <= out["val_conformal_q_t2"] for i in evaluation) / len(evaluation)
    assert out["val_conformal_cover_t2"] == covered
    assert out["val_conformal_ncal_t2"] == len(cal)
    assert "val_conformal_q_t1" not in out  # no residual was recorded for T1


def test_too_few_calibration_subjects_give_an_infinite_interval() -> None:
    acc = SubjectResiduals()
    for i in range(4):
        _feed(acc, f"s{i}", 1.0)
    out = acc.summary(alpha=0.1, beta=0.1, n_splits=1, names=("t1", "t2", "flair"))
    assert out["val_conformal_q_t2"] == math.inf  # ceil((n+1)(1-a)) > n: never clamp


def test_a_subject_with_many_slices_counts_once() -> None:
    # Subjects are the exchangeable unit: one large subject must not outweigh the rest.
    acc = SubjectResiduals()
    for i in range(30):
        _feed(acc, f"s{i:02d}", 1.0)
    for _ in range(200):
        _feed(acc, "s00", 1.0)
    out = acc.summary(alpha=0.2, beta=0.1, n_splits=1, names=("t1", "t2", "flair"))
    assert out["val_conformal_q_t2"] <= 1.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_histograms_stay_on_the_gpu_until_the_summary() -> None:
    # They were built on the CPU: every validation batch copied each signal voxel to the
    # host (.cpu()) and synced, in every q-map arm with a gauge prior.
    acc = SubjectResiduals()
    resid = torch.rand(2, 1, 8, 8, device="cuda")
    acc.add(["a", "b"], 0, resid, torch.ones_like(resid, dtype=torch.bool))
    assert {c.device.type for c in acc._counts.values()} == {"cuda"}


def test_only_masked_voxels_are_scored() -> None:
    acc = SubjectResiduals()
    for i in range(30):
        resid = torch.full((1, 1, 4, 4), 100.0)
        resid[..., 0, 0] = 2.0
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        mask[..., 0, 0] = True  # air elsewhere would inflate or deflate coverage for free
        acc.add([f"s{i}"], 0, resid, mask)
    out = acc.summary(alpha=0.2, beta=0.1, n_splits=1, names=("t1", "t2", "flair"))
    assert out["val_conformal_q_t1"] <= 2.25


def test_a_non_finite_residual_outside_the_mask_is_ignored() -> None:
    # A NaN floored to a negative index made bincount raise, though its weight was 0.
    acc = SubjectResiduals()
    resid = torch.full((1, 1, 4, 4), float("nan"))
    resid[..., 0, 0] = 2.0
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[..., 0, 0] = True
    acc.add(["s"], 0, resid, mask)
    assert float(acc._counts[("s", 0)].sum()) == 1.0


def test_a_non_finite_residual_inside_the_mask_counts_as_unbounded() -> None:
    acc = SubjectResiduals()
    resid = torch.tensor([[[[float("nan"), float("inf")]]]])
    acc.add(["s"], 0, resid, torch.ones_like(resid, dtype=torch.bool))
    assert acc._counts[("s", 0)][-1].item() == 2  # the overflow bin, never a finite one


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_histograms_are_deterministic_on_the_gpu(recwarn: pytest.WarningsRecorder) -> None:
    # A weighted CUDA bincount has no deterministic kernel and warned on every validation
    # pass under the training loop's deterministic mode.
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        resid = torch.rand(2, 1, 8, 8, device="cuda")
        SubjectResiduals().add(["a", "b"], 0, resid, resid > 0.5)
    finally:
        torch.use_deterministic_algorithms(previous)
    assert not [w for w in recwarn if "deterministic" in str(w.message)]


def test_subject_half_uses_a_non_security_sha1(monkeypatch):
    import hashlib

    real = hashlib.sha1
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("usedforsecurity"))
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha1", spy)
    assert subject_half("sub-01", 0) in (0, 1)
    assert seen == [False]
