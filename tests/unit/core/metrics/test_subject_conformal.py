"""Tests for held-out-subject conformal calibration (VFU-09)."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.anchor_conformal import AnchorConformalCalibrator
from spectramr.core.metrics.subject_conformal import SubjectResidualBank


def _fill(bank: SubjectResidualBank, subject: str, scale: float, n: int = 3) -> None:
    g = torch.Generator().manual_seed(hash(subject) % 1000)
    for _ in range(n):
        r = torch.randn(2, 1, 16, 16, generator=g) * scale
        d = torch.rand(2, 1, 16, 16, generator=g)
        bank.add([subject, subject], r, d)


def test_calibrates_on_the_first_sorted_subjects_and_tests_on_the_rest() -> None:
    bank = SubjectResidualBank(n_calibration=1)
    for s in ("s2", "s1", "s3"):
        _fill(bank, s, 1.0)
    assert bank.split() == (["s1"], ["s2", "s3"])
    out = bank.evaluate(AnchorConformalCalibrator(alpha=0.1, n_strata=1), train_subjects=["s9"])
    assert out["conformal_n_calibration_subjects"] == 1.0
    assert out["conformal_n_test_subjects"] == 2.0
    assert abs(out["conformal_worst_coverage"] - 0.9) < 0.05  # exchangeable residuals


def test_coverage_drops_when_test_subjects_are_harder() -> None:
    """The guarantee is TESTED on other subjects, so a shift shows up."""
    bank = SubjectResidualBank(n_calibration=1)
    _fill(bank, "a", 1.0)
    _fill(bank, "b", 3.0)
    out = bank.evaluate(AnchorConformalCalibrator(alpha=0.1, n_strata=1), train_subjects=[])
    assert out["conformal_worst_coverage"] < 0.6
    assert out["conformal_guaranteed"] == 0.0


def test_a_calibration_subject_seen_in_training_raises() -> None:
    bank = SubjectResidualBank(n_calibration=1)
    _fill(bank, "a", 1.0)
    _fill(bank, "b", 1.0)
    with pytest.raises(ValueError, match="overlaps training"):
        bank.evaluate(AnchorConformalCalibrator(), train_subjects=["a"])


def test_too_few_validation_subjects_raises() -> None:
    bank = SubjectResidualBank(n_calibration=1)
    _fill(bank, "only", 1.0)
    with pytest.raises(ValueError, match="needs more than 1"):
        bank.split()


def test_points_are_subsampled_per_sample() -> None:
    bank = SubjectResidualBank(n_calibration=1, points_per_sample=16)
    bank.add(["a"], torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, 16, 16))
    assert bank._bank["a"][0][0].numel() == 16


def test_shape_and_count_mismatches_raise() -> None:
    bank = SubjectResidualBank(n_calibration=1)
    with pytest.raises(ValueError, match="must match"):
        bank.add(["a"], torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 5))
    with pytest.raises(ValueError, match="subject ids for a batch of 2"):
        bank.add(["a"], torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 4))


def test_banked_points_do_not_keep_the_full_batch_alive() -> None:
    """The bank holds O(points) per sample, not a view onto every validation batch."""
    bank = SubjectResidualBank(1, points_per_sample=16)
    r = torch.randn(2, 1, 32, 32)
    bank.add(["a", "b"], r, torch.rand_like(r))
    stored = bank._bank["a"][0][0]
    assert stored.untyped_storage().nbytes() < r.untyped_storage().nbytes() // 8
