"""Paired tests against a baseline: pair subjects, never fake a p-value."""

import pandas as pd

from spectramr.infrastructure.reporting.tables._significance import paired_significance


def _row(method, values, subjects):
    series = pd.Series(values, index=subjects, dtype=float)
    return {"method": method, "psnr": (series.mean(), series.std(), series)}


def test_pairs_by_subject_not_by_position():
    subjects = ["s1", "s2", "s3", "s4", "s5"]
    base = _row("a", [30, 31, 29, 32, 30], subjects)
    # same per-subject gain of ~1 dB, listed in reverse order
    better = _row("b", [31.1, 33.0, 29.9, 32.1, 31.0][::-1], subjects[::-1])
    sig = paired_significance([base, better], ["a", "b"], ["psnr"], "a")
    assert sig[("b", "psnr")] is True


def test_no_shared_subjects_means_no_test():
    base = _row("a", [30, 31], ["s1", "s2"])
    other = _row("b", [33, 34], ["s3", "s4"])
    assert paired_significance([base, other], ["a", "b"], ["psnr"], "a") == {}


def test_a_constant_difference_is_decided_without_scipy():
    subjects = ["s1", "s2", "s3"]
    base = _row("a", [30, 31, 32], subjects)
    same = _row("b", [30, 31, 32], subjects)
    shifted = _row("c", [31, 32, 33], subjects)
    sig = paired_significance([base, same, shifted], ["a", "b", "c"], ["psnr"], "a")
    assert sig[("b", "psnr")] is False and sig[("c", "psnr")] is True
