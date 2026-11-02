"""Certified Blackwell relations on diagonal profiles."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.blackwell.catalogue import build_catalogue
from spectramr.core.metrics.meta_evaluation.blackwell.order import Relation, certify, compare

pytestmark = pytest.mark.unit
NEG = -math.inf


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 2.0], [1.0, 2.0], Relation.EQUIVALENT),
        ([1.0, 3.0], [1.0, 2.0], Relation.DOMINATES),
        ([0.0, 2.0], [1.0, 2.0], Relation.DOMINATED),
        ([2.0, 0.0], [1.0, 1.0], Relation.INCOMPARABLE),
        ([1.0, NEG], [1.0, NEG], Relation.EQUIVALENT),
        ([1.0, 0.0], [1.0, NEG], Relation.DOMINATES),
    ],
)
def test_compare(a: list[float], b: list[float], expected: Relation) -> None:
    assert (
        compare(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
        is expected
    )


def test_compare_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        compare(torch.zeros(2), torch.zeros(3))


def test_within_family_severities_form_chains() -> None:
    cat = build_catalogue((32, 32))
    graph = certify(cat)
    strict = set(graph.strict)
    for i, a in enumerate(cat):
        for j, b in enumerate(cat):
            if i < j and a.family == b.family:
                assert (i, j) in strict  # severities are listed mildest first


def test_noise_at_base_level_dominates_blur_at_the_same_noise() -> None:
    cat = build_catalogue((32, 32), base_noise=0.1)
    names = [e.name for e in cat]
    noise, blur = names.index("noise_0.1"), names.index("blur_1")
    assert (noise, blur) in set(certify(cat).strict)


def test_low_noise_blur_and_plain_noise_are_incomparable() -> None:
    """Less noise wins at low frequency, no blur wins at high frequency (Theorem C1.2)."""
    cat = build_catalogue((32, 32))
    names = [e.name for e in cat]
    pair = tuple(sorted((names.index("noise_0.1"), names.index("blur_low_noise_1"))))
    assert pair in set(certify(cat).incomparable)


def test_summary_counts_cover_every_pair() -> None:
    cat = build_catalogue((32, 32))
    s = certify(cat).summary(cat)
    assert s["n_strict"] + s["n_equivalent"] + s["n_incomparable"] == 25 * 24 // 2
    assert s["cross_family_pairs"] == 250
    assert 0.0 < s["cross_family_task_free_fraction"] < 1.0
