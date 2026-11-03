"""Unit tests for the equivariance-conformal scoring helpers (tiny CPU tensors)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.equivariance_conformal_scoring import (
    run_equivariance_calibration,
    score_loader,
    two_sample_ks,
)

_RAMP = torch.linspace(-1.0, 1.0, 16).view(1, 1, 1, 16)


def _non_equivariant(y, csm=None):
    return y * _RAMP


def _loader(n_batches, seed, b=2):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(b, 1, 16, 16, generator=gen) for _ in range(n_batches)]


def test_score_loader_one_score_per_sample_float64_cpu():
    scores = score_loader(
        _non_equivariant,
        _loader(3, 0),
        num_rotations=2,
        group="SO2",
        seed=0,
        max_shift_px=0,
        device="cpu",
    )
    assert scores.shape == (6,) and scores.dtype == torch.float64
    assert (scores > 0).all()


def test_held_out_coverage_and_ks_use_separate_test_scores():
    report = run_equivariance_calibration(
        _non_equivariant,
        _loader(4, 0),
        num_rotations=2,
        seed=3,
        exchangeability_split="held_out",
        test_dataloader=_loader(3, 1),
    )
    test = score_loader(
        _non_equivariant,
        _loader(3, 1),
        num_rotations=2,
        group="SO2",
        seed=3 + 8,
        max_shift_px=4,
        device="cpu",
    )
    assert report["n_test"] == 6
    assert report["coverage_at_alpha"] == pytest.approx(
        float((test <= report["quantile"]).double().mean())
    )
    assert report["mean_test_score"] == pytest.approx(float(test.mean()))


def test_held_out_flags_a_shifted_test_cohort():
    """A scaled test cohort shifts the score distribution; the cal-vs-test KS flags it."""
    shifted = [10.0 * x for x in _loader(8, 1)]
    held = run_equivariance_calibration(
        _non_equivariant,
        _loader(8, 0),
        num_rotations=2,
        exchangeability_split="held_out",
        test_dataloader=shifted,
    )
    assert held["exchangeability"]["flagged"] is True
    assert held["exchangeability"]["ks_statistic"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"exchangeability_split": "bogus"}, "exchangeability_split"),
        ({"exchangeability_test": "bogus"}, "exchangeability_test"),
        ({"exchangeability_split": "held_out"}, "held-out test loader"),
        ({"exchangeability_split": "held_out", "test_dataloader": []}, "no samples"),
    ],
)
def test_invalid_declarations_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        run_equivariance_calibration(_non_equivariant, _loader(2, 0), num_rotations=2, **kwargs)


def test_two_sample_ks_matches_closed_form_for_disjoint_sets():
    stat, p = two_sample_ks(torch.zeros(50), torch.ones(50))
    assert stat == pytest.approx(1.0)
    assert p < 1e-6
