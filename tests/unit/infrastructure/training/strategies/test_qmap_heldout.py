"""Readings for a held-out contrast: the label-free prediction must not see its target."""

from __future__ import annotations

import torch

from spectramr.infrastructure.training.strategies.qmap_conformal import SubjectResiduals
from spectramr.infrastructure.training.strategies.qmap_gauge_prior import parse_gain_ratio_prior
from spectramr.infrastructure.training.strategies.qmap_heldout import (
    heldout_pass,
    heldout_readings,
)

PRIOR = parse_gain_ratio_prior(
    {
        "weight": 1.0,
        "T2": {"median": 4.2, "std_log": 0.16},
        "FLAIR": {"median": 0.51, "std_log": 0.29},
    }
)


def _setup():
    torch.manual_seed(0)
    renders = [torch.rand(2, 1, 6, 6) + 0.5 for _ in range(3)]
    targets = [g * r for g, r in zip((2.0, 8.4, 1.02), renders, strict=True)]
    masks = [torch.ones(2, 1, 6, 6, dtype=torch.bool) for _ in range(3)]
    return renders, targets, masks


def _readings(renders, targets, masks, k, seen):
    available = torch.ones(2, 3)
    available[:, k] = 0.0
    return heldout_readings(
        k,
        renders,
        available,
        targets=targets,
        masks=masks,
        fitted_prediction=lambda j, r: r,
        psnr=lambda pred, target: seen.append(pred.clone()) or 0.0,
        prior=PRIOR,
        residuals=None,
        subjects=["a", "b"],
    )


def test_the_prior_gain_prediction_ignores_the_held_out_target() -> None:
    renders, targets, masks = _setup()
    first, second = [], []
    _readings(renders, targets, masks, 1, first)
    moved = list(targets)
    moved[1] = 50.0 * targets[1]
    _readings(renders, moved, masks, 1, second)
    assert torch.equal(first[1], second[1])  # [0] is the target-fitted reading, [1] the prior's


def test_the_prior_gain_prediction_recovers_a_consistent_scale() -> None:
    renders, targets, masks = _setup()
    seen = []
    _readings(renders, targets, masks, 2, seen)
    assert torch.allclose(seen[1], targets[2], rtol=1e-4)  # gains sit on the medians here


def test_without_a_prior_only_the_pre_registered_reading_is_made() -> None:
    renders, targets, masks = _setup()
    out = heldout_readings(
        0,
        renders,
        torch.tensor([[0.0, 1.0, 1.0]] * 2),
        targets=targets,
        masks=masks,
        fitted_prediction=lambda j, r: r,
        psnr=lambda p, t: 1.0,
        prior=None,
        residuals=None,
        subjects=["a", "b"],
    )
    assert out == {"val_psnr_loo": 1.0}


def test_the_pass_feeds_the_conformal_residuals_per_subject() -> None:
    renders, targets, masks = _setup()
    acc = SubjectResiduals()
    out = heldout_pass(
        lambda available: renders,
        torch.cat(targets, dim=1),
        masks,
        fitted_prediction=lambda j, r: r,
        psnr=lambda p, t: 1.0,
        prior=PRIOR,
        residuals=acc,
        subjects=["a", "b"],
    )
    assert {"val_psnr_loo", "val_psnr_loo_prior_gain"} <= set(out)
    assert {key[0] for key in acc._counts} == {"a", "b"}
