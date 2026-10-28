"""The gauge prior: a soft prior on each image's implied T2/T1 and FLAIR/T1 gain ratios.

The series-gain gate (#2614) measured these ratios per subject on M4Raw: stable medians
(T2/T1 about 4.2, FLAIR/T1 about 0.51) with a per-subject spread (std of log ratio 0.16 and
0.29). The two ratios are exactly the two directions an unknown gain leaves free, so a prior
centred on the medians with the measured spread closes the gauge from data.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.training.strategies.qmap_gauge_prior import (
    gain_ratio_penalty,
    parse_gain_ratio_prior,
    prior_predicted_gains,
    projected_gains,
)

PRIOR = {
    "weight": 1.0,
    "T2": {"median": 4.2, "std_log": 0.16},
    "FLAIR": {"median": 0.51, "std_log": 0.29},
}


def _renders_and_targets(g=(1.0, 4.2, 0.51)):
    torch.manual_seed(0)
    renders = [torch.rand(2, 1, 8, 8) + 0.5 for _ in range(3)]
    targets = [g[k] * r for k, r in enumerate(renders)]
    masks = [torch.ones(2, 1, 8, 8, dtype=torch.bool) for _ in range(3)]
    return renders, targets, masks


def test_projected_gains_recover_the_scale_of_each_series() -> None:
    renders, targets, masks = _renders_and_targets((2.0, 8.4, 1.02))
    g = projected_gains(renders, targets, masks)
    assert torch.allclose(g, torch.tensor([[2.0, 8.4, 1.02]] * 2), rtol=1e-5)


def test_gains_at_the_measured_medians_cost_nothing() -> None:
    renders, targets, masks = _renders_and_targets((3.0, 3.0 * 4.2, 3.0 * 0.51))
    prior = parse_gain_ratio_prior(PRIOR)
    assert float(gain_ratio_penalty(renders, targets, masks, prior)) == pytest.approx(0.0, abs=1e-8)


def test_the_penalty_is_the_squared_z_score_of_the_log_ratio() -> None:
    # T2/T1 one measured std above its median, FLAIR/T1 on its median: penalty 1 per image.
    r_t2 = 4.2 * math.exp(0.16)
    renders, targets, masks = _renders_and_targets((1.0, r_t2, 0.51))
    prior = parse_gain_ratio_prior(PRIOR)
    assert float(gain_ratio_penalty(renders, targets, masks, prior)) == pytest.approx(1.0, rel=1e-4)


def test_the_penalty_reaches_the_maps_through_the_renders() -> None:
    renders, targets, masks = _renders_and_targets((1.0, 6.0, 0.51))
    renders = [r.requires_grad_() for r in renders]
    gain_ratio_penalty(renders, targets, masks, parse_gain_ratio_prior(PRIOR)).backward()
    assert all(r.grad is not None and r.grad.abs().sum() > 0 for r in renders[:2])


def test_absent_prior_is_none() -> None:
    assert parse_gain_ratio_prior(None) is None


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"weight": 1.0, "T2": {"median": 4.2, "std_log": 0.16}}, "FLAIR"),
        ({"weight": -1.0, **{k: v for k, v in PRIOR.items() if k != "weight"}}, "weight"),
        ({**PRIOR, "T2": {"median": 0.0, "std_log": 0.16}}, "median"),
        ({**PRIOR, "T2": {"median": 4.2, "std_log": 0.0}}, "std_log"),
        ({**PRIOR, "PD": {"median": 1.0, "std_log": 0.1}}, "PD"),
    ],
)
def test_a_malformed_prior_is_refused(bad, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_gain_ratio_prior(bad)


def test_a_held_out_gain_is_predicted_from_the_available_ones_and_the_medians() -> None:
    prior = parse_gain_ratio_prior(PRIOR)
    fitted = torch.tensor([[2.0, 999.0, 2.0 * 0.51], [2.0, 2.0 * 4.2, 999.0], [999.0, 8.4, 1.02]])
    available = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    g = prior_predicted_gains(fitted, available, prior)
    # the 999s are the held-out contrasts' own fits: never used
    assert g[0, 1] == pytest.approx(2.0 * 4.2, rel=1e-5)
    assert g[1, 2] == pytest.approx(2.0 * 0.51, rel=1e-5)
    assert g[2, 0] == pytest.approx(2.0, rel=1e-5)


def test_a_held_out_gain_does_not_move_with_its_own_target() -> None:
    # Planted leak check: the held-out contrast's fitted gain changing must not move its
    # prediction, otherwise the interval and the PSNR would have seen the label.
    prior = parse_gain_ratio_prior(PRIOR)
    available = torch.tensor([[1.0, 0.0, 1.0]])
    a = prior_predicted_gains(torch.tensor([[2.0, 1.0, 1.0]]), available, prior)
    b = prior_predicted_gains(torch.tensor([[2.0, 50.0, 1.0]]), available, prior)
    assert torch.equal(a[:, 1], b[:, 1])
