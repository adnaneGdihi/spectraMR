"""Tests for ``spectramr.models.losses.pmps_losses``: the paired-distribution MMD.

``mmd_paired_distribution_match`` is the calibration objective and primary metric
of ``pmps/stage_b_corruption_calibration``. With absolute bandwidths on raw pixels
it was the constant 2/B with a vanishing gradient (#2277).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.pmps_losses import MMDPairedDistributionMatch


def _batches(shifted: bool) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, 1, 128, 128, generator=g)
    y = torch.rand(4, 1, 128, 128, generator=g)
    if shifted:
        y = y * 0.5 + 0.5
    return x, y


def test_defaults() -> None:
    loss = MMDPairedDistributionMatch()
    assert loss.bandwidths == (0.5, 1.0, 2.0, 4.0)
    assert loss.bandwidth_mode == "median"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bandwidths": ()}, "non-empty"),
        ({"bandwidths": (1.0, 0.0)}, "> 0"),
        ({"bandwidth_mode": "fixed"}, "bandwidth_mode"),
    ],
)
def test_invalid_construction_raises(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MMDPairedDistributionMatch(**kwargs)  # type: ignore[arg-type]


def test_identical_batches_score_zero() -> None:
    x, _ = _batches(shifted=False)
    assert float(MMDPairedDistributionMatch()(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_shifted_distribution_scores_higher_with_gradient() -> None:
    """On 128^2 images a shifted distribution scores clearly higher and has a gradient.

    With absolute bandwidths 0.5..4 both pairs scored 0.5 (= 2/B) and the
    largest gradient entry was below 1e-38.
    """
    loss = MMDPairedDistributionMatch()
    x_same, y_same = _batches(shifted=False)
    x, y = _batches(shifted=True)
    x.requires_grad_(True)
    shifted = loss(x, y)
    shifted.backward()
    same = float(loss(x_same, y_same))
    assert float(shifted) > same + 0.05
    assert x.grad is not None and float(x.grad.abs().max()) > 1e-6


def test_absolute_mode_is_available() -> None:
    """The absolute bandwidths remain selectable for pre-scaled features."""
    x = torch.zeros(2, 1, 1, 1)
    y = torch.ones(2, 1, 1, 1)
    loss = MMDPairedDistributionMatch(bandwidths=(1.0,), bandwidth_mode="absolute")
    expected = 2.0 - 2.0 * float(torch.exp(torch.tensor(-0.5)))
    assert float(loss(x, y)) == pytest.approx(expected, rel=1e-6)
