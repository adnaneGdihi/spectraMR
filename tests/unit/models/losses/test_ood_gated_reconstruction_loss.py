"""OODGatedReconstructionLoss normalises by the weights it actually applies."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.ood_gated_reconstruction_loss import OODGatedReconstructionLoss


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_zero_boost_is_the_plain_base_mean_for_any_channel_count(channels: int) -> None:
    """The weights are [B, 1, H, W]; dividing by their unbroadcast sum scaled the loss by C."""
    torch.manual_seed(channels)
    pred, target = torch.rand(2, channels, 8, 8), torch.rand(2, channels, 8, 8)
    loss = OODGatedReconstructionLoss(boost=0.0)(pred, target)
    torch.testing.assert_close(loss, (pred - target).abs().mean())


def test_uniform_context_score_matches_the_missing_context_fallback() -> None:
    torch.manual_seed(0)
    pred, target = torch.rand(2, 2, 8, 8), torch.rand(2, 2, 8, 8)
    loss_fn = OODGatedReconstructionLoss(score_source="context", require_context=False)
    fallback = loss_fn(pred, target)
    uniform = loss_fn(pred, target, context={"ood_score": torch.zeros(1, 1, 8, 8)})
    torch.testing.assert_close(uniform, fallback)


def test_a_flagged_region_is_up_weighted() -> None:
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    target[..., 0, 0] = 1.0
    score = torch.zeros(1, 1, 4, 4)
    score[..., 0, 0] = 1.0
    loss_fn = OODGatedReconstructionLoss(score_source="context", boost=3.0)
    weighted = loss_fn(pred, target, context={"ood_score": score})
    # One error of 1 at weight 4 over a total weight of 15 + 4.
    torch.testing.assert_close(weighted, torch.tensor(4.0 / 19.0))


def test_missing_context_raises_when_required() -> None:
    with pytest.raises(ValueError, match="ood_score"):
        OODGatedReconstructionLoss(score_source="context")(
            torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4)
        )
