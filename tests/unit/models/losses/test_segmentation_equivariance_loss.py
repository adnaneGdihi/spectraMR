"""SegmentationEquivarianceLoss: the detached anchor is computed without a graph."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.losses.segmentation_equivariance_loss import (
    SegmentationEquivarianceLoss,
)


class _Segmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.grad_enabled_calls: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        out: torch.Tensor = self.conv(x)
        return out


def _reference(distance: str, anchor: torch.Tensor, translated: torch.Tensor) -> torch.Tensor:
    if distance == "kl":
        log_p, log_q = F.log_softmax(anchor, dim=1), F.log_softmax(translated, dim=1)
        return 0.5 * (
            F.kl_div(log_q, log_p.exp(), reduction="batchmean")
            + F.kl_div(log_p, log_q.exp(), reduction="batchmean")
        )
    p, q = F.softmax(anchor, dim=1), F.softmax(translated, dim=1)
    return (p - q).abs().mean() if distance == "l1" else (p - q).pow(2).mean()


@pytest.mark.parametrize("distance", ["kl", "l1", "l2"])
def test_detached_anchor_matches_the_detach_after_forward_reference(distance: str) -> None:
    seg = _Segmenter()
    source = torch.rand(2, 1, 8, 8)
    translated = torch.rand(2, 1, 8, 8, requires_grad=True)
    loss = SegmentationEquivarianceLoss(distance=distance)(
        translated, source, context={"segmenter": seg}
    )
    expected = _reference(distance, seg(source).detach(), seg(translated))
    torch.testing.assert_close(loss, expected)
    (grad,) = torch.autograd.grad(loss, translated)
    (grad_ref,) = torch.autograd.grad(expected, translated)
    torch.testing.assert_close(grad, grad_ref)


def test_the_detached_anchor_builds_no_graph() -> None:
    seg = _Segmenter()
    SegmentationEquivarianceLoss()(
        torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), context={"segmenter": seg}
    )
    assert seg.grad_enabled_calls == [False, True]


def test_an_attached_anchor_keeps_its_graph() -> None:
    seg = _Segmenter()
    source = torch.rand(1, 1, 8, 8, requires_grad=True)
    loss = SegmentationEquivarianceLoss(detach_anchor=False)(
        torch.rand(1, 1, 8, 8), source, context={"segmenter": seg}
    )
    (grad,) = torch.autograd.grad(loss, source)
    assert grad.abs().sum() > 0


def test_unknown_distance_raises() -> None:
    with pytest.raises(ValueError, match="distance"):
        SegmentationEquivarianceLoss(distance="cosine")


def test_missing_segmenter_raises() -> None:
    with pytest.raises(ValueError, match="segmenter"):
        SegmentationEquivarianceLoss()(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4))
