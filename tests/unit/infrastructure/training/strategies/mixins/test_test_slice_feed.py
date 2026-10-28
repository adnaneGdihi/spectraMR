"""The slice writer gets the scoring-domain views; an off-grid input is drawn blank."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.mixins.test_slice_feed import (
    feed_test_slice_writer,
)
from spectramr.infrastructure.training.utils.metric_views import MetricViews

pytestmark = pytest.mark.unit


class _Writer:
    def __init__(self) -> None:
        self.seen: dict = {}

    def observe(self, **kw) -> None:
        self.seen = kw


class _Strategy:
    def _metric_transform_views(self, pred, target, cfg, batch_data=None):
        if pred.shape[-1] != target.shape[-1]:
            raise ValueError("off grid")
        return MetricViews(pred.abs(), target.abs(), None, None)


def _feed(inp):
    writer = _Writer()
    feed_test_slice_writer(
        _Strategy(),
        writer,
        input_batch=inp,
        prediction=torch.rand(1, 1, 8, 8),
        target=torch.rand(1, 1, 8, 8),
        val_config=None,
        batch_data=None,
        batch_index=2,
    )
    return writer.seen


def test_the_input_goes_through_the_same_transform_as_the_prediction() -> None:
    inp = -torch.rand(1, 1, 8, 8)
    seen = _feed(inp)
    assert torch.equal(seen["input_view"], inp.abs()) and seen["batch_index"] == 2


def test_an_input_the_transform_cannot_place_is_left_out() -> None:
    assert _feed(torch.rand(1, 1, 4, 4))["input_view"] is None
