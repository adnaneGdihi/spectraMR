"""Hand the held-out test pass's slice writer the scoring-domain views of one batch."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def feed_test_slice_writer(
    strategy: Any,
    writer: Any,
    *,
    input_batch: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    val_config: Any,
    batch_data: Any,
    batch_index: int | None,
) -> None:
    """Render ``prediction``/``target`` in the domain the metrics grade, and the input with them.

    The input goes through the same transform as the prediction, so a k-space arm's panel shows
    its zero-filled image. An input that transform cannot bring to the target's grid (a
    super-resolution arm's smaller image) is not comparable and is drawn blank.
    """
    with torch.no_grad():
        views = strategy._metric_transform_views(
            prediction, target, val_config, batch_data=batch_data
        )
        try:
            input_view = strategy._metric_transform_views(
                input_batch, target, val_config, batch_data=batch_data
            ).pred
        except (RuntimeError, ValueError) as exc:
            if not getattr(writer, "_input_view_reported", False):
                writer._input_view_reported = True
                logger.info("[Test] input panel not comparable to the target for this arm: %s", exc)
            input_view = None
        if input_view is not None and input_view.shape != views.target.shape:
            input_view = None
        writer.observe(
            input_view=input_view,
            pred_view=views.pred,
            target_view=views.target,
            batch_data=batch_data,
            batch_index=batch_index,
        )


__all__ = ["feed_test_slice_writer"]
