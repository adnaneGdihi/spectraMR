"""The validation visual reuses a strategy's own render instead of forwarding twice."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from spectramr.pipelines.train import _run_validation
from tests.unit.pipelines import test_validation_fail_loud as _base

# Ramps spanning [0, 1] survive the logger's min-max scaling unchanged, and differ from the
# random input a raw generator call would echo back.
_PRED = torch.linspace(0, 1, 8).expand(1, 1, 8, 8).clone()
_TARGET = _PRED.transpose(-1, -2).contiguous()


class _Strategy:
    sets_visual = True

    def __init__(self) -> None:
        cfg = _base.block_stub(
            "logging",
            log_validation_images=True,
            save_validation_images=False,
            validation_image_interval=1,
            max_images_per_batch=4,
            log_input_images=False,
            log_difference_images=True,
        )
        self.config = SimpleNamespace(logging=cfg)
        self.env = SimpleNamespace(generator=torch.nn.Identity())
        self.forwards = 0
        self._last_visual_pred = None
        self._last_visual_target = None

    def _validation_forward(self, x, ctx=None):
        self.forwards += 1
        return x

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        if self.sets_visual:
            self._last_visual_pred = _PRED.expand_as(input_batch).clone()
            self._last_visual_target = _TARGET.expand_as(target_batch).clone()
        return {"val_loss": 0.5}

    def finalize_validation(self) -> dict:
        return {}


class _NoVisualStrategy(_Strategy):
    sets_visual = False


class _PredOnlyStrategy(_Strategy):
    """Confluence-like: renders a prediction but no target."""

    def validation_step(self, input_batch, target_batch, batch_idx=0):
        self._last_visual_pred = _PRED.expand_as(input_batch).clone()
        return {"val_loss": 0.5}


def _run(strategy) -> MagicMock:
    x = torch.rand(1, 1, 8, 8)
    pipeline = _base._make_pipeline([{"input": x, "target": x.clone()}])
    writer = MagicMock()
    _run_validation(
        pipeline, strategy, iteration=10, epoch=0, logging_service=None, tb_writer=writer
    )
    return writer


def _logged(writer: MagicMock, tag: str) -> torch.Tensor:
    """The tensor a writer received under ``tag`` (``images`` takes a tag -> tensor dict)."""
    for call in writer.method_calls:
        if call.args and isinstance(call.args[0], dict) and tag in call.args[0]:
            return call.args[0][tag]
    raise AssertionError(f"nothing logged under {tag!r}: {writer.method_calls}")


def test_strategy_visual_skips_the_extra_forward() -> None:
    strategy = _Strategy()
    writer = _run(strategy)
    assert strategy.forwards == 0
    # The skip is only sound if the strategy's own render is what gets logged.
    assert torch.allclose(_logged(writer, "val/predictions").float(), _PRED)
    assert torch.allclose(_logged(writer, "val/targets").float(), _TARGET)


def test_strategy_without_visual_still_forwards_once() -> None:
    strategy = _NoVisualStrategy()
    _run(strategy)
    assert strategy.forwards == 1


def test_visuals_set_before_validation_still_forward_and_feed_the_override() -> None:
    """Trajectory-like: visuals come from training, so behaviour matches the base."""
    strategy = _NoVisualStrategy()
    strategy._last_visual_pred = _PRED.clone()
    strategy._last_visual_target = _TARGET.clone()
    writer = _run(strategy)
    assert strategy.forwards == 1
    assert torch.allclose(_logged(writer, "val/predictions").float(), _PRED)


def test_prediction_only_strategy_still_forwards() -> None:
    strategy = _PredOnlyStrategy()
    _run(strategy)
    assert strategy.forwards == 1
