"""The input-as-prediction (zero-filled / identity) baseline on the generic validation path.

Cohort review 2026-09-02, T0.7: the baseline used to exist only on the
cold-diffusion branch, so an attention arm that scored below its own
zero-filled input at every rung stayed green on every other strategy.
"""

from __future__ import annotations

import logging

import pytest
import torch

from spectramr.infrastructure.training.strategies.mixins.model_validation import (
    ModelValidationMixin,
)


class _Computer:
    """Records the ``only`` subset it was asked for; returns fixed psnr/ssim/hfen."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[str, ...] | None]] = []

    def compute(self, pred, target, *, only=None, **_kw):
        self.calls.append((tuple(pred.shape), tuple(only) if only else None))
        return {"psnr": 20.0, "ssim": 0.6, "hfen": 0.5}


class _Host(ModelValidationMixin):
    def _validation_forward(self, *a, **k):  # pragma: no cover - abstract hook
        raise NotImplementedError


def _identity_transform(pred, target, _cfg):
    return pred, target


def test_baseline_grades_the_input_on_its_three_metrics() -> None:
    """The fires-test: ``val_zf_*`` appears, computed from the INPUT, on psnr/ssim/hfen.

    SSIM joined the baseline when the EI arms began selecting on ``val_ssim``
    against it (8a14b7ed7, #2601).
    """
    host = _Host()
    computer = _Computer()
    x = torch.rand(2, 1, 8, 8)
    y = torch.rand(2, 1, 8, 8)
    out = host._zero_filled_baseline_metrics(x, y, computer, _identity_transform, None)
    assert out == {"val_zf_psnr": 20.0, "val_zf_ssim": 0.6, "val_zf_hfen": 0.5}
    assert computer.calls == [((2, 1, 8, 8), ("psnr", "ssim", "hfen"))]


def test_five_d_input_is_flattened_like_the_target() -> None:
    host = _Host()
    computer = _Computer()
    x = torch.rand(1, 1, 8, 8, 3)
    y = torch.rand(1, 1, 8, 8, 3)
    host._zero_filled_baseline_metrics(x, y, computer, _identity_transform, None)
    assert computer.calls[0][0] == (3, 1, 8, 8)


def test_shape_mismatch_is_not_applicable_reported_once(caplog) -> None:
    """An SR arm whose input is smaller than its target has no identity baseline."""
    host = _Host()
    computer = _Computer()
    x = torch.rand(2, 1, 4, 4)
    y = torch.rand(2, 1, 8, 8)
    with caplog.at_level(logging.INFO):
        first = host._zero_filled_baseline_metrics(x, y, computer, _identity_transform, None)
        second = host._zero_filled_baseline_metrics(x, y, computer, _identity_transform, None)
    assert first == {} and second == {}
    assert computer.calls == []
    assert sum("not applicable" in r.getMessage() for r in caplog.records) == 1


def test_missing_input_or_target_yields_nothing() -> None:
    host = _Host()
    assert (
        host._zero_filled_baseline_metrics(None, torch.rand(1, 1, 2, 2), _Computer(), None, None)
        == {}
    )


def test_cuda_oom_is_never_swallowed() -> None:
    class _Boom(_Computer):
        def compute(self, *a, **k):
            raise torch.cuda.OutOfMemoryError("boom")

    host = _Host()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        host._zero_filled_baseline_metrics(
            torch.rand(1, 1, 2, 2), torch.rand(1, 1, 2, 2), _Boom(), None, None
        )


def test_the_diffusion_strategy_owns_its_own_baseline() -> None:
    """The cold branch grades the MASKED measurement; the generic one must not double-emit."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

    assert DiffusionTrainingStrategy._owns_zero_filled_baseline is True
    assert getattr(_Host, "_owns_zero_filled_baseline", False) is False
