"""``BaseTrainingStrategy._handle_anomalies`` runs every step and feeds
``_last_step_metrics``, so it must keep losses on the device (#707): each value
is reduced to a scalar tensor there and converted to a float only at the
training loop's log gate.

The method touches no construction state, so it is exercised on a bare instance
built via ``object.__new__`` with a duck-typed ``env`` (the existing
strategy-unit-test pattern, see ``test_base_strategy_owned_state_2026_06``).
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from tests.unit.infrastructure.training.strategies.conftest import _SyncCounter


class _Strat(BaseTrainingStrategy):
    """Concrete subclass so ``object.__new__`` is not blocked by the abstract
    method; the behaviour under test lives entirely on the base/mixins."""

    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - stub
        return {}


def _bare() -> _Strat:
    s = object.__new__(_Strat)
    s.env = SimpleNamespace(generator=None, discriminator=None, losses={})
    return s


def test_handle_anomalies_return_annotation_admits_tensors() -> None:
    annot = inspect.signature(BaseTrainingStrategy._handle_anomalies).return_annotation
    assert annot == "dict[str, torch.Tensor | float]"


def test_handle_anomalies_returns_device_tensors() -> None:
    result = _bare()._handle_anomalies({"g_total_loss": torch.tensor(1.5)}, {}, 0)
    assert isinstance(result["g_total_loss"], torch.Tensor)
    assert result["g_total_loss"].ndim == 0
    assert result["g_total_loss"].item() == 1.5


def test_handle_anomalies_reduces_non_scalar_tensor_on_device() -> None:
    result = _bare()._handle_anomalies(
        {"d_total_loss": torch.tensor([2.0, 4.0])}, {}, 0
    )
    assert result["d_total_loss"].ndim == 0
    assert result["d_total_loss"].item() == 3.0


def test_handle_anomalies_takes_real_part_of_complex() -> None:
    result = _bare()._handle_anomalies(
        {"c": torch.tensor([1 + 5j, 3 - 2j])}, {}, 0
    )
    assert not torch.is_complex(result["c"])
    assert result["c"].item() == 2.0


def test_handle_anomalies_passes_python_numbers_through() -> None:
    result = _bare()._handle_anomalies({"a": 2, "b": 0.5}, {}, 0)
    assert result == {"a": 2.0, "b": 0.5}


def test_handle_anomalies_rejects_unconvertible_value() -> None:
    with pytest.raises(TypeError):
        _bare()._handle_anomalies({"bad": "x"}, {}, 0)


def test_handle_anomalies_does_not_sync() -> None:
    losses = {f"m{i}": _SyncCounter.make(float(i)) for i in range(12)}
    _SyncCounter.calls = 0
    result = _bare()._handle_anomalies(losses, {}, 0)
    assert _SyncCounter.calls == 0
    assert all(isinstance(v, torch.Tensor) for v in result.values())
