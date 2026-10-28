"""``BaseLossComputer._stack_components``: the accumulator seed and the finiteness check.

Two properties of the one function every computer's total goes through:

* The seed is not a ``requires_grad`` leaf. Seeded with
  ``torch.tensor(0.0, requires_grad=True)``, a total built from components that
  carry no graph (a prediction made under ``no_grad``) still came back with an
  ``AddBackward0``, so ``ensure_backward_ready`` let it through and ``backward()``
  updated nothing.
* The per-component NaN/Inf check costs one device-to-host transfer per call, not
  two per component, and decides exactly what the old per-component test decided.
"""

from __future__ import annotations

import inspect
import logging

import pytest
import torch

from spectramr.infrastructure.training.backward_guard import ensure_backward_ready
from spectramr.models.losses.computers import (
    BaseLossComputer,
    LossComputerAdapter,
    UnifiedDiffusionLossComputer,
    UnifiedDisentangledLossComputer,
    UnifiedGANLossComputer,
    UnifiedReconstructionLossComputer,
    UnifiedVAELossComputer,
    UnifiedVQVAELossComputer,
)
from spectramr.models.losses.computers.base import DEFAULT_LOSS_DEVICE


class _Probe(BaseLossComputer):
    """Concrete computer with a fixed weight table, to drive the real stack."""

    def _initialize_losses(self) -> None:
        pass

    def compute(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        raise NotImplementedError

    def _get_loss_weight(
        self, loss_name: str, epoch: int = 0, iteration: int = 0, **kwargs: object
    ) -> float:
        return self._weights.get(loss_name, 1.0)


@pytest.fixture
def probe() -> _Probe:
    p = _Probe(config=None)
    p._weights = {}
    return p


def _reference_stack(
    weights: dict[str, float], components: dict[str, torch.Tensor | None]
) -> tuple[float, int, bool]:
    """The pre-change arithmetic: ``(total, n_skipped, all_skipped)``, by the old test."""
    total = 0.0
    skipped = 0
    candidates = 0
    for name, val in components.items():
        if val is None:
            continue
        candidates += 1
        if bool(torch.isnan(val).any()) or bool(torch.isinf(val).any()):
            skipped += 1
            continue
        total += weights.get(name, 1.0) * float(val.detach())
    return total, skipped, candidates > 0 and skipped == candidates


class TestTheSeedDoesNotManufactureAGraph:
    def test_a_graph_free_total_is_refused_by_the_backward_guard(self, probe) -> None:
        """Fails on the old seed: the total carried an ``AddBackward0`` and passed."""
        generator = torch.nn.Linear(4, 4)
        with torch.no_grad():
            prediction = generator(torch.ones(2, 4))
        component = prediction.pow(2).mean()
        assert component.grad_fn is None

        total = probe._stack_components({"l1": component})

        assert total.grad_fn is None
        with pytest.raises(RuntimeError, match="not connected to the autograd graph"):
            ensure_backward_ready(total, name="gen", global_step=0)

    def test_a_connected_component_still_reaches_the_parameters(self, probe) -> None:
        weight = torch.nn.Parameter(torch.tensor(3.0))
        probe._weights = {"l1": 2.0}
        total = probe._stack_components({"l1": weight * 1.5})
        total.backward()
        assert weight.grad is not None
        assert float(weight.grad) == pytest.approx(3.0)

    def test_the_seed_does_not_change_the_dtype_of_the_total(self, probe) -> None:
        total = probe._stack_components({"a": torch.tensor(2.0, dtype=torch.float64)})
        assert total.dtype == torch.float64


class TestOneFinitenessDecisionPerCall:
    @pytest.mark.parametrize(
        "values",
        [
            {"a": 1.0, "b": 2.0},
            {"a": float("nan"), "b": 2.0},
            {"a": float("inf"), "b": -float("inf")},
            {"a": float("nan"), "b": float("nan")},
            {"a": 0.5, "b": None, "c": float("-inf")},
        ],
    )
    def test_matches_the_per_component_reference(self, probe, values, caplog) -> None:
        components = {
            k: (None if v is None else torch.tensor(v, requires_grad=True))
            for k, v in values.items()
        }
        probe._weights = {"a": 0.5, "b": 3.0, "c": 2.0}
        expected_total, expected_skipped, all_skipped = _reference_stack(probe._weights, components)

        with caplog.at_level(logging.WARNING):
            total = probe._stack_components(components)

        assert float(total.detach()) == pytest.approx(expected_total)
        skips = [r for r in caplog.records if "Skipping NaN/Inf" in r.getMessage()]
        assert len(skips) == expected_skipped
        collapse = any("SILENT NaN COLLAPSE" in r.getMessage() for r in caplog.records)
        assert collapse is all_skipped
        if all_skipped:
            assert total.grad_fn is None and total.requires_grad is True

    def test_a_non_scalar_component_with_one_nan_is_skipped(self, probe) -> None:
        bad = torch.tensor([1.0, float("nan"), 2.0])
        good = torch.tensor(4.0)
        total = probe._stack_components({"bad": bad, "good": good})
        assert float(total) == pytest.approx(4.0)

    def test_the_check_transfers_to_the_host_once(self, probe, monkeypatch) -> None:
        """``.tolist()`` once, never a per-component ``Tensor.__bool__``."""
        bool_calls: list[int] = []
        original_bool = torch.Tensor.__bool__

        def counting_bool(self: torch.Tensor) -> bool:
            bool_calls.append(1)
            return original_bool(self)

        monkeypatch.setattr(torch.Tensor, "__bool__", counting_bool)
        components = {name: torch.tensor(1.0) for name in "abcdef"}
        probe._stack_components(components)
        assert bool_calls == []


def test_every_computer_defaults_to_the_one_cpu_device() -> None:
    for cls in (
        BaseLossComputer,
        LossComputerAdapter,
        UnifiedDiffusionLossComputer,
        UnifiedDisentangledLossComputer,
        UnifiedGANLossComputer,
        UnifiedReconstructionLossComputer,
        UnifiedVAELossComputer,
        UnifiedVQVAELossComputer,
    ):
        default = inspect.signature(cls.__init__).parameters["device"].default
        assert default is DEFAULT_LOSS_DEVICE, cls.__name__
    assert torch.device("cpu") == DEFAULT_LOSS_DEVICE
