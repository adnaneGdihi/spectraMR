"""Tests for the TALL locality helper and its wiring in the reconstruction strategy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategies.tall_locality import (
    TALL_LOCALITY_LOSS_KEY,
    add_tall_locality,
    arm_tall_locality,
    tall_locality_term,
)


class _StubTALLGenerator(nn.Module):
    """Minimal stand-in exposing a grad-carrying locality term and the arming switch."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.track_locality = False
        self.calls = 0

    def last_locality_loss(self) -> torch.Tensor:
        self.calls += 1
        return self.scale * 0.5


class _NoLocalityGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))


def test_zero_weight_returns_none_without_calling_generator() -> None:
    gen = _StubTALLGenerator()
    assert tall_locality_term(gen, 0.0) is None
    assert gen.calls == 0


def test_zero_weight_does_not_arm_tracking() -> None:
    gen = _StubTALLGenerator()
    arm_tall_locality(gen, 0.0)
    assert gen.track_locality is False


def test_positive_weight_arms_tracking() -> None:
    gen = _StubTALLGenerator()
    arm_tall_locality(gen, 0.01)
    assert gen.track_locality is True


def test_positive_weight_returns_weighted_differentiable_term() -> None:
    gen = _StubTALLGenerator()
    term = tall_locality_term(gen, 0.01)
    assert term is not None and term.requires_grad
    assert torch.isclose(term.detach(), torch.tensor(0.01))
    term.backward()
    assert gen.scale.grad is not None and float(gen.scale.grad) == pytest.approx(0.005)


def test_unwraps_ddp_style_module_attribute() -> None:
    inner = _StubTALLGenerator()
    wrapper = nn.Module()
    wrapper.module = inner  # type: ignore[assignment]
    arm_tall_locality(wrapper, 1.0)
    assert inner.track_locality is True
    assert tall_locality_term(wrapper, 1.0) is not None and inner.calls == 1


def test_positive_weight_without_locality_method_raises() -> None:
    with pytest.raises(TypeError, match="track_locality"):
        arm_tall_locality(_NoLocalityGenerator(), 0.01)
    with pytest.raises(TypeError, match="no last_locality_loss"):
        tall_locality_term(_NoLocalityGenerator(), 0.01)


def test_add_tall_locality_none_leaves_dict_and_total_untouched() -> None:
    losses: dict[str, torch.Tensor] = {}
    total = torch.tensor(3.0, requires_grad=True)
    assert add_tall_locality(total, losses, None) is total
    assert losses == {}


def test_add_tall_locality_folds_term_and_records_key() -> None:
    losses: dict[str, torch.Tensor] = {}
    total = torch.tensor(3.0, requires_grad=True)
    loc = torch.tensor(0.25, requires_grad=True)
    out = add_tall_locality(total, losses, loc)
    assert torch.isclose(out.detach(), torch.tensor(3.25))
    assert losses[TALL_LOCALITY_LOSS_KEY] is loc


class _Computer:
    """Stands in for UnifiedReconstructionLossComputer: returns a fixed leaf total."""

    def __init__(self, total: torch.Tensor) -> None:
        self._total = total

    def compute(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(total=self._total, components={"l1": self._total})


def _wired_strategy(weight: float, gen: nn.Module, total: torch.Tensor):
    """Build the reconstruction strategy without __init__; stub only its seams."""
    s = object.__new__(ReconstructionTrainingStrategy)
    s.pinn_module = None
    s.env = SimpleNamespace(generator=gen, losses={})
    s.config = SimpleNamespace(
        training=SimpleNamespace(tall_sr=SimpleNamespace(lambda_locality=weight)),
        model=SimpleNamespace(model_type="tall_sr"),
    )
    s.loss_computer = _Computer(total)
    s._loss_dict_reuse = {}
    s._prepare_batch_context = lambda inp, tgt, **_kw: {"hr": tgt}  # type: ignore[method-assign]
    s._prepare_generator_inputs = lambda ctx, inp: (inp, {})  # type: ignore[method-assign]
    s._generate_predictions = lambda lr, kw, ctx: (lr, None)  # type: ignore[method-assign]
    s._compute_training_metrics = lambda **_kw: {}  # type: ignore[method-assign]
    return s


def test_strategy_adds_locality_term_to_g_total_and_logs_it() -> None:
    gen = _StubTALLGenerator()
    total = torch.tensor(3.0, requires_grad=True)
    s = _wired_strategy(0.5, gen, total)
    x = torch.zeros(1, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0, iteration=0)
    assert gen.track_locality is True
    assert TALL_LOCALITY_LOSS_KEY in out
    expected_term = 0.5 * 2.0 * 0.5  # weight * last_locality_loss() of the stub
    assert torch.isclose(out["g_total_loss"].detach(), total.detach() + torch.tensor(expected_term))
    out["g_total_loss"].backward()
    assert gen.scale.grad is not None and float(gen.scale.grad) == pytest.approx(0.25)


def test_strategy_zero_weight_leaves_loss_dict_untouched() -> None:
    gen = _StubTALLGenerator()
    total = torch.tensor(3.0, requires_grad=True)
    s = _wired_strategy(0.0, gen, total)
    x = torch.zeros(1, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0, iteration=0)
    assert TALL_LOCALITY_LOSS_KEY not in out
    assert out["g_total_loss"] is total
    assert gen.calls == 0 and gen.track_locality is False
