"""Unit tests for training strategy interface contracts.

Targets ``spectramr.infrastructure.training.strategy_interfaces``:
- ``ILossComputer`` abstract base class
- ``IAMPPolicy`` abstract base class
- ``IMetricsReporter`` abstract base class
- ``IOptimizerStepper`` re-export

These are abstract interfaces; we verify:
1. Concrete subclasses that implement all methods can be instantiated.
2. Attempting to instantiate abstract classes directly raises ``TypeError``.
3. Abstract methods raise ``NotImplementedError`` when called on an
   incomplete subclass (safety net: pitfall #9).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers: minimal concrete implementations
# ---------------------------------------------------------------------------


class _ConcreteAMPPolicy:
    """Minimal concrete IAMPPolicy."""

    def should_use_amp(self, model_type, device, config) -> bool:
        return False

    def get_autocast_device(self, device) -> str:
        return "cpu"


class _ConcreteMetricsReporter:
    """Minimal concrete IMetricsReporter."""

    def report_losses(self, losses, epoch, step_type="train"):
        return {k: float(v) if hasattr(v, "item") else float(v) for k, v in losses.items()}

    def detect_anomalies(self, losses):
        return False, False

    def compute_validation_metrics(self, hr_fakes, hr_reals, device):
        return {}


class _ConcreteLossComputer:
    """Minimal concrete ILossComputer."""

    def compute_loss(self, lr_reals, hr_reals, hr_fakes, discriminator_outputs=None, **kwargs):
        return {"total": torch.tensor(0.0)}


# ---------------------------------------------------------------------------
# Canary: import succeeds
# ---------------------------------------------------------------------------


def test_canary_import_interfaces() -> None:
    from spectramr.infrastructure.training.strategy_interfaces import (
        IAMPPolicy,
        ILossComputer,
        IMetricsReporter,
    )

    assert IAMPPolicy is not None
    assert ILossComputer is not None
    assert IMetricsReporter is not None


# ---------------------------------------------------------------------------
# Abstract classes cannot be instantiated directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interface_name", ["ILossComputer", "IAMPPolicy", "IMetricsReporter"])
def test_abstract_class_cannot_be_instantiated(interface_name: str) -> None:
    import spectramr.infrastructure.training.strategy_interfaces as mod

    cls = getattr(mod, interface_name)
    with pytest.raises(TypeError):
        cls()


# ---------------------------------------------------------------------------
# IOptimizerStepper re-export
# ---------------------------------------------------------------------------


def test_ioptimizerstepper_is_exported() -> None:
    from spectramr.infrastructure.training.strategy_interfaces import IOptimizerStepper

    assert IOptimizerStepper is not None


# ---------------------------------------------------------------------------
# Concrete implementations: runtime behaviour
# ---------------------------------------------------------------------------


def test_concrete_amp_policy_should_use_amp() -> None:
    policy = _ConcreteAMPPolicy()
    assert policy.should_use_amp("gan", torch.device("cpu"), {}) is False


def test_concrete_amp_policy_get_autocast_device() -> None:
    policy = _ConcreteAMPPolicy()
    assert policy.get_autocast_device(torch.device("cpu")) == "cpu"


def test_concrete_metrics_reporter_detect_anomalies() -> None:
    reporter = _ConcreteMetricsReporter()
    has_nan, has_inf = reporter.detect_anomalies({"loss": torch.tensor(0.5)})
    assert has_nan is False
    assert has_inf is False


def test_concrete_loss_computer_returns_dict() -> None:
    comp = _ConcreteLossComputer()
    losses = comp.compute_loss(
        lr_reals=torch.randn(2, 1, 8, 8),
        hr_reals=torch.randn(2, 1, 8, 8),
        hr_fakes=torch.randn(2, 1, 8, 8),
    )
    assert isinstance(losses, dict)
    assert "total" in losses


# ---------------------------------------------------------------------------
# Edge: abstract method raises NotImplementedError for partial implementation
# ---------------------------------------------------------------------------


def test_loss_computer_abstract_method_raises() -> None:
    from spectramr.infrastructure.training.strategy_interfaces import ILossComputer

    class _Partial(ILossComputer):
        pass  # does not implement compute_loss

    with pytest.raises(TypeError):
        _Partial()


def test_amp_policy_abstract_methods_raise() -> None:
    from spectramr.infrastructure.training.strategy_interfaces import IAMPPolicy

    class _Partial(IAMPPolicy):
        pass

    with pytest.raises(TypeError):
        _Partial()


# ---------------------------------------------------------------------------
# guard_loss agrees across ranks (a lone raise strands peers in the next collective)
# ---------------------------------------------------------------------------


class _FakeGroup:
    """Patches torch.distributed to a 2-rank group whose peer reports ``peer_flag``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, peer_flag: float) -> None:
        self.reduces = 0
        dist = torch.distributed
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)

        def _all_reduce(tensor: torch.Tensor, op: object = None, **_: object) -> None:
            self.reduces += 1
            tensor.copy_(torch.maximum(tensor, torch.tensor([peer_flag])))

        monkeypatch.setattr(dist, "all_reduce", _all_reduce)


def _guard(loss: torch.Tensor, scaler: object | None = None) -> None:
    from spectramr.infrastructure.training.optimizers import AMPPolicy

    policy = AMPPolicy(max_grad_norm=1.0, enable_gradient_clipping=False)
    policy.guard_loss(loss, name="gen", global_step=7, scaler=scaler)


def test_guard_loss_single_process_raises_without_a_collective() -> None:
    with pytest.raises(RuntimeError, match=r"on this rank"):
        _guard(torch.tensor(float("nan")))


def test_guard_loss_peer_nan_raises_on_a_finite_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    group = _FakeGroup(monkeypatch, peer_flag=1.0)
    with pytest.raises(RuntimeError, match=r"on a peer rank"):
        _guard(torch.tensor(0.5))
    assert group.reduces == 1


def test_guard_loss_local_nan_raises_after_agreeing(monkeypatch: pytest.MonkeyPatch) -> None:
    group = _FakeGroup(monkeypatch, peer_flag=0.0)
    with pytest.raises(RuntimeError, match=r"on this rank"):
        _guard(torch.tensor(float("inf")))
    assert group.reduces == 1


def test_guard_loss_finite_everywhere_passes_with_one_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _FakeGroup(monkeypatch, peer_flag=0.0)
    _guard(torch.tensor(0.5))
    assert group.reduces == 1


def test_guard_loss_reads_the_flags_back_in_one_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeGroup(monkeypatch, peer_flag=1.0)
    calls: list[str] = []
    real = torch.Tensor.tolist

    def _counting_tolist(self: torch.Tensor) -> object:
        calls.append("tolist")
        return real(self)

    def _no_item(self: torch.Tensor) -> object:
        raise AssertionError("guard_loss must not call .item() in the step")

    monkeypatch.setattr(torch.Tensor, "tolist", _counting_tolist)
    monkeypatch.setattr(torch.Tensor, "item", _no_item)
    with pytest.raises(RuntimeError, match=r"on a peer rank"):
        _guard(torch.tensor(0.5))
    assert calls == ["tolist"]


def test_guard_loss_with_a_monitor_records_without_a_collective_or_a_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop's path: the agreement moves to the monitor's end_step, so the
    guard itself enters no collective, reads nothing back and never raises."""
    from spectramr.infrastructure.training.nonfinite_monitor import NonFiniteMonitor
    from spectramr.infrastructure.training.optimizers import AMPPolicy

    group = _FakeGroup(monkeypatch, peer_flag=1.0)
    monitor = NonFiniteMonitor(torch.device("cpu"), distributed=False)

    def _no_readback(self: torch.Tensor, *a: object) -> object:
        raise AssertionError("guard_loss must not read a tensor back under a monitor")

    for attr in ("tolist", "item", "__bool__"):
        monkeypatch.setattr(torch.Tensor, attr, _no_readback)
    policy = AMPPolicy(max_grad_norm=1.0, enable_gradient_clipping=False)
    policy.guard_loss(torch.tensor(float("nan")), name="gen", global_step=7, scaler=None, monitor=monitor)
    assert group.reduces == 0
    monkeypatch.undo()
    monitor.end_step(7)
    assert monitor.resolve().names == ("gen",)


def test_guard_loss_defers_to_the_scaler_without_a_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _FakeGroup(monkeypatch, peer_flag=1.0)
    _guard(torch.tensor(float("nan")), scaler=object())
    assert group.reduces == 0
