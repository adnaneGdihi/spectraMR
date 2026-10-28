"""The ``IStepPolicy`` seam: who owns backward, the step, and the bookkeeping.

``backward_and_step`` was the de-facto seam of the training loop -- called
exactly once, from ``StepExecutor.execute_step``, with nothing else driving an
optimizer -- but it was never declared. That is fine with one implementation and
breaks the moment a second thing needs to own the step.

Two do. An engine-owned backend (DeepSpeed) performs its own loss scaling,
accumulation and ``zero_grad`` inside ``engine.backward``/``engine.step``, so
running the executor's copies as well gives **1/N^2 loss scaling and one real
optimizer step per N^2 micro-batches**. SAM needs two forward+backward passes.

The tests below pin the capability negotiation rather than any backend name,
because the whole point is that ``StepExecutor`` never learns what backend it is
driving.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from spectramr.infrastructure.training.optimizers import AMPPolicy  # noqa: E402
from spectramr.infrastructure.training.step_executor import StepExecutor  # noqa: E402
from spectramr.infrastructure.training.strategy_interfaces import (  # noqa: E402
    IStepPolicy,
)


def _policy(**kw) -> AMPPolicy:
    return AMPPolicy(max_grad_norm=1.0, enable_gradient_clipping=True, **kw)


class _Helper:
    """Minimal stand-in for MixedPrecisionIntegrationHelper."""

    def __init__(self, scaler=None) -> None:
        self.scaler = scaler


class TestDefaultsPreserveExistingBehaviour:
    def test_amp_policy_is_a_step_policy(self) -> None:
        assert isinstance(_policy(), IStepPolicy)

    def test_both_capability_flags_default_false(self) -> None:
        """A policy that declares neither must behave exactly as before."""
        policy = _policy()
        assert policy.owns_gradient_accumulation is False
        assert policy.owns_zero_grad is False
        assert policy.owns_lr_scheduler is False

    def test_executor_keeps_the_configured_accumulation(self) -> None:
        executor = StepExecutor(_Helper(), _policy(), gradient_accumulation_steps=4)
        assert executor.gradient_accumulation_steps == 4
        assert executor.requested_gradient_accumulation_steps == 4


class TestCapabilityNegotiation:
    """``StepExecutor`` asks the policy; it never tests a backend name."""

    class _EngineOwned(AMPPolicy):
        owns_gradient_accumulation = True
        owns_zero_grad = True

    def test_executor_stops_dividing_when_the_policy_owns_accumulation(self) -> None:
        """Both dividing gives 1/N^2 scaling and one step per N^2 micro-batches
        -- and every single-host test would still pass with that bug present."""
        executor = StepExecutor(
            _Helper(),
            self._EngineOwned(max_grad_norm=1.0),
            gradient_accumulation_steps=8,
        )
        assert executor.gradient_accumulation_steps == 1
        # The declared value is still recorded: the engine needs it, and
        # provenance must not claim accumulation of 1.
        assert executor.requested_gradient_accumulation_steps == 8

    def test_ownership_flags_are_read_from_the_policy(self) -> None:
        executor = StepExecutor(_Helper(), self._EngineOwned(max_grad_norm=1.0))
        assert executor._policy_owns_zero_grad is True
        assert executor._policy_owns_accumulation is True
        assert executor._policy_owns_lr_scheduler is False

    def test_lr_scheduler_ownership_is_read_from_the_policy(self) -> None:
        class _SchedOwned(AMPPolicy):
            owns_lr_scheduler = True

        executor = StepExecutor(_Helper(), _SchedOwned(max_grad_norm=1.0))
        assert executor._policy_owns_lr_scheduler is True

    def test_a_truthy_accident_does_not_hand_over_the_scheduler(self) -> None:
        from unittest.mock import MagicMock

        executor = StepExecutor(_Helper(), MagicMock())
        assert executor._policy_owns_lr_scheduler is False

    def test_deepspeed_policy_declares_scheduler_ownership(self) -> None:
        """Planted shape: engine.step() steps the scheduler it was handed, so a
        DeepSpeed policy that did not declare this would be stepped twice."""
        from spectramr.infrastructure.distributed.deepspeed_backend.runtime import (
            DeepSpeedStepPolicy,
        )

        assert DeepSpeedStepPolicy.owns_lr_scheduler is True

    def test_a_policy_lacking_the_flags_entirely_still_works(self) -> None:
        """Duck-typed: a hand-rolled policy object predating the ABC must not
        crash the executor."""

        class _Bare:
            pass

        executor = StepExecutor(_Helper(), _Bare(), gradient_accumulation_steps=3)
        assert executor.gradient_accumulation_steps == 3
        assert executor._policy_owns_zero_grad is False


class TestGuardLoss:
    """The finite-loss rule moved off the executor onto the policy."""

    def test_non_finite_loss_without_a_scaler_raises(self) -> None:
        """No scaler => non-finite grads that clip_grad_norm_ does NOT clamp =>
        optimizer.step() permanently poisons the weights."""
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            _policy().guard_loss(
                torch.tensor(float("nan")), name="g", global_step=7, scaler=None
            )

    def test_non_finite_loss_with_a_scaler_defers_to_the_scaler(self) -> None:
        """GradScaler already skips inf/nan steps; raising here would also add a
        per-step device sync."""
        _policy().guard_loss(
            torch.tensor(float("inf")), name="g", global_step=7, scaler=object()
        )

    def test_a_finite_loss_passes(self) -> None:
        _policy().guard_loss(torch.tensor(1.5), name="g", global_step=0, scaler=None)

    def test_the_message_names_the_config_and_the_step(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            _policy().guard_loss(
                torch.tensor(float("nan")),
                name="discriminator",
                global_step=42,
                scaler=None,
            )
        assert "discriminator" in str(excinfo.value)
        assert "42" in str(excinfo.value)

    def test_a_backend_can_no_op_it(self) -> None:
        """A backend with its own overflow handling must be able to opt out --
        a hardcoded raise in the executor would kill runs it is designed to
        survive."""

        class _OwnOverflow(AMPPolicy):
            def guard_loss(self, loss, *, name, global_step, scaler, monitor=None):
                return

        _OwnOverflow(max_grad_norm=1.0).guard_loss(
            torch.tensor(float("nan")), name="g", global_step=1, scaler=None
        )

    def test_the_executor_hands_its_monitor_to_the_policy(self) -> None:
        """The loop attaches the monitor to the executor, never to the policy, so
        the policy must receive it per call -- a no-op backend then records
        nothing, and the default records instead of syncing to raise."""
        seen = []

        class _Capturing(AMPPolicy):
            def guard_loss(self, loss, *, name, global_step, scaler, monitor=None):
                seen.append(monitor)

        executor = StepExecutor(_Helper(), _Capturing(max_grad_norm=1.0))
        sentinel = object()
        executor.nonfinite_monitor = sentinel
        loss = torch.tensor(1.0, requires_grad=True) * 1.0
        executor._guard_loss(loss, name="g", global_step=3)
        assert seen == [sentinel]

    def test_the_default_records_into_a_monitor_without_raising(self) -> None:
        from spectramr.infrastructure.training.nonfinite_monitor import NonFiniteMonitor

        monitor = NonFiniteMonitor(torch.device("cpu"), distributed=False)
        _policy().guard_loss(
            torch.tensor(float("nan")), name="g", global_step=7, scaler=None, monitor=monitor
        )
        monitor.end_step(7)
        assert monitor.resolve().names == ("g",)


class TestGradientClippingIsPolicyOwned:
    """The default policy's clip is the global norm on every backend (FSDP2's
    DTensor case runs on two real ranks in test_state_io_fsdp.py)."""

    def test_default_policy_clips_over_parameters(self) -> None:
        model = nn.Linear(2, 2)
        model.weight.grad = torch.full_like(model.weight, 10.0)
        model.bias.grad = torch.full_like(model.bias, 10.0)
        _policy().clip_gradients(model, 1.0)
        total = torch.cat([model.weight.grad.flatten(), model.bias.grad.flatten()])
        assert total.norm().item() == pytest.approx(1.0, rel=1e-4)
