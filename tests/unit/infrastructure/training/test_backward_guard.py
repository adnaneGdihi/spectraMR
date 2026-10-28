"""Planted violations for the pre-backward autograd-graph guard (#1952).

Every test here plants a *shape* the guard claims to catch and asserts it turns
red, per non-negotiable 15: a gate is only a gate for the violation shape you
have watched it fail on. The shapes are:

1. a bare leaf reaching the executor (the ``_stack_components`` guard-return);
2. that leaf under a **no-op policy guard** (the DeepSpeed shape) -- proves the
   check is not delegated to a backend;
3. that leaf with a **live GradScaler** -- proves the check runs before
   ``scaler.scale()``, which would otherwise wrap it in a ``MulBackward``;
4. that leaf under **gradient accumulation** -- same argument for ``loss / N``;
5. a non-finite leaf -- proves the *existing* finite guard still reports first.

The control in each case is a grad-connected loss, which must pass untouched.
"""

from unittest.mock import MagicMock

import pytest
import torch

from spectramr.infrastructure.training.backward_guard import ensure_backward_ready
from spectramr.infrastructure.training.step_executor import StepExecutor


@pytest.fixture
def model():
    return torch.nn.Linear(4, 4)


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def amp_helper():
    helper = MagicMock()
    helper.device_type = "cpu"
    helper.enabled = False
    helper.scaler = None
    return helper


def _leaf() -> torch.Tensor:
    """The exact shape ``_stack_components`` returns when it has nothing to stack."""
    return torch.tensor(0.0, requires_grad=True)


def _connected(model: torch.nn.Module) -> torch.Tensor:
    return model(torch.ones(2, 4)).sum()


# ---------------------------------------------------------------------------
# The helper in isolation
# ---------------------------------------------------------------------------


class TestEnsureBackwardReady:
    def test_leaf_raises(self):
        with pytest.raises(RuntimeError, match="not connected to the autograd graph"):
            ensure_backward_ready(_leaf(), name="gen", global_step=7)

    def test_message_names_the_config_and_step(self):
        with pytest.raises(RuntimeError, match=r"'gen'.*step 7"):
            ensure_backward_ready(_leaf(), name="gen", global_step=7)

    def test_connected_loss_passes(self, model):
        ensure_backward_ready(_connected(model), name="gen", global_step=0)

    def test_requires_grad_false_is_also_a_leaf(self):
        """A ``requires_grad=False`` total has no ``grad_fn`` either.

        ``backward()`` would raise on it unaided, but the guard reports it
        earlier and with a message that names the cause.
        """
        with pytest.raises(RuntimeError, match="not connected to the autograd graph"):
            ensure_backward_ready(torch.tensor(0.0), name="gen", global_step=0)

    def test_scaling_a_leaf_hides_it(self):
        """Why the guard must run BEFORE ``scaler.scale(loss)``.

        Scaling is a multiply, so it gives the leaf a ``grad_fn`` -- a check
        placed after scaling is structurally blind to every shape above.
        """
        scaled = _leaf() * 65536.0
        assert scaled.grad_fn is not None
        ensure_backward_ready(scaled, name="gen", global_step=0)  # blind, by construction


# ---------------------------------------------------------------------------
# The executor seam
# ---------------------------------------------------------------------------


class TestExecutorRefusesASeveredGraph:
    def _run(self, executor, optimizer, model, closure):
        return executor.execute_step(
            [{"optimizer": optimizer, "closure": closure, "model": model, "name": "gen"}],
            epoch=0,
            global_step=0,
        )

    def test_leaf_loss_raises_at_the_executor(self, amp_helper, model, optimizer):
        policy = MagicMock()
        executor = StepExecutor(amp_helper=amp_helper, amp_policy=policy)
        with pytest.raises(RuntimeError, match="not connected to the autograd graph"):
            self._run(executor, optimizer, model, _leaf)
        assert not policy.backward_and_step.called

    def test_a_no_op_policy_guard_does_not_disable_it(self, amp_helper, model, optimizer):
        """The DeepSpeed shape.

        ``DeepSpeedStepPolicy.guard_loss`` is a full no-op -- legitimately, since
        DeepSpeed owns its own overflow detect and step skip. No backend can make
        a severed graph correct, so this check is never delegated.
        """
        policy = MagicMock()
        policy.guard_loss = MagicMock(return_value=None)  # accepts everything
        executor = StepExecutor(amp_helper=amp_helper, amp_policy=policy)
        with pytest.raises(RuntimeError, match="not connected to the autograd graph"):
            self._run(executor, optimizer, model, _leaf)
        policy.guard_loss.assert_called_once()  # the no-op guard DID run and passed it
        assert not policy.backward_and_step.called

    def test_leaf_raises_under_gradient_accumulation(self, amp_helper, model, optimizer):
        """``loss / N`` is a multiply too -- the guard must precede it."""
        policy = MagicMock()
        executor = StepExecutor(
            amp_helper=amp_helper, amp_policy=policy, gradient_accumulation_steps=2
        )
        with pytest.raises(RuntimeError, match="not connected to the autograd graph"):
            self._run(executor, optimizer, model, _leaf)
        assert not policy.backward_and_step.called

    def test_leaf_raises_with_a_mock_scaler_present(self, amp_helper, model, optimizer):
        amp_helper.scaler = MagicMock()
        policy = MagicMock()
        executor = StepExecutor(amp_helper=amp_helper, amp_policy=policy)
        with pytest.raises(RuntimeError, match="not connected to the autograd graph"):
            self._run(executor, optimizer, model, _leaf)
        assert not policy.backward_and_step.called

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GradScaler needs CUDA")
    def test_leaf_raises_with_a_live_gradscaler(self, amp_helper, model, optimizer):
        """The same claim against a real ``torch.amp.GradScaler``, not a mock."""
        amp_helper.scaler = torch.amp.GradScaler("cuda")
        policy = MagicMock()
        executor = StepExecutor(amp_helper=amp_helper, amp_policy=policy)
        with pytest.raises(RuntimeError, match="not connected to the autograd graph"):
            self._run(executor, optimizer, model, _leaf)
        assert not policy.backward_and_step.called

    def test_connected_loss_still_reaches_backward(self, amp_helper, model, optimizer):
        policy = MagicMock()
        executor = StepExecutor(amp_helper=amp_helper, amp_policy=policy)
        result = self._run(executor, optimizer, model, lambda: _connected(model))
        assert policy.backward_and_step.called
        assert "gen_loss" in result


class TestGuardOrdering:
    """A loss that is both non-finite AND severed reports the finite failure.

    The numerically actionable message is the more useful one, and the finite
    guard is the established owner of it -- so the graph check runs after it.
    """

    def test_non_finite_leaf_reports_non_finite_first(self, amp_helper, model, optimizer):
        from spectramr.infrastructure.training.optimizers import AMPPolicy

        amp_helper.scaler = None
        policy = AMPPolicy(max_grad_norm=1.0, enable_gradient_clipping=False)
        policy.backward_and_step = MagicMock()  # type: ignore[method-assign]
        executor = StepExecutor(amp_helper=amp_helper, amp_policy=policy)
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            executor.execute_step(
                [
                    {
                        "optimizer": optimizer,
                        "closure": lambda: torch.tensor(float("nan"), requires_grad=True),
                        "model": model,
                        "name": "gen",
                    }
                ],
                epoch=0,
                global_step=42,
            )
        assert not policy.backward_and_step.called
