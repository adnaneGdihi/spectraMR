"""Unit tests for the unified StepExecutor orchestrator.

Tests the StepExecutor class that abstracts PyTorch training loop boilerplate
(zero_grad, autocast, backward, clip, step) away from individual strategies.
"""

import contextlib
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.infrastructure.training.step_executor import (
    OptimizationStepConfig,
    StepExecutor,
)

# ---------------------------------------------------------------------------
# Loss fixtures
# ---------------------------------------------------------------------------


def _connected_loss(value: float) -> torch.Tensor:
    """A scalar loss carrying a real ``grad_fn``.

    Every closure below used to return ``torch.tensor(v, requires_grad=True)``,
    which is a **leaf**: ``requires_grad`` is True but ``grad_fn`` is None and
    there is no path to any parameter. ``backward()`` on one succeeds while
    every gradient stays None -- the silent-no-op failure ``StepExecutor`` now
    refuses to perform (#1952). A leaf fixture would therefore test the guard
    instead of the behaviour each test names.

    Multiplying by a constant attaches a ``MulBackward0`` node while preserving
    the value exactly, including ``nan``/``inf`` -- so the non-finite tests
    still carry the value they are about, now on a realistic loss rather than
    on one no training step could produce.

    The guard's own behaviour, and its ordering against the non-finite check,
    are owned by ``test_backward_guard.py`` and are deliberately not
    re-asserted here (NN17).
    """
    return torch.tensor(value, requires_grad=True) * 1.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def amp_helper():
    """Mock AMP helper with CPU config."""
    helper = MagicMock()
    helper.device_type = "cpu"
    helper.enabled = False
    return helper


@pytest.fixture
def amp_policy():
    """Mock AMP policy."""
    return MagicMock()


@pytest.fixture
def trainer(amp_helper, amp_policy):
    """Standard StepExecutor with no gradient accumulation."""
    return StepExecutor(
        amp_helper=amp_helper,
        amp_policy=amp_policy,
        gradient_accumulation_steps=1,
    )


@pytest.fixture
def simple_model():
    return torch.nn.Linear(4, 4)


@pytest.fixture
def simple_optimizer(simple_model):
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestStepExecutorInit:
    def test_instantiation(self, amp_helper, amp_policy):
        trainer = StepExecutor(amp_helper=amp_helper, amp_policy=amp_policy)
        assert trainer.gradient_accumulation_steps == 1

    def test_gradient_accumulation_clamp(self, amp_helper, amp_policy):
        trainer = StepExecutor(
            amp_helper=amp_helper,
            amp_policy=amp_policy,
            gradient_accumulation_steps=0,
        )
        assert trainer.gradient_accumulation_steps == 1


# ---------------------------------------------------------------------------
# Single-step execution
# ---------------------------------------------------------------------------


class TestSingleStep:
    def test_single_closure_execution(self, trainer, amp_policy, simple_model, simple_optimizer):
        loss_val = _connected_loss(1.0)
        config: OptimizationStepConfig = {
            "optimizer": simple_optimizer,
            "closure": lambda: loss_val,
            "model": simple_model,
            "name": "generator",
        }

        result = trainer.execute_step([config], epoch=0, global_step=0)

        assert "generator_loss" in result
        assert amp_policy.backward_and_step.called
        kw = amp_policy.backward_and_step.call_args.kwargs
        assert kw["model_name"] == "generator"
        assert kw["optimizer"] is simple_optimizer
        assert kw["model"] is simple_model
        assert kw["perform_step"] is True

    def test_dict_config_auto_wrapped(self, trainer, amp_policy, simple_model, simple_optimizer):
        """A single dict (not list) should be auto-wrapped."""
        loss_val = _connected_loss(0.5)
        config = {
            "optimizer": simple_optimizer,
            "closure": lambda: loss_val,
            "model": simple_model,
            "name": "gen",
        }

        result = trainer.execute_step(config, epoch=0, global_step=0)
        assert "gen_loss" in result


# ---------------------------------------------------------------------------
# Multi-step (GAN-style D then G)
# ---------------------------------------------------------------------------


class TestMultiStep:
    def test_discriminator_then_generator(self, trainer, amp_policy):
        d_model = torch.nn.Linear(4, 4)
        g_model = torch.nn.Linear(4, 4)
        d_opt = torch.optim.SGD(d_model.parameters(), lr=0.01)
        g_opt = torch.optim.SGD(g_model.parameters(), lr=0.01)

        d_loss = _connected_loss(0.5)
        g_loss = _connected_loss(0.3)

        configs = [
            {
                "optimizer": d_opt,
                "closure": lambda: d_loss,
                "model": d_model,
                "name": "discriminator",
            },
            {
                "optimizer": g_opt,
                "closure": lambda: g_loss,
                "model": g_model,
                "name": "generator",
            },
        ]

        result = trainer.execute_step(configs, epoch=1, global_step=10)

        assert "discriminator_loss" in result
        assert "generator_loss" in result
        assert amp_policy.backward_and_step.call_count == 2


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    def test_no_step_at_non_boundary(self, amp_helper, amp_policy):
        trainer = StepExecutor(
            amp_helper=amp_helper,
            amp_policy=amp_policy,
            gradient_accumulation_steps=4,
        )
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = _connected_loss(1.0)

        config = {
            "optimizer": opt,
            "closure": lambda: loss,
            "model": model,
            "name": "gen",
        }

        # global_step=2 → (2+1)%4 = 3 ≠ 0 → no step
        trainer.execute_step([config], epoch=0, global_step=2)
        kw = amp_policy.backward_and_step.call_args.kwargs
        assert kw["perform_step"] is False

    def test_step_at_boundary(self, amp_helper, amp_policy):
        trainer = StepExecutor(
            amp_helper=amp_helper,
            amp_policy=amp_policy,
            gradient_accumulation_steps=4,
        )
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = _connected_loss(1.0)

        config = {
            "optimizer": opt,
            "closure": lambda: loss,
            "model": model,
            "name": "gen",
        }

        # global_step=3 → (3+1)%4 = 0 → step!
        trainer.execute_step([config], epoch=0, global_step=3)
        kw = amp_policy.backward_and_step.call_args.kwargs
        assert kw["perform_step"] is True

    def test_loss_scaled_by_accumulation(self, amp_helper, amp_policy):
        trainer = StepExecutor(
            amp_helper=amp_helper,
            amp_policy=amp_policy,
            gradient_accumulation_steps=4,
        )
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        raw_loss = _connected_loss(4.0)

        config = {
            "optimizer": opt,
            "closure": lambda: raw_loss,
            "model": model,
            "name": "gen",
        }

        trainer.execute_step([config], epoch=0, global_step=0)
        kw = amp_policy.backward_and_step.call_args.kwargs
        # Loss should be raw_loss / 4 = 1.0
        assert kw["loss"].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


class TestGradientClipping:
    def test_clip_fn_passed_when_provided(
        self, trainer, amp_policy, simple_model, simple_optimizer
    ):
        clip_fn = MagicMock()
        loss = _connected_loss(1.0)

        config = {
            "optimizer": simple_optimizer,
            "closure": lambda: loss,
            "model": simple_model,
            "name": "gen",
        }

        trainer.execute_step(
            [config],
            epoch=5,
            global_step=10,
            clip_and_log_fn=clip_fn,
        )

        kw = amp_policy.backward_and_step.call_args.kwargs
        assert kw["gradient_clipping_fn"] is not None

    def test_no_clip_fn_when_no_model(self, trainer, amp_policy, simple_optimizer):
        clip_fn = MagicMock()
        loss = _connected_loss(1.0)

        config = {
            "optimizer": simple_optimizer,
            "closure": lambda: loss,
            # no "model" key
            "name": "gen",
        }

        trainer.execute_step(
            [config],
            epoch=0,
            global_step=0,
            clip_and_log_fn=clip_fn,
        )

        kw = amp_policy.backward_and_step.call_args.kwargs
        assert kw["gradient_clipping_fn"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_tuple_loss_unpacked(self, trainer, amp_policy, simple_model, simple_optimizer):
        """Closures returning (loss, aux_outputs) should unpack."""
        loss = _connected_loss(1.0)
        aux = {"some": "data"}

        config = {
            "optimizer": simple_optimizer,
            "closure": lambda: (loss, aux),
            "model": simple_model,
            "name": "gen",
        }

        result = trainer.execute_step([config], epoch=0, global_step=0)
        assert "gen_loss" in result
        assert result["gen_loss"].item() == pytest.approx(1.0)

    def test_default_name_when_missing(self, trainer, amp_policy, simple_model, simple_optimizer):
        loss = _connected_loss(1.0)
        config = {
            "optimizer": simple_optimizer,
            "closure": lambda: loss,
            "model": simple_model,
            # no "name"
        }

        result = trainer.execute_step([config], epoch=0, global_step=0)
        assert "step_0_loss" in result


# ---------------------------------------------------------------------------
# Non-finite loss guard (fp32 divergence protection)
# ---------------------------------------------------------------------------


class TestNonFiniteLossGuard:
    def test_raises_on_non_finite_loss_fp32(
        self, amp_helper, amp_policy, simple_model, simple_optimizer
    ):
        """fp32 (scaler is None): a non-finite loss must raise BEFORE backward,
        never reaching the optimizer — otherwise it silently poisons weights
        (clip_grad_norm_ does not clamp NaN). The exp_11 divergence guard.

        Uses a REAL AMPPolicy rather than the MagicMock fixture. The rule now
        lives on ``IStepPolicy.guard_loss`` (so a backend with its own overflow
        handling can no-op it), and a MagicMock auto-creates ``guard_loss`` as a
        do-nothing method — the mock would assert that the guard is wired while
        proving nothing about whether it fires.
        """
        from spectramr.infrastructure.training.optimizers import AMPPolicy

        amp_helper.scaler = None
        real_policy = AMPPolicy(max_grad_norm=1.0, enable_gradient_clipping=False)
        real_policy.backward_and_step = MagicMock()  # type: ignore[method-assign]
        amp_policy = real_policy
        trainer = StepExecutor(amp_helper=amp_helper, amp_policy=amp_policy)
        config = {
            "optimizer": simple_optimizer,
            "closure": lambda: _connected_loss(float("nan")),
            "model": simple_model,
            "name": "gen",
        }
        with pytest.raises(RuntimeError, match="Non-finite loss"):
            trainer.execute_step([config], epoch=0, global_step=42)
        assert not amp_policy.backward_and_step.called

    def test_defers_to_gradscaler_under_amp(
        self, amp_helper, amp_policy, simple_model, simple_optimizer
    ):
        """With an active GradScaler (scaler not None) a non-finite loss is NOT
        raised — the scaler already skips inf/nan steps, and we avoid the
        per-step sync (`scaler is None` short-circuits before isfinite)."""
        amp_helper.scaler = MagicMock()
        trainer = StepExecutor(amp_helper=amp_helper, amp_policy=amp_policy)
        config = {
            "optimizer": simple_optimizer,
            "closure": lambda: _connected_loss(float("inf")),
            "model": simple_model,
            "name": "gen",
        }
        trainer.execute_step([config], epoch=0, global_step=0)  # no raise
        assert amp_policy.backward_and_step.called

    def test_finite_loss_passes_fp32(self, amp_helper, amp_policy, simple_model, simple_optimizer):
        amp_helper.scaler = None
        trainer = StepExecutor(amp_helper=amp_helper, amp_policy=amp_policy)
        config = {
            "optimizer": simple_optimizer,
            "closure": lambda: _connected_loss(1.0),
            "model": simple_model,
            "name": "gen",
        }
        result = trainer.execute_step([config], epoch=0, global_step=0)
        assert "gen_loss" in result


class TestDDPNoSyncUnderAccumulation:
    """``no_sync`` on the non-boundary micro-batches.

    Numerically a no-op (gradient averaging is linear, so N reductions of
    partial gradients and one reduction of their sum agree), which is exactly
    why its ABSENCE was invisible: the run trained correctly and paid N-1
    unnecessary all-reduces per step, presenting only as poor scaling.
    """

    class _SyncSpy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.entered = 0

        def forward(self, x):
            return self.linear(x)

        @contextlib.contextmanager
        def no_sync(self):
            self.entered += 1
            yield

    def _run(self, amp_helper, amp_policy, steps: int, accumulation: int):
        model = self._SyncSpy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = StepExecutor(
            amp_helper=amp_helper,
            amp_policy=amp_policy,
            gradient_accumulation_steps=accumulation,
        )
        for step in range(steps):
            trainer.execute_step(
                [
                    {
                        "optimizer": optimizer,
                        "closure": lambda: _connected_loss(1.0),
                        "model": model,
                        "name": "gen",
                    }
                ],
                epoch=0,
                global_step=step,
            )
        return model

    def test_suppressed_on_every_micro_batch_but_the_boundary(self, amp_helper, amp_policy):
        model = self._run(amp_helper, amp_policy, steps=4, accumulation=4)
        # steps 0,1,2 accumulate; step 3 is the boundary and must sync.
        assert model.entered == 3

    def test_never_suppressed_without_accumulation(self, amp_helper, amp_policy):
        """Every step is a boundary at accumulation=1, so no_sync must not fire
        -- entering it there would drop the reduction entirely."""
        model = self._run(amp_helper, amp_policy, steps=4, accumulation=1)
        assert model.entered == 0

    def test_a_model_without_no_sync_is_unaffected(
        self, amp_helper, amp_policy, simple_model, simple_optimizer
    ):
        """Single-process models have no such method; the helper must fall
        through to a null context rather than AttributeError."""
        trainer = StepExecutor(
            amp_helper=amp_helper, amp_policy=amp_policy, gradient_accumulation_steps=2
        )
        result = trainer.execute_step(
            [
                {
                    "optimizer": simple_optimizer,
                    "closure": lambda: _connected_loss(1.0),
                    "model": simple_model,
                    "name": "gen",
                }
            ],
            epoch=0,
            global_step=0,
        )
        assert "gen_loss" in result

    def test_helper_is_duck_typed_so_fsdp_is_covered(self):
        """FSDP exposes no_sync with the same meaning; naming DDP in an
        isinstance check would have excluded it."""
        from spectramr.infrastructure.training.step_executor import _suppress_ddp_sync

        class _OnlyNoSync:
            def __init__(self):
                self.entered = 0

            @contextlib.contextmanager
            def no_sync(self):
                self.entered += 1
                yield

        obj = _OnlyNoSync()
        with _suppress_ddp_sync(obj, perform_step=False):
            pass
        assert obj.entered == 1
        with _suppress_ddp_sync(obj, perform_step=True):
            pass
        assert obj.entered == 1, "boundary step must NOT suppress the reduction"


# ---------------------------------------------------------------------------
# adopt_step_policy — the seam that was built and never used
# ---------------------------------------------------------------------------


class TestAdoptStepPolicy:
    """The parallel plugin resolves a step policy, but the strategy (and this
    executor) is constructed BEFORE the parallel runtime exists. Without a
    re-negotiating install, the executor kept the generic ``AMPPolicy`` and a
    DeepSpeed run never reached ``engine.backward()``/``engine.step()``.
    """

    def _executor(self, accumulation: int = 4) -> StepExecutor:
        return StepExecutor(
            amp_helper=MagicMock(),
            amp_policy=MagicMock(owns_gradient_accumulation=False, owns_zero_grad=False),
            gradient_accumulation_steps=accumulation,
        )

    def test_engine_owned_policy_hands_over_accumulation(self) -> None:
        ex = self._executor(accumulation=4)
        assert ex.gradient_accumulation_steps == 4

        ex.adopt_step_policy(MagicMock(owns_gradient_accumulation=True, owns_zero_grad=True))

        # The engine divides the loss and decides boundaries itself; running our
        # copy as well gives 1/N^2 scaling.
        assert ex.gradient_accumulation_steps == 1
        assert ex._policy_owns_accumulation is True
        assert ex._policy_owns_zero_grad is True
        # What the config asked for is retained for provenance.
        assert ex.requested_gradient_accumulation_steps == 4

    def test_plain_policy_leaves_accumulation_with_the_executor(self) -> None:
        ex = self._executor(accumulation=3)
        ex.adopt_step_policy(MagicMock(owns_gradient_accumulation=False, owns_zero_grad=False))
        assert ex.gradient_accumulation_steps == 3
        assert ex._policy_owns_accumulation is False

    def test_none_is_a_no_op(self) -> None:
        ex = self._executor(accumulation=2)
        before = ex.amp_policy
        ex.adopt_step_policy(None)
        assert ex.amp_policy is before
        assert ex.gradient_accumulation_steps == 2

    def test_a_class_is_rejected_rather_than_negotiated(self) -> None:
        """getattr() on a CLASS returns the descriptor, not the value, so a
        class here would silently negotiate every capability flag to False."""
        ex = self._executor()

        class _Policy:
            owns_gradient_accumulation = True
            owns_zero_grad = True

        with pytest.raises(TypeError, match="INSTANCE"):
            ex.adopt_step_policy(_Policy)

    def test_adopted_policy_replaces_the_generic_one(self) -> None:
        ex = self._executor()
        policy = MagicMock(owns_gradient_accumulation=False, owns_zero_grad=False)
        ex.adopt_step_policy(policy)
        assert ex.amp_policy is policy
