"""Optimizer-step executor for the training loop.

The low-level engine that runs one training iteration's optimizer step(s) for a
strategy: AMP autocast, gradient accumulation, gradient clipping, and
``backward_and_step``. Strategies return step configurations (which optimizer
steps with which loss closure) and this class executes them uniformly, so no
strategy hand-rolls ``loss.backward()`` / ``optimizer.step()``.

Renamed from ``Trainer`` on 2026-06-18: the name overclaimed — this object runs
a training *step*, it is not the user-facing trainer — and it collided with the
public scripting ``Trainer`` orchestrator. The legacy ``spectramr.pipelines.trainer``
re-export shim (a 2026-05-14 layer-move leftover) was removed in the same change;
see ``docs/deprecation_journal.rst``. Import directly from
``spectramr.infrastructure.training.step_executor``.

Lives in ``infrastructure/training/`` (next to its dependencies) so strategies
construct it without a reverse layer-direction import into ``pipelines/``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

import torch

from spectramr.infrastructure.training.backward_guard import ensure_backward_ready
from spectramr.infrastructure.training.memory_context_managers import (
    gpu_memory_monitor,
    memory_cleanup_context,
)

logger = logging.getLogger(__name__)


def _suppress_ddp_sync(model: Any, perform_step: bool) -> Any:
    """``model.no_sync()`` on accumulation micro-batches, else a null context.

    DDP all-reduces gradients inside every ``backward()``. Under gradient
    accumulation only the boundary micro-batch's reduction is needed: the
    result is identical (gradient averaging is linear, so N reductions of
    partial gradients and one reduction of their sum agree), and the other
    N-1 are pure network cost.

    Duck-typed on ``no_sync`` rather than an ``isinstance`` check against
    ``DistributedDataParallel``, so FSDP -- which exposes the same method with
    the same meaning -- is covered without naming it, and any model without
    the method (the single-process case) falls through to the null context.
    """
    if perform_step:
        return contextlib.nullcontext()
    no_sync = getattr(model, "no_sync", None)
    if no_sync is None:
        return contextlib.nullcontext()
    return no_sync()


# Defining what a step configuration looks like.
OptimizationStepConfig = dict[str, Any]
# Expected keys:
#   "optimizer": torch.optim.Optimizer
#   "closure": Callable[[], torch.Tensor]  (must return loss tensor)
#   "model": nn.Module
#   "name": str (optional, defaults to "model")


class StepExecutor:
    """Executes a strategy's optimizer step(s) for one training iteration.

    Instead of strategies managing ``loss.backward()`` and
    ``optimizer.step()``, strategies yield/return configurations
    detailing which optimizer should step using which loss closure.
    This executor runs them consistently with AMP, gradient
    scaling, clipping, and anomaly detection.
    """

    def __init__(
        self,
        amp_helper: Any,
        amp_policy: Any,
        gradient_accumulation_steps: int = 1,
        enable_memory_monitoring: bool = False,
    ) -> None:
        """Initialize the unified trainer.

        Args:
            amp_helper: MixedPrecisionIntegrationHelper.
            amp_policy: AMPPolicy for backward and step.
            gradient_accumulation_steps: Steps to accumulate grads.
            enable_memory_monitoring: Flag to enable rigorous memory leak tracking.
        """
        self.amp_helper = amp_helper
        self.amp_policy = amp_policy
        self.enable_memory_monitoring = enable_memory_monitoring

        # Capability negotiation, read once. Polymorphism on an injected object,
        # NOT a backend name test -- an engine-owned backend (DeepSpeed) divides
        # the loss and decides accumulation boundaries inside its own
        # backward/step, so running our copy as well yields 1/N^2 scaling and one
        # real optimizer step per N^2 micro-batches. Both defaults are False, so a
        # policy that declares neither behaves exactly as before.
        # ``is True``, not ``bool(...)``: these flags hand ownership of loss
        # scaling and zero_grad to the policy, so only a literal True may do it.
        # ``bool()`` accepts any truthy accident -- a MagicMock attribute in a
        # test, a stray object -- and would silently switch the executor into
        # engine-owned mode, which still looks like training and is off by a
        # factor of N in the accumulation.
        self.requested_gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self._negotiate_capabilities()

    def _negotiate_capabilities(self) -> None:
        """(Re-)read the capability flags off ``self.amp_policy``.

        Split out of ``__init__`` so ``adopt_step_policy`` can re-run it: the
        parallel strategy is resolved AFTER the strategy (and therefore this
        executor) is constructed, so the flags have to be re-derived when the
        real policy finally arrives.
        """
        self._policy_owns_accumulation = (
            getattr(self.amp_policy, "owns_gradient_accumulation", False) is True
        )
        self._policy_owns_zero_grad = getattr(self.amp_policy, "owns_zero_grad", False) is True
        self.gradient_accumulation_steps = (
            1 if self._policy_owns_accumulation else self.requested_gradient_accumulation_steps
        )

    def adopt_step_policy(self, policy: Any) -> None:
        """Install the parallel strategy's step policy and re-negotiate.

        Building the policy is only half the wiring; without this call the
        executor keeps the generic ``AMPPolicy`` and a DeepSpeed run never
        reaches ``engine.backward()``/``engine.step()`` -- it trains, every test
        passes, and only the numbers are wrong (1/N^2 loss scaling). An FSDP run
        keeps the per-shard gradient clip that ``FSDPStepPolicy`` exists to fix.

        Raises:
            TypeError: if *policy* is a class rather than an instance. The
                registry may hand back either; resolving that ambiguity belongs
                to the caller, which knows the clip settings a class needs.
        """
        if policy is None:
            return
        if isinstance(policy, type):
            raise TypeError(
                "adopt_step_policy expects an INSTANCE, got the class "
                f"{policy.__name__!r}. Construct it first -- a class here would "
                "make every getattr() for a capability flag read the descriptor "
                "instead of the value, and silently negotiate to False."
            )
        self.amp_policy = policy
        self._negotiate_capabilities()

    def _guard_loss(self, loss: Any, *, name: str, global_step: int) -> None:
        """Run the policy's pre-backward loss guard, or the default one.

        The rule (fail fast without a GradScaler, defer to the scaler with one)
        lives on ``IStepPolicy.guard_loss`` so a backend with its own overflow
        handling can no-op it. But the executor keeps a fallback for a policy
        object that predates the ABC: dropping the guard because an injected
        object lacks a method would silently reinstate the exact failure it
        exists to prevent -- non-finite loss -> non-finite grads that
        clip_grad_norm_ does NOT clamp -> optimizer.step() poisons the weights.
        """
        scaler = self.amp_helper.scaler
        guard = getattr(self.amp_policy, "guard_loss", None)
        if callable(guard):
            guard(loss, name=name, global_step=global_step, scaler=scaler)
        elif scaler is None and not torch.isfinite(loss).all():
            raise RuntimeError(
                f"Non-finite loss for config {name!r} at step {global_step} "
                "before backward; refusing to poison weights (check coil maps / "
                "fidelity terms)."
            )

        # Second invariant, and deliberately NOT delegated to the policy: the
        # loss must be connected to the autograd graph. DeepSpeed legitimately
        # no-ops the finite check above (it owns its own overflow detect + step
        # skip), but no backend can make a severed graph correct, so routing
        # this through ``guard_loss`` would make it silently absent on exactly
        # one backend -- the failure shape it exists to prevent. Runs last so a
        # loss that is both non-finite and severed still reports the
        # numerically actionable failure first. Runs here, before the
        # accumulation divide and before ``scaler.scale()``, because both are
        # multiplies that give a leaf a ``grad_fn`` and hide it (#1952).
        ensure_backward_ready(loss, name=name, global_step=global_step)

    def execute_step(
        self,
        step_configs: OptimizationStepConfig | list[OptimizationStepConfig],
        epoch: int,
        global_step: int,
        model_type: str = "unknown",
        clip_and_log_fn: Callable | None = None,
    ) -> dict[str, torch.Tensor]:
        """Execute one full training iteration.

        May comprise multiple optimizer steps (e.g. D then G).

        Args:
            step_configs: Config(s) from the strategy.
            epoch: Current training epoch.
            global_step: Current global training step.
            model_type: Model type string for logging.
            clip_and_log_fn: Optional gradient clip+log fn.

        Returns:
            Dict mapping step names to their loss tensors.
        """
        if isinstance(step_configs, dict):
            step_configs = [step_configs]

        losses_record: dict[str, torch.Tensor] = {}

        for idx, config in enumerate(step_configs):
            optimizer = config["optimizer"]
            closure = config["closure"]
            model = config.get("model")
            name = config.get("name", f"step_{idx}")
            is_last_config = idx == len(step_configs) - 1

            # F-OPT-NONE-LOUD / 2026-05-20 — a None optimizer here used
            # to crash with the cryptic ``AttributeError: 'NoneType'
            # object has no attribute 'zero_grad'`` at the very first
            # iteration (smoke 20260516 11:48 bloch_cycle_network).
            # That message gave no clue WHICH optimizer was missing
            # or WHERE the wiring broke. Surface the gap with a clear
            # message naming the step + the likely YAML/strategy fix.
            if optimizer is None:
                raise RuntimeError(
                    f"[StepExecutor.execute_step] step_configs[{idx}] "
                    f"(name={name!r}) has ``optimizer=None``. This "
                    f"typically means the strategy's ``build_optimizers`` "
                    f"failed to construct one for this sub-model "
                    f"(e.g., missing ``losses.gan`` block for a GAN "
                    f"discriminator, or ``optimization.optimizer.type`` "
                    f"set to an unknown value). Fix the underlying "
                    f"builder or YAML rather than letting training "
                    f"proceed with a phantom optimizer."
                )

            # 1. Zero gradients only at the start of an accumulation cycle --
            # unless the policy zeroes them inside its own step, in which case
            # doing it here as well discards the accumulation the engine is
            # mid-way through building.
            if not self._policy_owns_zero_grad and (
                global_step % self.gradient_accumulation_steps == 0
            ):
                optimizer.zero_grad(set_to_none=True)

            # 2. Forward pass & loss inside AMP context.
            with contextlib.ExitStack() as stack:
                if self.enable_memory_monitoring:
                    stack.enter_context(
                        memory_cleanup_context(description=f"trainer_step_{name}", log_memory=False)
                    )
                    stack.enter_context(gpu_memory_monitor(name=f"trainer_step_{name}"))

                # Route through the AMP helper's policy instead of a hand-rolled
                # fp16 autocast: for complex/k-space flows it returns a disabled
                # context (fp16 is incompatible with complex tensors) and bf16
                # gets the TF32-safe path. The previous unconditional fp16
                # autocast defeated that guard for every complex-domain arm.
                stack.enter_context(self.amp_helper.get_autocast_context())

                loss = closure()

                # Handle tuple returns
                if isinstance(loss, tuple):
                    loss = loss[0]

            # Record loss for metrics before scaling
            if loss is not None:
                # Pre-backward finite guard. The rule (fail fast without a
                # GradScaler, defer to the scaler with one) now lives on the
                # policy as ``guard_loss``, so a backend with its own overflow
                # handling can no-op it instead of having a raise it cannot
                # reach hardcoded into the executor.
                self._guard_loss(loss, name=name, global_step=global_step)

                losses_record[f"{name}_loss"] = loss.detach()

                # Handle Gradient Accumulation scaling
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                perform_step = (global_step + 1) % self.gradient_accumulation_steps == 0

                # 3. Backward & Step via AMP policy
                clip_fn = self._make_clip_fn(clip_and_log_fn, model, epoch, global_step, name)

                # Defer scaler.update() until all configs are done to avoid
                # 'unscale_() already called' when multiple configs share
                # the same GradScaler.
                #
                # `no_sync` on the non-boundary micro-batches suppresses DDP's
                # per-micro-batch gradient all-reduce. Numerically identical
                # (averaging is linear, so reducing N times and reducing once
                # over the accumulated buffer agree), but it removes N-1 of
                # every N all-reduces -- the dominant cost of a
                # gradient_accumulation_steps > 1 DDP run.
                with _suppress_ddp_sync(model, perform_step):
                    self.amp_policy.backward_and_step(
                        loss=loss,
                        optimizer=optimizer,
                        model=model,
                        model_name=name,
                        epoch=epoch,
                        model_type=model_type,
                        scaler=self.amp_helper.scaler,
                        perform_step=perform_step,
                        gradient_clipping_fn=clip_fn,
                        skip_scaler_update=not is_last_config,
                    )

        return losses_record

    @staticmethod
    def _make_clip_fn(
        clip_and_log_fn: Callable | None,
        model: Any,
        epoch: int,
        global_step: int,
        name: str,
    ) -> Callable | None:
        """Build a gradient-clipping closure (avoids B023)."""
        if clip_and_log_fn is not None and model is not None:

            def _clip(m: Any) -> None:
                clip_and_log_fn(m, epoch, global_step, model_name=name)

            return _clip
        return None


__all__ = ["OptimizationStepConfig", "StepExecutor"]
