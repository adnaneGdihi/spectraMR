import logging
from collections.abc import Callable
from typing import Any

import torch

from spectramr.infrastructure.training.strategy_interfaces import IAMPPolicy

logger = logging.getLogger(__name__)


def _scaler_recorded_inf_checks(scaler: Any, optimizer: Any) -> bool | None:
    """Read GradScaler's private per-optimizer inf-check record, tolerantly.

    ``_per_optimizer_states`` is a private torch attribute; a torch upgrade
    that renames it must NOT crash every AMP step. Returns ``None`` when the
    private API is unavailable — callers then fall back to the standard
    public ``scaler.step`` + ``scaler.update`` pattern.
    """
    try:
        opt_state = scaler._per_optimizer_states.get(id(optimizer), {})
    except AttributeError:
        logger.warning(
            "GradScaler private state (_per_optimizer_states) unavailable on "
            "%s — falling back to the standard scaler.step/update pattern. "
            "The zero-grad-params corner case is no longer special-cased.",
            type(scaler).__name__,
        )
        return None
    return len(opt_state.get("found_inf_per_device", {})) > 0


def _clear_scaler_optimizer_state(scaler: Any, optimizer: Any) -> None:
    """Drop the stuck per-optimizer scaler entry, tolerating its absence."""
    try:
        scaler._per_optimizer_states.pop(id(optimizer), None)
    except AttributeError:
        pass  # already warned in _scaler_recorded_inf_checks


class AMPPolicy(IAMPPolicy):
    """Automatic mixed precision policy."""

    def __init__(
        self,
        force_disable_amp: bool = False,
        max_grad_norm: float = 1.0,
        enable_gradient_clipping: bool = True,
    ):
        """__init__.

        Args:
            force_disable_amp (bool): Description.
            max_grad_norm (float): Description.
            enable_gradient_clipping (bool): Description.
        """
        self.force_disable_amp = force_disable_amp
        self.max_grad_norm = max_grad_norm
        self.enable_gradient_clipping = enable_gradient_clipping

    def should_use_amp(
        self,
        model_type: str,
        device: torch.device,
        config: Any,
    ) -> bool:
        """Whether AMP is on, delegating the decision to the AMP SSOT.

        This used to be a **second AMP resolver** and it disagreed with the live
        one on two counts (#806):

        1. It read ``optimization.precision.enabled`` and never
           ``precision.dtype``, so it could not see the third state
           ``PrecisionConfigSchema`` documents -- ``dtype: 'float32'`` disables
           AMP even when ``enabled`` is true. ``resolve_amp_precision(True,
           "float32")`` returns ``(False, 'fp16')``; this returned ``True``.
        2. It force-disabled AMP for any ``model_type`` whose *string* contained
           "diffusion" -- a hardcoded paradigm branch (pitfall #5) silently
           overriding an explicit ``precision.enabled: true``. An arm that needs
           fp32 declares ``precision.dtype: float32``, where the choice is
           validated and stamped into provenance instead of inferred from a
           model-name substring.

        ``resolve_amp_precision`` is what ``BaseTrainingStrategy`` and
        ``build_deepspeed_config`` already use, so there is now one answer to
        "is AMP on?" (pitfall #13b).

        The old ``import torch.amp`` availability probe is gone with them: this
        module imports ``torch`` at load time and ``torch.amp`` has shipped in
        every version this repo supports, so the branch could not be taken.
        """
        from spectramr.infrastructure.training.mixed_precision import (
            resolve_amp_precision,
        )

        if self.force_disable_amp:
            return False

        # `config is None` keeps the historical "no config means don't object"
        # behaviour; a config object that is not TrainingSettings-shaped still
        # fails loudly rather than reading a stale key.
        enabled = True
        if config is not None:
            if not hasattr(config, "optimization"):
                raise TypeError(
                    "AMPPolicy.should_use_amp received a config object without "
                    "an 'optimization' attribute. Expected TrainingSettings. "
                    f"Got: {type(config).__name__}"
                )
            precision = config.optimization.precision
            enabled, _ = resolve_amp_precision(precision.enabled, precision.dtype)

        return device.type == "cuda" and enabled

    def get_autocast_device(self, device: torch.device) -> str:
        """Get device type for autocast."""
        return "cuda" if device.type == "cuda" else "cpu"

    def clip_gradients(self, model: torch.nn.Module, max_norm: float) -> None:
        """Clip by global norm over the model's own parameters.

        Correct for unsharded models (single-GPU, DP, DDP): every rank holds
        every parameter, so the norm each rank computes is the global norm.

        It is NOT correct under FSDP, where each rank holds a shard and this
        would compute a per-shard norm -- clipping inconsistently across ranks,
        silently, in a way that reads as training instability. See
        ``FSDPStepPolicy``.
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def _apply_gradient_clipping(self, model: torch.nn.Module) -> None:
        """Apply gradient clipping if enabled.

        AUDIT FIX: This was completely missing, causing gradient explosion!
        """
        if not self.enable_gradient_clipping:
            return
        if self.max_grad_norm is None or self.max_grad_norm <= 0:
            return

        # Delegates so a sharded subclass overrides ONE method rather than
        # re-implementing the enable/threshold gating.
        self.clip_gradients(model, self.max_grad_norm)

    def _ensure_gradients_arrived(self, optimizer: torch.optim.Optimizer, model_name: str) -> None:
        """Raise when the first boundary backward gave none of *optimizer*'s parameters a gradient.

        The executor's pre-backward guard reads only ``loss.grad_fn``, so a severed
        loss that went through one more op (``leaf * weight``, ``sum([leaf])``)
        reached backward and stepped nothing (#2309). The graph is fixed per step
        config, so this runs once per name, on host-side ``grad is None`` checks.
        """
        checked: set[str] = self.__dict__.setdefault("_gradients_checked", set())
        if model_name in checked:
            return
        trainable = [p for g in optimizer.param_groups for p in g["params"] if p.requires_grad]
        if trainable and all(p.grad is None for p in trainable):
            raise RuntimeError(
                f"Step config {model_name!r}: backward() ran but none of its optimizer's "
                f"{len(trainable)} trainable parameters received a gradient, so this step "
                "would train nothing. The loss is disconnected from the model -- e.g. a "
                "detached or constant term scaled by a weight, or summed with Python sum()."
            )
        checked.add(model_name)

    def backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        model_name: str,
        epoch: int,
        model_type: str,
        scaler: Any | None = None,
        perform_step: bool = True,
        gradient_clipping_fn: Callable[[torch.nn.Module], float] | None = None,
        skip_scaler_update: bool = False,
    ) -> None:
        """Perform backward pass and optimizer step with AMP support.

        Args:
            loss: Loss tensor to backpropagate
            optimizer: Optimizer to step (can be None for dummy configs)
            model: Model being trained
            model_name: Name of the model for logging
            epoch: Current epoch
            model_type: Type of model (gan, reconstruction, diffusion, etc.)
            scaler: GradScaler for mixed precision (optional)
            perform_step: Whether to perform optimizer step (for gradient accumulation)
            gradient_clipping_fn: Optional callback for custom gradient clipping/logging

        """
        # [FIX] AMP decision is made by the Trainer (amp_helper.enabled).
        # If scaler is provided, AMP is active; otherwise, standard precision.
        use_amp = scaler is not None

        if optimizer is None:
            # Fallback for dummy/test configs without optimizer
            # Just perform backward pass without stepping.
            # NOTE: Do NOT call scaler.update() here — no unscale_() was called,
            # so no inf checks are recorded, causing AssertionError.
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            return

        if use_amp and scaler is not None:
            # Mixed precision backward
            scaler.scale(loss).backward()

            if perform_step:
                self._ensure_gradients_arrived(optimizer, model_name)
                # Unscale gradients to fp32 for clipping and inf/nan checking
                scaler.unscale_(optimizer)

                # Clip gradients (operating on fp32 unscaled grads)
                if gradient_clipping_fn:
                    gradient_clipping_fn(model)
                else:
                    self._apply_gradient_clipping(model)

                # Guard: scaler.step() asserts found_inf_per_device > 0
                # which fails if optimizer has zero parameters with gradients
                # (e.g., zero-loss first step or a model whose parameters all
                # received None grads). The record lives in private torch
                # state; ``None`` means the private API is gone (torch
                # upgrade) — treat as recorded and use the public pattern.
                recorded = _scaler_recorded_inf_checks(scaler, optimizer)
                has_inf_checks = True if recorded is None else recorded

                if has_inf_checks:
                    scaler.step(optimizer)

                # CRITICAL: if has_inf_checks is False, scaler.unscale_() left
                # the optimizer state at OptState.UNSCALED with no inf record.
                # Without resetting that state the next iteration's unscale_()
                # raises "unscale_() has already been called". When stepping is
                # also skipped we must still advance / reset the scaler state
                # — either via scaler.update() (also resets all per-optimizer
                # states) or by clearing this optimizer's entry directly.
                if not skip_scaler_update:
                    if has_inf_checks:
                        scaler.update()
                    else:
                        # Clear stuck per-optimizer state without touching the
                        # global scale (no inf signal was actually observed).
                        _clear_scaler_optimizer_state(scaler, optimizer)
                elif not has_inf_checks:
                    # Multi-optimizer iteration where update() is deferred to
                    # the last config — still need to clear our own stuck state.
                    _clear_scaler_optimizer_state(scaler, optimizer)
                optimizer.zero_grad(set_to_none=True)
        else:
            # Standard precision backward (fp32)
            loss.backward()

            if perform_step:
                self._ensure_gradients_arrived(optimizer, model_name)
                if gradient_clipping_fn:
                    gradient_clipping_fn(model)
                else:
                    self._apply_gradient_clipping(model)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
