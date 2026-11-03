import logging as std_logging
from typing import Any

import torch


class StandardOptimizerStepper:
    """Standard optimizer stepper with gradient scaling and optional clipping."""

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        *,
        enable_gradient_clipping: bool = True,
        gradient_clip_method: str = "norm",
        logger: Any | None = None,
    ):
        """__init__.

        Args:
            max_grad_norm (float): Description.
            enable_gradient_clipping (bool): Description.
            gradient_clip_method (str): Description.
            logger (Optional[Any]): Description.
        """
        # Pitfall #9 — validate the advertised set rather than silently
        # treating any unknown method (typo, or ``norm_type`` from a template)
        # as norm-clipping.
        _valid_clip = {"norm", "value"}
        if gradient_clip_method not in _valid_clip:
            raise ValueError(
                f"Unknown gradient_clip_method {gradient_clip_method!r}. "
                f"Supported: {sorted(_valid_clip)}."
            )
        self.max_grad_norm = max_grad_norm
        self.enable_gradient_clipping = enable_gradient_clipping
        self.gradient_clip_method = gradient_clip_method
        self.logger = logger
        self._clip_warning_emitted = False

    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer | None,
        loss: torch.Tensor,
        grad_scaler: Any,
        model: torch.nn.Module,
        model_name: str,
        epoch: int,
        model_type: str,
        **kwargs,
    ) -> None:
        """Perform optimizer step with gradient scaling.

        Args:
            optimizer: Optimizer to step. If None, does nothing.
            loss: Loss tensor to backpropagate.
            grad_scaler: Optional gradient scaler for AMP.
            model: Model being optimized.
            model_name: Name of model for logging.
            epoch: Current epoch.
            model_type: Type of model for logging.
        """
        if optimizer is None:
            return

        # Backprop and optimizer step
        if grad_scaler is not None:
            # Mixed precision path
            grad_scaler.scale(loss).backward()

            # Unscale for clipping
            grad_scaler.unscale_(optimizer)
            self._apply_gradient_clipping(model, model_name, epoch, model_type)

            # Step and update scaler
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            # Standard precision path
            loss.backward()
            self._apply_gradient_clipping(model, model_name, epoch, model_type)
            optimizer.step()

        # Zero gradients using the interface method
        self.zero_grad(optimizer)

    def zero_grad(self, optimizer: torch.optim.Optimizer | None) -> None:
        """Zero gradients for the provided optimizer in a stable way.

        Args:
            optimizer: Optimizer to zero gradients for. If None, does nothing.
        """
        if optimizer is None:
            return

        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            # Some optimizers may not accept set_to_none
            optimizer.zero_grad()

    def _apply_gradient_clipping(
        self,
        model: torch.nn.Module,
        model_name: str,
        epoch: int,
        model_type: str,
    ) -> None:
        """Apply gradient clipping when enabled and configured."""

        if not self.enable_gradient_clipping:
            self._maybe_warn_disabled_clipping(model_name, epoch, model_type)
            return

        if self.max_grad_norm is None:
            return

        if self.gradient_clip_method == "value":
            torch.nn.utils.clip_grad_value_(model.parameters(), self.max_grad_norm)
        else:
            from spectramr.infrastructure.distributed.grad_norm import clip_grad_norm_global_

            clip_grad_norm_global_(model, self.max_grad_norm)

    def _maybe_warn_disabled_clipping(
        self,
        model_name: str,
        epoch: int,
        model_type: str,
    ) -> None:
        """Emit a single warning when clipping is disabled in config."""

        if self._clip_warning_emitted:
            return

        message = (
            "Gradient clipping disabled via configuration; clip_grad_* calls "
            f"skipped for model '{model_name}' at epoch {epoch}"
        )

        if self.logger is not None:
            try:
                # Prefer the stable API; fall back if signature differs
                try:
                    self.logger.log_warning(message, model_type=model_type, epoch=epoch)
                except TypeError:
                    self.logger.log_warning(message)
            except Exception:
                # If logger fails, emit to standard library logger
                std_logging.getLogger(__name__).warning(message)
        else:
            std_logging.getLogger(__name__).warning(message)

        self._clip_warning_emitted = True
