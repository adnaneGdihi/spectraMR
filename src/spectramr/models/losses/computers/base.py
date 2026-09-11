"""Base Loss Computer Module

Unified SSOT pattern for loss computation across all training paradigms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from spectramr.models.losses.weights import (
    LossWeightTable,
    build_loss_weight_table,
    resolve_loss_weight,
)


@dataclass
class LossOutput:
    """Standard loss computation output - SSOT for all loss computers.

    Attributes:
        total: Main scalar loss for the backward pass, carried through exactly as
            the computer produced it -- this class does not adjust
            ``requires_grad``. It used to: a ``__post_init__`` "ensured
            gradients" with ``total.detach().requires_grad_(True)``, which does
            not restore a graph but *manufactures a leaf* (``grad_fn is None``).
            ``backward()`` on that leaf succeeds, every ``param.grad`` stays
            ``None``, and the run checkpoints having learned nothing (#1952).
            The graph contract belongs at the backward seam
            (``infrastructure.training.backward_guard.ensure_backward_ready``),
            not here: at validation a grad-free total is *legitimate*, because
            the prediction was produced under ``no_grad``, so asserting it here
            would fire on every validation batch.
        components: Detailed loss components for logging (detached)
        metrics: Non-differentiable metrics (e.g., PSNR, SSIM)
    """

    total: torch.Tensor  # Scalar loss for backward
    components: dict[str, torch.Tensor] = field(default_factory=dict)  # Detailed losses
    metrics: dict[str, float] = field(default_factory=dict)  # Monitoring metrics

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary for logging.

        Returns:
            Dictionary with total loss and all components.
        """
        result = {"total_loss": self.total}
        result.update(self.components)
        return result

    def detach(self) -> "LossOutput":
        """Detach all tensors (for metrics collection).

        Returns:
            New LossOutput with detached tensors.
        """
        return LossOutput(
            total=self.total.detach(),
            components={
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in self.components.items()
            },
            metrics=self.metrics.copy(),
        )


class BaseLossComputer(torch.nn.Module, ABC):
    """Abstract base for all loss computation - SSOT pattern.

    Responsibilities:
    - Compute loss components WITHOUT backward
    - Return losses in unified LossOutput format
    - Handle loss weighting from config
    - Manage complex tensor conversions

    Subclasses MUST implement:
    - _initialize_losses()
    - compute()
    """

    def __init__(self, config: Any, device: torch.device = torch.device("cpu")):
        """Initialize loss computer.

        Args:
            config: Configuration object with loss settings
            device: Compute device (cpu or cuda)
        """
        super().__init__()
        self.config = config
        self.device = device
        # Dynamic per-step loss-term weight overrides. The training strategy
        # assigns this each step from ``loop_state.loss_weight_overrides`` (the
        # LossScheduleController seam) before calling :meth:`compute`. Empty by
        # default => no scheduling => existing static-config behavior.
        self.scheduled_weights: dict[str, float] = {}
        self._initialize_losses()

    def to(self, *args, **kwargs):
        """Override to() to also update self.device."""
        # Use torch's internal parsing if possible, or just look for device in args/kwargs
        device = kwargs.get("device")
        if device is None and args:
            if isinstance(args[0], (torch.device, str, int)):
                device = args[0]

        if device is not None:
            self.device = torch.device(device)

        return super().to(*args, **kwargs)

    @abstractmethod
    def _initialize_losses(self) -> None:
        """Initialize all loss functions based on config.

        This is called in __init__ and should set up:
        - Reconstruction losses (L1, L2, etc.)
        - Adversarial losses (if applicable)
        - Perceptual losses (if enabled)
        - Other specialized losses
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute losses WITHOUT backward.

        This method computes all loss components and returns them in
        standardized LossOutput format. The training strategy is responsible
        for calling backward() after this returns.

        Args:
            pred: Model prediction/generation
            target: Ground truth target
            epoch: Current epoch (for scheduling)
            iteration: Current iteration (for scheduling)
            **kwargs: Additional context (discriminator, etc.)

        Returns:
            LossOutput with total loss and components

        Note:
            - Do NOT call backward() here
            - Do NOT call optimizer.step() here
            - Return requires_grad=True for total loss
            - Detach intermediate tensors for metrics
        """
        raise NotImplementedError

    def _scheduled_override(self, loss_name: str) -> float | None:
        """Return a controller-supplied dynamic weight for ``loss_name``, or None.

        Consulted FIRST and UNCACHED by :meth:`_get_loss_weight` so a scheduled
        enable/disable interacts correctly with the ``if lambda > 0:`` compute
        gates and re-evaluates every step (mirrors the spatial-loss warmup gate).
        Source: :attr:`scheduled_weights`, assigned per step by the strategy from
        ``loop_state.loss_weight_overrides``. Absent => None => static behavior.
        """
        overrides = getattr(self, "scheduled_weights", None)
        if overrides and loss_name in overrides:
            return float(overrides[loss_name])
        return None

    @property
    def _weight_table(self) -> LossWeightTable:
        """This arm's resolved loss weights — built once from the frozen config."""
        table = getattr(self, "_loss_weight_table", None)
        if table is None:
            table = build_loss_weight_table(getattr(self.config, "losses", None))
            self._loss_weight_table = table
        return table

    def _get_loss_weight(
        self, loss_name: str, epoch: int = 0, iteration: int = 0, **kwargs: Any
    ) -> float:
        """The weight for ``loss_name``. Delegates to the loss-weight SSOT.

        Was one of eight resolvers. Besides disagreeing with the other seven, it called
        ``config.losses.model_dump()`` — a full dump of a ~200-field schema — once per
        loss term, per step (``performance.md``). The SSOT resolves every declared weight
        once, at build time; this is now a dict lookup.

        Subclasses may still override to add a *schedule* (the VAE KL annealer, the
        disentangled curriculum). They must not re-add a static-weight lookup.
        """
        return resolve_loss_weight(
            self._weight_table,
            loss_name,
            scheduled=getattr(self, "scheduled_weights", None),
            iteration=iteration,
        )

    def _weighted_loss(
        self,
        loss: torch.Tensor,
        loss_name: str,
        epoch: int = 0,
        iteration: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply weighting to loss.

        Args:
            loss: Loss tensor
            loss_name: Loss name for weighting lookup
            epoch: Current epoch
            iteration: Current iteration
            **kwargs: Additional context

        Returns:
            Weighted loss tensor
        """
        weight = self._get_loss_weight(loss_name, epoch=epoch, iteration=iteration, **kwargs)
        return loss * weight

    def _stack_components(
        self,
        components: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
        epoch: int = 0,
        iteration: int = 0,
    ) -> torch.Tensor:
        """Stack and sum multiple loss components.

        Args:
            components: Dictionary of loss components
            weights: Optional custom weights (overrides config)
            epoch: Current epoch for scheduling
            iteration: Current iteration for scheduling

        Returns:
            Total weighted loss
        """
        import logging as _logging

        _logger = _logging.getLogger(__name__)

        if not components:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total = torch.tensor(0.0, device=self.device, requires_grad=True)

        n_candidates = 0  # real-tensor components (not None)
        n_skipped = 0  # of those, how many were NaN/Inf
        applied_weight_sum = 0.0  # Σ|weight| over finite (surviving) components
        for loss_name, loss_val in components.items():
            if loss_val is None:
                continue
            n_candidates += 1

            # [STABILITY FIX] NaN/Inf guard per component.
            # A single diverged loss (e.g., from an unstable GNN output)
            # must not poison the total and crash backward().
            if isinstance(loss_val, torch.Tensor) and (
                torch.isnan(loss_val).any() or torch.isinf(loss_val).any()
            ):
                n_skipped += 1
                _logger.warning(
                    "[LossComputer] Skipping NaN/Inf loss component '%s' "
                    "(value=%s). This may indicate numerical instability "
                    "in the model forward pass.",
                    loss_name,
                    loss_val.item() if loss_val.numel() == 1 else "tensor",
                )
                continue

            # Get weight from custom dict or from config
            if weights and loss_name in weights:
                weight = weights[loss_name]
            else:
                weight = self._get_loss_weight(loss_name, epoch=epoch, iteration=iteration)

            applied_weight_sum += abs(float(weight))
            total = total + weight * loss_val

        # [SILENT-NAN-COLLAPSE GUARD / 2026-05-21]
        # Per-component skipping is benign when *some* component survives.
        # But if EVERY candidate was NaN/Inf, ``total`` is still exactly 0.0
        # with no graph contribution → ``backward()`` yields zero gradients
        # and the model silently stops learning while ``loss=0.0`` looks
        # perfect and the experiment "PASSES" smoke. This is the CLAUDE.md
        # #10 "passed-with-warnings is the most dangerous outcome" trap
        # (surfaced by experiment_120_capsule_networks: every step
        # ``loss=0.0000 train_psnr=nan``). Escalate to a distinct, grep-able
        # ERROR so smoke-log triage reclassifies it as a failure. We do not
        # raise — a one-off AMP overflow on a single step can recover, so
        # killing training here would be too aggressive; the loud signal +
        # log-parser gate is the correct layer.
        if n_candidates > 0 and n_skipped == n_candidates:
            _logger.error(
                "[LossComputer] SILENT NaN COLLAPSE: all %d loss component(s) "
                "were NaN/Inf — total loss is 0.0 and the model is NOT "
                "training (zero gradient). This is numerical collapse in the "
                "model forward pass, not a benign single-component skip.",
                n_candidates,
            )

        # [DEAD-LOSS GUARD / 2026-06-27] Parallel hole to the NaN collapse: every
        # *finite* surviving component can still be weighted to exactly 0.0, so
        # ``total`` is a zero-gradient leaf and training silently stalls while
        # ``loss=0.0`` looks fine. The NaN guard above misses this because the
        # component is finite (not skipped). Root failure mode: the spatial-loss
        # warmup gate holds l1 at weight 0 while it is the only finite loss (its
        # ms_ssim sibling went NaN on an off-scale operator output and was
        # skipped) — the cs_mno dead_loss cohort, 2026-06-27. Same loud,
        # grep-able ERROR + log-parser gate as the NaN case (we do not raise: a
        # transient single-step all-zero is recoverable; the persistent case is
        # caught by the diagnostics dead_loss detector).
        n_surviving = n_candidates - n_skipped
        if n_surviving > 0 and applied_weight_sum == 0.0:
            _logger.error(
                "[LossComputer] DEAD LOSS: all %d finite loss component(s) were "
                "weighted to 0.0 (Σ|weight|=0) — total loss is 0.0 and the model "
                "is NOT training (zero gradient). Typically the spatial-loss "
                "warmup gate (warmup_iterations) is holding the only surviving "
                "loss at weight 0 for the whole run; set "
                "losses.reconstruction.warmup_iterations lower or add a "
                "non-gated loss.",
                n_surviving,
            )

        return total

    def validate_config(self) -> None:
        """Validate that config has required attributes.

        Override in subclasses to check for required config attributes.
        Raise ValueError if validation fails.
        """
        pass


__all__ = [
    "BaseLossComputer",
    "LossOutput",
]
