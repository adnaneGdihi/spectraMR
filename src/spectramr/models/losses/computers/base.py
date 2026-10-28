"""Base Loss Computer Module

Unified SSOT pattern for loss computation across all training paradigms.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import torch

from spectramr.models.losses.weights import (
    LossWeightTable,
    build_loss_weight_table,
    resolve_loss_weight,
)

_logger = logging.getLogger(__name__)

#: Default ``device`` for every computer's constructor (a call in the signature
#: would be evaluated once at import anyway; this names the one instance).
DEFAULT_LOSS_DEVICE = torch.device("cpu")


def _all_finite_flags(tensors: list[torch.Tensor]) -> list[bool]:
    """``[bool(torch.isfinite(t).all()) for t in tensors]`` with one host transfer."""
    if not tensors:
        return []
    device = tensors[0].device
    flags = torch.stack([torch.isfinite(t).all().to(device) for t in tensors])
    result: list[bool] = flags.tolist()
    return result


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
        metrics: Non-differentiable metrics (e.g., PSNR, SSIM). A producer on the
            training path stores detached tensors rather than ``float``s, so that
            filling this dict does not force a device sync every step.
    """

    total: torch.Tensor  # Scalar loss for backward
    components: dict[str, torch.Tensor] = field(default_factory=dict)  # Detailed losses
    metrics: dict[str, float | torch.Tensor] = field(default_factory=dict)  # Monitoring metrics

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary for logging.

        Returns:
            Dictionary with total loss and all components.
        """
        result: dict[str, torch.Tensor] = {"total_loss": self.total}
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

    #: Component keys :meth:`compute` stamps on every call beyond the loss terms
    #: themselves. A strategy that forwards ``LossOutput.components`` into its
    #: step dict declares these as CSV columns
    #: (``BaseTrainingStrategy.forwards_loss_components``, read by
    #: ``training_loop.resolve_producer_declared_columns``).
    stamped_component_keys: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: Any, device: torch.device = DEFAULT_LOSS_DEVICE):
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

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Override to() to also update self.device."""
        device = kwargs.get("device")
        if device is None and args and isinstance(args[0], (torch.device, str, int)):
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
            Total weighted loss. When no component survives (none given, or every
            one non-finite) this is a ``requires_grad`` leaf with no ``grad_fn``,
            which the backward guard (``ensure_backward_ready``) refuses.
        """
        present = [(name, val) for name, val in components.items() if val is not None]
        n_candidates = len(present)
        if not n_candidates:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # One device->host transfer for the whole finiteness check, rather than an
        # ``if nan.any() or inf.any()`` sync pair per component on every step.
        tensors = [val for _, val in present if isinstance(val, torch.Tensor)]
        finite_iter = iter(_all_finite_flags(tensors))
        finite = [
            next(finite_iter) if isinstance(val, torch.Tensor) else True for _, val in present
        ]

        # The seed must not require grad: a leaf seed gives the total a ``grad_fn``
        # even when no component is connected to the model (a prediction made under
        # ``no_grad``), which lets that total past the backward guard.
        total = torch.zeros((), device=self.device)
        n_skipped = 0
        applied_weight_sum = 0.0  # Σ|weight| over finite (surviving) components
        for (loss_name, loss_val), is_finite in zip(present, finite, strict=True):
            if not is_finite:
                n_skipped += 1
                _logger.warning(
                    "[LossComputer] Skipping NaN/Inf loss component '%s' "
                    "(value=%s). This may indicate numerical instability "
                    "in the model forward pass.",
                    loss_name,
                    loss_val.item() if loss_val.numel() == 1 else "tensor",
                )
                continue

            if weights and loss_name in weights:
                weight = weights[loss_name]
            else:
                weight = self._get_loss_weight(loss_name, epoch=epoch, iteration=iteration)

            applied_weight_sum += abs(float(weight))
            total = total + weight * loss_val

        # Neither exit raises here. The all-NaN leaf is refused at the backward
        # guard (whether it should raise earlier is #1986); the dead-loss total is
        # grad-connected, and its persistent form is the dead_loss detector's job.
        if n_skipped == n_candidates:
            _logger.error(
                "[LossComputer] SILENT NaN COLLAPSE: all %d loss component(s) "
                "were NaN/Inf — total loss is 0.0 and the model is NOT "
                "training (zero gradient). This is numerical collapse in the "
                "model forward pass, not a benign single-component skip.",
                n_candidates,
            )
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        if applied_weight_sum == 0.0:
            _logger.error(
                "[LossComputer] DEAD LOSS: all %d finite loss component(s) were "
                "weighted to 0.0 (Σ|weight|=0) — total loss is 0.0 and the model "
                "is NOT training (zero gradient). Typically the spatial-loss "
                "warmup gate (warmup_iterations) is holding the only surviving "
                "loss at weight 0 for the whole run; set "
                "losses.reconstruction.warmup_iterations lower or add a "
                "non-gated loss.",
                n_candidates - n_skipped,
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
