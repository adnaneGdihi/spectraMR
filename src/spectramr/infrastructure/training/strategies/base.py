"""Base Training Strategies Module

This module contains the base training strategies and foundational classes,
including TrainingStepStrategy and BaseTrainingStrategy implementing the Template Method pattern.
"""

from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, final

import torch
import torch.nn as nn
from torch.cuda import OutOfMemoryError as CudaOutOfMemoryError

# Type checking imports (avoid circular dependencies)
if TYPE_CHECKING:
    from spectramr.domain.interfaces.service_interfaces import ILoggingService
    from spectramr.infrastructure.training.debug_snapshot import ChannelSegment


# Phase 2: Use frozen TrainingEnvironment from builders (SSOT)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.loop_state import LoopState
from spectramr.infrastructure.training.metrics.metrics_reporter import MetricsReporter
from spectramr.infrastructure.training.mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionIntegrationHelper,
    resolve_amp_precision,
)
from spectramr.infrastructure.training.optimizers.amp_policy import AMPPolicy
from spectramr.infrastructure.training.strategies.loss_folding import scheduled_overrides
from spectramr.infrastructure.training.strategies.mixins.batch_preparation import (
    BatchPreparationMixin,
)
from spectramr.infrastructure.training.strategies.mixins.conditioning_mixin import (
    ConditioningMixin,
)
from spectramr.infrastructure.training.strategies.mixins.ema import EMAMixin
from spectramr.infrastructure.training.strategies.mixins.kspace import KspaceMixin
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.strategies.mixins.optimizer import OptimizerMixin
from spectramr.infrastructure.training.strategies.mixins.utils import (  # noqa: F401  # noqa: F401
    _CALLABLE_KWARG_CACHE,
    _callable_accepts_kwarg,
    _get_config_value,
    pick_present,
)
from spectramr.infrastructure.training.strategies.mixins.validation import ValidationMixin
from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator
from spectramr.infrastructure.training.utils.transform_ops import (
    ComplexTensorHandler,
    FFTTransformer,
)
from spectramr.models.capabilities import StrategyCapabilities
from spectramr.models.losses.reporting import active_loss_names, format_loss_objective
from spectramr.models.losses.weights import (
    LossWeightTable,
    build_loss_weight_table,
    canonical_loss_name,
    resolve_loss_weight,
)

logger = logging.getLogger(__name__)


@dataclass
class LossResult:
    """Typed container for loss computation results."""

    losses: dict[str, torch.Tensor]
    metrics: dict[str, float] | None = None
    g_total_loss: torch.Tensor | None = None

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Return losses dict for backward pass."""
        return self.losses

    def get_scalar(self, key: str, default: float = 0.0) -> float:
        """Safely get scalar value from metrics."""
        if self.metrics is None:
            return default
        return self.metrics.get(key, default)


# Tuple of exceptions treated as expected training-time failures
HANDLED_TRAINING_ERRORS = (RuntimeError, ValueError, CudaOutOfMemoryError)


class TrainingStepStrategy(ABC, MetricsMixin, BatchPreparationMixin, ValidationMixin):
    """Abstract base class for different training task strategies.

    This class provides foundational functionality for all training strategies,
    including batch preparation, validation, and perceptual loss computation.
    It follows the Template Method design pattern where subclasses implement
    strategy-specific loss computation logic.

    Attributes:
        logging_service: Service for structured logging operations.
        env: Training environment containing configuration and services.
        state: Training state object (legacy, prefer env).
        device: PyTorch device for tensor operations (CPU or CUDA).

    Note:
        This class should not be instantiated directly. Instead, use concrete
        strategy implementations like GANTrainingStrategy, DiffusionTrainingStrategy, etc.

    Example:
        >>> from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy
        >>> strategy = GANTrainingStrategy(env=training_env)
        >>> losses = strategy.train_step(batch, epoch=0)
    """

    logging_service: Any  # Typed as Any to avoid circular imports, but effectively ILoggingService

    def __init__(
        self,
        env: TrainingEnvironment,
        device: torch.device | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize training step strategy.

        Args:
            env: Training environment containing all components (REQUIRED).
                Built by TrainingEnvironmentDirector.
            **kwargs: Additional keyword arguments (for logging_service injection).

        Raises:
            TypeError: If env is None or not a TrainingEnvironment.

        Note:
            Phase 2 Migration: This method now requires TrainingEnvironment.
            The legacy state= parameter has been removed. All strategies must
            use the frozen TrainingEnvironment from builders.environment.
        """
        if env is None:
            raise TypeError(
                "env is required. Use TrainingEnvironmentDirector to build environment. "
                "Legacy state= parameter removed in Phase 2 migration. "
                "See docs/MIGRATION_PHASE2.md for migration guide."
            )

        # Duck typing check: Accept any object with required attributes (test-friendly)
        required_attrs = ["device", "config", "models"]
        missing = [attr for attr in required_attrs if not hasattr(env, attr)]
        if missing:
            raise TypeError(
                f"env must have attributes: {', '.join(required_attrs)}. "
                f"Missing: {', '.join(missing)}. "
                "Use TrainingEnvironmentDirector to build environment."
            )

        self.env = env
        self.state = env
        self.device = env.device
        self._telemetry_logged = False
        # WS-3 PR-3: the mutable live-iteration seam. The training loop assigns
        # this once and advances ``loop_state.iteration`` each step; strategies
        # read it instead of the frozen, perpetually-zero ``env.step``. Defaults
        # to iteration 0 so direct construction (tests / scripting) is safe.
        self.loop_state = LoopState()

    @property
    def generator_model(self) -> nn.Module:
        """Access the generator model (nn.Module) from environment.

        Returns:
            nn.Module: The generator neural network model.

        Raises:
            AttributeError: If generator cannot be resolved from environment.

        Note:
            Phase 2: Simplified to use env.generator only (no fallback).
            TrainingEnvironment.generator is a property that returns models["generator"].
        """
        return self.env.generator

    @property
    def discriminator_model(self) -> nn.Module | None:
        """Access the discriminator model (nn.Module) from environment.

        Returns:
            Optional[nn.Module]: The discriminator model, or None if not used.

        Note:
            Phase 2: Simplified to use env.discriminator only (no fallback).
            TrainingEnvironment.discriminator returns models.get("discriminator").
        """
        return self.env.discriminator

    # ------------------------------------------------------------------
    # Shared error-handling helpers
    # ------------------------------------------------------------------
    # NOTE: Error handling methods (_attempt_cuda_recovery, _handle_train_step_failure,
    # _verify_strategy_config, _log_config_features) are now provided by ValidationMixin


# Mixins

logger = logging.getLogger(__name__)


def _first_tensor(output: Any) -> torch.Tensor | None:
    """Extract the primary tensor from a model output and detach it.

    Generators return a bare tensor, a tuple ``(recon, aux...)``, or a dict
    (e.g. ``{"reconstruction": ...}`` from the disentangled / multi-head
    models). The model-output debug snapshot only needs the principal
    reconstruction, detached so it never pins the autograd graph across steps.
    Returns ``None`` when no tensor is present (the caller then skips).
    """
    if isinstance(output, torch.Tensor):
        return output.detach()
    if isinstance(output, (tuple, list)):
        for o in output:
            if isinstance(o, torch.Tensor):
                return o.detach()
        return None
    if isinstance(output, dict):
        for key in ("reconstruction", "output", "prediction"):
            v = output.get(key)
            if isinstance(v, torch.Tensor):
                return v.detach()
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v.detach()
    return None


class BaseTrainingStrategy(
    KspaceMixin, EMAMixin, OptimizerMixin, ConditioningMixin, TrainingStepStrategy
):
    """Base training strategy implementing the Template Method pattern.

    This class orchestrates the complete training workflow for MRI reconstruction
    models. It provides:
    - Automatic Mixed Precision (AMP) integration
    - Gradient clipping and anomaly detection
    - K-space operations and physics-based constraints
    - Exponential Moving Average (EMA) for model weights
    - Configurable optimizer stepping strategies

    The Template Method pattern is realized by `train_step()`, which orchestrates
    the fixed execution flow below. (Note: `train_step()` is NOT yet decorated
    `@final` — that enforcement is pending the `@accepts_step_io` signature
    migration tracked in ``tests/architecture/baselines/step_signature_drift.txt``;
    subclasses currently override it.) The flow:
    1. Batch preparation and device transfer
    2. Model forward pass with AMP
    3. Loss computation (strategy-specific via `_compute_losses_impl`)
    4. Backward pass with gradient clipping
    5. Optimizer stepping with anomaly detection

    Attributes:
        config: Immutable Pydantic configuration (Single Source of Truth).
        logging_service: Structured logging interface.
        amp_policy: Automatic Mixed Precision configuration.
        amp_helper: Helper for AMP context management.
        metrics_reporter: Metrics aggregation and logging.
        optimizer_stepper: Optimizer step orchestration.
        trainer: Unified orchestrator for executing optimization closures.
        fft: FFT/IFFT operations for k-space transforms.
        mask_generator: K-space undersampling mask generation.
        complex_handler: Complex-valued tensor utilities.
        _global_step: Training step counter.

    Raises:
        ValueError: If configuration is missing required sections.
        AttributeError: If generator model cannot be resolved from state.

    Example:
        >>> from spectramr.infrastructure.training.strategies.reconstruction import ReconstructionTrainingStrategy
        >>> strategy = ReconstructionTrainingStrategy(env=training_env)
        >>> losses_record = trainer.execute_step(strategy.train_step(batch, epoch=0, step=100))
    """

    #: Declarative capability contract: the imaging-regime × task tags
    #: (``workflows`` / ``tasks``), ``supported_paradigms``, and the dotted
    #: ``required_config_fields`` that must resolve to non-None at audit time.
    #: The default is the empty contract — "unannotated, skip the check"
    #: (mirrors the ``ModelCapabilities`` all-None convention); a strategy opts
    #: in by overriding this ClassVar. Read by the maturity ledger
    #: (``workflows``, via ``workflow_ledger.strategies_tagged``) and by
    #: :meth:`TrainingStrategyFactory.get_strategy_capabilities` (config
    #: fields). See :mod:`spectramr.models.capabilities`.
    #:
    #: DECLARE THIS EXACTLY ONCE. Two workstreams each added a copy of this
    #: ClassVar and the second silently shadowed the first — harmless only
    #: because both defaults were equal. Pinned by an ast-level test, since a
    #: duplicate declaration has no runtime signature to assert against.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities()

    #: True when the strategy builds its OWN undersampling masks from the
    #: ``undersampling:`` block (cold-diffusion schedules, the k-space mixin's
    #: batch masks, the VF digital twin). The audit's
    #: ``undersampling_block_is_applied`` witness reads it: an arm whose block
    #: reaches neither the loader (``data.trajectory`` /
    #: ``data.image_undersampling``) nor a strategy that says so here is
    #: advertising an acceleration that nothing applies (cohort review
    #: 2026-09-02, T0.6: 106 image-domain arms).
    applies_undersampling: ClassVar[bool] = False

    #: Canonical image-loss names this strategy computes INSIDE its own
    #: ``_compute_losses_impl`` (``l1`` for an inline fidelity term; a registered
    #: name such as ``cocycle_consistency`` when the strategy owns that term).
    #: ``None`` = ownership not declared: the audit's
    #: ``image_losses_reach_the_objective`` witness reports such a strategy as
    #: UNVERIFIED and errors only on declared ones, and ``LossBuilder.validate()``
    #: refuses its empty loss stack -- silence is not a claim of ownership. Declared
    #: here alongside ``folds_image_losses = False``, an empty stack is the design
    #: instead and the build proceeds (``loss_folding.declares_inline_objective``,
    #: the one owner of that pairing). The fold
    #: (``loss_folding.fold_builder_image_losses``) skips exactly this set, so a
    #: declared inline term is never counted twice (mrixfields review 2026-09-03).
    inline_losses: ClassVar[frozenset[str] | None] = None

    #: True when every other ``losses.image_losses`` entry reaches the objective.
    #: There are FIVE routes, all enumerated in ``test_loss_ownership.ROUTE_MARKERS``
    #: (the one owner of that vocabulary): the parent's builder path
    #: (``super()._compute_losses_impl``), the strategy's own
    #: ``_apply_builder_image_losses`` call, the shared ``fold_builder_image_losses``
    #: helper, forwarding ``env.losses`` as ``losses_dict`` to a unified loss
    #: computer, and iterating ``env_losses.items()`` to accumulate
    #: ``weight * loss``. A strategy that overrides ``_compute_losses_impl`` with
    #: none of them declares False, and the witness then rejects any declared entry
    #: outside ``inline_losses`` as a decoy.
    #:
    #: The fourth route is a HAND-OFF, not a fold: it only reaches the objective if
    #: the receiving computer reads the dict. ``UnifiedMAELossComputer`` and
    #: ``UnifiedDisentangledLossComputer`` accept ``losses_dict`` and discard it, so
    #: a strategy handing off to either declares False (issue #1918).
    #:
    #: ``test_loss_ownership`` pins a True value against the source for every one of
    #: the 153 strategy classes. The pin is ONE-DIRECTIONAL by design: declaring
    #: False costs you nothing but honesty -- the witness then reports every
    #: non-inline declared entry through ``unreachable_image_losses`` -- while
    #: declaring True buys a silent PASS. Only the silent direction is gated, which
    #: also lets a strategy whose fold is FILTERED (``SSDUReconstructionStrategy``
    #: runs only names containing ``ssdu``) declare False truthfully even though it
    #: matches a route marker textually.
    folds_image_losses: ClassVar[bool | None] = None

    logging_service: ILoggingService

    #: Set True by strategies whose generator emits MORE output channels than the
    #: target by design — distributional / parametric heads (heteroscedastic
    #: ``[mean, logvar]``, evidential, variational priors) where the strategy
    #: self-computes the likelihood from the extra channels. The strict
    #: out_channels-vs-target width guard in :meth:`train_step` would otherwise
    #: reject these (e.g. the mrixfields ``b29_heteroscedastic_ulf`` crash:
    #: "Model outputs 2 channels, but target dataset provided 1"). The Open-Closed
    #: replacement for the hard-coded ``evidential_unet`` model-type escape.
    predicts_distribution_params: ClassVar[bool] = False

    def __init__(
        self,
        env: TrainingEnvironment,
        device: torch.device | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the base training strategy.

        Args:
            env: The training environment containing configuration and services (REQUIRED).
            **kwargs: Additional keyword arguments.
                logging_service: Optional logging service injection.

        Raises:
            TypeError: If env is None.
            ValueError: If configuration is missing required sections.

        Note:
            Phase 2 Migration: Requires TrainingEnvironment (frozen, immutable).
            Legacy device= and state= parameters removed.
        """
        # Abstract base — concrete strategies must subclass and implement
        # ``_compute_losses_impl``. We can't mark that method ``@abstractmethod``
        # (≈14 concrete strategies legitimately override ``_compute_losses`` /
        # ``train_step`` instead and would be falsely blocked), so guard direct
        # instantiation here: fail loudly at construction rather than defer to a
        # call-time NotImplementedError (CLAUDE.md #9 — fail at the earliest point).
        if type(self) is BaseTrainingStrategy:
            raise TypeError(
                "BaseTrainingStrategy is abstract; instantiate a concrete "
                "subclass that implements _compute_losses_impl."
            )

        super().__init__(env=env, **kwargs)

        # SSOT: config from environment
        self.config = self.env.config

        # Model-output debug-snapshot capture (parity with the pre-forward
        # ``first_steps`` auto-snapshot). A lazily-registered forward hook on
        # the generator stashes its latest output so ``_compute_losses`` can
        # auto-emit a ``model_output`` snapshot for EVERY paradigm — closing
        # the 2026-06-21 forensics "model output missing" gap for the field /
        # VF / recon / GAN / VAE arms that previously emitted inputs only.
        self._last_generator_output: torch.Tensor | None = None
        self._capture_gen_output: bool = False
        # Keyed by ``id(module)`` -> ``(module, handle)``, NOT a single handle:
        # a strategy may own a generator that is not ``env.generator`` (cut /
        # cyclegan / stargan_v2 all build their own when the env-built primary
        # is an incompatible type), and one global handle meant whichever
        # module armed first won forever. The module is stored beside its
        # handle so the ``id()`` key cannot be recycled by a later allocation
        # at the same address.
        self._gen_output_hook_handles: dict[int, tuple[Any, Any]] = {}
        self._model_output_snapshot_done: bool = False

        # Model-INPUT declaration for strategies that degrade the prepared input
        # further inside the step (``snapshot_prepared_is_model_input = False``).
        # The wrapper cannot capture this itself: the tensor the model receives
        # is built by each ``_compute_losses_impl`` and never leaves it -- flow
        # matching's interpolant, EDM's preconditioned ``c_in * noised`` and
        # ambient's corruption are different tensors from different math. So the
        # impl hands it over and the wrapper emits it. See
        # ``_snapshot_declared_model_input`` for the enforcement.
        self._declared_model_input: (
            tuple[dict[str, Any], dict[str, Any] | None, set[str] | None] | None
        ) = None
        # Channel-domain decomposition for the declared tensors, kept beside the
        # triple above rather than inside it (sibling strategies' tests unpack
        # that triple positionally).
        self._declared_channel_segments: Mapping[str, Sequence[ChannelSegment]] | None = None
        # Which of the declared tensors are ACTUALLY ``log1p``-compressed. Same
        # sidecar treatment, same reason. ``None`` keeps the historical
        # "every declared-k-space tensor is decompressed" default.
        self._declared_log_scaled_keys: set[str] | None = None
        self._model_input_snapshot_done: bool = False
        # Backbone input width, resolved lazily and cached. The resolution walks
        # ``modules()``, and the wrapper that needs it runs on EVERY step (the
        # write gates live downstream, in the writer), so re-walking would put
        # an O(len(modules)) loop in the training step for a diagnostic --
        # non-negotiable 9. ``None`` here means "not looked up yet"; a genuine
        # "could not be read" caches as ``(None, "unresolved")``, so a backbone
        # whose width is unreadable is walked once, not once per step.
        self._model_input_width: tuple[int | None, str] | None = None
        self._model_input_contract_warned: bool = False

        # Initialize logging service (DI or from kwargs)
        if "logging_service" in kwargs:
            self.logging_service = kwargs.pop("logging_service")  # type: ignore
        else:
            from spectramr.domain.interfaces.service_interfaces import ILoggingService
            from spectramr.infrastructure.di.di_container import resolve_service

            try:
                self.logging_service = resolve_service(ILoggingService)
            except ValueError:
                self.logging_service = None

        # Initialize metrics service (DI or from kwargs)
        if "metrics_service" in kwargs:
            self.metrics_service = kwargs.pop("metrics_service")  # type: ignore
        else:
            from spectramr.domain.interfaces.service_interfaces import IMetricsService
            from spectramr.infrastructure.di.di_container import resolve_service

            try:
                self.metrics_service = resolve_service(IMetricsService)
            except ValueError:
                self.metrics_service = None

            # FALLBACK: If resolve_service returns None, try to get it from container attribute.
            # Narrow except to ValueError (DIContainer.resolve raises ValueError when a
            # service is unregistered or abstract). Any other exception (RuntimeError for
            # circular dependency, AttributeError, etc.) indicates a real wiring bug and
            # must propagate per CLAUDE.md #9 (no silent fallbacks).
            if self.metrics_service is None and hasattr(self, "container"):
                try:
                    self.metrics_service = self.container.resolve(IMetricsService)
                except ValueError:
                    self.metrics_service = None  # genuine "not registered" path

            # DEBUG: Log if metrics_service is None (indicates DI issue)
            if self.metrics_service is None and hasattr(self, "logging_service"):
                self.logging_service.log_warning(
                    "[BaseStrategy] metrics_service is None - validation images will NOT be saved. "
                    "This typically means IMetricsService was not registered in the DI container."
                )

        # Config is already validated by Pydantic; direct access safe
        opt_config = self.config.optimization
        enable_clip = opt_config.gradient.clip.enabled
        # Use gradient_clip_value as the SSOT for max_grad_norm
        max_grad_norm = (
            opt_config.gradient.clip.value if opt_config.gradient.clip.value is not None else 1.0
        )

        self.amp_policy = AMPPolicy(
            max_grad_norm=max_grad_norm,
            enable_gradient_clipping=enable_clip,
        )
        self.metrics_reporter = MetricsReporter()
        self.optimizer_stepper = self._build_optimizer_stepper(self.logging_service)

        # Physics and Domain resolution (SSOT: model and physics sections)
        model_domain = self.config.model.target_domain or "image"

        physics_config = self.config.physics
        physics_kspace = (
            physics_config.kspace.enable_kspace_recon
            if physics_config and physics_config.kspace
            else False
        )
        is_complex = (str(model_domain).lower() == "kspace") or physics_kspace

        # Thread optimization.precision.dtype into the AMP policy (pitfall #15: the
        # knob was previously inert and precision was hardcoded fp16).
        amp_enabled, amp_precision = resolve_amp_precision(
            opt_config.precision.enabled, opt_config.precision.dtype
        )

        self.logging_service.log_info(
            f"[Mixed Precision] enabled={amp_enabled} precision={amp_precision} "
            f"(precision.dtype={opt_config.precision.dtype}, "
            f"precision.enabled={opt_config.precision.enabled}) | enable_complex_support={is_complex} "
            f"(domain={model_domain}, kspace_recon={physics_kspace})"
        )

        amp_config = MixedPrecisionConfig(
            enabled=amp_enabled,
            precision=amp_precision,
            optimize_memory=True,
            loss_scaling=1024.0,
            dynamic_loss_scaling=True,
            enable_complex_support=is_complex,
        )
        self.amp_helper = MixedPrecisionIntegrationHelper(amp_config, self.device)
        self.amp_helper.configure_model_for_amp(self.generator_model)
        if self.discriminator_model is not None:
            self.amp_helper.configure_model_for_amp(self.discriminator_model)

        from ..utils.strategy_context import create_strategy_context

        # Use environment losses if available (from environment context)
        losses = getattr(self.env, "losses", {}) or {}

        self.context = create_strategy_context(
            device=self.device,
            config=self.config,
            fft_transformer=FFTTransformer(device=self.device),
            complex_handler=ComplexTensorHandler(),
            loss_fn=losses,
            mask_generator=KSpaceMaskGenerator(device=self.device),
        )

        self._g_losses_pool: dict[str, torch.Tensor] = {}
        self._d_losses_pool: dict[str, torch.Tensor] = {}
        self._loss_dict_reuse: dict[str, torch.Tensor] = {}
        self._all_losses_tensor: dict[str, torch.Tensor] = {}

        # Read the declared field directly. The old `hasattr(opt_config,
        # "gradient_accumulation_steps")` guard went permanently False when the
        # key folded to `optimization.gradient.accumulation_steps`, which pinned
        # accumulation to 1 for every arm regardless of config.
        self._acc_steps: int = max(1, int(opt_config.gradient.accumulation_steps))

        from spectramr.infrastructure.training.step_executor import StepExecutor

        # Check if memory monitoring is enabled in config
        memory_monitoring = self.config.optimization.memory.enable_monitoring

        self.step_executor = StepExecutor(
            amp_helper=self.amp_helper,
            amp_policy=self.amp_policy,
            gradient_accumulation_steps=self._acc_steps,
            enable_memory_monitoring=memory_monitoring,
        )

        # Resolved lazily by `_weight_table` (the per-name cache is gone: the SSOT
        # resolves EVERY declared weight once, so there is nothing left to memoize).
        self._loss_weight_table: LossWeightTable | None = None

        # Build declarative adapter chains from `config.adapters:` (Phase 4
        # of the experiment-spec-card design). Empty dict if not declared,
        # so subclasses can call `self.apply_adapters(hook, x)` unconditionally.
        # See docs/superpowers/specs/2026-05-05-experiment-spec-card-and-adapters-design.md
        from spectramr.infrastructure.builders.leaf.adapter_builders import (
            AdapterChainBuilder,
        )

        self.adapter_chains: dict[str, list[Any]] = AdapterChainBuilder(
            getattr(self.config, "adapters", None)
        ).build()

        # Call the strategy-specific setup hook.
        # Strategies that also call this in their own __init__ will execute it
        # twice, but all setup methods are idempotent (re-assign attributes).
        self._setup_strategy_specific_components()

    def _sync_schedulers_after_param_group_addition(self, new_lr: float) -> None:
        """Re-sync LR-scheduler state after ``optimizer.add_param_group()``.

        PyTorch LR schedulers snapshot ``base_lrs`` (one entry per param group)
        at construction. A strategy that adds a param group AFTER its scheduler
        is built (e.g. TTO registering ``motion_traj`` on ``opt_g``) leaves the
        scheduler's per-group lists shorter than ``optimizer.param_groups``, so
        the next ``scheduler.step()`` raises
        ``zip() argument 2 is longer than argument 1`` (smoke audit 2026-06-03,
        F7d). This appends ``new_lr`` to every active scheduler's per-group
        lists — a :class:`WarmupScheduler`'s own ``base_lrs`` /
        ``warmup_start_lr`` / ``warmup_end_lr`` AND its wrapped
        ``main_scheduler.base_lrs`` — so ``step()`` zips equal lengths.

        Call EXACTLY ONCE per added param group (it appends; repeated calls
        over-grow the lists). 29 strategies call ``add_param_group``; any that
        do so after scheduler construction must route through this helper.
        """
        schedulers = getattr(self.env, "schedulers", None)
        if not schedulers:
            return
        # Dedup by LIST identity, not scheduler identity: a WarmupScheduler
        # aliases ``warmup_end_lr`` to ``base_lrs`` when no explicit end-LR is
        # given (scheduler_system.py), and the same scheduler can appear under
        # several keys ("opt_g"/"main") — appending per-attribute would grow a
        # shared list more than once.
        seen_lists: set[int] = set()
        for sched in schedulers.values():
            if sched is None:
                continue
            # A WarmupScheduler wraps an inner scheduler — sync both objects.
            for obj in (sched, getattr(sched, "main_scheduler", None)):
                if obj is None:
                    continue
                for attr in ("base_lrs", "warmup_start_lr", "warmup_end_lr"):
                    seq = getattr(obj, attr, None)
                    if isinstance(seq, list) and id(seq) not in seen_lists:
                        seen_lists.add(id(seq))
                        seq.append(new_lr)

    def _setup_strategy_specific_components(self) -> None:
        """Hook for subclasses to initialize strategy-specific components.

        Override this method to set up strategy-specific resources like:
        - Custom loss functions (e.g., LPIPS, FID calculator)
        - Physics simulators (e.g., Bloch equation solver)
        - Specialized networks (e.g., VAE encoder, diffusion scheduler)
        - Additional optimizers

        This method is called during __init__() after common components are initialized.

        Example:
            >>> def _setup_strategy_specific_components(self):
            ...     self._lpips_loss = LPIPS(net='vgg').to(self.device, non_blocking=True)
            ...     self._bloch_simulator = BlochSimulator(config=self.config.physics)
        """

    def apply_adapters(self, hook: str, x: Any) -> Any:
        """Apply the declared adapter chain at ``hook`` to ``x``.

        No-op when no adapter chain is declared at this hook (the
        common case for legacy experiments). Subclasses opt in by
        wrapping their tensor flow at the relevant boundary, e.g.::

            target = self.apply_adapters("pre_loss_target", target)

        Per CLAUDE.md item #9, adapters fire only when the YAML opts
        in via the ``adapters:`` block — never silently. The audit's
        :func:`check_data_model_compatibility` has already validated
        that the chain bridges the declared mismatch; this method
        just runs it.
        """
        from spectramr.infrastructure.builders.leaf.adapter_builders import apply_chain

        chain = self.adapter_chains.get(hook, [])
        if not chain:
            return x
        return apply_chain(chain, x)

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the start of each epoch.

        Args:
            epoch: The index of the epoch about to start (0-indexed).
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Hook called at the end of each epoch.

        Args:
            epoch: The index of the epoch that just finished.
            metrics: Dictionary of aggregated metrics for the epoch.
        """
        pass

    def on_validation_start(self) -> None:
        """Hook called at the start of the validation phase."""
        pass

    def on_validation_end(self, metrics: dict[str, float]) -> None:
        """Hook called at the end of the validation phase.

        Args:
            metrics: Dictionary of aggregated validation metrics.
        """
        pass

    # ------------------------------------------------------------------ #
    # Domain seam (SSOT): bring a k-space dataloader target into image domain
    # ------------------------------------------------------------------ #
    def _ensure_image_domain_target(self, target: torch.Tensor) -> torch.Tensor:
        """IFFT a k-space dataloader target to image domain — once, idempotently.

        An image-domain strategy — one whose reconstruction loss, ``val_psnr``
        and cached ``_last_visual_*`` are defined on *images* — must convert a
        k-space target to image domain before using it, or:

          * ``_last_visual_target`` becomes ``|k-space|`` → the saved REAL
            reference renders as a centre-bright k-space blob instead of the
            brain ("k-space-as-the-real-image"), and
          * the motion / marker corruption (which assumes an image input) is
            applied to k-space → garbage, so the FAKE collapses to black and
            ``val_psnr`` goes NaN.

        This is the "k-space-real + black-fake" failure on the svd-coil image-
        output arms exp_p3 / hyper_mamba_meta / method_c (smoke audit
        2026-06-13). ``ConcreteVirtualFiducialStrategy`` / ``IBVFStrategy`` carry
        an equivalent inline guard; this is the shared SSOT seam every other
        image-domain strategy routes through.

        **Domain decision is delegated to the SSOT**
        :func:`~spectramr.infrastructure.training.utils.domain_inference.needs_ifft_for_visualization`,
        NOT to a raw ``dataset_type`` check: ``coil_processing_mode: rss_image`` /
        ``magnitude`` already IFFT inside the dataset's TorchIO pipeline, so those
        arms read ``dataset_type: kspace`` yet deliver an *image*. A naive
        ``dataset_type == "kspace"`` guard would re-FFT that image into k-space —
        the mirror-image regression. The result is cached (config is frozen).

        Only ``ifft2c`` from :mod:`spectramr.infrastructure.physics.fft_ops` is used
        (centred, ``norm="ortho"`` — CLAUDE.md #2; raw ``torch.fft`` calls would
        de-centre the image). The representation is preserved: complex in →
        complex out; ``[B, 2C, H, W]`` real-interleaved in → real-interleaved out.
        A real tensor whose channel count is odd (e.g. a ``target_channels: 1``
        config that stripped the imaginary half) has no complex pair to invert and
        is returned unchanged — such an arm must set ``data.target_channels: 2``
        so the pair survives the collate.

        Idempotent for image-domain data: returns ``target`` untouched whenever
        the SSOT says the targets are not k-space.
        """
        cached = getattr(self, "_target_needs_image_ifft", None)
        if cached is None:
            from spectramr.infrastructure.training.utils.domain_inference import (
                needs_ifft_for_visualization,
            )

            # needs_ifft_for_visualization returns (preds, targets); the target
            # bool already folds in the coil_processing_mode override.
            cached = bool(needs_ifft_for_visualization(self.config)[1])
            self._target_needs_image_ifft = cached
        if not cached:
            return target

        from spectramr.infrastructure.physics.fft_ops import ifft2c

        if torch.is_complex(target):
            return ifft2c(target)

        channels = target.shape[1]
        if channels >= 2 and channels % 2 == 0:
            complex_ksp = torch.complex(target[:, 0::2], target[:, 1::2])
            image = ifft2c(complex_ksp)
            out = torch.empty_like(target)
            out[:, 0::2] = image.real
            out[:, 1::2] = image.imag
            return out

        # Odd / single-channel real target: no imaginary half to invert. The
        # config must keep the complex pair (data.target_channels: 2); we refuse
        # to fabricate phase, so the tensor passes through unchanged.
        return target

    @final
    def save_debug_snapshot(
        self,
        tensors: dict[str, Any],
        *,
        step: int,
        tag: str = "train_step",
        extra: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
        in_kspace_keys: set[str] | None = None,
        log_scaled_keys: set[str] | None = None,
        channel_segments: Mapping[str, Sequence[ChannelSegment]] | None = None,
        model_input_key: str | None = None,
    ) -> None:
        """Paradigm-agnostic per-step debug snapshot.

        Any strategy can call this from its ``_compute_losses_impl`` to
        write a comprehensive textual + JSON + image-preview snapshot to
        ``<run_dir>/debug_snapshots/`` for the first few training steps.
        Gated on ``config.logging.snapshots.enabled`` (default True) and
        capped by ``config.logging.snapshots.max_calls`` (default 8, per
        ``(run_dir, tag)``) so disk usage stays bounded across paradigms.

        See :mod:`spectramr.infrastructure.training.debug_snapshot` for the
        full contract and rendering rules. The base class also auto-
        calls this once at the first training step of every experiment
        — strategies only need to call it explicitly to capture
        paradigm-specific intermediate tensors.

        ``extra`` may be a zero-arg callable (or a dict of them) when building
        the values costs a device->host sync; it is then resolved only if a
        write actually happens. See
        :func:`~spectramr.infrastructure.training.debug_snapshot._resolve_deferred_extra`.

        ``log_scaled_keys`` names the tensors that are actually ``log1p``-
        compressed, for arms that set ``data.processing.enable_log_scaling``.
        Pass it whenever a snapshot mixes domains — the two emitters in this
        class both capture ``target``/``input`` as they entered ``train_step``,
        i.e. BEFORE :meth:`apply_kspace_normalization` compresses them, so the
        arm-level flag alone would have them decompressed with ``expm1`` and
        rendered as a ringing phase-only artifact (#682's blind spot). Omitting
        it keeps the historical all-keys behaviour, which is right for the
        strategy-side emitters that capture only post-normalization tensors.

        ``model_input_key`` names which of ``tensors`` the model is ACTUALLY fed.
        Required when ``tag`` is this strategy's ``snapshot_model_input_tag``,
        ignored on every other snapshot. A real parameter rather than one more
        ``extra`` convention because ``extra`` may be a deferred callable:
        reading a key out of it here would force the device sync the deferral
        exists to avoid, so a contract built on ``extra`` could not be checked
        on the self-emitting route -- which is precisely the route that produced
        #1298. The five ``_declare_model_input`` sites keep writing it into their
        (plain-dict) ``extra``; ``_snapshot_declared_model_input`` lifts it out
        and forwards it here.
        """
        try:
            from spectramr.infrastructure.training.debug_snapshot import (
                save_debug_snapshot,
            )
            from spectramr.infrastructure.training.utils.kspace_view import (
                log_scaling_enabled,
            )
        except Exception as exc:
            logger.debug("Debug snapshot helper unavailable: %s", exc)
            return

        # Resolve run dir (mirrors diffusion._save_debug_snapshots logic)
        run_dir = (
            getattr(self.env, "run_output_dir", None)
            or getattr(self.config.training, "output_dir", None)
            or "experiments/outputs"
        )
        logging_cfg = getattr(self.config, "logging", None)

        # Mark which tensors carry k-space so the previewer IFFTs them for a
        # human-readable image. Two gaps motivated the config-derived union:
        #  * the "first_steps" auto-snapshot passes ``in_kspace_keys=None``, so
        #    the module fell back to a ``"kspace"``-substring match that misses
        #    the canonical names ``input_raw`` / ``input_prepared`` / ``target``
        #    (May 2026 experiment_11 "black with white dot").
        #  * the diffusion "model_output" snapshot passes an EXPLICIT set that
        #    omitted ``"target"``, so its ``target.png`` rendered as raw
        #    k-space (2026-06-17 downloaded-run triage).
        # Fix both: when the config says the dataloader yields k-space, UNION
        # the canonical k-space key names into whatever the caller supplied
        # (mark every key when the caller supplied none). The previewer's
        # spectrum-based check (``_is_in_kspace_domain``) still corrects any
        # image-domain tensor mis-flagged here, so over-marking is safe.
        try:
            from spectramr.infrastructure.training.utils.domain_inference import (
                needs_ifft_for_visualization,
            )

            _, targets_are_kspace = needs_ifft_for_visualization(self.config)
        except Exception as exc:
            logger.debug("first_steps in_kspace inference skipped: %s", exc)
            targets_are_kspace = False
        # Keys whose k-space domain is declared by the config SSOT (not a name
        # guess). The previewer IFFTs these UNCONDITIONALLY, bypassing the
        # fragile spectrum veto that false-negated on real normalized multicoil
        # M4Raw k-space and rendered experiment_11 ``input_prepared`` /
        # ``model_output`` as raw, off-centre k-space (2026-06-27 audit).
        authoritative_kspace_keys: set[str] = set()
        if targets_are_kspace:
            _canonical = {
                "input",
                "input_raw",
                "input_prepared",
                "target",
                # Mixed-domain in at least one paradigm: cold diffusion's
                # ``model_input`` is ``cat([x_t, smaps], dim=1)``, so half its
                # channels are k-space and half are image-domain coil maps.
                # Membership here is still right -- the k-space half genuinely
                # needs the veto bypassed, and the veto false-negates on this
                # very data -- but a whole-tensor render then IFFTs the maps too
                # and draws the experiment_11 crosshair. The emitter declares
                # ``channel_segments`` so each half renders in its own domain;
                # see ``debug_snapshot._split_channel_segments``.
                "model_input",
                "model_output",
                # The cold-diffusion forward process degrades k-space by
                # zero-filling (``q_sample`` is ``x_0 * mask``), so its output
                # is k-space by construction under exactly the same SSOT that
                # declares ``input``/``target`` to be. Without this the ONLY
                # snapshot carrying the tensor the model is actually fed fell
                # back to the spectrum veto -- the same veto the 2026-06-27
                # audit above records as false-negating on this very data --
                # and rendered the accelerated input as raw, off-centre k-space
                # instead of the zero-filled image it is.
                #
                # ``noisy_kspace`` ONLY. Not ``noisy_images``: no emitter writes
                # that key (the diffusion capture names it ``noisy_kspace``), and
                # the name is bound to a LATENT in the latent-diffusion branch
                # -- ``q_sample(latent_z0, ...)`` at diffusion.py. Membership
                # here bypasses the veto unconditionally, so a latent added
                # under its own variable name would be IFFT'd with nothing left
                # to catch it. "Over-marking is safe" is true of
                # ``in_kspace_keys``, which the veto still corrects; it is not
                # true of this set. ``noisy_images`` stays in the veto-corrected
                # set at diffusion._save_debug_snapshots, which is the right
                # home for a name whose domain depends on the branch.
                "noisy_kspace",
                # diffusion pre/post-DC outputs are k-space by construction;
                # without them the veto renders them as raw k-space over image tiles.
                "model_output_pre_dc",
                "model_output_post_dc",
            }
            authoritative_kspace_keys = _canonical & set(tensors.keys())
            if in_kspace_keys is None:
                in_kspace_keys = set(tensors.keys())
            else:
                in_kspace_keys = set(in_kspace_keys) | authoritative_kspace_keys

        # A strategy that emits its OWN model-output snapshot (diffusion's
        # richer pre/post-DC capture, tagged ``model_output_dc``) marks the step
        # done so the base wrapper's generic emission in
        # ``_snapshot_model_output`` skips the duplicate. Prefix-matched rather
        # than compared to one literal: the richer captures carry their own tag
        # so they get their own per-tag call budget, and an exact match would
        # have stopped suppressing the duplicate the moment they did.
        if tag.startswith("model_output"):
            self._model_output_snapshot_done = True

        # Same idiom for the model-INPUT contract: a strategy that already emits
        # its own snapshot under the tag it declares (VF's ``vf_twin``) satisfies
        # ``_snapshot_declared_model_input`` here rather than declaring a second
        # copy of the same tensors.
        #
        # Set on ATTEMPT, before the writer's gates -- exactly like the flag
        # above. Setting it on a successful WRITE would make the per-(run_dir,
        # tag) budget produce a false violation: once ``max_calls`` is spent the
        # writer stops writing, the flag would stop being set, and the wrapper
        # would raise at step ``max_calls + 1`` of a run that is behaving
        # correctly. Invisible to a one-step unit test, fatal to a real run.
        declared_tag = self.snapshot_model_input_tag
        model_input_contract: dict[str, Any] | None = None
        if declared_tag and tag == declared_tag:
            self._model_input_snapshot_done = True
            # Both routes into the model-input snapshot pass through here -- the
            # declared one (``_snapshot_declared_model_input`` calls this
            # method) and the self-emitting one that only sets the flag above.
            # The self-emitting route is the one that mislabelled #1298, so a
            # check placed on the declared branch alone would have missed it.
            model_input_contract = self._verify_model_input_snapshot(
                tensors=tensors, tag=tag, model_input_key=model_input_key
            )

        try:
            save_debug_snapshot(
                run_dir=run_dir,
                step=step,
                tag=tag,
                tensors=tensors,
                paradigm=type(self).__name__,
                config_section=logging_cfg,
                in_kspace_keys=in_kspace_keys,
                authoritative_kspace_keys=authoritative_kspace_keys,
                # The one place that can supply this: `debug_snapshot` receives
                # `config.logging` and cannot reach `config.data`, so it had no
                # way to know the k-space it was handed was log1p-compressed and
                # IFFT'd it as-is (#682).
                log_scaled=log_scaling_enabled(self.config),
                log_scaled_keys=log_scaled_keys,
                channel_segments=channel_segments,
                extra=extra,
                provenance=self._snapshot_provenance(self._snapshot_phase),
                model_input_contract=model_input_contract,
            )
        except Exception as exc:
            logger.debug("save_debug_snapshot raised: %s", exc)

    def _verify_model_input_snapshot(
        self, *, tensors: dict[str, Any], tag: str, model_input_key: str | None
    ) -> dict[str, Any] | None:
        """Check that the model-input snapshot describes the tensor the model got.

        Two tiers, matching the static / artifact split
        :meth:`_snapshot_declared_model_input` already draws:

        * the NAMING half raises. A snapshot that does not say which of its
          tensors is the model input, or that names one it does not carry, is
          wrong before any batch is loaded, and it is the failure that let
          #1298's mislabel read as a passing contract.
        * the WIDTH half warns and stamps. Resolving a backbone's input width is
          a heuristic (see
          :func:`~spectramr.infrastructure.training.model_input_contract.resolve_model_in_channels`),
          and a false positive here would abort a real run over a diagnostic.

        The verdict goes into the artifact either way -- including
        ``unresolved``. Non-negotiable 14 makes the declared-vs-applied
        divergence *itself* the finding, and a check whose negative result is
        only a log line is one clamped log level away from invisible (the
        attention_shootout arms run at ``level: warning``).

        Returns ``None`` only when the whole verification could not run, so the
        writer omits the key rather than stamping a hollow one.
        """
        from spectramr.infrastructure.training.model_input_contract import (
            STATUS_MISMATCH,
            require_model_input_key,
            resolve_model_in_channels,
            verify_model_input,
        )

        key = require_model_input_key(
            strategy_name=type(self).__name__,
            tag=tag,
            tensors=tensors,
            model_input_key=model_input_key,
        )

        if self._model_input_width is None:
            try:
                module = self.generator_model
            except Exception as exc:  # a strategy whose env has no generator yet
                logger.debug("model-input contract: generator unavailable: %s", exc)
                module = None
            self._model_input_width = resolve_model_in_channels(module)
        in_channels, in_channels_source = self._model_input_width

        try:
            verdict = verify_model_input(
                tensors=tensors,
                model_input_key=key,
                in_channels=in_channels,
                in_channels_source=in_channels_source,
            )
        except Exception as exc:  # a diagnostic must never break a training step
            logger.debug("model-input contract: verification skipped: %s", exc)
            return None

        if verdict.status == STATUS_MISMATCH and not self._model_input_contract_warned:
            # Latched: this wrapper runs every step (the write budget lives
            # downstream), so an unlatched warning would emit once per iteration
            # for the whole run.
            self._model_input_contract_warned = True
            logger.warning(
                "[model-input contract] %s snapshot %r: %s",
                type(self).__name__,
                tag,
                verdict.detail,
            )
        return verdict.as_record()

    #: Does ``first_steps/input_prepared`` hold the tensor the model is fed?
    #: True for every strategy whose preparation finishes before the forward
    #: pass. A strategy that transforms the prepared input FURTHER inside the
    #: step (cold diffusion degrades it there) overrides this to False and
    #: emits its own snapshot of the real model input -- see
    #: ``docs/debug_snapshot_contract.rst``. A class attribute, not a config
    #: knob: it states what the code does, so it is not the user's to set.
    snapshot_prepared_is_model_input: bool = True

    #: Where the real model input lives when the flag above is False. The
    #: contract is only useful if the artifact says where to look instead;
    #: "this is not the model input" alone sends the reader back to the code.
    snapshot_model_input_tag: str | None = None

    #: Channel-domain decomposition declared alongside the model input, for a
    #: tensor whose channel axis superposes domains (cold diffusion's
    #: ``model_input`` is ``cat([x_t, smaps], dim=1)``). A CLASS-level default so
    #: it exists even on an instance whose ``__init__`` was bypassed -- the
    #: emitter reads it unconditionally, and the strategy tests build doubles
    #: that never run ``__init__``. Defaulting it here rather than reaching for a
    #: ``getattr(..., None)`` at the read site keeps "absent" from becoming a
    #: silent synonym for "declared nothing" in production code.
    _declared_channel_segments: Mapping[str, Sequence[ChannelSegment]] | None = None
    #: Companion to the above, defaulted here for the same reason.
    _declared_log_scaled_keys: set[str] | None = None

    def _first_steps_extra(self, epoch: int) -> dict[str, Any]:
        """Context stamped on the ``first_steps`` snapshot.

        States the canonical-key contract in the artifact itself:
        ``input_prepared`` is captured BEFORE the forward pass, so for a
        strategy that degrades further inside the step it is not the tensor the
        model receives. Saying so — and naming where the real one is — is what
        stops a reader concluding the arm was fed fully-sampled data (the
        experiment_11 report that produced this contract).
        """
        prepared_is_input = bool(self.snapshot_prepared_is_model_input)
        extra: dict[str, Any] = {
            "epoch": int(epoch),
            "prepared_equals_model_input": prepared_is_input,
        }
        if not prepared_is_input:
            tag = self.snapshot_model_input_tag or "see strategy docs"
            extra["model_input_snapshot_tag"] = tag
        return extra

    def _snapshot_provenance(self, source: str = "train") -> dict[str, Any] | None:
        """Build the data-provenance record once PER SOURCE and reuse it.

        Cached because the walk touches the dataset wrapper chain and the
        module tree, and snapshots fire inside the training step -- non-
        negotiable 9 forbids paying that per step. Nothing it reads changes
        after the environment is built.

        Keyed on ``source``, and the loader is looked up under the SAME key.
        A single train-built record reused for every snapshot is what the
        ``source`` field exists to prevent: train and val are built by
        *different* ``tio.Compose`` objects, so a val snapshot carrying the
        train chain asserts augmentation that never touched the batch.

        The cache stores failures too (as ``None``), via a sentinel rather
        than ``getattr(..., None)``. Treating "built and failed" as "not built
        yet" re-ran the whole dataset/model walk on every subsequent snapshot
        -- the exact per-step cost the cache exists to avoid.
        """
        cache = self.__dict__.setdefault("_snapshot_provenance_cache", {})
        if source in cache:
            return cache[source]
        try:
            from spectramr.infrastructure.training.snapshot_provenance import (
                build_snapshot_provenance,
            )

            loaders = getattr(self.env, "data_loaders", None)
            loader = loaders.get(source) if isinstance(loaders, dict) else None
            record = build_snapshot_provenance(
                self.config,
                dataset=getattr(loader, "dataset", None),
                model=getattr(self.env, "generator", None),
                source=source,
            )
        except Exception as exc:
            logger.debug("snapshot provenance unavailable (source=%s): %s", source, exc)
            record = None
        cache[source] = record
        return record

    #: Which split the snapshots being written right now came from. Class
    #: default is the training path; a validation path declares its own via
    #: :meth:`snapshot_source`. Not derived from ``torch.is_grad_enabled()`` or
    #: ``model.training`` -- both are toggled by diagnostics that are still on
    #: the train path, so neither can carry a correctness claim.
    _snapshot_phase: str = "train"

    @contextmanager
    def snapshot_source(self, source: str) -> Iterator[None]:
        """Declare which split the snapshots written in this block belong to.

        Validation paths that reach a snapshot emitter must wrap the call, or
        the record claims the training data chain. ``virtual_fiducial``'s
        ``validation_step`` calls ``_compute_losses_impl`` directly, and that
        impl emits ``vf_twin`` -- so without this the val snapshot inherits the
        train ``Compose``.

        Restores the previous value rather than resetting to ``"train"``, so
        nesting cannot silently relabel an outer block.
        """
        previous = self._snapshot_phase
        self._snapshot_phase = source
        try:
            yield
        finally:
            self._snapshot_phase = previous

    def _ensure_generator_output_capture(self, module: Any = None) -> None:
        """Register a forward hook on the generator that stashes its latest
        output, enabling the paradigm-agnostic ``model_output`` snapshot in
        :meth:`_compute_losses` and :meth:`_capture_model_output`.

        The hook captures the REAL model output regardless of how each strategy
        invokes the generator (``gen(x)`` for recon, ``gen(x, field_strength=…)``
        for the field family, the digital-twin-stacked input for VF, …), so no
        per-strategy wiring is needed. It only stashes while
        ``self._capture_gen_output`` is set (the training forward inside
        ``_compute_losses``), so validation / witness forwards are ignored.

        Args:
            module: The module whose forward to hook. ``None`` means the env's
                primary ``env.generator``. A strategy that owns a generator
                DIFFERENT from the env's must pass its own -- ``cut``
                (``cut_strategy.py:117``), ``cyclegan``
                (``cyclegan_strategy.py:158``) and ``stargan_v2``
                (``stargan_v2_strategy.py:150``) each build their own whenever
                the env-built primary is an incompatible type, and hooking
                ``env.generator`` for those installs the hook on a module the
                strategy never forwards: a snapshot that is wired and can still
                never fire (pitfall #16).

        Registration is guarded PER MODULE. The previous single
        ``_gen_output_hook_handle`` meant whichever module armed first won
        forever, so the explicit ``module`` above would have been silently
        ignored for any strategy whose base ``__init__`` path armed the env
        generator first.

        No raise when there is nothing to hook: silence is this method's
        pre-existing contract (a mock-fed strategy has no hookable module), and
        every wired call site passes a concrete one.
        """
        handles = getattr(self, "_gen_output_hook_handles", None)
        if handles is None:
            handles = {}
            self._gen_output_hook_handles = handles
        gen = module if module is not None else getattr(self.env, "generator", None)
        if gen is None or not hasattr(gen, "register_forward_hook"):
            return
        if id(gen) in handles:
            return

        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            if getattr(self, "_capture_gen_output", False):
                self._last_generator_output = _first_tensor(output)

        try:
            handles[id(gen)] = (gen, gen.register_forward_hook(_hook))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("generator-output hook registration skipped: %s", exc)

    @contextmanager
    def _capture_model_output(
        self,
        *,
        module: Any,
        input_batch: Any,
        target_batch: Any,
        step: int = 0,
    ) -> Iterator[None]:
        """Arm the ``model_output`` capture around a forward, and emit on exit.

        :meth:`_compute_losses` is the wrapper that normally does this, but 13
        call sites invoke ``_compute_losses_impl`` DIRECTLY -- a custom
        ``train_step`` that needs the losses inside an optimizer closure, or a
        ``validation_step`` reusing the training loss. Those bypass the wrapper
        and so never reached :meth:`_snapshot_model_output` at all (#1190).
        This is the wrapper's snapshot half, extracted so both paths reach it
        through ONE mechanism instead of two divergent copies (non-negotiable 6).

        Deliberately NOT included here, because the bypass sites already have
        both and duplicating them would change behaviour:

        * **AMP.** ``StepExecutor.execute_step`` enters
          ``amp_helper.get_autocast_context()`` around every closure it runs
          (``step_executor.py``), so a train-path bypass is already inside
          autocast.
        * **Total-loss key normalization.** Every bypass site reads
          ``losses["g_total_loss"]`` itself.

        Args:
            module: The module that produces the model output. REQUIRED and
                explicit -- pass ``None`` only to mean "the env's primary
                generator". See :meth:`_ensure_generator_output_capture` for why
                defaulting would make this a no-op for exactly the strategies
                that own their generator.
            input_batch: Batch as it entered the step, for the snapshot's
                ``input`` row.
            target_batch: Ground truth, for the snapshot's ``target`` row.
            step: Training iteration the snapshot is stamped with.

        Arm TIGHTLY around the forward that produces the model output. The
        adversarial strategies also run the GENERATOR inside their discriminator
        closure to make fakes (``cut_strategy.py``, ``cyclegan_strategy.py``,
        ``stargan_v2_strategy.py`` -- all under ``torch.no_grad()``); a flag left
        armed across the whole step would let that no-grad pass overwrite the
        stash, and the snapshot would show D's fake rather than G's output.

        :meth:`_ensure_input_contract_guard` is deliberately NOT called here even
        though :meth:`_compute_losses` installs it alongside this capture. That
        guard is the TRAINING dimension contract and can raise under
        ``SPECTRAMR_DIMENSION_CONTRACT=enforce``; installing it from here would
        extend it to the seven validation paths, which is a runtime behaviour
        change well outside a snapshot-coverage fix. It stays in the wrapper.
        """
        self._ensure_generator_output_capture(module=module)
        self._model_output_snapshot_done = False
        self._capture_gen_output = True
        try:
            yield
        finally:
            self._capture_gen_output = False
        # After the try/finally, NOT inside it: an exception escaping the body
        # must propagate without emitting a snapshot of a forward that never
        # completed. Same ordering as :meth:`_compute_losses`.
        self._snapshot_model_output(input_batch=input_batch, target_batch=target_batch, step=step)

    def _ensure_input_contract_guard(self) -> None:
        """Install a forward PRE-hook enforcing the model's declared dimension
        contract (Layer 2 of the dimension-contract plan).

        Twin of :meth:`_ensure_generator_output_capture`: one
        ``register_forward_pre_hook`` fires for every paradigm regardless of how
        the strategy invokes the generator, so it never needs per-strategy wiring.

        No-op when the model leaves ``ModelCapabilities`` unannotated (zero risk
        to legacy models) or the guard mode is ``off``. The mode is the validated
        ``SPECTRAMR_DIMENSION_CONTRACT`` knob (``observe`` default → log-only;
        ``enforce`` → raise at iteration 1). The guard asserts only declared,
        universally-safe invariants — see ``dimension_contract.assert_input_contract``.
        """
        if getattr(self, "_input_contract_hook_handle", None) is not None:
            return
        from spectramr.infrastructure.validation.dimension_contract import (
            DimensionContractError,
            assert_input_contract,
            first_tensor,
            resolve_contract_mode,
        )

        mode = resolve_contract_mode()
        if mode == "off":
            return
        gen = getattr(self.env, "generator", None)
        if gen is None or not hasattr(gen, "register_forward_pre_hook"):
            return
        model_type = self.config.model.model_type
        if not model_type:
            return
        try:
            from spectramr.models.registry import get_model_capabilities

            caps = get_model_capabilities(str(model_type))
        except Exception:  # pragma: no cover - defensive
            caps = None
        if caps is None:
            return  # unannotated → no guard installed (zero legacy risk)
        enforce = mode == "enforce"

        def _pre(_module: Any, inputs: Any) -> None:
            x = first_tensor(inputs)
            if x is None:
                return
            try:
                assert_input_contract(x, caps, where=str(model_type))
            except DimensionContractError as exc:
                if enforce:
                    raise
                logger.warning("dimension-contract (observe): %s", exc)

        try:
            self._input_contract_hook_handle = gen.register_forward_pre_hook(_pre)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("input-contract guard registration skipped: %s", exc)

    def _declare_model_input(
        self,
        tensors: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
        in_kspace_keys: set[str] | None = None,
        channel_segments: Mapping[str, Sequence[ChannelSegment]] | None = None,
        log_scaled_keys: set[str] | None = None,
    ) -> None:
        """Hand the wrapper the tensor the model is ACTUALLY fed.

        For a strategy that sets ``snapshot_prepared_is_model_input = False``,
        the ``first_steps`` snapshot's ``input_prepared`` is captured before the
        forward pass and so is *not* the model input. This declares the real
        one; :meth:`_snapshot_declared_model_input` emits it under
        ``snapshot_model_input_tag`` -- the tag ``first_steps`` already points
        readers to.

        Call it from ``_compute_losses_impl`` at the point the model input
        exists, UNCONDITIONALLY. There is deliberately no cadence check here:
        the stash is a dict-of-references assignment (non-negotiable 9 -- no
        copy, no device sync), and ``save_debug_snapshot`` is the single
        authority on enabled/interval/budget. A second gate at the call site is
        precisely how #706 and the ``intervals.log * 5`` mis-gating happened.

        Args:
            tensors: Key -> tensor. Name each key for what the tensor IS: the
                names drive the previewer's domain inference and are what a
                reader concludes from the rendered PNG.
            extra: JSON-serialisable provenance stamped into the snapshot.
            channel_segments: For a key whose channel axis superposes domains,
                its ``[(label, width, is_kspace[, log_compressed]), ...]``
                decomposition. See
                :func:`~spectramr.infrastructure.training.debug_snapshot.save_debug_snapshot`.
            in_kspace_keys: Which keys hold k-space, so the previewer IFFTs
                them. Pass it EXPLICITLY -- ``None`` falls back to a ``"kspace"``
                substring match over key names, which mis-renders an
                image-domain tensor that borrowed a k-space name (findings
                booklet 2026-05-05 VIS-1). An empty set is a valid, explicit
                "none of these are k-space".
            log_scaled_keys: Which keys are ACTUALLY ``log1p``-compressed, for
                the arms where the compression is on. ``None`` keeps the
                historical "every declared-k-space key is decompressed", which
                is right only when every tensor in ``tensors`` was captured
                post-normalization. Pass it explicitly as soon as one is not --
                a key that is k-space but uncompressed (cold diffusion's
                ``fft2c``'d S-maps) gets ``expm1`` applied to a spectrum that
                never saw ``log1p``, which flattens every bin above
                ``DECOMPRESS_MAGNITUDE_CEILING`` onto one value.
        """
        self._declared_model_input = (tensors, extra, in_kspace_keys)
        # Deliberately NOT a fourth tuple slot: several strategies' tests
        # unpack this triple positionally, so widening it breaks modules
        # whose own paired test never runs here. Set and cleared in lockstep
        # with the triple below.
        self._declared_channel_segments = channel_segments
        self._declared_log_scaled_keys = log_scaled_keys

    def _snapshot_declared_model_input(self, *, step: int) -> None:
        """Emit the declared model input, or RAISE when the declaration is absent.

        This is what makes non-negotiable 14's carve-out structural instead of a
        convention. ``snapshot_prepared_is_model_input = False`` is a *pointer*:
        it tells whoever reads a ``first_steps`` artifact "the real model input
        is under <tag>". Nothing used to check that the target existed -- and it
        did not. ``DiffusionTrainingStrategy`` declares ``diffusion_step``, but
        the sole emitter sat in ``_prepare_diffusion_inputs``, reachable only
        through ``diffusion._compute_losses_impl`` -- the hook seven subclasses
        override. Each inherited the pointer and emitted nothing at its target,
        so the artifact named a tag that was never written.

        Lives in the WRAPPER rather than the overridable hook for exactly that
        reason: a subclass cannot silently drop what it does not implement.

        Two checks with deliberately different scopes:

        * ``tag`` unset while the carve-out is declared is a STATIC defect in
          the class -- wrong before any run starts -- so it raises regardless of
          config.
        * A missing declaration is an ARTIFACT defect, so it is enforced on the
          runs that write artifacts. With snapshots disabled no ``first_steps``
          record makes the claim, so there is nothing to contradict. That gate
          reads ``enabled`` ONLY, never a step-dependent predicate -- see
          :func:`~spectramr.infrastructure.training.debug_snapshot.snapshots_are_enabled`.

        Raises rather than skipping quietly per non-negotiable 3: a silent skip
        is the exact failure mode this method exists to close.
        """
        if self.snapshot_prepared_is_model_input:
            return  # No carve-out -- ``first_steps/input_prepared`` IS the input.

        declared = self._declared_model_input
        # Read the sidecar BEFORE clearing it -- it is consumed further down.
        declared_segments = self._declared_channel_segments
        declared_log_keys = self._declared_log_scaled_keys
        self._declared_model_input = None  # Never pin one step's tensors.
        self._declared_channel_segments = None
        self._declared_log_scaled_keys = None

        tag = self.snapshot_model_input_tag
        if not tag:
            raise ValueError(
                f"{type(self).__name__} sets snapshot_prepared_is_model_input="
                "False but leaves snapshot_model_input_tag unset. The flag tells"
                " a reader that 'input_prepared' is not the model input; without"
                " a tag it does not say where the real one is, which sends them"
                " back to the code (non-negotiable 14). Set both, or neither."
            )

        from spectramr.infrastructure.training.debug_snapshot import (
            snapshots_are_enabled,
        )

        if not snapshots_are_enabled(getattr(self.config, "logging", None)):
            return

        if declared is not None:
            tensors, extra, in_kspace_keys = declared
            channel_segments = declared_segments
            log_scaled_keys = declared_log_keys
            # ``extra`` is a plain dict on this route by construction (the five
            # declare sites pass dict literals), so the key can be lifted out of
            # it here and handed to the wrapper as a real parameter. That keeps
            # the declaration where its five callers already write it while
            # giving the wrapper something it can also demand of the deferred,
            # self-emitting route.
            self.save_debug_snapshot(
                tensors,
                step=step,
                tag=tag,
                extra=extra,
                in_kspace_keys=in_kspace_keys,
                channel_segments=channel_segments,
                log_scaled_keys=log_scaled_keys,
                model_input_key=(
                    extra.get("model_input_key") if isinstance(extra, Mapping) else None
                ),
            )
            return

        if self._model_input_snapshot_done:
            return  # The strategy emitted under ``tag`` itself.

        raise RuntimeError(
            f"{type(self).__name__} declares snapshot_prepared_is_model_input="
            f"False and snapshot_model_input_tag={tag!r}, but its "
            "_compute_losses_impl neither called _declare_model_input() nor "
            f"emitted a snapshot tagged {tag!r}. The 'first_steps' artifact "
            f"therefore points at a {tag!r} snapshot that does not exist. "
            "Declare the tensor the model is fed at the point it is built:\n"
            "    self._declare_model_input({'<name>': <tensor>}, "
            "in_kspace_keys=set())\n"
            "If this strategy does NOT degrade its input inside the step, drop "
            "the two class attributes instead and let the base contract stand."
        )

    def _snapshot_model_output(self, *, input_batch: Any, target_batch: Any, step: int) -> None:
        """Emit a ``model_output`` snapshot from the hooked generator output.

        Skipped when the strategy already emitted its own ``model_output``
        snapshot this step (``_model_output_snapshot_done``) or when no output
        was captured. Always clears the stash so a forward's output is never
        pinned across training steps.
        """
        out = getattr(self, "_last_generator_output", None)
        self._last_generator_output = None
        if out is None or getattr(self, "_model_output_snapshot_done", False):
            return
        try:
            self.save_debug_snapshot(
                {"model_output": out, "target": target_batch, "input": input_batch},
                step=step,
                tag="model_output",
                # Only the generator's output is in network units (compressed
                # when the arm log-scales). ``target``/``input`` are the batch as
                # it entered ``train_step`` -- pre-normalization, so physical.
                # The same split the scale-context note below describes.
                log_scaled_keys={"model_output"},
                # DEFERRED, not called here: `_model_output_scale_context` does
                # two `float(tensor)` device->host syncs, and this method runs
                # once per training step for every strategy routed through
                # `_compute_losses`. Building it eagerly paid both syncs on
                # every step while the budget only suppressed the WRITE -- the
                # same shape as #1188's `vf_twin` defect, in the base emitter.
                extra=lambda: self._model_output_scale_context(out, target_batch),
            )
        except Exception as exc:  # pragma: no cover - never break training
            logger.debug("model_output auto-snapshot skipped: %s", exc)

    @staticmethod
    def _model_output_scale_context(model_output: Any, target_batch: Any) -> dict[str, Any]:
        """Describe the SCALE each row of the model_output snapshot lives on.

        This fallback snapshot pairs the generator's raw output — network units,
        i.e. whatever space the model was handed — with ``target``/``input`` as
        they entered ``train_step``, BEFORE any normalization the strategy
        applies internally (``_prepare_diffusion_inputs`` normalizes when the
        batch carries no ``kspace_scale``). The two can therefore sit on
        different scales in adjacent rows of the same stats table.

        Read at face value that manufactures an alarming mismatch. On exp_11
        attention_none it showed ``model_output`` at abs_max 3.8 against a
        ``target`` at 2401 — a 630x gap that reads as a broken model, when in
        fact ``expm1(4.707) * 22.207 = 2435``: the target was simply displayed
        in physical units while the output stayed log-compressed. The loss
        itself compares like with like. Issue #587.

        Emitting the ratio makes the table state this rather than imply the
        reader will notice.
        """
        context: dict[str, Any] = {
            "model_output_scale": "generator output (network units)",
            "target_input_scale": "batch as received by train_step (pre-strategy-normalization)",
        }
        try:
            out_max = float(model_output.detach().abs().max())
            tgt_max = float(target_batch.detach().abs().max())
        except Exception:  # pragma: no cover - defensive, never break training
            return context
        context["abs_max_model_output"] = out_max
        context["abs_max_target"] = tgt_max
        if out_max > 0.0 and tgt_max > 0.0:
            ratio = max(out_max, tgt_max) / min(out_max, tgt_max)
            context["abs_max_ratio"] = round(ratio, 2)
            if ratio > 10.0:
                context["scale_warning"] = (
                    f"model_output and target differ by {ratio:.0f}x in peak magnitude. "
                    "These rows are NOT on a common scale — compare them only after "
                    "putting both in the same space. This is expected when the "
                    "strategy normalizes internally (#587), and is NOT by itself "
                    "evidence of a model or loss defect."
                )
        return context

    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Template method returning the generator step closures for the Trainer.

        Returns a list containing the optimization configuration for the generator.
        Subclasses should typically override `_compute_losses_impl`.

        Note:
            For every paradigm, the base class writes a per-experiment
            debug snapshot at the first few training steps via
            :meth:`save_debug_snapshot`. This generalises the
            diffusion-only ``_save_debug_snapshots`` mechanism so the
            same diagnostics (shape/dtype/range/NaN-count + IFFT-aware
            image previews) are produced for GAN / VAE / recon / SSL /
            cycle-Bloch / disentangled / etc. experiments.
        """
        # Unpack batch if lr/hr not provided
        if input_batch is None or target_batch is None:
            lr, hr = self._unpack_batch(batch)
            input_batch = pick_present(input_batch, lr)
            target_batch = pick_present(target_batch, hr)

        if input_batch is not None:
            input_batch = self._to_device(input_batch)
        if target_batch is not None:
            target_batch = self._to_device(target_batch)

        # Ensure model is in training mode
        if not self.generator_model.training:
            self.generator_model.train()
        if self.discriminator_model is not None and not self.discriminator_model.training:
            self.discriminator_model.train()

        input_batch_prepared = self._prepare_model_input(input_batch)

        # [FIX] Complex→Real guard: standard nn.Conv2d and GroupNorm have no
        # CUDA kernels for ComplexFloat tensors (crashes with NotImplementedError).
        # Convert complex64 [B,C,H,W] → real-stacked float32 [B,2C,H,W].
        # See docs/DOMAIN_HANDLING_RULES.md, Rule 5.
        # IMPORTANT: Convert BOTH input AND target to prevent 'mse_cuda not
        # implemented for ComplexFloat' in loss computation.
        if input_batch_prepared is not None and torch.is_complex(input_batch_prepared):
            if not hasattr(self, "_complex_guard_logged"):
                logger.info(
                    "[ComplexGuard] Complex input detected (dtype=%s, shape=%s). "
                    "Converting to real-interleaved [B,2C,H,W] for nn.Conv2d compatibility.",
                    input_batch_prepared.dtype,
                    input_batch_prepared.shape,
                )
                self._complex_guard_logged = True
            # Interleaved layout [R0,I0,R1,I1,...] matches ComplexToRealTransform and
            # all downstream consumers (data_consistency_layer, normalization, attention).
            _b, _c, _h, _w = input_batch_prepared.shape
            _ri = torch.empty(
                _b,
                _c * 2,
                _h,
                _w,
                dtype=torch.float32,
                device=input_batch_prepared.device,
            )
            _ri[:, 0::2] = input_batch_prepared.real
            _ri[:, 1::2] = input_batch_prepared.imag
            input_batch_prepared = _ri

        if target_batch is not None and torch.is_complex(target_batch):
            if not hasattr(self, "_complex_target_guard_logged"):
                logger.info(
                    "[ComplexGuard] Complex target detected (dtype=%s, shape=%s). "
                    "Converting to real-interleaved [B,2C,H,W] for loss compatibility.",
                    target_batch.dtype,
                    target_batch.shape,
                )
                self._complex_target_guard_logged = True
            _b, _c, _h, _w = target_batch.shape
            _ri = torch.empty(_b, _c * 2, _h, _w, dtype=torch.float32, device=target_batch.device)
            _ri[:, 0::2] = target_batch.real
            _ri[:, 1::2] = target_batch.imag
            target_batch = _ri

        # [CoilAdapter] v6.0 ``adapters.pre_model`` opt-in: when the YAML
        # declares a ``pre_model:`` chain (e.g. ``rss_coils_to_magnitude``
        # to collapse the m4raw dataset's cross-contrast 4-ch tensor down
        # to the 1-ch image-magnitude that image-domain reconstruction
        # models expect), apply it to **both** the input and target
        # *before* the strict DomainMismatch check below.  Without this
        # early application the audit would pass (it derives channels from
        # the declarative chain) but the runtime check at lines 633-663
        # would still see the un-adapted shape and crash.  Idempotent
        # adapters (rss_coils_to_magnitude is identity on 1-ch input)
        # mean re-running them later in the loss path is safe; chains
        # declared exclusively under ``pre_loss_pred`` / ``pre_loss_target``
        # (e.g. the bloch-cycle iFFT+magnitude+RSS chain that is **not**
        # idempotent) are untouched here because they live on different
        # hooks.
        if self.adapter_chains.get("pre_model"):
            if input_batch_prepared is not None:
                input_batch_prepared = self.apply_adapters("pre_model", input_batch_prepared)
            if target_batch is not None:
                target_batch = self.apply_adapters("pre_model", target_batch)

        # [FIX] Channel adaptation guard: strict domain mismatch check.
        # - actual != expected: Raise ValueError to prevent doomed GPU runs.
        # We no longer apply 1x1 convs or zero-padding, as this destroys
        # physics integrity (e.g. multi-coil k-space into single-channel models).
        #
        # Complex-stacking strategies (distillation / virtual-fiducial family)
        # IGNORE ``input_batch`` and synthesise the model input internally from
        # the target — they real-stack the digital-twin-corrupted complex image
        # via cat([real, imag]), so the model sees 2× the raw dataset channels
        # (e.g. a 1-ch rss_image magnitude → 2-ch model input). Checking the raw
        # ``input_batch_prepared`` width against ``in_channels`` here is checking
        # a tensor the model never consumes, so it wrongly rejected the correct
        # in_channels=2 for eval_c2/eval_c3/eval_c7/exp_c4 (cluster smoke
        # 20260605). The real check is deferred to the generator's first conv
        # (and to the audit's strategy-aware domain_alignment). Mirror of
        # config_health_checker._COMPLEX_STACKING_STRATEGY_MARKERS — keep in sync.
        _own_input_markers = (
            "virtual_fiducial_strategy",
            "motion_meta_strategy",
            "vf_admm_strategy",
            "ib_vf_strategy",
            "distillation_strategy",
        )
        _strategy_class = getattr(self.config.training, "strategy_class", None) or ""
        _builds_own_model_input = any(m in _strategy_class for m in _own_input_markers)
        if (
            input_batch_prepared is not None
            and hasattr(self.config.model, "in_channels")
            and not _builds_own_model_input
        ):
            expected_ch = self.config.model.in_channels
            actual_ch = input_batch_prepared.shape[1]
            if actual_ch != expected_ch:
                raise ValueError(
                    f"[DomainMismatch] Model expects {expected_ch} input channels, "
                    f"but dataset provided {actual_ch} channels. "
                    "Please configure coil combination (e.g., RSS or SENSE) in your experiment "
                    "or ensure the dataset matches the model's expected domain."
                )

        # Target batch channel mismatch check (skipped for complex-stacking
        # strategies: they real-stack the clean target internally too, so the
        # raw target width here is half the model's out_channels — see the
        # input-side note above).
        if (
            target_batch is not None
            and hasattr(self.config.model, "out_channels")
            and not _builds_own_model_input
        ):
            expected_target_ch = self.config.model.out_channels
            actual_target_ch = target_batch.shape[1]
            if actual_target_ch != expected_target_ch:
                # Distributional / parametric heads emit more channels than the
                # target BY DESIGN and self-compute the likelihood from them:
                #  - evidential_unet → 4 params (mean, var, alpha, beta) per 1-ch target
                #  - strategies with ``predicts_distribution_params`` (heteroscedastic
                #    [mean, logvar]; variational priors) → 2-ch out per 1-ch target.
                # The strategy validates its own channel contract (e.g.
                # HeteroscedasticULFStrategy._as_mean_logvar), so the strict width
                # guard must defer to it rather than reject (mrixfields b29).
                if self.config.model.model_type == "evidential_unet" or getattr(
                    self, "predicts_distribution_params", False
                ):
                    pass
                else:
                    raise ValueError(
                        f"[DomainMismatch] Model outputs {expected_target_ch} channels, "
                        f"but target dataset provided {actual_target_ch} channels. "
                        "Cannot reconcile shape for loss computation."
                    )

        # ── Auto debug-snapshot at the first few training steps ──────
        # Captures (input, prepared_input, target) for every paradigm so
        # the same diagnostics (shape/dtype/range/NaN/IFFT-preview) are
        # produced for GAN / VAE / recon / SSL / cycle-Bloch / etc.
        # Strategies may also call self.save_debug_snapshot directly to
        # capture paradigm-specific intermediates (e.g. noisy_images
        # for diffusion).
        try:
            current_step = int(kwargs.get("iteration", 0) or 0)
            self.save_debug_snapshot(
                {
                    "input_raw": input_batch,
                    "input_prepared": input_batch_prepared,
                    "target": target_batch,
                },
                step=current_step,
                tag="first_steps",
                # Every tensor here is the batch as the dataloader delivered it,
                # captured before any strategy-side normalization -- so none of
                # them is log-compressed, whatever the arm declares.
                log_scaled_keys=set(),
                # The canonical-key contract, stated in the artifact itself.
                # ``input_prepared`` is captured BEFORE the forward pass, so for
                # a strategy that degrades further inside the step it is NOT the
                # tensor the model receives; saying so here is what stops a
                # reader concluding the arm was fed fully-sampled data (the
                # experiment_11 report that produced this contract).
                extra=self._first_steps_extra(epoch),
            )
        except Exception as _exc:
            logger.debug("Auto debug-snapshot skipped: %s", _exc)

        # Build closure that performs the forward pass and loss computation
        def g_closure() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """g_closure.

            Returns:
                tuple[torch.Tensor, dict[str, torch.Tensor]]: Description.
            """
            losses, metrics_tensor = self._compute_losses(
                input_batch_prepared, target_batch, epoch, batch=batch, **kwargs
            )
            loss_total = losses.get("g_total_loss")
            if loss_total is None:
                loss_total = losses.get("g_loss_total")
                if loss_total is None:
                    raise RuntimeError("No g_total_loss or g_loss_total found in losses dict.")

            # Save detailed components for metric tracking (detached)
            # Use safe detach to handle both Tensors and non-tensor scalars
            detached_losses = {
                k: v.detach() if hasattr(v, "detach") else v for k, v in losses.items()
            }
            self._last_step_metrics = self._handle_anomalies(detached_losses, metrics_tensor, epoch)

            return loss_total

        step_config = {
            "optimizer": self.env.opt_g,
            "closure": g_closure,
            "model": self.generator_model,
            "name": "generator",
        }

        return [step_config]

    def get_last_metrics(self) -> dict[str, Any]:
        """Return the detached component metrics from the most recent step closure.

        Values stay **on-device** (#707). This used to be
        ``{k: float(v) ...}``, and `float(cuda_tensor)` IS `.item()` -- one host
        round-trip per component metric. `training_loop` calls this on EVERY
        iteration, outside the `log_interval` gate, so a GAN step publishing ~8-15
        component metrics paid 8-15 syncs per step and discarded the result on
        every non-logging step.

        What makes that notable is that the closures already keep these tensors
        on-device *deliberately*, with a "no sync: NN#9" comment naming this method
        as where conversion moved to. The conversion moved; the per-step call site
        did not, so the net sync count never changed.

        The loop's own batched conversion (`training_loop.py`, gated on
        `iteration % log_interval`) is now the only converter. Both consumers
        tolerate tensors: the loop detaches and defers, and `runners/run_explicit`
        already `.item()`s for its CSV and formats 0-dim tensors directly.
        """
        return dict(getattr(self, "_last_step_metrics", {}))

    def declared_metric_keys(self) -> frozenset[str]:
        """Keys this strategy stamps that no config traversal can derive (#1682).

        The training CSV header is built ONCE at startup from the config
        (``training_loop`` ~:920) and the row writer is
        ``csv.DictWriter(..., extrasaction="ignore")`` -- so a stamped value
        whose key is absent from that header is measured and then silently
        thrown away. Most stamped keys ARE derivable: a loss named in
        ``config.losses`` arrives under its own name. A few are not, because the
        producer renames between the knob and the metric --
        ``losses.reconstruction.lambda_pre_dc_kspace`` is stamped as
        ``pre_dc_kspace_l1``. No resolver reading the config can bridge that
        rename, so the PRODUCER declares the key here instead of the header
        builder guessing it.

        Declare a key only under the same condition the step actually stamps it.
        An always-empty column is pitfall #15 in artifact form, and the header
        builder deliberately dropped two families of promised-but-unpopulated
        columns for exactly that reason.

        Base declares nothing, so this is a no-op for every strategy that does
        not override it.
        """
        return frozenset()

    # ------------------------------------------------------------------
    # Strategy-owned learnable state (checkpoint round-trip)
    # ------------------------------------------------------------------
    #
    # A strategy is NOT an ``nn.Module``, but several strategies build their own
    # learnable modules / parameters (e.g. the inline ``BeltramiSFCBlock`` in
    # ``adaptive_sfc_hssc``, ``spin_sde``'s diffusion ``nn.Parameter``, the ib_vf
    # critics, ``mri_slam`` trajectory deltas) and register them on ``opt_g`` via
    # ``add_param_group`` so they train. The checkpoint path persists only
    # ``env.generator``; without the seam below those weights are NOT saved, so a
    # resume re-inits them randomly while the optimizer state — keyed by parameter
    # INDEX, not value — is restored against the fresh weights (a silent corrupt
    # resume). These methods round-trip that owned state as a sibling checkpoint
    # key. See ``docs/strategy_owned_state_checkpointing_design_2026_06.rst``.

    def _env_owned_param_ids(self) -> set[int]:
        """``id()`` of every parameter owned by the environment (generator /
        discriminator / loss modules). Owned state is anything NOT in this set —
        so a freshly-built strategy module is captured while a module that merely
        aliases the generator is excluded. ``isinstance`` guards keep it safe when
        ``env`` is a unit-test mock (a ``MagicMock`` is not an ``nn.Module``)."""
        ids: set[int] = set()
        env = getattr(self, "env", None)
        if env is None:
            return ids
        candidates: list[Any] = [
            getattr(env, "generator", None),
            getattr(env, "discriminator", None),
        ]
        losses = getattr(env, "losses", None)
        if isinstance(losses, dict):
            candidates += list(losses.values())
        for module in candidates:
            if isinstance(module, nn.Module):
                ids.update(id(p) for p in module.parameters())
        return ids

    def strategy_state_dict(self) -> dict[str, Any]:
        """Serialisable state for every strategy-OWNED module / parameter.

        Auto-discovered from instance attributes (sorted for determinism) so a new
        strategy that builds an owned module is checkpointed WITHOUT a per-strategy
        edit. Objects reachable from ``self.env`` are excluded — they round-trip
        through their own checkpoint paths and must not be double-saved. Handles
        both ``nn.Module`` (state-dict) and bare ``nn.Parameter`` (e.g. spin_sde's
        diffusion coefficient).
        """
        env_ids = self._env_owned_param_ids()
        modules: dict[str, Any] = {}
        params: dict[str, Any] = {}
        for name in sorted(vars(self)):
            obj = vars(self)[name]
            if isinstance(obj, nn.Parameter):
                if id(obj) not in env_ids:
                    params[name] = obj.detach().cpu()
            elif isinstance(obj, nn.Module):
                # Owns state iff it has >=1 parameter the env does not.
                if any(id(p) not in env_ids for p in obj.parameters()):
                    modules[name] = obj.state_dict()
        return {"modules": modules, "params": params}

    def load_strategy_state_dict(self, state: dict[str, Any] | None) -> None:
        """Restore owned modules / parameters captured by ``strategy_state_dict``.

        A no-op for ``None``/empty state (older checkpoints, or strategies that own
        nothing), so it is forward/backward compatible. Modules are restored via
        ``load_state_dict``; bare parameters via an in-place ``copy_`` onto the
        live (already re-created in ``__init__``) parameter so the optimizer's
        param references stay valid."""
        if not state:
            return
        for name, sd in state.get("modules", {}).items():
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.load_state_dict(sd)
        for name, tensor in state.get("params", {}).items():
            param = getattr(self, name, None)
            if isinstance(param, nn.Parameter):
                with torch.no_grad():
                    param.copy_(tensor.to(param.device))

    # ------------------------------------------------------------------
    # Hook methods for subclasses to override
    # ------------------------------------------------------------------

    def _zero_gradients(self) -> None:
        """Zero gradients for all optimizers. Override to customize."""
        opt_g = self.env.opt_g
        if opt_g is not None:
            self.optimizer_stepper.zero_grad(opt_g)
        opt_d = self.env.opt_d
        if opt_d is not None:
            self.optimizer_stepper.zero_grad(opt_d)

    @staticmethod
    def _resolve_legacy_batch(batch: Any, kwargs: dict) -> Any:
        """F6 (2026-05-17 round 5) — resolve the ``batch`` dict from kwargs.

        Legacy strategies (mri_slam, diffeomorphic_recon, coord_kspace_gen,
        and others — see TODO/audit/smoke_audit_20260516.md §F6) declared
        ``_compute_losses_impl(self, batch, *args, **kwargs)`` because they
        intentionally read metadata from the batch dict (e.g.
        ``batch.get("subject_id")``, ``batch.get("velocity_field")``).

        The base orchestrator at ``BaseTrainingStrategy._compute_losses``
        calls the implementation with named kwargs:
        ``_compute_losses_impl(input_batch=..., target_batch=...,
        epoch=..., **kwargs)``. Under that calling convention, the
        legacy ``batch`` positional parameter is unbound (it doesn't
        match any kwarg name), producing
        ``TypeError: missing 1 required positional argument: 'batch'``.

        This helper centralizes the resolution logic the JEPA strategy
        already implements inline: look for ``batch`` first (the original
        kwarg name), then ``input_batch`` (the orchestrator's new name).
        Return whatever was found, or ``None`` if neither yields a dict.

        Returns:
            The resolved batch (typically a dict from the dataloader
            collate, but may also be a Tensor for legacy callers).
        """
        if isinstance(batch, dict):
            return batch
        # Use explicit ``is not None`` rather than ``a or b``: the values may be
        # tensors, and ``bool(multi_element_tensor)`` raises "Boolean value of
        # Tensor is ambiguous". (pick_present is the mixin-level SSOT for this,
        # but importing it here would create a base↔mixin cycle.)
        batch_kw = kwargs.get("batch")
        if batch_kw is None:
            batch_kw = kwargs.get("input_batch")
        return batch_kw

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for the training step (strategy-specific implementation).

        This is the primary extension point for custom training strategies. Subclasses
        must implement this method to define their specific loss computation logic.

        Args:
            input_batch: Input tensor (typically low-resolution or undersampled MRI data).
                Shape: (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.
            target_batch: Target tensor (typically fully-sampled or high-resolution MRI).
                Shape: Same as input_batch.
            epoch: Current training epoch (0-indexed), useful for curriculum learning.
            **kwargs: Additional keyword arguments that may include:
                - batch: Original batch dict from DataLoader (for metadata access).
                - step: Global training step counter.
                - model: Generator model (nn.Module).

        Returns:
            Dictionary of loss tensors with string keys. MUST include:
                - 'g_total_loss': Scalar tensor for total generator loss (used for backward).
                - Additional component losses (e.g., 'loss_l1', 'loss_perceptual', 'loss_adversarial').

            All tensors must:
                - Be scalar (0-dimensional) or single-element tensors.
                - Require gradients (for loss components contributing to backprop).
                - Be on the same device as input_batch.

        Raises:
            NotImplementedError: If subclass does not implement this method.
            RuntimeError: If loss computation fails (e.g., NaN/Inf values, shape mismatches).

        Note:
            - This method is called within an autocast context (for AMP).
            - DO NOT call .backward() here - handled by _backward_and_step().
            - Return raw loss tensors, not .item() values.
            - Use self.config.losses.* for loss weights (SSOT principle).

        Example:
            >>> def _compute_losses_impl(self, input_batch, target_batch, epoch, **kwargs):
            ...     model = kwargs.get('model', self.generator_model)
            ...     pred = model(input_batch)
            ...
            ...     loss_l1 = F.l1_loss(pred, target_batch)
            ...     loss_l2 = F.mse_loss(pred, target_batch)
            ...
            ...     # Weighted combination
            ...     g_total_loss = (
            ...         self.config.losses.reconstruction.lambda_l1 * loss_l1 +
            ...         self.config.losses.reconstruction.lambda_l2 * loss_l2
            ...     )
            ...
            ...     return {
            ...         'g_total_loss': g_total_loss,
            ...         'loss_l1': loss_l1,
            ...         'loss_l2': loss_l2,
            ...     }
        """
        raise NotImplementedError("Subclasses must implement _compute_losses_impl")

    def sync_scheduled_loss_weights(self) -> None:
        """Publish ``loop_state.loss_weight_overrides`` into the loss computer.

        This is the single paradigm-agnostic seam for the ``loss_schedule:``
        feature. The training loop calls it each step BEFORE ``train_step`` so
        EVERY strategy (not just reconstruction) honors a dynamic loss
        curriculum: previously the override was copied only inside
        ``ReconstructionTrainingStrategy``, making the whole feature a silent
        no-op for GAN / diffusion / VAE / etc. while still logging fire events
        and writing provenance (CLAUDE.md pitfall #16). Calling it here, on the
        same ``self.loss_computer`` object the strategy later passes to
        ``compute(...)``, covers strategies regardless of whether they override
        ``train_step`` / ``_compute_losses`` / ``_compute_losses_impl``.

        No-op when the strategy owns no ``loss_computer`` or has no overrides
        (empty dict => static-config behavior). The computer consults the map
        first, uncached, in ``_get_loss_weight``.
        """
        loss_computer = getattr(self, "loss_computer", None)
        if loss_computer is None:
            return
        loop_state = getattr(self, "loop_state", None)
        loss_computer.scheduled_weights = getattr(loop_state, "loss_weight_overrides", None) or {}

    def _compute_losses(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Orchestrate loss computation with Automatic Mixed Precision (AMP).

        This template method wraps the strategy-specific `_compute_losses_impl`
        to ensure all operations occur within the correct AMP autocast context.
        It also handles fail-safes for missing total loss keys.

        Subclasses should typically override `_compute_losses_impl` instead of this method.

        Args:
            input_batch: The input tensor batch.
            target_batch: The target tensor batch.
            epoch: The current epoch index.
            **kwargs: Additional arguments passed to the implementation hook.

        Returns:
            A tuple containing:
                - `losses`: A dictionary of loss tensors (including `g_total_loss`).
                - `metrics_tensor`: A dictionary of additional metric tensors (currently empty).
        """
        # Initialize empty metrics tensor dict (will be populated by subclass)
        metrics_tensor: dict[str, torch.Tensor] = {}

        # Defensive: drop any duplicate ``input_batch`` / ``target_batch`` /
        # ``epoch`` keys that may have travelled in ``**kwargs`` from
        # upstream callers (e.g. a strategy that builds a ``batch`` dict
        # and passes both the dict AND its constituents). Passing them
        # again as keyword args would raise
        # ``TypeError: _compute_losses_impl() got multiple values for
        # argument 'input_batch'`` (observed 2026-05-14 smoke run on
        # ``configurable_unet``).
        for _dup in ("input_batch", "target_batch", "epoch"):
            kwargs.pop(_dup, None)

        # Capture the generator output for the paradigm-agnostic ``model_output``
        # snapshot (parity with the pre-forward ``first_steps`` auto-snapshot),
        # and auto-emit it on exit -- budget-capped by
        # ``logging.snapshots.max_calls``, and suppressed when the strategy
        # already emitted its own richer ``model_output`` snapshot during the
        # forward. ``module=None`` keeps this path on ``env.generator``, which is
        # what it has always hooked; the 13 sites that bypass this wrapper enter
        # the SAME context manager with their own module (#1190).
        #
        # The total-loss normalization below sits INSIDE the block so a missing
        # key still raises before any snapshot is written -- the emit is placed
        # after the manager's try/finally, so an exception escaping here skips it.
        #
        # The dimension-contract guard stays HERE rather than inside the context
        # manager: it is the training contract, and it can raise under
        # ``enforce``. Installing it from the shared manager would silently apply
        # it to the seven validation paths that manager also serves.
        self._ensure_input_contract_guard()
        # The model-INPUT contract's per-step resets. `_capture_model_output`
        # below owns the model-OUTPUT half (`_model_output_snapshot_done`,
        # `_capture_gen_output`) since #1190, so only the input flags are set
        # here. Both the declaration and the "the strategy emitted it itself"
        # flag are per-step: a stale value from the previous step must never
        # satisfy this step's check.
        self._model_input_snapshot_done = False
        self._declared_model_input = None
        self._declared_channel_segments = None
        self._declared_log_scaled_keys = None
        with self._capture_model_output(
            module=None,
            input_batch=input_batch,
            target_batch=target_batch,
            step=int(kwargs.get("iteration", 0) or 0),
        ):
            with self.amp_helper.get_autocast_context():
                losses = self._compute_losses_impl(
                    input_batch=input_batch,
                    target_batch=target_batch,
                    epoch=epoch,
                    **kwargs,
                )

            # Normalize the total-loss key. Strategies historically used a
            # mix of "g_total_loss" / "g_loss_total" / "loss_total" / "total"
            # (per TODO/audit/07_strategies_v6_2_v6_3_exotic.md F2 + F18).
            # The pre-fix behaviour silently substituted ``torch.tensor(0.0)``
            # when the canonical key was missing, producing a zero gradient
            # that looks like training is happening — exactly the
            # CLAUDE.md #9 silent fallback class. Now we accept any of the
            # synonyms but raise if NONE are present.
            if "g_total_loss" not in losses:
                for synonym in ("g_loss_total", "loss_total", "total"):
                    aliased = losses.get(synonym)
                    if aliased is not None:
                        losses["g_total_loss"] = aliased
                        break
                else:
                    raise ValueError(
                        f"_compute_losses_impl returned no recognized total-loss key. "
                        f"Expected one of {{'g_total_loss', 'g_loss_total', "
                        f"'loss_total', 'total'}}; got {sorted(losses.keys())}. "
                        "Strategies must include a scalar total loss for backprop."
                    )

        # Emit the model INPUT for strategies that build it inside the step, and
        # raise if one declares the carve-out without handing anything over.
        # Placed in the wrapper so an overriding ``_compute_losses_impl`` cannot
        # drop it -- that override is what let the diffusion pointer dangle.
        self._snapshot_declared_model_input(
            step=int(kwargs.get("iteration", 0) or 0),
        )

        return losses, metrics_tensor

    def _backward_and_step(
        self, losses: dict[str, torch.Tensor], epoch: int, step: int = 0
    ) -> None:
        """DEPRECATED: Backward pass and optimizer step are now managed by Trainer."""
        pass

    def _handle_anomalies(
        self,
        losses: dict[str, torch.Tensor],
        metrics_tensor: dict[str, torch.Tensor],
        epoch: int,
    ) -> dict[str, float]:
        """Handle NaN/Inf detection with fail-fast behavior."""
        batch_results = self._convert_metrics_to_floats(losses)
        return batch_results

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @property
    def _weight_table(self) -> LossWeightTable:
        """This arm's resolved loss weights — built once from the frozen config."""
        table = getattr(self, "_loss_weight_table", None)
        if table is None:
            table = build_loss_weight_table(getattr(self.config, "losses", None))
            self._loss_weight_table = table
        return table

    @property
    def _declared_loss_keys(self) -> frozenset[str]:
        """The loss names THIS strategy must guarantee in its per-step loss dict.

        The CSV header is built ONCE at startup and the row writer drops any key the
        header lacks (``extrasaction="ignore"``), so a declared loss that never reaches
        the dict leaves a blank column for the whole run (#1919).

        Derived from the loss-weight SSOT, never hand-listed. The lists this replaced
        read one schema block each, so they were empty for every arm that declares its
        objective in ``losses.image_losses`` / ``losses.kspace_losses`` — 58 of the
        kspace_filling cohort. Builder-owned losses are subtracted via the existing
        skip-set SSOT so they are not zero-filled here and then reported missing.
        """
        keys = getattr(self, "_declared_loss_key_cache", None)
        if keys is None:
            # Function-local, mirroring ``infrastructure/loss_audit.py``: the builder
            # module owns the skip set, and importing it at module scope would put a
            # builder import into every strategy.
            from spectramr.infrastructure.training.builders.loss_builder import (
                STRATEGY_MANAGED_LOSSES,
            )

            keys = active_loss_names(self._weight_table) - STRATEGY_MANAGED_LOSSES
            self._declared_loss_key_cache = keys
        return keys

    @property
    def _warmup_gated_loss_keys(self) -> frozenset[str]:
        """Declared keys that are LEGITIMATELY absent while the warm-up gate holds.

        ``adversarial`` is the common case: it is declared active, but
        :func:`resolve_loss_weight` returns 0.0 while ``iteration <
        warmup_iterations``, so the branch that would produce it never runs. Absent is a
        state to report, never to infer (NN18) — but a *gated* absence is expected, and
        reporting it would emit one line per step for the whole warm-up.
        """
        keys = getattr(self, "_warmup_gated_loss_key_cache", None)
        if keys is None:
            table = self._weight_table
            keys = frozenset(
                name
                for name in self._declared_loss_keys
                if name in table and table[name].warmup_gated
            )
            self._warmup_gated_loss_key_cache = keys
        return keys

    def _log_loss_objective(self, prefix: str) -> None:
        """Announce this arm's declared objective, once, at strategy setup.

        The banner is the only surface that shows the weight the user actually
        wrote: a strategy-inline term such as ``pre_dc_kspace`` reaches the CSV
        under a renamed key, and a warm-up-gated term reads back 0.0 from any
        resolver called at iteration 0.
        """
        for line in format_loss_objective(self._weight_table, prefix=prefix):
            self.logging_service.log_info(line)

    def _ensure_declared_losses_present(
        self, loss_dict: dict[str, torch.Tensor], *, iteration: int
    ) -> frozenset[str]:
        """Zero-fill declared losses this step did not produce; report the real drops.

        Presence is tested on the CANONICAL name: an arm declaring ``mse`` resolves to
        ``l2`` in the table while the computer may key either, and comparing raw strings
        would fill a duplicate column and report a phantom miss every step.

        Returns the names that were filled, so callers and tests can assert on it
        instead of scraping the log.
        """
        declared = self._declared_loss_keys
        if not declared:
            return frozenset()

        present = {canonical_loss_name(key) for key in loss_dict}
        missing = declared - present
        if missing:
            device = next(
                (v.device for v in loss_dict.values() if isinstance(v, torch.Tensor)),
                None,
            )
            for loss_name in missing:
                loss_dict[loss_name] = torch.zeros((), device=device)

        # Anything missing that the warm-up gate does not explain is reported: the arm
        # declared it at a live weight and this step did not produce it under that name.
        # Two shapes reach here and the message must not pick one — the term may be
        # genuinely uncomputed (a silent drop, NN3), or computed and keyed differently
        # (``losses.diffusion.lambda_mse`` canonicalises to ``l2`` while the computer
        # emits ``diffusion``). Reported once per distinct set, not per step (NN9).
        expected_absent = (
            self._warmup_gated_loss_keys
            if iteration < self._weight_table.warmup_iterations
            else frozenset()
        )
        unexplained = missing - expected_absent
        if unexplained and unexplained != getattr(self, "_reported_missing_losses", None):
            self._reported_missing_losses = unexplained
            self.logging_service.log_warning(
                f"[Train] Declared but absent from the loss dict ({len(unexplained)}): "
                f"{sorted(unexplained)} — filled with 0.0 so the CSV column exists. "
                "Either nothing on this path computes them, or the producer keys them "
                "under a different name; both report as zeros for the whole run."
            )
        return frozenset(missing)

    def _get_loss_weight(self, loss_name: str, epoch: int = 0, **kwargs: Any) -> float:
        """The weight for ``loss_name``. Delegates to the loss-weight SSOT.

        This was one of eight resolvers, each with its own precedence and its own magic
        default table, so the same YAML resolved differently depending on which code path
        ran and an undeclared loss silently materialised at 1.0 (pitfall #9). Resolution
        — including the warm-up gate and the curriculum override — now lives in
        :mod:`spectramr.models.losses.weights`, and an undeclared loss RAISES.
        """
        return resolve_loss_weight(
            self._weight_table,
            loss_name,
            scheduled=scheduled_overrides(self),
            iteration=kwargs.get("iteration", 1_000_000),
        )

    def _verify_strategy_config(self, expected_modes: tuple[str, ...]) -> None:
        """Verify that the strategy configuration is valid.

        Legacy method - strategy selection is now handled by TrainingStrategyFactory.
        This method only verifies config exists, not mode coherence.

        Args:
            expected_modes: Ignored (legacy parameter)
        """
        if self.config is None:
            raise ValueError("Training environment must have a valid config")
        # Strategy selection now schema-driven via TrainingStrategyFactory
        # No mode validation needed here

    def _log_config_features(self, logging_service: Any) -> None:
        """Log advanced configuration features if present."""

        # Log physics knobs if present (compact summary, not full object dump)
        if self.config.physics:
            phys = self.config.physics
            parts = []
            if phys.data_consistency:
                dc = phys.data_consistency
                parts.append(f"DC({dc.method}, w={dc.weight})")
            if phys.kspace:
                ks = phys.kspace
                parts.append(
                    f"kspace(recon={ks.enable_kspace_recon}, hermitian={ks.enforce_hermitian_symmetry})"
                )
            if phys.compressed_sensing and phys.compressed_sensing.enabled:
                parts.append(f"CS(R={phys.compressed_sensing.acceleration_factor})")
            if phys.pinn and phys.pinn.enabled:
                parts.append(f"PINN({phys.pinn.pde_type})")
            if phys.digital_twin and phys.digital_twin.enabled:
                parts.append("DigitalTwin")
            summary = " | ".join(parts) if parts else "default"
            logging_service.log_info(
                f"🔬 Physics config: {summary}",
                model_type=self.env.model_type,
            )

        # Log specialized knobs from config (direct access)
        # Note: These are optional sections in TrainingSettings v6.0
        optional_sections = ["contrast", "volumetric", "adaptation", "low_field"]
        for section in optional_sections:
            if hasattr(self.config, section) and getattr(self.config, section):
                logging_service.log_info(
                    f"📋 {section} config: enabled",
                    model_type=self.env.model_type,
                )

    def _get_gradient_clipping_config(self) -> tuple[float, bool]:
        """Extract gradient clipping configuration.

        Returns:
            (clip_value, enable_clipping) tuple
        """
        opt_config = self.config.optimization
        clip_value = (
            opt_config.gradient.clip.value if opt_config.gradient.clip.value is not None else 1.0
        )
        enable_clip = opt_config.gradient.clip.enabled
        return clip_value, enable_clip

    def _get_gradient_logging_config(self) -> tuple[bool, int]:
        """Extract gradient logging configuration.

        Returns:
            (should_log_gradients, log_interval) tuple
        """
        log_config = self.config.logging
        return log_config.log_gradients, log_config.intervals.log

    def _compute_gradient_norm(
        self, model: torch.nn.Module, enable_clip: bool, clip_value: float
    ) -> torch.Tensor:
        """Compute and optionally clip gradient norm.

        Returns the norm as a **device tensor**, deliberately unmaterialised.
        ``float(total_norm)`` is a host synchronise, and this runs once per
        optimiser step: a Scalene profile of experiment_11_attention_none
        charged it 4.64 % of the run (~43 s over 150 steps) while no production
        caller reads the value — ``amp_policy.backward_and_step`` and
        ``step_executor`` both invoke the clip hook as a bare statement. The
        caller decides when a host copy is actually owed; see
        :meth:`_clip_and_log_gradients`.

        Args:
            model: PyTorch model
            enable_clip: Whether to actually clip gradients
            clip_value: Maximum gradient norm

        Returns:
            Total gradient norm, on the parameters' device.
        """
        if enable_clip:
            return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

    def _check_and_log_gradient_anomalies(
        self,
        total_norm: float,
        model_name: str,
        epoch: int,
        enable_clip: bool,
        clip_value: float,
        should_log: bool,
        check_anomalies: bool,
    ) -> None:
        """Check for gradient anomalies and log warnings.

        Args:
            total_norm: Total gradient norm
            model_name: Name of the model for logging
            epoch: Current epoch
            enable_clip: Whether gradient clipping is enabled
            clip_value: Gradient clipping threshold
            should_log: Whether logging is enabled
            check_anomalies: Whether to check for anomalies
        """
        if not hasattr(self, "logging_service"):
            return

        GRADIENT_EXPLOSION_THRESHOLD = 1000.0
        GRADIENT_VANISHING_THRESHOLD = 1e-7

        # Check for gradient explosion
        # F-GRADEXPLODE-HINT / 2026-05-20 — point out whether clipping
        # will catch this. A 1596 gradient at iter-0 is alarming when
        # clipping is disabled (training will diverge) but benign when
        # the YAML sets ``optimization.gradient.clip.enabled=true``
        # with ``gradient_clip_value≤1`` (the clip silently rescales to
        # 1 before the optimizer step). The smoke-time warning was
        # noisy because both cases logged the same message.
        if total_norm > GRADIENT_EXPLOSION_THRESHOLD:
            clipping_active = (
                enable_clip and clip_value > 0 and clip_value < GRADIENT_EXPLOSION_THRESHOLD
            )
            if clipping_active:
                hint = (
                    f" — clipping active ({clip_value:g}); "
                    f"this is being rescaled before the optimizer step."
                )
            else:
                hint = (
                    " — clipping is DISABLED or set above the explosion "
                    "threshold; training will likely diverge. Set "
                    "``optimization.gradient.clip.enabled: true`` and "
                    "``optimization.gradient.clip.value: 1.0`` in the YAML."
                )
            # F-GRADEXPLODE-RATELIMIT / 2026-05-20 — `rectified_flow` and
            # `stable_diffusion_adapter` produced 10+ per-step explosion
            # warnings in smoke 20260519 even though clipping was on and
            # rescaling them. When clipping IS active the warning is
            # informational and per-step repetition is pure noise.
            # Rate-limit to one notice per strategy instance for the
            # clipping-active case; the clipping-disabled case still
            # fires every step (because there IS no protection and the
            # user needs to act).
            if clipping_active:
                if getattr(self, "_warned_grad_explode_under_clip", False):
                    return
                self._warned_grad_explode_under_clip = True
            self.logging_service.log_warning(
                f"GRADIENT EXPLOSION DETECTED in {model_name}: total_norm={total_norm:.4f}{hint}",
                model_type=self.env.model_type,
                epoch=epoch,
            )

        # Check for vanishing gradients
        if total_norm > 0 and total_norm < GRADIENT_VANISHING_THRESHOLD and check_anomalies:
            self.logging_service.log_warning(
                f"VANISHING GRADIENTS DETECTED in {model_name}: total_norm={total_norm:.10f}",
                model_type=self.env.model_type,
                epoch=epoch,
            )

        # Log clipping event
        if enable_clip and total_norm > clip_value:
            if should_log or check_anomalies:
                self.logging_service.log_info(
                    f"Gradient clipping triggered for {model_name}: norm {total_norm:.4f} -> {clip_value:.4f}",
                    model_type=self.env.model_type,
                    epoch=epoch,
                )

        # Log general gradient stats
        if should_log:
            self.logging_service.log_info(
                f"Gradient norms for {model_name}: total={total_norm:.4f}",
                model_type=self.env.model_type,
                epoch=epoch,
            )

    def _clip_and_log_gradients(
        self,
        model: torch.nn.Module,
        epoch: int,
        step: int,
        model_name: str = "model",
    ) -> float | None:
        """Clip gradients and log statistics/anomalies.

        Refactored for clarity: delegates to helper methods for config extraction,
        gradient computation, and anomaly logging.

        Returns:
            The gradient norm when it was actually materialised, or ``None`` when
            this step deliberately skipped the host copy. It used to return
            ``0.0`` there, which is a VALID gradient norm -- the sentinel and a
            genuinely vanished gradient were the same value, and the ``-> float``
            annotation said the number always meant something. No caller reads
            the result today (``step_executor`` calls it as a bare statement,
            ``training_loop`` passes it through as ``clip_and_log_fn``, and the
            federated DP override just forwards it), so this narrows the contract
            to the truth before someone starts logging it.
        """
        # Extract configuration
        clip_value, enable_clip = self._get_gradient_clipping_config()
        log_gradients, log_interval = self._get_gradient_logging_config()

        should_log = log_gradients and (step % log_interval == 0)
        check_anomalies = step % 100 == 0

        # Compute gradient norm (with optional clipping)
        if enable_clip or should_log or check_anomalies:
            norm_tensor = self._compute_gradient_norm(model, enable_clip, clip_value)
        else:
            # Nothing to clip and nothing to log: no norm was computed, so there
            # is no number to report (see the Returns note above).
            return None

        # Materialise only when a warning could actually be emitted, because
        # ``float()`` on a CUDA tensor blocks until the queue drains.
        #
        # Every step that gets here without one of these flags did so because
        # ``enable_clip`` is set, i.e. the norm was computed to CLIP with, and
        # the clip is already done — the host copy would serve only
        # ``_check_and_log_gradient_anomalies``. Deferring it moves the
        # explosion check from every step to the existing ``check_anomalies``
        # cadence; with clipping active that warning is rate-limited to once
        # per strategy instance (``_warned_grad_explode_under_clip``) anyway, so
        # what is lost is latency on a once-per-run notice. With clipping OFF
        # nothing changes: that path only reaches here on a log/anomaly step,
        # having returned 0.0 above otherwise.
        if not (should_log or check_anomalies):
            # The norm exists as a device tensor but was deliberately not
            # materialised; reporting 0.0 would claim a measurement never taken.
            return None

        total_norm = float(norm_tensor)

        # Check for anomalies and log results
        self._check_and_log_gradient_anomalies(
            total_norm=total_norm,
            model_name=model_name,
            epoch=epoch,
            enable_clip=enable_clip,
            clip_value=clip_value,
            should_log=should_log,
            check_anomalies=check_anomalies,
        )

        return total_norm
