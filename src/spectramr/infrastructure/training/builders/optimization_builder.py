"""Optimization Builder

Creates optimizers, learning rate schedulers, and gradient scalers
for training based on configuration.
"""

import logging
from typing import Any, ClassVar

import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from spectramr.core.parameter_free import declares_parameter_free
from spectramr.infrastructure.builders.context import (
    BuilderContext,
    accepts_builder_context,
)
from spectramr.infrastructure.training.optimizer_registry import (
    accepted_optimizer_kwargs,
    create_optimizer,
)
from spectramr.infrastructure.training.optimizer_resolution import (
    build_optimizer_from_spec,
    resolve_optimizer_spec,
)
from spectramr.infrastructure.training.scheduler_resolution import (
    SchedulerSpec,
    build_scheduler_from_spec,
    resolve_scheduler_spec,
)

from .base import Builder

logger = logging.getLogger(__name__)

# ``_EXPLICITLY_HANDLED_OPTIMIZER_KWARGS`` / ``_unconsumed_optimizer_kwargs``
# lived here and are gone. They validated ``optimizer_kwargs`` against
# ``inspect.signature(cls.__init__)`` — the right idea — but they also stripped
# ``betas``/``eps``/``momentum`` from the declared set before validating, on the
# assumption that ``_create_optimizer`` forwarded those itself. It only partly
# did, so ``optimizer_kwargs: {betas: [...]}`` was stripped here AND dropped
# there: two independent paths to the same silent loss.
#
# ``optimizer_resolution.resolve_optimizer_spec`` now owns the whole decision
# (which knobs exist, which the optimizer accepts, which were explicitly asked
# for), and it raises on a declared-but-unaccepted key instead of stripping.


class OptimizationBuilder(Builder):
    """Builds optimization components (optimizers, schedulers, scalers).

    Creates optimizers for each trainable model, learning rate schedulers,
    and gradient scalers for mixed precision training.

    Attributes:
        _config: Training configuration
        _models: Dictionary of models to optimize
        _optimizers: Dictionary of created optimizers
        _schedulers: Dictionary of created schedulers
        _scaler: Optional gradient scaler for AMP

    Example:
        >>> builder = OptimizationBuilder(config, models=models)
        >>> optimizers, schedulers, scaler = (builder
        ...     .build_optimizers()
        ...     .build_schedulers()
        ...     .build_grad_scaler()
        ...     .build())
    """

    # ============================================================================
    # SSOT: Single-method factory interface for direct optimizer creation
    # ============================================================================
    #: Factory defaults, applied when the caller passes ``None``. The signature
    #: takes sentinels rather than these values directly so that "the caller
    #: asked for it" stays distinguishable from "the caller took the default" --
    #: the same split ``resolve_optimizer_spec`` gets from ``model_fields_set``,
    #: and the thing that decides whether an unaccepted knob raises or is dropped.
    _FACTORY_DEFAULTS: ClassVar[dict[str, Any]] = {
        "betas": (0.5, 0.999),
        "weight_decay": 0.0,
        "momentum": 0.9,
    }

    @staticmethod
    def create_single_optimizer(
        parameters,
        learning_rate: float = 1e-4,
        optimizer_type: str = "adam",
        betas: tuple | None = None,
        weight_decay: float | None = None,
        momentum: float | None = None,
        alpha: float | None = None,
    ) -> Optimizer:
        """Create a single optimizer directly (factory-like interface).

        For callers that have parameters and a learning rate but no
        ``OptimizationConfigSchema`` (test-time adaptation, inner optimisation
        loops, INR fitting). Config-driven construction goes through
        ``resolve_optimizer_spec`` / ``build_optimizer_from_spec`` instead.

        **Which knobs reach the optimizer is decided by the registry's
        accepted-kwarg table, not by a family tuple here.** This method used to
        forward ``betas`` for ``adam``/``adamw``/``nadam``/``radam`` only, and
        ``weight_decay`` unconditionally -- verbatim the defect
        ``optimizer_resolution``'s module docstring says it was written to
        delete. Only the config path was converted, so the static factory kept
        the old behaviour and 13 of the 21 registered names were mishandled:
        ``lbfgs``/``rprop``/``sparseadam`` raised ``TypeError`` on the
        unconditional ``weight_decay``, while ``adamax``/``lamb``/``lion``/
        ``adam8bit``/``adamw8bit``/``schedulefree_adamw`` silently lost their
        ``betas`` and ``rmsprop``/``lars``/``schedulefree_sgd`` their
        ``momentum``. Forwarding is now signature-driven via
        :func:`accepted_optimizer_kwargs`, so a newly-registered optimizer works
        without editing anything here.

        ``alpha`` is deliberately NOT generalised, and that is not an oversight.
        It is the one name in this signature whose *meaning* is not portable:
        RMSprop's ``alpha`` is the squared-gradient smoothing constant (0.99),
        ASGD's is the power for its eta update (0.75). Forwarding by name match
        would quietly hand ASGD a smoothing constant as a decay exponent. The
        config schema has no ``alpha`` field for exactly this reason, so keeping
        it RMSprop-scoped is what agrees with the SSOT.

        Args:
            parameters: Model parameters to optimize.
            learning_rate: Learning rate. Forwarded unconditionally, as
                ``build_optimizer_from_spec`` does -- every registered optimizer
                accepts ``lr``.
            optimizer_type: Any name in the ``OptimizerRegistry`` (21 today),
                not just the four this docstring used to list.
            betas: Adam-family beta pair. Defaults to ``(0.5, 0.999)``; forwarded
                to every optimizer whose constructor accepts ``betas``.
            weight_decay: Weight decay. Defaults to ``0.0``; forwarded to every
                optimizer that accepts it, dropped for those that do not.
            momentum: Momentum factor. Defaults to ``0.9``; forwarded to every
                optimizer that accepts ``momentum`` (sgd, rmsprop, lars,
                schedulefree_sgd).
            alpha: RMSprop's smoothing constant. Defaults to ``0.99`` and is
                forwarded to RMSprop only -- see above.

        Returns:
            Optimizer: Created optimizer.

        Raises:
            ValueError: If ``optimizer_type`` is unknown, or if a knob the
                caller passed EXPLICITLY cannot be accepted by that optimizer.
                A knob left at its default is dropped quietly instead: the
                caller never asked for it, so silently honouring nothing is not
                a broken promise (the policy ``_partition_kwargs`` applies).
        """
        optimizer_type = optimizer_type.lower()
        # Raises on an unknown name, with the available list -- no silent
        # fallback (NN#3). Doubles as the dispatch-table membership check.
        accepted = accepted_optimizer_kwargs(optimizer_type)

        requested = {"betas": betas, "weight_decay": weight_decay, "momentum": momentum}
        explicit = {k for k, v in requested.items() if v is not None}
        candidates = {
            k: (v if v is not None else OptimizationBuilder._FACTORY_DEFAULTS[k])
            for k, v in requested.items()
        }
        # Meaning is not portable (see the docstring), so this one stays scoped
        # to the optimizer whose knob it actually is.
        if optimizer_type == "rmsprop":
            candidates["alpha"] = 0.99 if alpha is None else alpha
            if alpha is not None:
                explicit.add("alpha")
        elif alpha is not None:
            raise ValueError(
                f"create_single_optimizer(alpha={alpha!r}) is RMSprop-only, but "
                f"optimizer_type={optimizer_type!r}. ASGD also takes an 'alpha', "
                "with an unrelated meaning (eta-update power, not a smoothing "
                "constant), so forwarding it by name match would silently "
                "misconfigure it. Pass it via the optimizer's own construction "
                "path if you really mean that knob."
            )

        rejected = sorted(k for k in explicit if k not in accepted)
        if rejected:
            raise ValueError(
                f"optimizer_type={optimizer_type!r} does not accept {rejected}, "
                f"but they were passed explicitly. It accepts {sorted(accepted)}. "
                "A declared knob that cannot reach the optimizer is a silent "
                "no-op (pitfall #15)."
            )

        kwargs = {k: v for k, v in candidates.items() if k in accepted}
        dropped = sorted(k for k in candidates if k not in accepted)
        if dropped:
            logger.info(
                "optimizer %s does not accept %s; dropped (left at factory "
                "defaults, not requested by the caller).",
                optimizer_type,
                dropped,
            )
        return create_optimizer(optimizer_type, parameters, lr=learning_rate, **kwargs)

    @accepts_builder_context
    def __init__(self, ctx: BuilderContext) -> None:
        """Initialize OptimizationBuilder.

        Args:
            ctx: Builder context carrying ``config`` (the immutable training
                configuration) and ``models`` (dictionary of models to create
                optimizers for).
        """
        self._config = ctx.config
        self._models = ctx.models
        self._optimizers: dict[str, Optimizer] = {}
        #: Resolved specs, keyed by role. Kept so provenance can record what
        #: each optimizer actually received rather than what was declared.
        self._optimizer_specs: dict[str, Any] = {}
        self._schedulers: dict[str, _LRScheduler] = {}
        self._scaler: GradScaler | None = None
        #: A declared parameter-free generator gets no optimizer; empty is then valid.
        self._parameter_free = False

    def build_optimizers(self) -> "OptimizationBuilder":
        """Create optimizers for each trainable model.

        Creates single optimizer for most paradigms (reconstruction, VAE, diffusion)
        or dual optimizers for GAN (generator + discriminator).

        Returns:
            self: For method chaining

        Raises:
            ValueError: If optimizer creation fails
        """
        # Try to get training_mode from config.training (extra field) or config.task or deprecated top-level
        training_mode = "reconstruction"
        if self._config.training:
            training_mode = getattr(
                self._config.training,
                "training_mode",
                getattr(self._config.training, "task", "reconstruction"),
            )

        if not training_mode:
            if hasattr(self._config, "training_mode"):
                training_mode = self._config.training_mode
            elif hasattr(self._config, "task"):
                training_mode = self._config.task
            else:
                training_mode = "reconstruction"

        training_mode = str(training_mode).lower()
        opt_config = self._config.optimization

        try:
            # Dual optimizers for GAN
            if training_mode in [
                "gan",
                "latent_gan",
                "cycle_bloch",
                "disentangled",
                "disentangled_vae",
                # MICCAI MRIxFields2026 idea 4.2: field_cocycle is a GAN-family arm
                # (generator = AnatomyFieldRenderer subclass, field-conditioned
                # discriminator), so it needs opt_g + opt_d. Absent from the line-186
                # raise-list, so a discriminator-less variant degrades gracefully.
                "field_cocycle",
            ]:
                # Generator optimizer
                if "generator" not in self._models:
                    raise ValueError("Generator model not found for GAN training")

                lr_multiplier_g = 1.0
                if (
                    self._config.training
                    and hasattr(self._config.training, "gan")
                    and self._config.training.gan
                ):
                    gan_conf = self._config.training.gan
                    if isinstance(gan_conf, dict):
                        lr_multiplier_g = gan_conf.get("generator_lr_multiplier", 1.0)
                    else:
                        lr_multiplier_g = (
                            gan_conf.generator_lr_multiplier
                            if hasattr(gan_conf, "generator_lr_multiplier")
                            else 1.0
                        )

                opt_g = self._create_optimizer(
                    self._models["generator"],
                    opt_config,
                    lr_multiplier=lr_multiplier_g,
                    role="generator",
                )
                # SSOT: Use canonical keys expected by TrainingEnvironment properties
                self._optimizers["opt_g"] = opt_g
                logger.info(
                    f"Created generator optimizer: {opt_config.optimizer.type} "
                    f"(lr={opt_config.optimizer.learning_rate * lr_multiplier_g})"
                )

                # Discriminator optimizer
                if "discriminator" not in self._models:
                    if training_mode in ["gan", "latent_gan"]:
                        raise ValueError("Discriminator model not found for GAN training")
                    else:
                        logger.info("No discriminator found, skipping opt_d creation")
                        return self

                # Build ``opt_d`` through the shared path (see #1670).
                self._build_discriminator_optimizer(opt_config)

            # Single optimizer for other paradigms
            else:
                # Find main model (generator, encoder, or first model).
                # Use explicit `is not None` checks instead of an `or` chain:
                # torch.compile's OptimizedModule defines `__len__` proxying to
                # the wrapped module, so `bool(model)` falls back to len() and
                # raises TypeError ("DisentangledMRI does not support len()")
                # when the wrapped module isn't Sized.
                main_model = self._models.get("generator")
                if main_model is None:
                    main_model = self._models.get("encoder")
                if main_model is None:
                    main_model = next(iter(self._models.values()))

                if declares_parameter_free(main_model):
                    # A classical solver: nothing to optimise, so no opt_g and no
                    # scheduler. train.py admits such a run only under --validate-only.
                    self._parameter_free = True
                    logger.info("Generator is parameter-free: no optimizer built.")
                    return self
                optimizer = self._create_optimizer(main_model, opt_config, role="generator")
                self._optimizers["opt_g"] = optimizer
                logger.info(
                    f"Created optimizer under canonical key 'opt_g': "
                    f"{opt_config.optimizer.type} (lr={opt_config.optimizer.learning_rate})"
                )

                # A discriminator outside the GAN-family allowlist above still
                # needs an optimizer. The allowlist decides whether a critic is
                # REQUIRED; the models dict decides whether one is PRESENT, and
                # only the second question is the right one to ask here (#1670).
                # Additive: an arm with no discriminator is untouched.
                if "discriminator" in self._models:
                    logger.info(
                        "Discriminator present under training_mode=%r (outside the "
                        "GAN-family list): building opt_d so the critic can be stepped.",
                        training_mode,
                    )
                    self._build_discriminator_optimizer(opt_config)

        except Exception as e:
            raise ValueError(f"Failed to create optimizers: {e}") from e

        return self

    def _build_discriminator_optimizer(self, opt_config) -> None:
        """Create ``opt_d`` for the discriminator already present in ``self._models``.

        Extracted so the GAN-family branch and every other paradigm share ONE
        construction path. It used to be inline in the GAN branch only, which is
        how a non-GAN arm with a configured discriminator ended up with a critic
        and no optimizer (#1670): ``ModelBuilder.build_discriminator`` is keyed
        on CONFIGURATION (``model.discriminator_component``), the caller was
        keyed on the paradigm NAME, and the two disagreed. ``env.opt_d`` is
        ``optimizers.get("opt_d")``, so the disagreement surfaced as ``None``
        and died later in ``StepExecutor``, naming the wrong cause.

        ``fit()`` never had that gate -- it builds ``opt_d`` whenever a
        discriminator is present, for any paradigm -- so the same config worked
        programmatically and failed under ``spectramr train``. Two owners for one
        question (non-negotiable 17); this is the surviving one.
        """
        lr_multiplier_d = 1.0
        if (
            self._config.training
            and hasattr(self._config.training, "gan")
            and self._config.training.gan
        ):
            gan_conf = self._config.training.gan
            if isinstance(gan_conf, dict):
                lr_multiplier_d = gan_conf.get("discriminator_lr_multiplier", 1.0)
            else:
                lr_multiplier_d = (
                    gan_conf.discriminator_lr_multiplier
                    if hasattr(gan_conf, "discriminator_lr_multiplier")
                    else 1.0
                )

        # TTUR: an explicit ``optimization.optimizer.discriminator_learning_rate``
        # overrides the generator LR for D. Previously this schema knob
        # was read by nobody, so D silently trained at G's LR.
        disc_lr = opt_config.optimizer.discriminator_learning_rate
        opt_d = self._create_optimizer(
            self._models["discriminator"],
            opt_config,
            role="discriminator",
            lr_multiplier=lr_multiplier_d,
            lr_override=disc_lr,
        )
        # SSOT: Use canonical keys expected by TrainingEnvironment properties
        self._optimizers["opt_d"] = opt_d
        effective_disc_lr = (
            disc_lr if disc_lr is not None else opt_config.optimizer.learning_rate * lr_multiplier_d
        )
        logger.info(
            f"Created discriminator optimizer: {opt_config.optimizer.type} (lr={effective_disc_lr})"
        )

    def build_schedulers(self) -> "OptimizationBuilder":
        """Create LR schedulers for optimizers.

        The scheduler request is resolved by the SSOT
        :func:`spectramr.infrastructure.training.scheduler_resolution.resolve_scheduler_spec`,
        which reads the **flat** ``scheduler:`` block every config in the corpus
        actually declares, honours ``lr_scheduler_strategy``, defaults a cosine
        period to ``training.max_iterations``, applies ``warmup_steps``, and
        RAISES on any knob the resolved factory cannot consume.

        This method used to read ``scheduler["type"]`` / ``scheduler["kwargs"]``
        only. No config in the corpus declares either, so every declared
        parameter was discarded and every arm silently got
        ``CosineAnnealingLR(T_max=100)`` — issue #533.

        Returns:
            self: For method chaining

        Raises:
            ConfigurationError: propagated from the resolver on an unroutable or
                doubly-declared scheduler knob.
        """
        spec = resolve_scheduler_spec(
            self._config.optimization,
            max_iterations=getattr(self._config.training, "max_iterations", None),
        )
        if spec is None:
            logger.debug("No scheduler configured")
            return self

        self._scheduler_spec = spec
        for name, optimizer in self._optimizers.items():
            self._schedulers[name] = build_scheduler_from_spec(spec, optimizer)
            logger.info("Created %s scheduler for %s: %s", spec.name, name, spec.as_provenance())

        return self

    @property
    def scheduler_spec(self) -> "SchedulerSpec | None":
        """The resolved scheduler request, for provenance stamping."""
        return getattr(self, "_scheduler_spec", None)

    def build_grad_scaler(self) -> "OptimizationBuilder":
        """Deprecated no-op: the run's scaler is owned by the AMP policy.

        This used to construct a ``GradScaler("cuda")`` and hand it to the
        pipeline, where **nothing ever called it**. The scaler that scales this
        run's losses is built by ``MixedPrecisionIntegrationHelper`` on the
        strategy (a ``NativeScaler``, or a ``ComplexGradScaler`` for complex
        arms) -- a different object, of a different class.

        The consequence was not a leaked allocation: the checkpoint writer
        persisted ``scaler_state`` from the unused one, so every fp16 checkpoint
        carried an untouched scale of 65536 and a zero growth-tracker while the
        live scale was silently dropped. On resume the run re-converged its loss
        scale from scratch, overflowing and skipping steps for the first few
        hundred iterations -- with a ``scaler_state`` key present and restored,
        so the checkpoint looked complete.

        Kept as a no-op rather than deleted because it is a documented step of
        the public builder chain; ``checkpointing.capture.resolve_scaler`` is now
        the single reader.
        """
        return self

    def _create_optimizer(
        self,
        model_or_parameters,
        config,
        lr_multiplier: float = 1.0,
        lr_override: float | None = None,
        role: str = "generator",
    ) -> Optimizer:
        """Internal helper that resolves and builds one optimizer.

        Args:
            model_or_parameters: Model to extract parameters from, or raw parameters
            config: Optimization configuration
            lr_multiplier: Learning rate multiplier (for GAN dual LR)
            lr_override: Explicit learning rate that bypasses
                ``config.learning_rate * lr_multiplier`` (used to honour
                ``optimization.optimizer.discriminator_learning_rate`` for TTUR).
            role: Which optimizer this is; recorded on the spec so provenance can
                distinguish the generator's settings from the discriminator's.

        Returns:
            Optimizer: Created optimizer
        """
        # One resolver, one dispatch table.
        #
        # This method used to hand-pick which knobs reached torch: a family tuple
        # for momentum, a second family tuple for eps, and `with_betas` set
        # unconditionally but honoured by the leaf builder ONLY for Adam/AdamW —
        # so beta1/beta2 were silently dropped for nadam, radam and adamax, while
        # weight_decay was forwarded to optimizers that cannot take it.
        # ``resolve_optimizer_spec`` derives the forwarded set from the
        # optimizer's own signature instead, so a newly-registered optimizer
        # works without editing a tuple here.
        spec = resolve_optimizer_spec(
            config,
            role=role,
            lr_multiplier=lr_multiplier,
            lr_override=lr_override,
            model=(model_or_parameters if isinstance(model_or_parameters, nn.Module) else None),
        )
        # Lazily initialised: several suites construct this builder via
        # ``__new__`` and set only the attributes they need, so relying on
        # ``__init__`` having run would make the spec ledger a test-only crash
        # rather than a recording.
        if not hasattr(self, "_optimizer_specs"):
            self._optimizer_specs = {}
        self._optimizer_specs[role] = spec
        return build_optimizer_from_spec(spec, model_or_parameters)

    def validate(self) -> "OptimizationBuilder":
        """Validate optimization components.

        Ensures at least one optimizer is created.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If no optimizers created
        """
        if not self._optimizers and not self._parameter_free:
            raise ValueError("No optimizers created")

        logger.info("Optimization validation passed")
        return self

    def build(
        self,
    ) -> tuple[dict[str, Optimizer], dict[str, _LRScheduler], GradScaler | None]:
        """Return all optimization components.

        Returns:
            Tuple containing:
                - Dictionary of optimizers
                - Dictionary of schedulers
                - Optional gradient scaler

        Raises:
            ValueError: If validation fails
        """
        if not self._optimizers and not self._parameter_free:
            raise ValueError("No optimizers created. Call build_optimizers() first.")

        return self._optimizers, self._schedulers, self._scaler
