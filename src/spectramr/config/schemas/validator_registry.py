"""Centralized validator registry for configuration validation rules.

This module provides a registry-based approach to validation, enabling:
- Extensible validation rules (add new validators without modifying core code)
- Composable validators (combine multiple validators for different scenarios)
- Centralized configuration (all rules in one place, easy to maintain)
- Type-safe validation (validators are strongly typed and composable)
- Dynamic validator selection (choose validators based on config content)

The registry pattern allows new validation rules to be registered without
modifying existing code, adhering to the Open/Closed Principle.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """A single validation rule with metadata.

    Attributes:
        name: Unique identifier for the rule (e.g., "batch_size_positive")
        validator_fn: Callable that performs validation
        category: Category for organizing rules (e.g., "data", "optimization")
        description: Human-readable description of what is validated
        severity: "error" (fails validation) or "warning" (informational)
        applies_to: Optional list of training modes this rule applies to
        applies_to_models: Optional list of model types this rule applies to
    """

    name: str
    validator_fn: Callable[[dict[str, Any]], list[str]]
    category: str
    description: str
    severity: str = "error"
    applies_to: list[str] | None = None
    applies_to_models: list[str] | None = None

    def applies(self, config: dict[str, Any]) -> bool:
        """Check if this rule applies to the given configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if rule should be applied, False otherwise
        """
        # Check training mode applicability
        if self.applies_to:
            training_mode = config.get("training", {}).get("training_mode")
            if training_mode not in self.applies_to:
                return False

        # Check model type applicability
        if self.applies_to_models:
            model_type = config.get("model", {}).get("model_type")
            if model_type not in self.applies_to_models:
                return False

        return True

    def execute(self, config: dict[str, Any]) -> list[str]:
        """Execute the validation rule on a configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation messages (errors/warnings)
        """
        try:
            return self.validator_fn(config)
        except Exception as e:
            logger.error(f"Error executing validation rule '{self.name}': {e}")
            return [f"Validation rule '{self.name}' failed: {e}"]


class ValidatorRegistry:
    """Registry for all configuration validation rules.

    Provides centralized, extensible validation with rule categorization,
    conditional rule application, and composable validator execution.

    Usage:
        registry = ValidatorRegistry()

        # Run all applicable validators
        errors = registry.validate(config)

        # Run validators for specific category
        data_errors = registry.validate(config, category="data")

        # Check if specific rule passes
        if registry.check_rule(config, "batch_size_positive"):
            print("Batch size is valid")
    """

    def __init__(self) -> None:
        """Initialize an empty validator registry."""
        self._validators: dict[str, ValidationRule] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, rule: ValidationRule, *, override: bool = False) -> None:
        """Register a new validation rule.

        Args:
            rule: ValidationRule instance to register
            override: Permit replacing an existing rule of the same name. Off by
                default so a duplicate registration is a loud error, not a
                silent last-write-wins overwrite (pitfall #9 — the docstring
                always promised this, the body only warned).

        Raises:
            ValueError: If rule name already exists and ``override`` is False.
        """
        if rule.name in self._validators and not override:
            raise ValueError(
                f"Validator '{rule.name}' is already registered. Rename one of "
                "the two rules, or pass override=True to replace deliberately."
            )

        self._validators[rule.name] = rule

        # Index by category
        if rule.category not in self._categories:
            self._categories[rule.category] = []
        if rule.name not in self._categories[rule.category]:
            self._categories[rule.category].append(rule.name)

    def get(self, name: str) -> ValidationRule | None:
        """Get a validator by name.

        Args:
            name: Name of the validator

        Returns:
            ValidationRule if found, None otherwise
        """
        return self._validators.get(name)

    def list_validators(self, category: str | None = None) -> list[ValidationRule]:
        """List all validators, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of ValidationRule instances
        """
        if category:
            rule_names = self._categories.get(category, [])
            return [self._validators[name] for name in rule_names]
        return list(self._validators.values())

    def list_categories(self) -> list[str]:
        """List all validator categories.

        Returns:
            Sorted list of category names
        """
        return sorted(self._categories.keys())

    def validate(
        self,
        config: dict[str, Any],
        category: str | None = None,
        severity: str | None = None,
    ) -> list[tuple[str, str]]:
        """Validate configuration using applicable rules.

        Runs all registered validators that apply to the given configuration,
        collecting all error and warning messages.

        Args:
            config: Configuration dictionary to validate
            category: Optional category to validate (e.g., "data")
            severity: Optional severity filter ("error" or "warning")

        Returns:
            List of (rule_name, message) tuples for each validation issue
        """
        results: list[tuple[str, str]] = []

        validators = self.list_validators(category=category)

        for rule in validators:
            # Filter by severity if specified
            if severity and rule.severity != severity:
                continue

            # Skip if rule doesn't apply
            if not rule.applies(config):
                continue

            # Execute validator
            messages = rule.execute(config)
            for message in messages:
                results.append((rule.name, message))

        return results

    def check_rule(self, config: dict[str, Any], rule_name: str) -> bool:
        """Check if a specific rule passes validation.

        Args:
            config: Configuration dictionary to validate
            rule_name: Name of the rule to check

        Returns:
            True if rule passes (no messages), False otherwise
        """
        rule = self.get(rule_name)
        if not rule:
            logger.warning(f"Rule '{rule_name}' not found in registry")
            return False

        if not rule.applies(config):
            logger.debug(f"Rule '{rule_name}' does not apply to this config")
            return True  # Consider as passing if doesn't apply

        messages = rule.execute(config)
        return len(messages) == 0

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with stats about registered validators
        """
        categories_stats = {cat: len(rules) for cat, rules in self._categories.items()}

        severity_stats = {
            "error": len([r for r in self._validators.values() if r.severity == "error"]),
            "warning": len([r for r in self._validators.values() if r.severity == "warning"]),
        }

        return {
            "total_validators": len(self._validators),
            "categories": categories_stats,
            "severity": severity_stats,
        }


# Global singleton registry instance
_registry: ValidatorRegistry | None = None


def get_validator_registry() -> ValidatorRegistry:
    """Get or create the global validator registry.

    Returns:
        Global ValidatorRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ValidatorRegistry()
        _initialize_default_validators(_registry)
        # audit_plan_novel.md — Tier-1 checks for the seven novel ideas.
        # Imported lazily here to avoid a top-level cycle (the audit-plan
        # module imports ValidationRule from this file).
        try:
            from spectramr.config.schemas.audit_plan_novel_validators import (
                register_audit_plan_novel_validators,
            )

            register_audit_plan_novel_validators(_registry)
        except ImportError as e:
            logger.warning(
                "Failed to register audit_plan_novel validators: %s. "
                "Tier-1 checks for the seven novel ideas will not fire.",
                e,
            )
        # SFC/conformal + fMRI/MRF 2026 checks (8 rules across the new arms).
        try:
            from spectramr.config.schemas.audit_plan_novel_2026_validators import (
                register_audit_plan_novel_2026_validators,
            )

            register_audit_plan_novel_2026_validators(_registry)
        except ImportError as e:
            logger.warning(
                "Failed to register audit_plan_novel_2026 validators: %s. "
                "SFC/conformal + fMRI/MRF Tier-1/2 checks will not fire.",
                e,
            )
    return _registry


def _initialize_default_validators(registry: ValidatorRegistry) -> None:
    """Initialize default validation rules in the registry.

    Args:
        registry: ValidatorRegistry to populate with default rules
    """
    # Data validation rules
    registry.register(
        ValidationRule(
            name="batch_size_positive",
            category="data",
            description="Batch size must be positive",
            validator_fn=_validate_batch_size_positive,
        )
    )

    registry.register(
        ValidationRule(
            name="num_workers_valid",
            category="data",
            description="Number of workers must be non-negative",
            validator_fn=_validate_num_workers,
        )
    )

    registry.register(
        ValidationRule(
            name="dataset_config",
            category="data",
            description="Dataset configuration validation",
            severity="warning",
            validator_fn=_validate_dataset_config,
        )
    )

    # Optimization validation rules
    registry.register(
        ValidationRule(
            name="learning_rate_positive",
            category="optimization",
            description="Learning rate must be positive",
            validator_fn=_validate_learning_rate,
        )
    )

    registry.register(
        ValidationRule(
            name="epochs_valid",
            category="optimization",
            description="Epochs must be >= 1",
            validator_fn=_validate_epochs,
        )
    )

    registry.register(
        ValidationRule(
            name="acceleration_factors",
            category="optimization",
            description="Acceleration configuration compatibility",
            severity="warning",
            validator_fn=_validate_acceleration_factors,
        )
    )

    # Training validation rules
    registry.register(
        ValidationRule(
            name="training_mode_specified",
            category="training",
            description="Training mode must be specified",
            validator_fn=_validate_training_mode_specified,
        )
    )

    # Model validation rules
    registry.register(
        ValidationRule(
            name="model_channels",
            category="model",
            description="Model input/output channels within bounds",
            validator_fn=_validate_model_in_out_channels,
        )
    )

    registry.register(
        ValidationRule(
            name="discriminator_architecture",
            category="model",
            description="GAN discriminator configuration",
            severity="warning",
            validator_fn=_validate_discriminator_architecture,
            applies_to_models=["gan"],
        )
    )

    # Loss-weight validation rules. `category` was "objectives" (a v4.0
    # block the v1.0 root schema rejects, issue #933); repointed to "losses",
    # the block the validator_fn actually reads.
    registry.register(
        ValidationRule(
            name="loss_weights",
            category="losses",
            description=(
                "lambda_adv/l1/l2/ssim/perceptual/kl within valid ranges -- the "
                "6 of 76 losses.*.lambda_* fields LOSS_WEIGHT_RANGES covers, "
                "not loss weights generally"
            ),
            validator_fn=_validate_loss_weights_positive,
        )
    )

    # Physics validation rules
    registry.register(
        ValidationRule(
            name="data_consistency",
            category="physics",
            description="Data consistency configuration",
            severity="warning",
            validator_fn=_validate_data_consistency_config,
        )
    )

    registry.register(
        ValidationRule(
            name="kspace_model_config",
            category="physics",
            description="K-space model configuration consistency",
            severity="warning",
            validator_fn=_validate_kspace_model_config,
        )
    )

    # Metrics validation rules
    registry.register(
        ValidationRule(
            name="metric_domain_compatibility",
            category="metrics",
            description="Metrics compatible with model/data configuration",
            severity="warning",
            validator_fn=_validate_metric_domain_compatibility,
        )
    )

    # Checkpoint validation rules
    registry.register(
        ValidationRule(
            name="checkpoint_config",
            category="checkpoint",
            description="Checkpoint configuration validity",
            severity="warning",
            validator_fn=_validate_checkpoint_config,
        )
    )

    # Logging validation rules
    registry.register(
        ValidationRule(
            name="logging_config",
            category="logging",
            description="Logging configuration validity",
            severity="warning",
            validator_fn=_validate_logging_config,
        )
    )

    # Device validation rules
    registry.register(
        ValidationRule(
            name="device_config",
            category="device",
            description="Device configuration validity",
            severity="warning",
            validator_fn=_validate_device_config,
        )
    )


# ============================================================================
# Default Validator Functions
# ============================================================================


def _validate_batch_size_positive(config: dict[str, Any]) -> list[str]:
    """Validate that batch size is positive."""
    messages = []
    # Accepts BOTH spellings: `data.batch_size` still parses (the rename is a
    # `fold`, not a removal), but a config already written canonically nests it
    # under `loader`, and reading only the flat key reported "got None".
    _data = config.get("data", {}) or {}
    batch_size = (_data.get("loader", {}) or {}).get("batch_size", _data.get("batch_size"))

    if batch_size is None or batch_size <= 0:
        messages.append(f"data.loader.batch_size must be positive, got {batch_size}")

    return messages


def _validate_num_workers(config: dict[str, Any]) -> list[str]:
    """Validate that num_workers is non-negative."""
    messages = []
    _data = config.get("data", {}) or {}
    num_workers = (_data.get("loader", {}) or {}).get("num_workers", _data.get("num_workers"))

    if num_workers is not None and num_workers < 0:
        messages.append(f"data.loader.num_workers must be non-negative, got {num_workers}")

    return messages


def _validate_learning_rate(config: dict[str, Any]) -> list[str]:
    """Validate that learning rate is positive."""
    messages = []
    # Accepts BOTH spellings, for the same reason as `_validate_batch_size_positive`
    # above: phase 8 folded the flat key under `optimizer`, so a resolved
    # `model_dump()` carries only the nested one and reading the flat key alone
    # reported "got None" for every arm in the corpus.
    _opt = config.get("optimization", {}) or {}
    lr = (_opt.get("optimizer", {}) or {}).get("learning_rate", _opt.get("learning_rate"))

    if lr is None or lr <= 0:
        messages.append(f"optimization.optimizer.learning_rate must be positive, got {lr}")

    return messages


def _validate_epochs(config: dict[str, Any]) -> list[str]:
    """Validate the training budget through the one shared predicate.

    ``optimization.epochs`` is the older spelling; ``training.epochs`` the live one.
    The rule itself (and the calibration modes it exempts) lives in
    :mod:`spectramr.config.training_budget`, which the audit witness reads too.
    """
    from spectramr.config.training_budget import zero_budget_defect

    training = dict(config.get("training", {}) or {})
    if training.get("epochs") is None and config.get("optimization", {}).get("epochs") is not None:
        training["epochs"] = config["optimization"]["epochs"]
    message = zero_budget_defect(training)
    return [message] if message else []


def _validate_training_mode_specified(config: dict[str, Any]) -> list[str]:
    """Validate that training mode is specified.

    Requires either strategy_class (v5.0+) or training_mode (v4.0) to be present.
    """
    messages = []
    training_config = config.get("training") or {}
    strategy_class = training_config.get("strategy_class")
    training_mode = training_config.get("training_mode")

    if not strategy_class and not training_mode:
        messages.append(
            "Training mode must be specified (either training.strategy_class or training.training_mode)"
        )

    return messages


#: Metrics the computer grades on complex k-space. ``psnr`` resolves its data
#: range from the target's peak magnitude under ``domain="kspace"``
#: (``core/metrics/evaluation_metrics.py``); the rest are k-space-native error
#: measures. SSIM raises ``TypeError`` on complex tensors and the perceptual
#: metrics (lpips, hfen, ms_ssim, fid, ...) read a real intensity field: those
#: are what the rule below exists to catch.
_KSPACE_GRADEABLE_METRICS: frozenset[str] = frozenset(
    {"kspace_error", "phase_mse", "nmse", "mse", "mae", "psnr"}
)


def _declared_metric_names(metrics_config: dict[str, Any]) -> list[str]:
    """The metrics an arm computes: ``metrics.compute`` first, else its ``compute_*`` flags.

    The same precedence as ``MetricsMixin._extract_metrics_from_config``: a
    non-empty ``compute`` list wins outright, the flags are the fallback for
    arms that predate it, and ``NON_METRIC_FLAGS`` (master switches) select
    nothing.
    """
    from spectramr.core.metrics.flag_map import NON_METRIC_FLAGS, metric_for_flag

    compute = metrics_config.get("compute")
    if isinstance(compute, (list, tuple)) and compute:
        return [str(getattr(entry, "value", entry)) for entry in compute]
    return [
        metric_for_flag(key)
        for key, enabled in metrics_config.items()
        if key.startswith("compute_") and key not in NON_METRIC_FLAGS and enabled is True
    ]


def _validate_kspace_model_config(config: dict[str, Any]) -> list[str]:
    """Flag image-intensity metrics declared under ``metrics.domain: kspace``.

    Before 2026-09-03 this warned on every k-space model whose ``metrics.domain``
    was not ``image`` -- including the ten arms that compute only k-space-native
    metrics there (``kspace_error``, ``phase_mse``), which is the right domain
    for those, and the arms that declare no domain at all (the computer then
    grades in image space). The three arms that compute SSIM / LPIPS / HFEN on
    k-space read the same message as the ten correct ones. The rule now reads
    the declared metric set and names the offending ones.
    """
    messages: list[str] = []

    # Safely extract config sections, handling None values
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        model_config = {}

    metrics_config = config.get("metrics")
    if not isinstance(metrics_config, dict):
        metrics_config = {}

    data_config = config.get("data")
    if not isinstance(data_config, dict):
        data_config = {}

    model_type = model_config.get("model_type", "")
    target_domain = model_config.get("target_domain")
    dataset_type = data_config.get("dataset_type", "")
    metrics_domain = metrics_config.get("domain")

    # Check if this is a k-space model
    is_kspace_model = (
        "kspace" in model_type.lower() or target_domain == "kspace" or dataset_type == "kspace"
    )

    if not is_kspace_model:
        return messages

    # An absent domain grades in image space (``MetricsMixin`` defaults the
    # training-metric domain to "image"), so only a declared k-space domain
    # can carry an image-intensity metric.
    if str(getattr(metrics_domain, "value", metrics_domain) or "") != "kspace":
        return messages

    offending = sorted(
        {
            name
            for name in _declared_metric_names(metrics_config)
            if name not in _KSPACE_GRADEABLE_METRICS
        }
    )
    if offending:
        messages.append(
            f"metrics.domain is 'kspace' but {offending} grade a real intensity field "
            "(SSIM is undefined on complex tensors; the perceptual metrics read image "
            "intensities). Compute them in the image domain (metrics.domain: image with "
            "a transform such as ifft_magnitude), or declare only k-space-gradeable "
            f"metrics: {sorted(_KSPACE_GRADEABLE_METRICS)}."
        )

    return messages


def _validate_model_in_out_channels(config: dict[str, Any]) -> list[str]:
    """Validate model input/output channels are within bounds."""
    from spectramr.config.validation_constants import (
        MAX_IN_CHANNELS,
        MAX_OUT_CHANNELS,
        MIN_IN_CHANNELS,
        MIN_OUT_CHANNELS,
    )

    messages = []
    model_config = config.get("model", {})

    in_channels = model_config.get("in_channels")
    out_channels = model_config.get("out_channels")

    if in_channels is not None:
        if not (MIN_IN_CHANNELS <= in_channels <= MAX_IN_CHANNELS):
            messages.append(
                f"model.in_channels must be between {MIN_IN_CHANNELS} and {MAX_IN_CHANNELS}, got {in_channels}"
            )

    if out_channels is not None:
        if not (MIN_OUT_CHANNELS <= out_channels <= MAX_OUT_CHANNELS):
            messages.append(
                f"model.out_channels must be between {MIN_OUT_CHANNELS} and {MAX_OUT_CHANNELS}, got {out_channels}"
            )

    return messages


def _validate_acceleration_factors(config: dict[str, Any]) -> list[str]:
    """Validate acceleration-specific configuration."""
    messages: list[str] = []

    acceleration_config = config.get("acceleration")
    if acceleration_config is None:
        return messages

    amp_enabled = acceleration_config.get("amp_enabled", False)

    if amp_enabled:
        # Only the canonical nested knob is read — the flat ``optimization.lr``
        # alias was removed in v6.0 (NN#1) and is rejected by extra="forbid"
        # before reaching this validator, so a fallback to it would be dead code.
        lr = config.get("optimization", {}).get("learning_rate")

        if lr is not None and lr > 0.1:
            messages.append(
                f"AMP enabled with learning_rate={lr}. Consider using lower LR (< 0.1) for stability"
            )

    return messages


def _validate_data_consistency_config(config: dict[str, Any]) -> list[str]:
    """Validate data consistency configuration."""
    messages = []

    physics_config = config.get("physics") or {}
    dc_config = physics_config.get("data_consistency") or {}

    if dc_config.get("enabled"):
        weight = dc_config.get("weight")
        if weight is not None and not (0.0 <= weight <= 1.0):
            messages.append(
                f"physics.data_consistency.weight must be between 0.0 and 1.0, got {weight}"
            )

    return messages


def _validate_loss_weights_positive(config: dict[str, Any]) -> list[str]:
    """Validate that loss weights are within valid ranges.

    Reads the canonical v1.0 ``losses.*`` block (issue #933). The
    pre-#933 version read ``config["objectives"]`` -- a block
    ``extra="forbid"`` rejects at the root schema and which 0/647
    ``experiments/inprogress`` arms declare -- so this rule had never
    fired on a real config. ``LOSS_WEIGHT_RANGES`` is keyed by the
    canonical field name; see its module docstring for the one rename
    (``lambda_adversarial`` -> ``lambda_adv``) and the two dropped keys.
    """
    from spectramr.config.validation_constants import LOSS_WEIGHT_RANGES

    messages: list[str] = []
    losses_config = config.get("losses") or {}

    # GAN adversarial loss weight.
    gan_config = losses_config.get("gan") or {}
    lambda_adv = gan_config.get("lambda_adv")
    if lambda_adv is not None:
        min_val, max_val = LOSS_WEIGHT_RANGES.get("lambda_adv", (0.0, float("inf")))
        if not (min_val <= lambda_adv <= max_val):
            messages.append(
                f"losses.gan.lambda_adv out of range [{min_val}, {max_val}], got {lambda_adv}"
            )

    # Reconstruction loss weights.
    recon_config = losses_config.get("reconstruction") or {}
    for field in ("lambda_l1", "lambda_l2", "lambda_ssim", "lambda_perceptual"):
        value = recon_config.get(field)
        if value is None:
            continue
        min_val, max_val = LOSS_WEIGHT_RANGES.get(field, (0.0, float("inf")))
        if not (min_val <= value <= max_val):
            messages.append(
                f"losses.reconstruction.{field} out of range [{min_val}, {max_val}], got {value}"
            )

    # Latent (VAE-family) KL weight.
    latent_config = losses_config.get("latent") or {}
    lambda_kl = latent_config.get("lambda_kl")
    if lambda_kl is not None:
        min_val, max_val = LOSS_WEIGHT_RANGES.get("lambda_kl", (0.0, float("inf")))
        if not (min_val <= lambda_kl <= max_val):
            messages.append(
                f"losses.latent.lambda_kl out of range [{min_val}, {max_val}], got {lambda_kl}"
            )

    return messages


def _validate_metric_domain_compatibility(config: dict[str, Any]) -> list[str]:
    """Validate metrics are compatible with training mode and model type."""
    messages: list[str] = []

    validation_config = config.get("validation") or {}
    metrics = validation_config.get("metrics", [])

    if not metrics:
        return messages

    data_config = config.get("data") or {}
    patch_size = data_config.get("patch_size")

    # FID requires high-resolution images
    if "fid" in metrics:
        if patch_size is not None:
            # Handle both scalar and sequence patch_size
            min_patch = patch_size
            if isinstance(patch_size, (list, tuple)):
                # For MRI, we care about spatial dimensions (H, W)
                min_patch = min(patch_size[:2]) if len(patch_size) >= 2 else patch_size[0]

            if min_patch < 64:
                messages.append(
                    f"Metric 'fid' requires minimum patch_size of 64 (spatial), got {patch_size}"
                )

    # LPIPS designed for RGB/grayscale
    if "lpips" in metrics:
        model_config = config.get("model") or {}
        model_domain = model_config.get("model_domain", "image")
        # K-space models apply iFFT before metric evaluation, so channel count
        # in model config is irrelevant — LPIPS receives magnitude images.
        if model_domain != "kspace":
            out_channels = model_config.get("out_channels", model_config.get("in_channels"))
            # The LPIPS metric routes through adapt_to_rgb(ChannelMode.AUTO),
            # which collapses EVEN C (real/imag-interleaved multi-coil) to a
            # 1-channel RSS magnitude and replicates to 3 — so even-channel
            # output is handled and must NOT warn. AUTO only *raises* on odd
            # C != 1, 3, so the warning is legitimate only there (smoke audit
            # 2026-06-03, F7c; see core/metrics/channel_adapter.py AUTO mode).
            if out_channels is not None and out_channels > 3 and out_channels % 2 != 0:
                messages.append(
                    f"Metric 'lpips' designed for RGB/grayscale, model has "
                    f"{out_channels} output channels (odd > 3 cannot be RSS-adapted; "
                    f"use validation channel_mode=grayscale_mean to average channels)"
                )

    return messages


def _validate_discriminator_architecture(config: dict[str, Any]) -> list[str]:
    """Validate GAN discriminator configuration."""
    from spectramr.config.validation_constants import (
        MAX_GAN_DISCRIMINATOR_LAYERS,
        MIN_GAN_DISCRIMINATOR_LAYERS,
    )

    messages = []
    training_mode = config.get("training", {}).get("training_mode")

    if training_mode == "gan":
        model_config = config.get("model", {})
        discriminator_layers = model_config.get("discriminator_layers")

        if discriminator_layers is not None:
            if not (
                MIN_GAN_DISCRIMINATOR_LAYERS <= discriminator_layers <= MAX_GAN_DISCRIMINATOR_LAYERS
            ):
                messages.append(
                    f"discriminator_layers must be between {MIN_GAN_DISCRIMINATOR_LAYERS} and {MAX_GAN_DISCRIMINATOR_LAYERS}, got {discriminator_layers}"
                )

    return messages


def _validate_checkpoint_config(config: dict[str, Any]) -> list[str]:
    """Validate checkpoint configuration."""
    messages = []

    checkpoint_config = config.get("checkpoint", {})

    if checkpoint_config:
        save_interval = checkpoint_config.get("save_interval")
        if save_interval is not None and save_interval < 1:
            messages.append("checkpoint.save_interval must be at least 1")

        keep_top_k = checkpoint_config.get("keep_top_k")
        if keep_top_k is not None and keep_top_k < 1:
            messages.append("checkpoint.keep_top_k must be at least 1")

    return messages


def _validate_logging_config(config: dict[str, Any]) -> list[str]:
    """Validate logging configuration."""
    messages = []

    logging_config = config.get("logging", {})

    if logging_config:
        log_interval = logging_config.get("log_interval")
        if log_interval is not None and log_interval < 1:
            messages.append("logging.log_interval must be at least 1")

        log_level = logging_config.get("log_level")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level is not None and log_level not in valid_levels:
            messages.append(f"logging.log_level must be one of {valid_levels}, got {log_level}")

    return messages


def _validate_device_config(config: dict[str, Any]) -> list[str]:
    """Validate device configuration."""
    messages: list[str] = []

    acceleration_config = config.get("acceleration")
    if acceleration_config is None:
        return messages

    device = acceleration_config.get("device")

    if device is not None:
        valid_devices = ["cuda", "cpu", "mps"]
        if device not in valid_devices:
            messages.append(f"acceleration.device must be one of {valid_devices}, got {device}")

    return messages


def _validate_dataset_config(config: dict[str, Any]) -> list[str]:
    """Validate dataset configuration."""
    messages = []

    data_config = config.get("data", {})

    split_ratio = data_config.get("split_ratio")
    if split_ratio is not None and not (0.0 <= split_ratio <= 1.0):
        messages.append(f"data.split_ratio must be between 0.0 and 1.0, got {split_ratio}")

    patch_size = data_config.get("patch_size")
    if patch_size is not None:
        # Handle both int and tuple (for 2D/3D)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0] if patch_size else None

        if patch_size is not None and patch_size < 16:
            messages.append(
                f"data.patch_size should be at least 16 for most MRI tasks, got {patch_size}"
            )

    return messages


__all__ = [
    "ValidationRule",
    "ValidatorRegistry",
    "get_validator_registry",
]
