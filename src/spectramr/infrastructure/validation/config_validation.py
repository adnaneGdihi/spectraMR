"""Centralized configuration validation module.

This module provides strict, consistent validation for all configuration objects.
All validation rules are defined here to ensure maintainability and consistency.

**Validation Strategy**:
- Pydantic schemas are the primary validation mechanism
- ValidatorRegistry provides extensible cross-field validation
- All validation errors are descriptive and actionable
"""

import difflib
import logging
from typing import Any

from spectramr.config.schemas.validator_registry import get_validator_registry
from spectramr.config.settings import TrainingSettings

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


def _dispatchable_training_modes() -> set[str]:
    """Every short name ``TrainingStrategyFactory`` can actually resolve.

    Includes out-of-tree plugin entry points, which ``_load_strategy_class``
    accepts -- checking only the static dict would reject a working plugin mode.
    """
    try:
        from spectramr.infrastructure.training.strategy_factory import (
            TrainingStrategyFactory,
            _load_plugin_strategy_paths,
        )
    except ImportError as exc:  # pragma: no cover - broken install only
        # Surface as ConfigValidationError so ConfigValidator.validate re-raises
        # it verbatim instead of flattening it to "Unexpected validation error".
        raise ConfigValidationError(
            f"Cannot verify training.training_mode: the strategy factory failed "
            f"to import ({exc}). This is an environment fault, not a config fault."
        ) from exc

    return set(TrainingStrategyFactory.STRATEGY_CLASS_PATHS) | set(_load_plugin_strategy_paths())


def _validate_training_mode_dispatchable(config: dict[str, Any]) -> list[str]:
    """`training.training_mode` must name a strategy the factory can resolve.

    This is the SSOT for training-mode *existence*. It lives here, rather than
    in ``config/schemas/validator_registry``, because the answer is owned by ``STRATEGY_CLASS_PATHS`` in ``infrastructure/``
    and ``config/`` may not import ``infrastructure/`` (non-negotiable #5).

    Previously existence rode on membership of ``TRAINING_MODE_CONSTRAINTS`` (since deleted), a
    hand-maintained copy that had drifted 90 keys behind the dispatch table --
    so real configs (``physics_driven``, ``cyclegan``, ``meta_learning``) were
    rejected as "Unknown training mode" while genuinely-undispatchable ones
    passed whenever ``strategy_class`` was also set.
    """
    training = config.get("training") or {}
    mode = training.get("training_mode")

    if mode is None:
        return []

    if not isinstance(mode, str):
        return [
            f"training.training_mode must be a string naming a strategy, got {type(mode).__name__}"
        ]

    dispatchable = _dispatchable_training_modes()
    if mode in dispatchable:
        return []

    close = difflib.get_close_matches(mode, dispatchable, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    return [
        f"training.training_mode={mode!r} names no registered strategy -- "
        f"TrainingStrategyFactory.get_strategy_class would raise at runtime."
        f"{hint}"
    ]


class ConfigValidator:
    """Centralized configuration validator with comprehensive checks.

    This validator uses the extensible ValidatorRegistry pattern to check
    configurations against all registered validation rules. Legacy validation
    methods are maintained for backward compatibility but delegate to the
    registry when possible.
    """

    # Pre-v6.0 had ``REQUIRED_FIELDS`` / ``MUST_NOT_BE_NONE`` /
    # ``LEGACY_FIELDS`` class attributes here. The live validation
    # pipeline runs entirely through ``ValidatorRegistry`` (see
    # :meth:`validate`); those class attributes had zero readers and one
    # of them (``REQUIRED_FIELDS``) still referenced the removed
    # ``objectives`` field — see ``TODO/audit/04_schemas.md`` F-S-013.
    # They have been removed to keep the validator from advertising a
    # contract it does not enforce (CLAUDE.md pitfall #9).

    @classmethod
    def validate(cls, config: TrainingSettings) -> None:
        """Validate entire configuration for consistency and completeness.

        **Validation Checks** (via ValidatorRegistry):
        1. All required fields present and not None
        2. Training mode is valid
        3. Model configuration is consistent
        4. Data configuration is valid
        5. No legacy fields detected
        6. Optimizer and scheduler are compatible
        7. All additional registered validators

        Args:
            config: TrainingSettings instance to validate

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # Convert TrainingSettings to dict for registry validation
            config_dict = cls._config_to_dict(config)

            # Use ValidatorRegistry for all validation
            registry = get_validator_registry()
            errors = registry.validate(config_dict, severity="error")

            # Collect all error messages
            error_messages: list[str] = []
            for rule_name, message in errors:
                error_messages.append(f"[{rule_name}] {message}")

            # Training-mode existence: owned by STRATEGY_CLASS_PATHS, so it runs
            # here rather than inside the (config-layer) ValidatorRegistry.
            for message in _validate_training_mode_dispatchable(config_dict):
                error_messages.append(f"[training_mode_dispatchable] {message}")

            # Raise if any errors found
            if error_messages:
                raise ConfigValidationError(
                    "Configuration validation failed:\n" + "\n".join(error_messages)
                )

            # Log warnings (non-blocking)
            warnings = registry.validate(config_dict, severity="warning")
            for rule_name, message in warnings:
                logger.warning(f"Configuration warning [{rule_name}]: {message}")

        except ConfigValidationError:
            raise
        except Exception as e:
            raise ConfigValidationError(f"Unexpected validation error: {e}")

    @classmethod
    def _config_to_dict(cls, config: TrainingSettings) -> dict[str, Any]:
        """Convert TrainingSettings instance to dictionary.

        Args:
            config: TrainingSettings instance

        Returns:
            Dictionary representation of config
        """
        return config.model_dump()


def validate_config_at_startup(config: TrainingSettings) -> None:
    """Validate configuration immediately after loading.

    This is called during bootstrap to ensure configuration is valid
    before any training or inference begins.

    Args:
        config: TrainingSettings to validate

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    try:
        ConfigValidator.validate(config)
    except ConfigValidationError as e:
        raise ConfigValidationError(f"Configuration validation failed at startup: {e}")


__all__ = [
    "ConfigValidationError",
    "ConfigValidator",
    "validate_config_at_startup",
]
