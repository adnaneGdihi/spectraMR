"""
Tests for the centralized validator registry.

This test module validates the validator registry infrastructure and all
registered validators to ensure they correctly validate configurations
according to the validation_constants.py specifications.
"""

from typing import Any

import pytest

from spectramr.config.schemas.validator_registry import (
    ValidationRule,
    ValidatorRegistry,
    _validate_batch_size_positive,
    _validate_checkpoint_config,
    _validate_dataset_config,
    _validate_device_config,
    _validate_epochs,
    _validate_learning_rate,
    _validate_logging_config,
    _validate_loss_weights_positive,
    _validate_model_in_out_channels,
    _validate_num_workers,
    _validate_training_mode_specified,
    get_validator_registry,
)


class TestValidationRule:
    """Test ValidationRule dataclass functionality."""

    def test_validation_rule_creation(self) -> None:
        """Test creating a ValidationRule instance."""
        rule = ValidationRule(
            name="test_rule",
            category="testing",
            description="A test rule",
            validator_fn=lambda x: [],
        )
        assert rule.name == "test_rule"
        assert rule.category == "testing"
        assert rule.description == "A test rule"
        assert rule.severity == "error"  # Default

    def test_validation_rule_with_severity(self) -> None:
        """Test ValidationRule with custom severity."""
        rule = ValidationRule(
            name="warning_rule",
            category="testing",
            description="A warning rule",
            severity="warning",
            validator_fn=lambda x: [],
        )
        assert rule.severity == "warning"

    def test_validation_rule_applies(self) -> None:
        """Test ValidationRule.applies() method."""
        config = {"training": {"training_mode": "gan"}}

        # Rule that applies to all configs
        rule = ValidationRule(
            name="global_rule",
            category="test",
            description="Applies everywhere",
            validator_fn=lambda x: [],
        )
        assert rule.applies(config) is True

        # Rule that applies only to specific training modes
        rule_mode = ValidationRule(
            name="gan_rule",
            category="test",
            description="GAN only",
            validator_fn=lambda x: [],
            applies_to=["gan"],
        )
        assert rule_mode.applies(config) is True

        # Rule that doesn't apply to this config
        config_diffusion = {"training": {"training_mode": "diffusion"}}
        assert rule_mode.applies(config_diffusion) is False

    def test_validation_rule_execute(self) -> None:
        """Test ValidationRule.execute() method."""

        def validator_fn(config: dict[str, Any]):
            return ["Error 1", "Error 2"]

        rule = ValidationRule(
            name="test_rule",
            category="test",
            description="Test",
            validator_fn=validator_fn,
        )
        messages = rule.execute({})
        assert len(messages) == 2
        assert "Error 1" in messages
        assert "Error 2" in messages


class TestValidatorRegistry:
    """Test ValidatorRegistry class functionality."""

    def test_registry_singleton(self) -> None:
        """Test that get_validator_registry returns a singleton."""
        registry1 = get_validator_registry()
        registry2 = get_validator_registry()
        assert registry1 is registry2

    def test_registry_register_validator(self) -> None:
        """Test registering a validator."""
        registry = ValidatorRegistry()

        rule = ValidationRule(
            name="test_validator",
            category="test",
            description="Test validator",
            validator_fn=lambda x: ["Test error"],
        )

        registry.register(rule)
        retrieved = registry.get("test_validator")
        assert retrieved is not None
        assert retrieved.name == "test_validator"

    def test_registry_get_validator(self) -> None:
        """Test retrieving a validator by name."""
        registry = ValidatorRegistry()

        rule = ValidationRule(
            name="my_rule",
            category="test",
            description="My rule",
            validator_fn=lambda x: [],
        )
        registry.register(rule)

        retrieved = registry.get("my_rule")
        assert retrieved is not None
        assert retrieved.name == "my_rule"

        # Non-existent rule
        assert registry.get("nonexistent") is None

    def test_registry_list_validators(self) -> None:
        """Test listing validators."""
        registry = ValidatorRegistry()

        rule1 = ValidationRule(
            name="rule1",
            category="data",
            description="Rule 1",
            validator_fn=lambda x: [],
        )
        rule2 = ValidationRule(
            name="rule2",
            category="optimization",
            description="Rule 2",
            validator_fn=lambda x: [],
        )
        rule3 = ValidationRule(
            name="rule3",
            category="data",
            description="Rule 3",
            validator_fn=lambda x: [],
        )

        registry.register(rule1)
        registry.register(rule2)
        registry.register(rule3)

        # All validators
        all_validators = registry.list_validators()
        assert len(all_validators) == 3

        # By category
        data_validators = registry.list_validators(category="data")
        assert len(data_validators) == 2
        assert all(v.category == "data" for v in data_validators)

    def test_registry_validate(self) -> None:
        """Test running validation."""
        registry = ValidatorRegistry()

        def check_batch_size(config: dict[str, Any]):
            batch_size = config.get("data", {}).get("batch_size")
            if batch_size and batch_size <= 0:
                return ["Batch size must be positive"]
            return []

        rule = ValidationRule(
            name="batch_size",
            category="data",
            description="Batch size check",
            validator_fn=check_batch_size,
        )
        registry.register(rule)

        # Valid config
        config_valid = {"data": {"batch_size": 32}}
        results = registry.validate(config_valid)
        assert len(results) == 0

        # Invalid config
        config_invalid = {"data": {"batch_size": -1}}
        results = registry.validate(config_invalid)
        assert len(results) == 1
        assert results[0][0] == "batch_size"

    def test_registry_check_rule(self) -> None:
        """Test check_rule method."""
        registry = ValidatorRegistry()

        def positive_check(config: dict[str, Any]):
            value = config.get("value")
            if value is not None and value <= 0:
                return ["Value must be positive"]
            return []

        rule = ValidationRule(
            name="positive",
            category="test",
            description="Positive check",
            validator_fn=positive_check,
        )
        registry.register(rule)

        # Pass
        assert registry.check_rule({"value": 5}, "positive") is True

        # Fail
        assert registry.check_rule({"value": -5}, "positive") is False

    def test_registry_get_stats(self) -> None:
        """Test registry statistics."""
        registry = ValidatorRegistry()

        for i in range(3):
            rule = ValidationRule(
                name=f"rule_{i}",
                category="test",
                description=f"Rule {i}",
                validator_fn=lambda x: [],
            )
            registry.register(rule)

        stats = registry.get_stats()
        assert stats["total_validators"] == 3
        assert "categories" in stats
        assert "severity" in stats


class TestValidationFunctions:
    """Test individual validation functions."""

    def test_validate_batch_size_positive(self) -> None:
        """Test batch size validation."""
        # Valid
        assert _validate_batch_size_positive({"data": {"batch_size": 32}}) == []

        # Invalid: zero
        errors = _validate_batch_size_positive({"data": {"batch_size": 0}})
        assert len(errors) > 0

        # Invalid: negative
        errors = _validate_batch_size_positive({"data": {"batch_size": -10}})
        assert len(errors) > 0

        # The CANONICAL spelling. `data.batch_size` folded to
        # `data.loader.batch_size` in phase 9a; the flat form still parses, so
        # both reach this validator and both must be read. Reading only the flat
        # key made an already-migrated config report "got None".
        assert _validate_batch_size_positive({"data": {"loader": {"batch_size": 32}}}) == []
        assert len(_validate_batch_size_positive({"data": {"loader": {"batch_size": 0}}})) > 0
        # Nested wins when both are present -- it is the post-fold truth.
        assert (
            _validate_batch_size_positive(
                {"data": {"batch_size": -10, "loader": {"batch_size": 32}}}
            )
            == []
        )

    def test_validate_num_workers(self) -> None:
        """Test num_workers validation."""
        # Valid
        assert _validate_num_workers({"data": {"num_workers": 4}}) == []
        assert _validate_num_workers({"data": {"num_workers": 0}}) == []

        # Invalid: negative
        errors = _validate_num_workers({"data": {"num_workers": -1}})
        assert len(errors) > 0

        # Canonical spelling (see the batch_size case above).
        assert _validate_num_workers({"data": {"loader": {"num_workers": 4}}}) == []
        assert len(_validate_num_workers({"data": {"loader": {"num_workers": -1}}})) > 0

    def test_validate_learning_rate(self) -> None:
        """Test learning rate validation."""
        # Valid
        assert _validate_learning_rate({"optimization": {"learning_rate": 0.001}}) == []
        assert _validate_learning_rate({"optimization": {"learning_rate": 1e-6}}) == []

        # Invalid: zero
        errors = _validate_learning_rate({"optimization": {"learning_rate": 0}})
        assert len(errors) > 0

        # Invalid: negative
        errors = _validate_learning_rate({"optimization": {"learning_rate": -0.1}})
        assert len(errors) > 0

    def test_validate_epochs(self) -> None:
        """Test epochs validation."""
        # Valid
        assert _validate_epochs({"optimization": {"epochs": 100}}) == []
        assert _validate_epochs({"optimization": {"epochs": 1}}) == []

        # Invalid: zero
        errors = _validate_epochs({"optimization": {"epochs": 0}})
        assert len(errors) > 0

        # Invalid: negative
        errors = _validate_epochs({"optimization": {"epochs": -10}})
        assert len(errors) > 0

    def test_validate_training_mode_specified(self) -> None:
        """Test training mode validation."""
        # Valid
        assert (
            _validate_training_mode_specified({"training": {"training_mode": "gan"}})
            == []
        )

        # Invalid: missing
        errors = _validate_training_mode_specified({"training": {}})
        assert len(errors) > 0

    def test_validate_model_in_out_channels(self) -> None:
        """Test model channels validation."""
        # Valid
        config = {"model": {"in_channels": 2, "out_channels": 2}}
        assert _validate_model_in_out_channels(config) == []

        # Invalid: in_channels too high
        config = {"model": {"in_channels": 300, "out_channels": 2}}
        errors = _validate_model_in_out_channels(config)
        assert len(errors) > 0

    def test_validate_loss_weights_positive(self) -> None:
        """Test loss weights validation.

        Reads `losses.*` -- the canonical v1.0 block (issue #933). The
        previous `objectives.*` shape read a block `extra="forbid"` rejects
        and 0/647 arms declare, so this test pinned an unreachable path.
        """
        # Valid
        config = {"losses": {"gan": {"lambda_adv": 0.5}}}
        assert _validate_loss_weights_positive(config) == []

        # Invalid: out of range
        config = {"losses": {"gan": {"lambda_adv": 20.0}}}
        errors = _validate_loss_weights_positive(config)
        assert len(errors) > 0

    def test_validate_dataset_config(self) -> None:
        """Test dataset configuration."""
        # Valid
        config = {"data": {"split_ratio": 0.8, "patch_size": 128}}
        assert _validate_dataset_config(config) == []

        # Invalid: split_ratio
        config = {"data": {"split_ratio": 1.5}}
        errors = _validate_dataset_config(config)
        assert len(errors) > 0

        # Warning: patch_size too small
        config = {"data": {"patch_size": 8}}
        warnings = _validate_dataset_config(config)
        assert len(warnings) > 0

    def test_validate_logging_config(self) -> None:
        """Test logging configuration."""
        # Valid
        config = {"logging": {"log_interval": 10, "log_level": "INFO"}}
        assert _validate_logging_config(config) == []

        # Invalid: log_interval
        config = {"logging": {"log_interval": 0}}
        errors = _validate_logging_config(config)
        assert len(errors) > 0

        # Invalid: log_level
        config = {"logging": {"log_level": "INVALID"}}
        errors = _validate_logging_config(config)
        assert len(errors) > 0

    def test_validate_checkpoint_config(self) -> None:
        """Test checkpoint configuration."""
        # Valid
        config = {"checkpoint": {"save_interval": 5, "keep_top_k": 3}}
        assert _validate_checkpoint_config(config) == []

        # Invalid: save_interval
        config = {"checkpoint": {"save_interval": 0}}
        errors = _validate_checkpoint_config(config)
        assert len(errors) > 0

    def test_validate_device_config(self) -> None:
        """Test device configuration."""
        # Valid
        config = {"acceleration": {"device": "cuda"}}
        assert _validate_device_config(config) == []

        config = {"acceleration": {"device": "cpu"}}
        assert _validate_device_config(config) == []

        # Invalid: unknown device
        config = {"acceleration": {"device": "invalid"}}
        errors = _validate_device_config(config)
        assert len(errors) > 0


class TestValidatorRegistryIntegration:
    """Integration tests for the validator registry."""

    def test_full_validation_pipeline(self) -> None:
        """Test a complete validation workflow."""
        registry = get_validator_registry()

        # Valid configuration
        valid_config = {
            "data": {"batch_size": 32, "num_workers": 4},
            "optimization": {"learning_rate": 0.001, "epochs": 100},
            "training": {"training_mode": "gan"},
            "objectives": {"gan": {}},
            "model": {"in_channels": 2, "out_channels": 2},
        }

        errors = registry.validate(valid_config)
        # Filter to errors only (not warnings)
        error_list = [e for e in errors if registry.get(e[0]).severity == "error"]
        assert len(error_list) == 0, f"Unexpected errors: {error_list}"

    def test_invalid_configuration_detection(self) -> None:
        """Test that invalid configurations are detected."""
        registry = get_validator_registry()

        # Invalid configuration
        invalid_config = {
            "data": {"batch_size": -1, "num_workers": 4},
            "optimization": {"learning_rate": -0.1, "epochs": 0},
            "training": {},  # Missing training_mode
            "model": {"in_channels": 300, "out_channels": 2},
        }

        errors = registry.validate(invalid_config)
        # Filter to errors only
        error_list = [e for e in errors if registry.get(e[0]).severity == "error"]
        assert len(error_list) > 0, "Should have detected errors"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
