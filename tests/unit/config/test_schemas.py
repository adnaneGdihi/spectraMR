"""
Unit tests for configuration schemas.

Tests validate:
1. Schema validation and parsing
2. Default values
3. Required field enforcement
4. Type coercion
5. Nested configuration

Run on CI, NOT local dev machine.
"""

import pytest
from pydantic import ValidationError


# === Schema Import Tests ===
class TestSchemaImports:
    """Test that all schemas can be imported."""

    # @# pytest.mark.timeout(10)
    def test_import_training_settings(self):
        from spectramr.config.settings import TrainingSettings

        assert TrainingSettings is not None

    # @# pytest.mark.timeout(10)
    def test_import_model_config_schema(self):
        try:
            from spectramr.config.schemas.model import ModelConfigSchema

            assert ModelConfigSchema is not None
        except ImportError:
            pytest.skip("ModelConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_data_config_schema(self):
        try:
            from spectramr.config.schemas.data import DataConfigSchema

            assert DataConfigSchema is not None
        except ImportError:
            pytest.skip("DataConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_optimization_schema(self):
        try:
            from spectramr.config.schemas.optimization import OptimizationConfigSchema

            assert OptimizationConfigSchema is not None
        except ImportError:
            pytest.skip("OptimizationConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_early_stopping_schema(self):
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

        assert EarlyStoppingConfigSchema is not None


# === Training Settings Tests ===
class TestTrainingSettings:
    """Test TrainingSettings schema."""

    # @# pytest.mark.timeout(30)
    def test_minimal_config_loads(self):
        """Minimal valid config should load."""
        from spectramr.config.settings import TrainingSettings

        config_dict = {
            "config_version": "1.0",
            "model": {
                "model_type": "standard_unet",
                "in_channels": 1,
                "out_channels": 1,
            },
            "training": {
                "training_mode": "reconstruction",
                "strategy_class": "reconstruction",
                "batch_size": 4,
                "epochs": 10,
                "output_dir": "/tmp/test",
            },
            "data": {
                "dataset_type": "fastmri_brain",
            },
            "optimization": {
                "generator_learning_rate": 1e-4,
            },
            "losses": {
                "reconstruction": {
                    "lambda_l1": 1.0,
                },
            },
            "logging": {
                "log_dir": "/tmp/logs",
            },
        }

        # Direct construction skips the YAML-aware ``config_version`` strip,
        # so pop it manually (``from_yaml`` is the production path).
        config_dict.pop("config_version")

        settings = TrainingSettings(**config_dict)
        assert settings.model.model_type == "standard_unet"

    # @# pytest.mark.timeout(30)
    def test_missing_required_field_raises(self):
        """Missing required fields should raise ValidationError."""
        from spectramr.config.settings import TrainingSettings

        # Missing model section
        config_dict = {
            "config_version": "1.0",
            "training": {
                "training_mode": "reconstruction",
            },
        }

        with pytest.raises((ValidationError, Exception)):
            TrainingSettings(**config_dict)

    # @# pytest.mark.timeout(30)
    def test_default_values_applied(self):
        """Default values should be applied for optional fields."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

        config = EarlyStoppingConfigSchema()

        assert config.enabled == False
        assert config.patience == 10
        assert config.min_delta == 0.0

    # @# pytest.mark.timeout(30)
    def test_getattr_access_pattern(self):
        """Config should support getattr access (not .get())."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

        config = EarlyStoppingConfigSchema(enabled=True, patience=15)

        # Correct pattern: getattr
        enabled = getattr(config, "enabled", False)
        assert enabled == True

        # This pattern should work
        patience = config.patience
        assert patience == 15

        # .get() should NOT exist on Pydantic models
        assert not hasattr(config, "get") or not callable(getattr(config, "get", None))


# === Early Stopping Schema Tests ===
class TestEarlyStoppingSchema:
    """Test EarlyStoppingConfigSchema."""

    # @# pytest.mark.timeout(30)
    def test_early_stopping_defaults(self):
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

        config = EarlyStoppingConfigSchema()

        assert config.enabled == False
        assert config.patience == 10
        assert config.metric == "val_loss"

    # @# pytest.mark.timeout(30)
    def test_early_stopping_custom_values(self):
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

        # ``mode`` must be stated: it defaults to MIN, and minimising PSNR is
        # the silent inversion the reconciliation validator now rejects (metrics
        # plan PR 2, 2.5). This fixture relied on that default.
        from spectramr.config.schemas.enums import MetricMode

        config = EarlyStoppingConfigSchema(
            enabled=True,
            patience=20,
            min_delta=0.001,
            metric="val_psnr",
            mode=MetricMode.MAX,
        )

        assert config.enabled == True
        assert config.patience == 20
        assert config.min_delta == 0.001


# === Physics Schema Tests ===
class TestPhysicsSchema:
    """Test physics configuration schemas."""

    # @# pytest.mark.timeout(30)
    def test_physics_schema_import(self):
        try:
            from spectramr.config.schemas.physics import PhysicsConfigSchema

            assert PhysicsConfigSchema is not None
        except ImportError:
            pytest.skip("PhysicsConfigSchema not available")

    # @# pytest.mark.timeout(30)
    def test_data_consistency_config(self):
        try:
            from spectramr.config.schemas.physics import PhysicsConfigSchema
        except ImportError:
            pytest.skip("PhysicsConfigSchema not available")

        config = PhysicsConfigSchema(data_consistency={"enabled": True, "weight": 0.5})

        assert config.data_consistency.enabled == True


# === Validation Tests ===
class TestSchemaValidation:
    """Test schema validation behavior."""

    # @# pytest.mark.timeout(30)
    def test_invalid_enum_value_raises(self):
        """Invalid enum values should raise ValidationError."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
        from spectramr.config.schemas.enums import MetricMode

        # Valid mode
        config = EarlyStoppingConfigSchema(mode=MetricMode.MIN)
        assert config.mode == MetricMode.MIN

        # Invalid mode should raise
        with pytest.raises((ValidationError, ValueError)):
            EarlyStoppingConfigSchema(mode="invalid_mode")

    # @# pytest.mark.timeout(30)
    def test_negative_patience_raises(self):
        """Patience must be >= 1."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

        with pytest.raises(ValidationError):
            EarlyStoppingConfigSchema(patience=0)

        with pytest.raises(ValidationError):
            EarlyStoppingConfigSchema(patience=-1)

    # @# pytest.mark.timeout(30)
    def test_type_coercion(self):
        """Types should be coerced when possible."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

        # String "true" might not coerce, but bool True should work
        config = EarlyStoppingConfigSchema(enabled=True)
        assert config.enabled == True

        # Float min_delta
        config = EarlyStoppingConfigSchema(min_delta=0.001)
        assert config.min_delta == 0.001


# === Nested Config Tests ===
class TestNestedConfigs:
    """Test nested configuration handling."""

    # @# pytest.mark.timeout(30)
    def test_objectives_nested_config(self):
        """Test losses section with nested configs (lambda fields moved from objectives)."""
        from spectramr.config.settings import TrainingSettings

        config_dict = {
            "config_version": "1.0",
            "model": {
                "model_type": "standard_unet",
                "in_channels": 1,
                "out_channels": 1,
            },
            "training": {
                "training_mode": "reconstruction",
                "strategy_class": "reconstruction",
                "batch_size": 4,
                "epochs": 10,
                "output_dir": "/tmp/test",
            },
            "data": {
                "dataset_type": "fastmri_brain",
            },
            "optimization": {
                "generator_learning_rate": 1e-4,
            },
            "losses": {
                "reconstruction": {
                    "lambda_l1": 1.0,
                    "lambda_ssim": 0.5,
                },
            },
            "logging": {
                "log_dir": "/tmp/logs",
            },
        }

        config_dict.pop("config_version")
        settings = TrainingSettings(**config_dict)

        # Lambda fields are now in losses, not objectives
        assert settings.losses.reconstruction.lambda_l1 == 1.0


# === YAML Loading Tests ===
class TestYAMLLoading:
    """Test loading configs from YAML."""

    # @# pytest.mark.timeout(30)
    def test_load_yaml_config(self, tmp_path):
        """Test loading config from YAML file."""
        import yaml

        from spectramr.config.settings import TrainingSettings

        config_dict = {
            "config_version": "1.0",
            "model": {
                "model_type": "standard_unet",
                "in_channels": 1,
                "out_channels": 1,
            },
            "training": {
                "training_mode": "reconstruction",
                "strategy_class": "reconstruction",
                "batch_size": 4,
                "epochs": 10,
                "output_dir": str(tmp_path),
            },
            "data": {
                "dataset_type": "fastmri_brain",
            },
            "optimization": {
                "generator_learning_rate": 1e-4,
            },
            "losses": {
                "reconstruction": {
                    "lambda_l1": 1.0,
                },
            },
            "logging": {
                "log_dir": str(tmp_path / "logs"),
            },
        }

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        # Production path: from_yaml strips config_version before passing
        # the rest of the dict to the schema.
        settings = TrainingSettings.from_yaml(str(yaml_path))
        assert settings.model.model_type == "standard_unet"


# === Frozen Config Tests ===
class TestFrozenConfigs:
    """Test that configs are immutable when frozen."""

    # @# pytest.mark.timeout(30)
    def test_early_stopping_is_frozen(self):
        """Frozen configs should not allow modification."""
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema

        config = EarlyStoppingConfigSchema(patience=10)

        # Should raise when trying to modify
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            config.patience = 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
