"""Unit tests for Builder Pattern implementation

Tests all builders independently and integration test for TrainingEnvironmentDirector.
"""

import textwrap

import pytest
import torch
import torch.nn as nn

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders import (
    LossBuilder,
    ModelBuilder,
    OptimizationBuilder,
    PhysicsBuilder,
    TrainingEnvironment,
    TrainingEnvironmentDirector,
)


class TestModelBuilder:
    """Test ModelBuilder independently."""

    @pytest.fixture
    def simple_config(self, tmp_path):
        """Create a simple reconstruction config."""
        config_yaml = textwrap.dedent("""
            config_version: '1.0'
            model:
              model_type: standard_unet
              in_channels: 1
              out_channels: 1
              model_kwargs:
                features: [32, 64, 128]

            training:
              training_mode: reconstruction
              epochs: 10

            optimization:
              # Nested form. `optimizer` is a BLOCK; declaring it as a scalar
              # leaves the loader unable to fold the siblings into it
              # ("cannot fold `optimization.learning_rate` ...: `optimizer` is
              # already str, not a block").
              optimizer:
                type: adam
                learning_rate: 0.001
                beta1: 0.9
                beta2: 0.999
                weight_decay: 0.0

            logging:
              log_dir: /tmp/logs
              tensorboard_enabled: false

            data:
              dataset_type: synthetic
              patch_size: [64, 64, 1]
              batch_size: 4
            """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)
        return TrainingSettings.from_yaml(str(config_file))

    def test_build_generator(self, simple_config):
        """Test building generator."""
        device = torch.device("cpu")
        builder = ModelBuilder(simple_config, device)

        models = builder.build_generator().build()

        assert "generator" in models
        assert isinstance(models["generator"], nn.Module)
        assert next(models["generator"].parameters()).device.type == "cpu"

    def test_validation_passes_for_reconstruction(self, simple_config):
        """Test validation passes for reconstruction mode."""
        device = torch.device("cpu")
        builder = ModelBuilder(simple_config, device)

        # Should not raise
        models = builder.build_generator().validate().build()
        assert "generator" in models

    def test_validation_fails_missing_generator(self, simple_config):
        """Test validation fails if generator not built."""
        device = torch.device("cpu")
        builder = ModelBuilder(simple_config, device)

        with pytest.raises(ValueError, match="No models created"):
            builder.build()


class TestOptimizationBuilder:
    """Test OptimizationBuilder independently."""

    @pytest.fixture
    def models(self):
        """Create dummy models."""
        return {"generator": nn.Linear(10, 10)}

    @pytest.fixture
    def simple_config(self, tmp_path):
        """Create simple config."""
        config_yaml = textwrap.dedent("""
            config_version: '1.0'
            model:
              model_type: standard_unet
              in_channels: 1
              out_channels: 1

            training:
              training_mode: reconstruction

            optimization:
              # Nested form. `optimizer` is a BLOCK; declaring it as a scalar
              # leaves the loader unable to fold the siblings into it
              # ("cannot fold `optimization.learning_rate` ...: `optimizer` is
              # already str, not a block").
              optimizer:
                type: adam
                learning_rate: 0.001
                beta1: 0.9
                beta2: 0.999
                weight_decay: 0.0
              use_amp: true
              scheduler: null

            logging:
              log_dir: /tmp/logs

            data:
              dataset_name: test
              batch_size: 4
            """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)
        return TrainingSettings.from_yaml(str(config_file))

    def test_build_optimizers(self, simple_config, models):
        """Test building optimizers."""
        builder = OptimizationBuilder(simple_config, models=models)

        optimizers, schedulers, scaler = builder.build_optimizers().validate().build()

        assert "opt_g" in optimizers
        assert isinstance(optimizers["opt_g"], torch.optim.Optimizer)
        assert scaler is None  # AMP disabled

    def test_build_grad_scaler_is_a_no_op(self, simple_config, models):
        """`build_grad_scaler` builds NOTHING, and must not start again.

        It used to construct a `GradScaler("cuda")` that nothing ever called.
        The scaler that scales this run's losses is built by
        `MixedPrecisionIntegrationHelper` on the strategy -- a different object
        of a different class. The unused one was not merely wasted:
        the pre-1.0 checkpoint writer persisted its `scaler_state`, so every fp16
        checkpoint carried an untouched scale of 65536 and a zero growth-tracker
        while the live scale was dropped, and a resumed run re-converged its loss
        scale from scratch, overflowing and skipping steps -- with a
        `scaler_state` key present and restored, so the checkpoint looked whole.

        This test previously asserted `scaler is not None`, i.e. it demanded the
        presence of the very object whose presence was the defect. It now pins
        the two things that actually matter: the step stays in the public
        builder chain (it is documented as part of it), and it yields no scaler.
        """
        builder = OptimizationBuilder(simple_config, models=models)
        chained = builder.build_optimizers().build_grad_scaler()
        assert chained is builder, "must remain chainable in the public builder API"

        optimizers, schedulers, scaler = chained.build()
        assert scaler is None, (
            "build_grad_scaler must stay a no-op; checkpointing.capture.resolve_scaler "
            "is the single reader of the run's real scaler"
        )


class TestLossBuilder:
    """Test LossBuilder independently."""

    @pytest.fixture
    def simple_config(self, tmp_path):
        """Create simple config."""
        config_yaml = textwrap.dedent("""
            config_version: '1.0'
            model:
              model_type: standard_unet

            training:
              training_mode: reconstruction

            optimization:
              optimizer:
                type: adam
                learning_rate: 0.001

            logging:
              log_dir: /tmp/logs

            data:
              dataset_name: test
              batch_size: 4

            losses:
              reconstruction:
                enable_l1: true
                lambda_l1: 10.0
                enable_l2: true
                lambda_l2: 1.0
            """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)
        return TrainingSettings.from_yaml(str(config_file))

    def test_build_reconstruction_losses(self, simple_config):
        """Test building reconstruction losses."""
        device = torch.device("cpu")
        builder = LossBuilder(simple_config, device)

        losses = builder.build_reconstruction_losses().build()

        assert "l1" in losses
        assert "l2" in losses
        assert isinstance(losses["l1"], nn.L1Loss)
        assert isinstance(losses["l2"], nn.MSELoss)


class TestPhysicsBuilder:
    """Test PhysicsBuilder independently."""

    @pytest.fixture
    def simple_config(self, tmp_path):
        """Create simple config."""
        config_yaml = textwrap.dedent("""
            config_version: '1.0'
            model:
              model_type: standard_unet

            training:
              training_mode: reconstruction

            optimization:
              optimizer:
                type: adam
                learning_rate: 0.001

            logging:
              log_dir: /tmp/logs

            data:
              dataset_name: test
              batch_size: 4
            """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)
        return TrainingSettings.from_yaml(str(config_file))

    def test_build_fft_transformer(self, simple_config):
        """Test building FFT transformer."""
        device = torch.device("cpu")
        builder = PhysicsBuilder(simple_config, device)

        physics = builder.build_fft_transformer().validate().build()

        assert "fft" in physics

    def test_validation_fails_without_fft(self, simple_config):
        """Test validation fails if FFT not built."""
        device = torch.device("cpu")
        builder = PhysicsBuilder(simple_config, device)

        with pytest.raises(ValueError, match="FFT transformer not created"):
            builder.build()


class TestTrainingEnvironmentDirector:
    """Integration test for TrainingEnvironmentDirector."""

    @pytest.fixture
    def full_config(self, tmp_path):
        """Create full reconstruction config."""
        config_yaml = textwrap.dedent("""
            config_version: '1.0'
            model:
              model_type: standard_unet
              in_channels: 1
              out_channels: 1
              model_kwargs:
                features: [32, 64]

            run:
              # The CANONICAL device key, and the only one that takes effect.
              # ``training.device`` alone was silently ignored: `run.device`
              # defaults to "cuda" and director.py:255-262 consults it FIRST,
              # falling back to `training.device` only when it is empty or
              # "auto". So this fixture asked for CPU, got "cuda", and raised
              # AcceleratorRequiredError on the CPU lane -- 2 of cluster job
              # 8004252's failures. The precedence defect itself is issue #844;
              # this declares the key that is actually read.
              device: cpu

            training:
              training_mode: reconstruction
              epochs: 10
              device: cpu

            optimization:
              # Nested form. `optimizer` is a BLOCK; declaring it as a scalar
              # leaves the loader unable to fold the siblings into it
              # ("cannot fold `optimization.learning_rate` ...: `optimizer` is
              # already str, not a block").
              optimizer:
                type: adam
                learning_rate: 0.001
                beta1: 0.9
                beta2: 0.999
                weight_decay: 0.0
              use_amp: false
              scheduler: null

            logging:
              log_dir: /tmp/logs
              tensorboard_enabled: false

            data:
              dataset_type: synthetic
              patch_size: [64, 64, 1]
              synthetic_samples: 16
              batch_size: 4

            losses:
              reconstruction:
                enable_l1: true
                lambda_l1: 1.0
            """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_yaml)
        return TrainingSettings.from_yaml(str(config_file))

    def test_build_environment(self, full_config):
        """Test building complete training environment."""
        director = TrainingEnvironmentDirector(full_config)
        env = director.build_environment()

        # Verify environment structure
        assert isinstance(env, TrainingEnvironment)
        assert env.models is not None
        assert env.optimizers is not None
        assert env.losses is not None
        assert env.physics is not None
        assert env.device is not None
        assert env.config is not None

        # Verify components
        assert "generator" in env.models
        assert "opt_g" in env.optimizers
        assert "l1" in env.losses
        assert "fft" in env.physics

    def test_environment_immutability(self, full_config):
        """Test that TrainingEnvironment is immutable."""
        director = TrainingEnvironmentDirector(full_config)
        env = director.build_environment()

        # Should raise error when trying to modify frozen dataclass
        with pytest.raises(Exception):  # FrozenInstanceError
            env.models = {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
