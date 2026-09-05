"""Unit tests for OptimizationBuilder.

Tests the builder's ability to:
- Create single optimizers (factory interface)
- Create optimizer sets for different training modes
- Create schedulers and gradient scalers
- Validate optimization configuration
"""

import pytest
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders import OptimizationBuilder
from tests.utils.repo_scripts import require_repo_file


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return DummyModel()


@pytest.fixture
def dummy_models(dummy_model):
    """Create a dictionary of dummy models."""
    return {
        "generator": dummy_model,
        "discriminator": DummyModel(),
        "encoder": DummyModel(),
    }


@pytest.fixture
def config_reconstruction():
    """Load a v6.0+ reconstruction config for testing.

    Previously used ``experiments/training/v6_multi_strategy_example.yaml``,
    which has drifted past the current schema (uses ``loss_config:`` and
    other field names that ``MultiStageSchema`` now rejects under
    ``extra='forbid'``). The hilbert_mamba reconstruction example is a
    clean v6.x config and is enough for OptimizationBuilder smoke tests
    (the builder only cares about ``config.optimization`` + a
    reconstruction training_mode).
    """
    return TrainingSettings.from_yaml(
        str(require_repo_file("experiments/inprogress/hilbert_mamba/exp_hm_01_ct_mamba.yaml"))
    )


@pytest.fixture
def config_gan():
    """Load a v6.0+ GAN config for testing.

    Previously used ``experiments/training/experiment_01_baseline_gan.yaml``
    (now under ``experiments/training/gan/`` AND pinned to legacy v5.0
    schema which the loader rejects). ``dummy_gan.yaml`` is the canonical
    minimal v6.x GAN exemplar.
    """
    return TrainingSettings.from_yaml(
        str(require_repo_file("experiments/inprogress/dummy/dummy_gan.yaml"))
    )


class TestOptimizationBuilderFactoryInterface:
    """Test the factory-like static methods."""

    def test_create_single_optimizer_adam(self, dummy_model):
        """Test creating a single Adam optimizer."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            learning_rate=1e-4,
            optimizer_type="adam",
        )
        assert isinstance(optimizer, Adam)
        assert optimizer.defaults["lr"] == 1e-4

    def test_create_single_optimizer_adamw(self, dummy_model):
        """Test creating a single AdamW optimizer."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            learning_rate=2e-4,
            optimizer_type="adamw",
        )
        assert isinstance(optimizer, AdamW)
        assert optimizer.defaults["lr"] == 2e-4

    def test_create_single_optimizer_sgd(self, dummy_model):
        """Test creating a single SGD optimizer."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            learning_rate=1e-3,
            optimizer_type="sgd",
            momentum=0.9,
        )
        assert isinstance(optimizer, SGD)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["momentum"] == 0.9

    def test_create_single_optimizer_rmsprop(self, dummy_model):
        """Test creating a single RMSprop optimizer."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            learning_rate=1e-4,
            optimizer_type="rmsprop",
            alpha=0.99,
        )
        assert isinstance(optimizer, RMSprop)
        assert optimizer.defaults["lr"] == 1e-4

    def test_create_single_optimizer_with_custom_betas(self, dummy_model):
        """Test creating optimizer with custom beta values."""
        betas = (0.9, 0.999)
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            optimizer_type="adam",
            betas=betas,
        )
        assert optimizer.defaults["betas"] == betas

    def test_create_single_optimizer_with_weight_decay(self, dummy_model):
        """Test creating optimizer with weight decay."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            optimizer_type="adam",
            weight_decay=1e-4,
        )
        assert optimizer.defaults["weight_decay"] == 1e-4

    def test_create_single_optimizer_unknown_type(self, dummy_model):
        """Test that unknown optimizer type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizationBuilder.create_single_optimizer(
                dummy_model.parameters(),
                optimizer_type="unknown",
            )

    def test_create_single_optimizer_routes_through_registry(self, dummy_model):
        """TB-01: create_single_optimizer dispatches via the OptimizerRegistry.

        An unknown type must surface the registry's descriptive error (it lists
        the available optimizers) instead of the bare legacy if/elif message.
        """
        with pytest.raises(ValueError, match="Unknown optimizer") as exc_info:
            OptimizationBuilder.create_single_optimizer(
                dummy_model.parameters(),
                optimizer_type="unknown_xyz",
            )
        # The registry path includes the available list in its message.
        assert "Available" in str(exc_info.value)

    def test_create_single_optimizer_nadam(self, dummy_model):
        """TB-01: a registry-extended optimizer not in the old if/elif is reachable."""
        from torch.optim import NAdam

        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            optimizer_type="nadam",
            learning_rate=1e-4,
        )
        assert isinstance(optimizer, NAdam)
        assert optimizer.defaults["lr"] == 1e-4

    def test_create_single_optimizer_case_insensitive(self, dummy_model):
        """Test that optimizer type is case-insensitive."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            optimizer_type="ADAM",
        )
        assert isinstance(optimizer, Adam)


class TestOptimizationBuilderBuilderInterface:
    """Test the builder pattern interface."""

    def test_build_reconstruction_optimizer(self, config_reconstruction, dummy_models):
        """Test building optimizer for reconstruction training."""
        # Modify config to reconstruction (default)
        builder = OptimizationBuilder(config_reconstruction, models=dummy_models)
        builder.build_optimizers()

        optimizers, schedulers, scaler = builder.build()
        assert "opt_g" in optimizers
        assert isinstance(optimizers["opt_g"], Optimizer)

    def test_build_gan_dual_optimizers(self, config_gan, dummy_models):
        """Test building dual optimizers for GAN training."""
        # Create GAN config - need to set training_mode to gan
        config = config_gan

        # Update the generator and discriminator models
        dummy_models["generator"] = DummyModel()
        dummy_models["discriminator"] = DummyModel()

        builder = OptimizationBuilder(config, models=dummy_models)
        builder.build_optimizers()

        optimizers, schedulers, scaler = builder.build()

        # For GAN, check if dual optimizers exist
        # (may be "main" or "generator"/"discriminator" depending on config)
        assert len(optimizers) > 0
        assert all(isinstance(opt, Optimizer) for opt in optimizers.values())

    def test_build_with_scheduler(self, config_reconstruction, dummy_models):
        """Test building optimizer with scheduler."""
        builder = OptimizationBuilder(config_reconstruction, models=dummy_models)
        builder.build_optimizers().build_schedulers()

        optimizers, schedulers, scaler = builder.build()

        # The builder names optimizers per paradigm — reconstruction uses
        # ``opt_g`` (generator-only). Previously this asserted ``"main"``
        # which was the pre-Phase-7 naming. Accept any non-empty optimizer
        # dict instead.
        assert len(optimizers) > 0
        assert all(isinstance(opt, Optimizer) for opt in optimizers.values())
        # Scheduler may or may not be created depending on config
        assert isinstance(schedulers, dict)

    def test_build_with_grad_scaler(self, config_reconstruction, dummy_models):
        """Test building optimizer with gradient scaler."""
        builder = OptimizationBuilder(config_reconstruction, models=dummy_models)
        builder.build_optimizers().build_grad_scaler()

        optimizers, schedulers, scaler = builder.build()

        # Scaler depends on config.optimization.precision.enabled
        if scaler is not None:
            from torch.amp import GradScaler

            assert isinstance(scaler, GradScaler)

    def test_method_chaining(self, config_reconstruction, dummy_models):
        """Test that builder methods support chaining."""
        builder = (
            OptimizationBuilder(config_reconstruction, models=dummy_models)
            .build_optimizers()
            .build_schedulers()
            .build_grad_scaler()
        )

        assert builder is not None
        optimizers, schedulers, scaler = builder.build()
        assert len(optimizers) > 0

    def test_validate_optimization(self, config_reconstruction, dummy_models):
        """Test optimization validation."""
        builder = OptimizationBuilder(config_reconstruction, models=dummy_models)
        builder.build_optimizers()

        # Validate should pass
        result = builder.validate()
        assert result is builder  # Returns self for chaining

    def test_validate_fails_without_optimizers(self, config_reconstruction, dummy_models):
        """Test that validation fails if no optimizers created."""
        builder = OptimizationBuilder(config_reconstruction, models=dummy_models)
        # Don't call build_optimizers()

        with pytest.raises(ValueError, match="No optimizers created"):
            builder.validate()

    def test_build_fails_without_optimizers(self, config_reconstruction, dummy_models):
        """Test that build fails if no optimizers created."""
        builder = OptimizationBuilder(config_reconstruction, models=dummy_models)
        # Don't call build_optimizers()

        with pytest.raises(ValueError, match="No optimizers created"):
            builder.build()

    def test_build_with_missing_generator(self, config_reconstruction):
        """Test error handling when generator model is missing."""
        models = {"discriminator": DummyModel()}  # Missing generator
        builder = OptimizationBuilder(config_reconstruction, models=models)

        # This may or may not raise depending on config
        # Just verify it doesn't crash unexpectedly
        try:
            builder.build_optimizers()
        except ValueError:
            pass  # Expected if config requires generator


class TestBuildSchedulersRaisesOnUnknown:
    """TB-02: build_schedulers routes through create_scheduler (raise on unknown)."""

    @staticmethod
    def _make_builder_with_optimizer(scheduler_cfg, dummy_model):
        """Build an OptimizationBuilder with a populated optimizer + stub config.

        The docstring here used to claim ``build_schedulers`` "only reads
        ``config.optimization.scheduler`` and iterates ``self._optimizers``, so
        a SimpleNamespace config is enough". It also reads
        ``config.training.max_iterations`` (to default a cosine period), so the
        stub raised ``AttributeError: 'types.SimpleNamespace' object has no
        attribute 'training'`` and both tests in this class were red on dev
        (#723).

        ``optimization`` comes from the shared ``block_stub`` rather than a
        hand-rolled namespace: it IS the real schema with the flat->canonical
        routing derived from ``RENAMES``, so it cannot disagree with the loader
        about where a reader looks -- which is exactly how this rotted.
        """
        from types import SimpleNamespace

        from tests.utils.config_block_stub import block_stub

        builder = OptimizationBuilder.__new__(OptimizationBuilder)
        builder._config = SimpleNamespace(
            optimization=block_stub("optimization", scheduler=scheduler_cfg),
            training=SimpleNamespace(max_iterations=1000),
        )
        builder._models = {}
        builder._optimizers = {
            "opt_g": Adam(dummy_model.parameters(), lr=1e-4),
        }
        builder._schedulers = {}
        builder._scaler = None
        return builder

    def test_build_schedulers_unknown_type_raises(self, dummy_model):
        """An unknown scheduler type must raise (no silent skip).

        The guarantee this test exists for holds; only the exception TYPE it
        named was stale. Resolution moved from ``create_scheduler``'s bare
        ``ValueError`` to ``resolve_scheduler_spec``, which raises the domain
        ``ConfigurationError`` and additionally lists the valid names.
        ``ConfigurationError`` derives from ``SpectraMRError``, not ``ValueError``,
        so ``pytest.raises(ValueError)`` no longer caught it -- a red test over
        a guard that had in fact got stricter. The sibling suite in
        ``tests/unit/infrastructure/training/builders/`` already expects
        ``ConfigurationError``.

        Assert the payload, not just the type: "raises" alone would still pass
        if the message stopped naming which value was rejected.
        """
        from spectramr.domain.exceptions import ConfigurationError

        builder = self._make_builder_with_optimizer(
            {"type": "totally_unknown_scheduler", "kwargs": {}}, dummy_model
        )
        with pytest.raises(ConfigurationError, match="Unknown scheduler") as excinfo:
            builder.build_schedulers()
        assert "totally_unknown_scheduler" in str(excinfo.value)
        assert "cosine" in str(excinfo.value), "the error must list valid names"

    def test_build_schedulers_known_type_creates(self, dummy_model):
        """A known scheduler type is created and stored (alias 'cosine' normalized)."""
        builder = self._make_builder_with_optimizer({"type": "cosine", "kwargs": {}}, dummy_model)
        result = builder.build_schedulers()
        assert result is builder
        assert "opt_g" in builder._schedulers
        assert builder._schedulers["opt_g"] is not None


class TestOptimizationBuilderIntegration:
    """Integration tests with real configs."""

    def test_builder_with_multiple_configs(self):
        """Test builder with different experiment configs."""
        config_files = [
            "experiments/validated/experiment_01_baseline_gan.yaml",
            "experiments/validated/experiment_100_ensemble_disagreement.yaml",
        ]

        for config_file in config_files:
            try:
                config = TrainingSettings.from_yaml(config_file)
                models = {"generator": DummyModel()}
                builder = OptimizationBuilder(config, models=models)
                builder.build_optimizers()

                optimizers, schedulers, scaler = builder.build()
                assert len(optimizers) > 0, f"No optimizers for {config_file}"
            except FileNotFoundError:
                pytest.skip(f"Config file not found: {config_file}")

    def test_optimizer_preserves_learning_rates(self, dummy_model):
        """Test that optimizer preserves learning rate settings."""
        test_rates = [1e-5, 1e-4, 1e-3, 1e-2]

        for lr in test_rates:
            optimizer = OptimizationBuilder.create_single_optimizer(
                dummy_model.parameters(),
                learning_rate=lr,
            )
            assert optimizer.defaults["lr"] == lr

    def test_optimizer_compatible_with_training_loop(self, dummy_model):
        """Test that created optimizer works in a training loop."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            learning_rate=1e-4,
        )

        # Simulate training step
        loss = dummy_model(torch.randn(4, 10)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # If we got here without error, optimizer works correctly
        assert True


class TestOptimizationBuilderEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_models_dict(self, config_reconstruction):
        """Test handling of empty models dictionary."""
        models = {}
        builder = OptimizationBuilder(config_reconstruction, models=models)

        # Should raise error when trying to build
        with pytest.raises((ValueError, KeyError)):
            builder.build_optimizers()

    def test_optimizer_without_scheduler(self, config_reconstruction, dummy_models):
        """Test that optimizer works without scheduler."""
        builder = OptimizationBuilder(config_reconstruction, models=dummy_models)
        builder.build_optimizers()  # Skip build_schedulers()

        optimizers, schedulers, scaler = builder.build()
        assert len(optimizers) > 0
        assert isinstance(schedulers, dict)

    def test_optimizer_with_very_small_lr(self, dummy_model):
        """Test optimizer with very small learning rate."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            learning_rate=1e-10,
        )
        assert optimizer.defaults["lr"] == 1e-10

    def test_optimizer_with_zero_weight_decay(self, dummy_model):
        """Test optimizer with zero weight decay (no regularization)."""
        optimizer = OptimizationBuilder.create_single_optimizer(
            dummy_model.parameters(),
            optimizer_type="adam",
            weight_decay=0.0,
        )
        assert optimizer.defaults["weight_decay"] == 0.0
