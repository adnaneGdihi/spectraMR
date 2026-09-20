from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from spectramr.config.schemas.run import RunConfigSchema
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.director import TrainingEnvironmentDirector
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


# Mock strategy concrete class for testing
class MockStrategy(BaseTrainingStrategy):
    def _compute_losses_impl(self, input_batch, target_batch, epoch, **kwargs):
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    # Override validation to simplify
    def validation_step(self, batch, batch_idx):
        return {}


def create_mock_config(mode="reconstruction"):
    """Create a minimal valid training configuration."""
    conf = MagicMock(spec=TrainingSettings)

    # The REAL schema, not a MagicMock child. `TrainingEnvironmentDirector`
    # reads `run.device` (director.py:255) on the path it takes whenever the DI
    # container is not built -- which is every test in this file, because none
    # of them call `build_container`. A spec'd mock does not auto-vivify it:
    # pydantic v2 keeps fields in `model_fields`, not on the class, so `run` is
    # absent from `dir(TrainingSettings)` and the getattr raises.
    #
    # That AttributeError surfaced UNDER the caught `ServiceResolutionError:
    # Service IDeviceManager not registered`, which is why 26 failures pointed
    # at a registration bug that does not exist -- bootstrap.py:233 registers
    # that service, under its `IDeviceService` alias. The resolve failing here
    # is the expected fallback, not the defect.
    #
    # `device="cpu"` is explicit, not a default: RunConfigSchema defaults to
    # "cuda", and train is a heavy pipeline, so a defaulted stub would raise
    # AcceleratorRequiredError on the CPU nodes this suite runs on.
    conf.run = RunConfigSchema(device="cpu")

    conf.model = MagicMock()
    conf.model.model_type = "mock_model"

    conf.training = MagicMock()
    conf.training.device = "cpu"
    conf.training.training_mode = mode
    conf.training.output_dir = "/tmp"

    conf.optimization = MagicMock()
    conf.optimization.optimizer.type = "adam"
    conf.optimization.optimizer.learning_rate = 1e-4
    conf.optimization.precision.enabled = False

    conf.losses = MagicMock()
    conf.losses.reconstruction = MagicMock()

    return conf


@pytest.fixture
def cold_diffusion_config():
    return create_mock_config("cold_diffusion")


@pytest.fixture
def gan_config():
    return create_mock_config("gan")


def setup_builder_mocks(
    mock_mb_cls, mock_ob_cls, mock_lb_cls, mock_pb_cls, mock_db_cls, mock_ib_cls
):
    """Helper to configure builder mocks for chaining."""
    # ModelBuilder
    mock_mb = mock_mb_cls.return_value
    mock_mb.build_generator.return_value = mock_mb
    mock_mb.build_discriminator.return_value = mock_mb
    mock_mb.build_encoder_decoder.return_value = mock_mb
    mock_mb.validate.return_value = mock_mb
    mock_mb.build.return_value = {"generator": nn.Linear(1, 1)}

    # OptimizationBuilder
    mock_ob = mock_ob_cls.return_value
    mock_ob.build_optimizers.return_value = mock_ob
    mock_ob.build_schedulers.return_value = mock_ob
    mock_ob.build_grad_scaler.return_value = mock_ob
    mock_ob.validate.return_value = mock_ob
    # build() return value will be set by test specific logic

    # LossBuilder
    mock_lb = mock_lb_cls.return_value
    mock_lb.build_reconstruction_losses.return_value = mock_lb
    mock_lb.build_adversarial_losses.return_value = mock_lb
    mock_lb.build_physics_losses.return_value = mock_lb
    mock_lb.build_regularization_losses.return_value = mock_lb
    mock_lb.validate.return_value = mock_lb
    mock_lb.build.return_value = {}

    # PhysicsBuilder
    mock_pb = mock_pb_cls.return_value
    mock_pb.build_fft_transformer.return_value = mock_pb
    mock_pb.build_data_consistency.return_value = mock_pb
    mock_pb.build_coil_sensitivity.return_value = mock_pb
    mock_pb.validate.return_value = mock_pb
    mock_pb.build.return_value = {}

    # DataBuilder
    mock_db = mock_db_cls.return_value
    mock_db.build_train_val_loaders.return_value = mock_db
    mock_db.build_test_loader.return_value = mock_db
    mock_db.validate.return_value = mock_db
    mock_db.build.return_value = {}

    # InfrastructureBuilder
    mock_ib = mock_ib_cls.return_value
    mock_ib.build_metrics.return_value = mock_ib
    mock_ib.validate.return_value = mock_ib
    mock_ib.build.return_value = {}

    return mock_mb, mock_ob, mock_lb, mock_pb, mock_db, mock_ib


@pytest.mark.parametrize(
    "mode,expected_main_key,expect_opt_d",
    [
        # --- Single Optimizer Strategies ---
        ("cold_diffusion", "opt_g", False),
        ("reconstruction", "opt_g", False),
        ("vae", "opt_g", False),
        ("vqvae", "opt_g", False),
        ("mae", "opt_g", False),
        ("ssl", "opt_g", False),
        ("domain_adaptation", "opt_g", False),
        ("physics_driven", "opt_g", False),
        ("pinn", "opt_g", False),
        ("disentangled", "opt_g", False),
        ("disentangled_vae", "opt_g", False),
        ("masked", "opt_g", False),
        ("padnet", "opt_g", False),
        ("swarm", "opt_g", False),
        ("cs", "opt_g", False),
        ("compressed_sensing", "opt_g", False),
        ("graph_cold_diffusion", "opt_g", False),
        ("b0_mapping", "opt_g", False),
        ("volumetric", "opt_g", False),
        ("trellis", "opt_g", False),
        ("ttt", "opt_g", False),
        ("meta_learning", "opt_g", False),
        ("n2n", "opt_g", False),
        # --- Dual Optimizer Strategies (GANs) ---
        ("gan", "opt_g", True),
        ("latent_gan", "opt_g", True),
        ("cycle_bloch", "opt_g", True),
    ],
)
def test_environment_optimizer_resolution(mode, expected_main_key, expect_opt_d):
    """Test optimizer resolution for various training modes."""
    # Create config
    conf = create_mock_config(mode)
    if mode == "latent_gan":
        conf.model.model_type = "latent_gan_model"
    elif mode == "gan":
        conf.model.model_type = "gan_model"
    elif mode == "cycle_bloch":
        conf.model.model_type = "cycle_bloch_model"

    with (
        patch(
            "spectramr.infrastructure.training.builders.director.ModelBuilder"
        ) as mock_mb_cls,
        patch(
            "spectramr.infrastructure.training.builders.director.OptimizationBuilder"
        ) as mock_ob_cls,
        patch(
            "spectramr.infrastructure.training.builders.director.LossBuilder"
        ) as mock_lb_cls,
        patch(
            "spectramr.infrastructure.training.builders.director.PhysicsBuilder"
        ) as mock_pb_cls,
        patch(
            "spectramr.infrastructure.training.builders.director.DataBuilder"
        ) as mock_db_cls,
        patch(
            "spectramr.infrastructure.training.builders.director.InfrastructureBuilder"
        ) as mock_ib_cls,
    ):
        mock_mb, mock_ob, _, _, _, _ = setup_builder_mocks(
            mock_mb_cls, mock_ob_cls, mock_lb_cls, mock_pb_cls, mock_db_cls, mock_ib_cls
        )

        # Configure OptimizationBuilder mock based on mode (simulating its internal logic)
        main_opt = MagicMock(spec=torch.optim.Optimizer)
        opt_d = MagicMock(spec=torch.optim.Optimizer)

        # We must mirror the logic we just patched in OptimizationBuilder
        if mode in ["gan", "latent_gan", "cycle_bloch"]:
            # OptimizationBuilder returns opt_g/opt_d for these modes
            mock_ob.build.return_value = ({"opt_g": main_opt, "opt_d": opt_d}, {}, None)
            # ModelBuilder usually returns discriminator too
            mock_mb.build.return_value = {
                "generator": nn.Linear(1, 1),
                "discriminator": nn.Linear(1, 1),
            }
        else:
            # Single-optimizer strategies get the SAME canonical key. There is
            # no "main" key anywhere in production: optimization_builder.py:280
            # writes `self._optimizers["opt_g"] = optimizer` and logs "Created
            # optimizer under canonical key 'opt_g'". This double mirrored a
            # contract that was retired under it -- hence the comment above it
            # promising to "mirror the logic we just patched".
            mock_ob.build.return_value = ({"opt_g": main_opt}, {}, None)
            mock_mb.build.return_value = {"generator": nn.Linear(1, 1)}

        director = TrainingEnvironmentDirector(conf)
        env = director.build_environment()

        # Check optimizer dictionary. `opt_g` is canonical for every mode; the
        # dual-optimizer ones merely add `opt_d` beside it.
        assert expected_main_key in env.optimizers

        # Check property access (Crucial Fix Verification)
        # env.opt_g should ALWAYS resolve to the main optimizer (generator/primary)
        assert env.opt_g is main_opt

        # Check discriminator optimizer
        if expect_opt_d:
            assert "opt_d" in env.optimizers
            assert env.opt_d is opt_d
        else:
            assert env.opt_d is None


def test_strategy_access_to_opt_g_manual_env():
    """Verify BaseTrainingStrategy logic with manually built valid environment."""
    # This minimal test ensures BaseTrainingStrategy can read from a properly constructed env
    # without needing to mock internal strategy dependencies like MetricLogger which caused issues.

    conf = create_mock_config("cold_diffusion")
    main_opt = MagicMock(spec=torch.optim.Optimizer)
    models = {"generator": nn.Linear(1, 1)}

    from spectramr.infrastructure.training.builders.environment import TrainingEnvironment

    env = TrainingEnvironment(
        models=models,
        # Canonical key, matching optimization_builder.py:280 -- "main" was
        # never a key production writes.
        optimizers={"opt_g": main_opt},
        schedulers={},
        losses={},
        physics={},
        data_loaders={},
        metrics={},
        scaler=None,
        device=torch.device("cpu"),
        config=conf,
    )

    # We don't instantiate the Full Strategy because it imports heavy things.
    # We just check the environment property that the Strategy WOULD use.
    # BaseTrainingStrategy code: self.state = env; opt = self.state.opt_g
    assert env.opt_g is main_opt
