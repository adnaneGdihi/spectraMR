from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)


class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_vae = True
        self.class_embedding = None
        self.enc_c = nn.Linear(1, 1)
        self.enc_s = nn.Linear(1, 2)

    def reparameterize(self, mu, logvar):
        return mu

    def gen(self, c, s):
        return c + s.sum()

    def forward(self, *args, **kwargs):
        pass


class AlwaysContainsDict(dict):
    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return MagicMock(return_value=torch.tensor(1.0, requires_grad=True))


class DummyEnvironment:
    def __init__(self, with_disc=True, enable_adv=False):
        self.device = torch.device("cpu")

        c = MagicMock()
        c.objectives.gan.enable_adversarial = enable_adv
        c.objectives.gan.enable_gradient_penalty = False
        c.objectives.gan.enable_feature_matching = False

        c.physics.kspace.enable_kspace_recon = False

        c.optimization.precision.enabled = False
        c.optimization.compile_models = False
        c.optimization.gradient.clip.enabled = False
        c.optimization.gradient_clip_val = 1.0
        c.optimization.gradient.clip.value = 1.0
        # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
        c.optimization.gradient.clip.method = "norm"
        # mixed_precision.py validates amp_dtype against a fixed set; a bare
        # MagicMock fails it (stale-fixture repair, 2026-07-01).
        c.optimization.precision.dtype = "float32"

        c.model.in_channels = 1
        c.model.target_domain = "image"
        # A real mapping: the strategy validates bloch_sequence_by_contrast and
        # refuses a MagicMock in its place (#2611).
        c.model.model_kwargs = {}
        # Single-process: a bare MagicMock reads as an unknown parallel backend (#2629).
        c.parallel = None

        self.config = c

        self.models = {}
        self.optimizers = {}

        self.generator = DummyGenerator()
        self.models["generator"] = self.generator
        self.optimizers["generator"] = MagicMock()
        self.optimizers["generator"].param_groups = [
            {"params": list(self.generator.parameters())}
        ]
        self.opt_g = self.optimizers["generator"]

        self.losses = AlwaysContainsDict(
            {"l1": MagicMock(return_value=torch.tensor(1.0))}
        )

        if with_disc:
            self.discriminator = nn.Linear(1, 1)
            self.models["discriminator"] = self.discriminator
            self.optimizers["discriminator"] = MagicMock()
            self.opt_d = self.optimizers["discriminator"]
        else:
            self.discriminator = None
            self.opt_d = None


@pytest.fixture
def patch_bloch():
    with patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    ) as mock:
        yield mock


@pytest.fixture
def patch_loss_computer():
    with patch(
        "spectramr.models.losses.computers.unified_disentangled.UnifiedDisentangledLossComputer"
    ) as mock:
        mock_instance = mock.return_value

        mock_instance.compute.return_value = {
            "loss": torch.tensor(1.5, requires_grad=True),
            "log_metrics": {"G_total": 1.5},
        }

        yield mock


@pytest.fixture
def patch_loss_builder():
    """Stub LossBuilder so the strategy's ADVERSARIAL loss computer (built when a
    discriminator exists) constructs without exercising real loss-building on the
    incompatible MagicMock config. This is a wiring test; loss composition is
    covered elsewhere. Needed since the 2026-07-01 L1 fix made the adversarial
    computer fail loud on an invalid config instead of silently falling back."""
    builder = MagicMock()
    builder.return_value.build_reconstruction_losses.return_value = builder.return_value
    builder.return_value.build_adversarial_losses.return_value = builder.return_value
    builder.return_value.build_regularization_losses.return_value = builder.return_value
    builder.return_value.build_physics_losses.return_value = builder.return_value
    builder.return_value.build_latent_losses.return_value = builder.return_value
    builder.return_value.build.return_value = {}
    with patch(
        "spectramr.infrastructure.training.builders.loss_builder.LossBuilder", builder
    ):
        yield builder


def test_disentangled_strategy_without_discriminator(
    patch_bloch, patch_loss_computer, patch_loss_builder
):
    """
    Validate that DisentangledTrainingStrategy initializes gracefully without an
    adversarial loss computer if Discriminator tools are omitted (Edge case).
    """
    env = DummyEnvironment(with_disc=False, enable_adv=False)

    strategy = DisentangledTrainingStrategy(
        env=env, metrics_service=MagicMock(), checkpoint_service=MagicMock()
    )

    assert getattr(strategy, "discriminator", None) is None
    assert getattr(strategy, "opt_d", None) is None
    assert getattr(strategy, "adv_loss_computer", None) is None


def test_disentangled_strategy_with_disc_disabled_in_config(
    patch_bloch, patch_loss_computer, patch_loss_builder
):
    """
    Validate that DisentangledTrainingStrategy initializes the adv_loss_computer
    because the modules exist, but the enable_adv flags represent the standard disablement.
    """
    env = DummyEnvironment(with_disc=True, enable_adv=False)

    strategy = DisentangledTrainingStrategy(
        env=env, metrics_service=MagicMock(), checkpoint_service=MagicMock()
    )

    assert getattr(strategy, "adv_loss_computer", None) is not None
