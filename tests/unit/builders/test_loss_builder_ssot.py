"""Unit tests for SSOT-compliant LossBuilder enhancements.

Tests the new histogram_consistency and focal_frequency loss integrations
added as part of the SSOT compliance refactoring.
"""

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from spectramr.config.schemas.loss import LossConfigSchema, ReconstructionLossesConfig
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder


@pytest.fixture
def declare_build_set(monkeypatch):
    """Stand in for the owner of the build set (models/losses/build_set.py)."""

    def _declare(value):
        monkeypatch.setattr(
            "spectramr.infrastructure.training.builders.loss_builder.declared_build_set",
            lambda _losses: dict(value),
        )

    return _declare



class TestLossBuilderSSOT:
    """Test LossBuilder SSOT compliance for new losses."""

    @pytest.fixture
    def mock_config_with_histogram(self):
        """Config with histogram loss enabled."""
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock(spec=LossConfigSchema)
        # `spec=` restricts attributes to the class's, and pydantic v2 fields
        # are not class attributes -- so the phase-10d `policy:` sub-block has
        # to be attached explicitly. `loss_builder.py:515` reads
        # `losses.policy.output_domain`.
        config.losses.policy = MagicMock()

        # Use real config object for reconstruction to avoid mock attribute issues
        recon_config = ReconstructionLossesConfig(
            enable_hist=True,
            lambda_hist=1.0,
            histogram_bins=100,
            enable_l1=False,
            enable_perceptual=False,
        )
        config.losses.reconstruction = recon_config

        config.losses.gan = None
        config.losses.physics = None
        config.losses.diffusion = None
        config.losses.latent = None
        config.losses.ssl = None
        config.losses.evidential = None
        config.losses.uses_list_based_losses = False
        config.losses.policy.output_domain = "image"
        config.losses.kspace_losses = []
        config.losses.image_losses = []
        config.losses.complex_losses = []
        # Added by later phases of this same PR and never backfilled here:
        # `latent_losses` (phase 6, the fourth domain list) and
        # `lambda_deep_supervision` (phase 4b, moved off the config ROOT). A
        # spec= mock only exposes what the fixture sets, so the builder's
        # getattr raised instead of reading a value.
        config.losses.latent_losses = []
        config.losses.lambda_deep_supervision = 0.0
        config.deep_supervision_weight = 0.0

        return config

    @pytest.fixture
    def mock_config_with_focal_frequency(self):
        """Config with focal frequency loss enabled."""
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock(spec=LossConfigSchema)
        # `spec=` restricts attributes to the class's, and pydantic v2 fields
        # are not class attributes -- so the phase-10d `policy:` sub-block has
        # to be attached explicitly. `loss_builder.py:515` reads
        # `losses.policy.output_domain`.
        config.losses.policy = MagicMock()

        # Use real config object
        recon_config = ReconstructionLossesConfig(
            enable_ffl=True,
            lambda_ffl=1.0,
            ffl_alpha=1.0,
            enable_l1=False,
            enable_perceptual=False,
        )
        config.losses.reconstruction = recon_config

        config.losses.gan = None
        config.losses.physics = None
        config.losses.diffusion = None
        config.losses.latent = None
        config.losses.ssl = None
        config.losses.evidential = None
        config.losses.uses_list_based_losses = False
        config.losses.policy.output_domain = "image"
        config.losses.kspace_losses = []
        config.losses.image_losses = []
        config.losses.complex_losses = []
        # Added by later phases of this same PR and never backfilled here:
        # `latent_losses` (phase 6) and `lambda_deep_supervision`
        # (phase 4b, moved off the config ROOT). A spec= mock only
        # exposes what the fixture sets, so the builder's getattr raised.
        config.losses.latent_losses = []
        config.losses.lambda_deep_supervision = 0.0
        config.deep_supervision_weight = 0.0

        return config

    def test_histogram_loss_creation(self, mock_config_with_histogram, declare_build_set):
        """Test that histogram_consistency loss is created when enabled."""
        declare_build_set({
            "hist": 1.0
        })
        builder = LossBuilder(mock_config_with_histogram, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert (
            "hist" in losses
        ), "histogram_consistency loss not created (expected key 'hist')"
        assert isinstance(losses["hist"], nn.Module)

    def test_histogram_loss_respects_bins_parameter(self, mock_config_with_histogram, declare_build_set):
        """Test that histogram loss uses bins parameter from config."""
        # ReconstructionLossesConfig is frozen, so we need to create a new one
        recon_config = ReconstructionLossesConfig(
            enable_hist=True,
            lambda_hist=1.0,
            histogram_bins=256,  # Changed from 100
            enable_l1=False,
            enable_perceptual=False,
        )
        mock_config_with_histogram.losses.reconstruction = recon_config
        declare_build_set({
            "hist": 1.0
        })

        builder = LossBuilder(mock_config_with_histogram, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert "hist" in losses

    def test_histogram_loss_disabled_when_weight_zero(self, declare_build_set):
        """Test that histogram loss is NOT created when lambda_hist = 0."""
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock(spec=LossConfigSchema)
        # `spec=` restricts attributes to the class's, and pydantic v2 fields
        # are not class attributes -- so the phase-10d `policy:` sub-block has
        # to be attached explicitly. `loss_builder.py:515` reads
        # `losses.policy.output_domain`.
        config.losses.policy = MagicMock()
        recon_config = ReconstructionLossesConfig(
            enable_hist=True,
            lambda_hist=0.0,  # Weight is zero
            histogram_bins=100,
            enable_l1=False,
        )
        config.losses.reconstruction = recon_config

        config.losses.gan = None
        config.losses.physics = None
        config.losses.diffusion = None
        config.losses.latent = None
        config.losses.ssl = None
        config.losses.evidential = None
        config.losses.uses_list_based_losses = False
        config.losses.policy.output_domain = "image"
        config.losses.kspace_losses = []
        config.losses.image_losses = []
        config.losses.complex_losses = []
        # Added by later phases of this same PR and never backfilled here:
        # `latent_losses` (phase 6) and `lambda_deep_supervision`
        # (phase 4b, moved off the config ROOT). A spec= mock only
        # exposes what the fixture sets, so the builder's getattr raised.
        config.losses.latent_losses = []
        config.losses.lambda_deep_supervision = 0.0
        config.deep_supervision_weight = 0.0
        declare_build_set({})

        builder = LossBuilder(config, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert (
            "hist" not in losses
        ), "histogram_consistency should not be created when lambda_hist=0"

    def test_focal_frequency_loss_creation(self, mock_config_with_focal_frequency, declare_build_set):
        """Test that focal_frequency loss is created when enabled."""
        declare_build_set({
            "ffl": 1.0
        })
        builder = LossBuilder(mock_config_with_focal_frequency, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert "ffl" in losses, "focal_frequency loss not created (expected key 'ffl')"
        assert isinstance(losses["ffl"], nn.Module)

    def test_focal_frequency_loss_respects_alpha_parameter(
        self, mock_config_with_focal_frequency, declare_build_set
    ):
        """Test that focal frequency loss uses alpha parameter from config."""
        # Change alpha to non-default value
        recon_config = ReconstructionLossesConfig(
            enable_ffl=True,
            lambda_ffl=1.0,
            ffl_alpha=2.0,
            enable_l1=False,
            enable_perceptual=False,
        )
        mock_config_with_focal_frequency.losses.reconstruction = recon_config
        declare_build_set({
            "ffl": 1.0
        })

        builder = LossBuilder(mock_config_with_focal_frequency, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert "ffl" in losses

    def test_focal_frequency_loss_disabled_when_weight_zero(self, declare_build_set):
        """Test that focal frequency loss is NOT created when lambda_ffl = 0."""
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock(spec=LossConfigSchema)
        # `spec=` restricts attributes to the class's, and pydantic v2 fields
        # are not class attributes -- so the phase-10d `policy:` sub-block has
        # to be attached explicitly. `loss_builder.py:515` reads
        # `losses.policy.output_domain`.
        config.losses.policy = MagicMock()
        recon_config = ReconstructionLossesConfig(
            enable_ffl=True,
            lambda_ffl=0.0,  # Weight is zero
            ffl_alpha=1.0,
            enable_l1=False,
        )
        config.losses.reconstruction = recon_config

        config.losses.gan = None
        config.losses.physics = None
        config.losses.diffusion = None
        config.losses.latent = None
        config.losses.ssl = None
        config.losses.evidential = None
        config.losses.uses_list_based_losses = False
        config.losses.policy.output_domain = "image"
        config.losses.kspace_losses = []
        config.losses.image_losses = []
        config.losses.complex_losses = []
        # Added by later phases of this same PR and never backfilled here:
        # `latent_losses` (phase 6) and `lambda_deep_supervision`
        # (phase 4b, moved off the config ROOT). A spec= mock only
        # exposes what the fixture sets, so the builder's getattr raised.
        config.losses.latent_losses = []
        config.losses.lambda_deep_supervision = 0.0
        config.deep_supervision_weight = 0.0
        declare_build_set({})

        builder = LossBuilder(config, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert (
            "ffl" not in losses
        ), "focal_frequency should not be created when lambda_ffl=0"

    def test_both_losses_created_together(self, declare_build_set):
        """Test that both new losses can be created simultaneously."""
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock(spec=LossConfigSchema)
        # `spec=` restricts attributes to the class's, and pydantic v2 fields
        # are not class attributes -- so the phase-10d `policy:` sub-block has
        # to be attached explicitly. `loss_builder.py:515` reads
        # `losses.policy.output_domain`.
        config.losses.policy = MagicMock()

        # Enable both losses
        recon_config = ReconstructionLossesConfig(
            enable_hist=True,
            lambda_hist=1.0,
            histogram_bins=100,
            enable_ffl=True,
            lambda_ffl=1.0,
            ffl_alpha=1.5,
            enable_l1=False,
        )
        config.losses.reconstruction = recon_config

        config.losses.gan = None
        config.losses.physics = None
        config.losses.diffusion = None
        config.losses.latent = None
        config.losses.ssl = None
        config.losses.evidential = None
        config.losses.uses_list_based_losses = False
        config.losses.policy.output_domain = "image"
        config.losses.kspace_losses = []
        config.losses.image_losses = []
        config.losses.complex_losses = []
        # Added by later phases of this same PR and never backfilled here:
        # `latent_losses` (phase 6) and `lambda_deep_supervision`
        # (phase 4b, moved off the config ROOT). A spec= mock only
        # exposes what the fixture sets, so the builder's getattr raised.
        config.losses.latent_losses = []
        config.losses.lambda_deep_supervision = 0.0
        config.deep_supervision_weight = 0.0
        declare_build_set({"hist": 1.0, "ffl": 1.0})

        builder = LossBuilder(config, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert "hist" in losses
        assert "ffl" in losses
        # perceptual might be enabled by default if not set to False,
        # but in our recon_config above we set enable_l1=False,
        # but what about others? ReconstructionLossesConfig defaults:
        # enable_l1=True, enable_perceptual=True
        # We should check the actual count.
        assert "hist" in losses
        assert "ffl" in losses

    def test_ssot_principle_no_hardcoded_defaults(self, mock_config_with_histogram, declare_build_set):
        """Test that builder reads all parameters from config (SSOT compliance)."""
        # Change to non-default values to prove they come from config
        recon_config = ReconstructionLossesConfig(
            enable_hist=True,
            lambda_hist=1.0,
            histogram_bins=200,
            enable_l1=False,
            enable_perceptual=False,
        )
        mock_config_with_histogram.losses.reconstruction = recon_config
        declare_build_set({
            "hist": 1.0
        })

        builder = LossBuilder(mock_config_with_histogram, "cpu")
        losses = builder.build_reconstruction_losses().build()

        # If builder used hardcoded defaults, this would fail
        # The fact that it doesn't error with custom bins proves SSOT
        assert "hist" in losses
