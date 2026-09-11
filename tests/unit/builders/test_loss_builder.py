from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from spectramr.config.schemas.loss import GANLossesConfig, LossConfigSchema
from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder


class TestLossBuilder:
    @pytest.fixture
    def mock_config(self):
        # Create a mock config structure
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock(spec=LossConfigSchema)
        # `spec=` excludes pydantic v2 fields (not class attributes), so the
        # phase-10d `policy:` sub-block is attached explicitly.
        config.losses.policy = MagicMock()
        config.losses.reconstruction = MagicMock()
        config.losses.gan = MagicMock(spec=GANLossesConfig)
        config.losses.physics = MagicMock()
        config.losses.diffusion = MagicMock()
        config.losses.latent = MagicMock()
        config.losses.ssl = MagicMock()
        # ``evidential`` is an optional sub-schema (None on real configs by
        # default). The builder reads it via getattr but the spec=Mock here
        # doesn't auto-vivify Pydantic optional fields — declare it.
        config.losses.evidential = None
        # v6.0 declarative-list path is gated on uses_list_based_losses; with
        # MagicMock that property returns a truthy Mock by default and the
        # builder enters the list-based branch and tries to read fields that
        # the legacy fixture doesn't populate. Force the legacy flag-based
        # path here (these tests target the legacy reconstruction/GAN flags).
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

        # Default settings
        config.losses.reconstruction.enable_l1 = False
        config.losses.reconstruction.enable_l2 = False
        config.losses.reconstruction.enable_perceptual = False
        config.losses.reconstruction.enable_ssim = False
        # Set default values for weights to avoid comparison with MagicMock
        config.losses.reconstruction.lambda_l1 = 0.0
        config.losses.reconstruction.lambda_l2 = 0.0
        config.losses.reconstruction.lambda_complex_l1 = 0.0
        config.losses.reconstruction.lambda_complex_mse = 0.0
        config.losses.reconstruction.lambda_ssim = 0.0
        config.losses.reconstruction.lambda_perceptual = 0.0
        config.losses.reconstruction.lambda_lpips = 0.0
        config.losses.reconstruction.lambda_hfen = 0.0
        config.losses.reconstruction.lambda_log_spectral = 0.0
        config.losses.reconstruction.lambda_spectral_kspace = 0.0
        config.losses.reconstruction.lambda_edge = 0.0
        config.losses.reconstruction.lambda_sobel = 0.0
        config.losses.reconstruction.lambda_mind_ssc = 0.0
        config.losses.reconstruction.lambda_hist = 0.0
        config.losses.reconstruction.lambda_ffl = 0.0
        config.losses.reconstruction.lambda_latent_consistency = 0.0
        config.losses.reconstruction.lambda_tissue_bounds = 0.0
        config.losses.reconstruction.lambda_dists = 0.0
        config.losses.reconstruction.lambda_ms_ssim = 0.0
        config.losses.reconstruction.lambda_frequency = 0.0
        config.losses.reconstruction.lambda_weighted_kspace_l1 = 0.0
        config.losses.reconstruction.lambda_kspace = 0.0
        config.losses.reconstruction.lambda_smooth_l1 = 0.0
        config.losses.reconstruction.lambda_energy_conservation = 0.0
        config.losses.reconstruction.lambda_phase_consistency = 0.0
        config.losses.reconstruction.lambda_magnitude_consistency = 0.0
        config.losses.reconstruction.lambda_phase_mismapping = 0.0
        config.losses.reconstruction.lambda_frequency_domain = 0.0
        config.losses.reconstruction.lambda_frequency_weighted_l1_kspace = 0.0
        config.losses.reconstruction.lambda_background_suppression = 0.0
        config.losses.reconstruction.spatial_losses_use_fourier_bridge = False

        config.losses.gan.enable_adversarial = False
        config.losses.gan.lambda_adv = 0.0
        config.losses.gan.lambda_gp = 0.0
        config.losses.gan.lambda_r1 = 0.0
        config.losses.gan.feature_matching = 0.0
        config.losses.gan.label_smoothing = 0.0

        config.training = MagicMock()
        config.training.strategy_class = "reconstruction"

        return config

    def test_build_empty(self, mock_config):
        builder = LossBuilder(mock_config, "cpu")
        losses = builder.build_reconstruction_losses().build()
        assert len(losses) == 0

    def test_build_l1_loss(self, mock_config):
        mock_config.losses.reconstruction.enable_l1 = True
        mock_config.losses.reconstruction.lambda_l1 = 1.0
        mock_config.losses.get_enabled_losses.return_value = {"l1": 1.0}

        builder = LossBuilder(mock_config, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert "l1" in losses
        assert isinstance(losses["l1"], nn.L1Loss)

    def test_build_l2_loss(self, mock_config):
        mock_config.losses.reconstruction.enable_l2 = True
        mock_config.losses.reconstruction.lambda_l2 = 1.0
        mock_config.losses.get_enabled_losses.return_value = {"l2": 1.0}

        builder = LossBuilder(mock_config, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert "l2" in losses
        # Note: LossBuilder creates "mse" (from registry) for l2
        assert isinstance(losses["l2"], nn.MSELoss)

    def test_build_ssim_loss(self, mock_config):
        mock_config.losses.reconstruction.enable_ssim = True
        mock_config.losses.reconstruction.lambda_ssim = 1.0
        mock_config.losses.get_enabled_losses.return_value = {"ssim": 1.0}

        builder = LossBuilder(mock_config, "cpu")
        losses = builder.build_reconstruction_losses().build()

        assert "ssim" in losses
        # Checking type might be tricky if it's from registry, but check it exists

    @patch("spectramr.infrastructure.training.builders.loss_builder.create_loss")
    def test_build_gan_loss(self, mock_create_loss, mock_config):
        mock_config.losses.gan.enable_adversarial = True
        mock_config.losses.gan.gan_loss_type = "vanilla"
        mock_config.losses.gan.lambda_adv = 1.0
        mock_config.training.strategy_class = "gan_training"
        mock_config.losses.get_enabled_losses.return_value = {"adversarial": 1.0}

        builder = LossBuilder(mock_config, "cpu")
        losses = builder.build_adversarial_losses().build()

        assert "adversarial" in losses

        # Verify create_loss was called for gan_composite
        # It might also be called for the strategy ("gan_standard" etc)
        calls = [args[0] for args, kwargs in mock_create_loss.call_args_list]
        assert "gan_composite" in calls

    def test_introspection(self, mock_config):
        mock_config.losses.get_enabled_losses.return_value = {"l1": 1.0}

        builder = LossBuilder(mock_config, "cpu")
        enabled = builder.get_enabled_losses()

        assert enabled == {"l1": 1.0}

    def test_build_composite_gan_unknown_type_raises(self, mock_config):
        """TB-03: an unknown gan_loss_type must raise, not silently default.

        Previously ``adv_registry_map.get(..., "gan_standard")`` silently fell
        back to the standard GAN loss for any unrecognized type (NN#3 violation).
        """
        from spectramr.domain.exceptions import ConfigurationError

        gan_config = MagicMock()
        gan_config.gan_loss_type = "not_a_real_loss_type"
        gan_config.label_smoothing = 0.0

        builder = LossBuilder(mock_config, "cpu")
        with pytest.raises(ConfigurationError, match="Unknown gan_loss_type") as exc_info:
            builder._build_composite_gan(gan_config, None)
        # The offending value is surfaced in the message.
        assert "not_a_real_loss_type" in str(exc_info.value)

    @patch("spectramr.infrastructure.training.builders.loss_builder.create_loss")
    def test_build_composite_gan_known_type_does_not_raise(self, mock_create_loss, mock_config):
        """TB-03: a valid gan_loss_type still resolves through the registry map."""
        gan_config = MagicMock()
        gan_config.gan_loss_type = "lsgan"
        gan_config.label_smoothing = 0.0
        gan_config.lambda_adv = 1.0
        gan_config.feature_matching = 0.0
        gan_config.lambda_gp = 0.0

        builder = LossBuilder(mock_config, "cpu")
        builder._build_composite_gan(gan_config, None)

        # lsgan maps to the gan_lsgan adversarial strategy in the registry map.
        adv_calls = [args[0] for args, kwargs in mock_create_loss.call_args_list]
        assert "gan_lsgan" in adv_calls

    def _list_based(self, mock_config):
        """Switch the fixture into the v6.0 declarative-list path (where the
        unmigrated-key guard runs) and isolate the guard from list-building."""
        mock_config.losses.uses_list_based_losses = True
        ssim_entry = MagicMock()
        ssim_entry.name = "ssim"
        ssim_entry.weight = 0.5
        ssim_entry.enabled = True
        mock_config.losses.image_losses = [ssim_entry]
        # A config double has no ``model_fields_set``, so ``build_loss_weight_table``
        # falls back to ``__dict__`` and reads every ``lambda_*`` the fixture set as
        # an author declaration. The fixture's ``lambda_ssim = 0.0`` would therefore
        # contradict the list entry above; make the two surfaces agree.
        mock_config.losses.reconstruction.lambda_ssim = ssim_entry.weight
        builder = LossBuilder(mock_config, "cpu")
        # Isolate the guard: don't actually build list losses (needs registry).
        builder._build_list_based_losses = lambda: None
        return builder

    def test_recon_managed_loss_not_flagged_as_unmigrated(self, mock_config):
        """direct_ulf_to_hf_sr pattern: a loss wired via reconstruction.enable_<x>
        (consumed by the reconstruction computer) and deliberately kept OUT of
        image_losses. The unmigrated-key guard must NOT reject it.

        ``hfen`` rather than ``l1``: ``losses.reconstruction.lambda_l1`` now sits in
        ``COMPUTER_RESOLVED_LAMBDA_SOURCES``, so an l1 case would pass through that
        exemption and never reach the recon-managed branch this test names.
        """
        mock_config.losses.reconstruction.enable_hfen = True
        mock_config.losses.reconstruction.lambda_hfen = 1.0
        # get_enabled_losses reports both the recon hfen and the list ssim.
        mock_config.losses.get_enabled_losses.return_value = {"hfen": 1.0, "ssim": 0.5}
        mock_config.losses.reconstruction_managed_losses.return_value = {"hfen", "ssim"}

        builder = self._list_based(mock_config)
        builder._build_all_dynamic()  # must NOT raise ConfigurationError

    def test_genuinely_unmigrated_key_still_raises(self, mock_config):
        """A key that is in NO computer path (not recon-managed, not in the
        declarative lists) must still be rejected — the guard's real purpose."""
        from spectramr.domain.exceptions import ConfigurationError

        mock_config.losses.get_enabled_losses.return_value = {"bogus_loss": 1.0}
        mock_config.losses.reconstruction_managed_losses.return_value = set()

        builder = self._list_based(mock_config)
        with pytest.raises(ConfigurationError, match="not in the v6.0"):
            builder._build_all_dynamic()
