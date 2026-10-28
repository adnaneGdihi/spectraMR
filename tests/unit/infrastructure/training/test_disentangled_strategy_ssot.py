"""Unit tests for SSOT-compliant DisentangledTrainingStrategy refactoring.

Tests the loss_weights property and _resolve_prefixed_loss_weight helper
added as part of the SSOT compliance refactoring.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.disentangled_strategy import (
    DisentangledTrainingStrategy,
)


@pytest.fixture(autouse=True)
def _patch_loss_builder_ssot():
    """Stub LossBuilder so the disentangled strategy's UnifiedDisentangledLossComputer
    constructs without exercising real loss-building on the MagicMock config. The
    2026-07-01 L1 fix made that computer fail loud on an invalid config (e.g. a
    MagicMock losses.output_domain) instead of swallowing it; these tests exercise
    the loss-weight helpers, not loss composition."""
    from unittest.mock import MagicMock as _MM
    builder = _MM()
    bi = builder.return_value
    for _m in (
        "build_reconstruction_losses",
        "build_adversarial_losses",
        "build_regularization_losses",
        "build_physics_losses",
        "build_latent_losses",
    ):
        getattr(bi, _m).return_value = bi
    # The computer now requires its declared reconstruction loss to be built (#2016);
    # the config doubles below declare loss_type "l1", so the stub builds exactly that.
    bi.build.return_value = {"l1": torch.nn.L1Loss()}
    with patch(
        "spectramr.infrastructure.training.builders.loss_builder.LossBuilder", builder
    ):
        yield



class TestDisentangledStrategySSOT:
    """Test DisentangledTrainingStrategy SSOT compliance."""

    @pytest.fixture
    def mock_config(self):
        """Create a minimal mock config for testing."""
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock()
        config.losses.reconstruction = MagicMock()
        config.losses.reconstruction.loss_type = "l1"
        config.losses.latent = MagicMock()

        # Set common weights
        config.losses.reconstruction.lambda_recon = 10.0
        config.losses.reconstruction.lambda_l1 = 10.0
        config.losses.reconstruction.lambda_style = 1.0
        config.losses.reconstruction.lambda_content = 1.0
        config.losses.reconstruction.lambda_bloch = 5.0
        config.losses.reconstruction.lambda_anat = 5.0
        config.losses.reconstruction.lambda_mind_ssc = 0.5
        config.losses.reconstruction.lambda_hist = 1.0
        config.losses.reconstruction.lambda_ffl = 1.0
        config.losses.reconstruction.lambda_perceptual = 0.1
        config.losses.reconstruction.lambda_lpips = 0.1
        config.losses.reconstruction.lambda_ssim = 0.1

        config.losses.latent.lambda_kl = 0.01
        config.losses.latent.kl_capacity_target = 20.0
        config.losses.latent.kl_capacity_warmup_steps = 10000
        config.losses.latent.use_capacity_scheduling = False

        # Enable flags
        config.losses.reconstruction.enable_mind_ssc = False
        config.losses.reconstruction.enable_hist = False
        config.losses.reconstruction.enable_ffl = False
        config.losses.reconstruction.enable_hfen = False
        config.losses.reconstruction.enable_complex_mse = False
        config.losses.reconstruction.enable_perceptual = False
        config.losses.reconstruction.enable_lpips = False
        config.losses.reconstruction.enable_ssim = False
        config.losses.reconstruction.enable_dists = False
        config.losses.reconstruction.enable_latent_consistency = False
        config.losses.reconstruction.enable_hist = False
        config.losses.reconstruction.use_magnitude_scaling = False
        config.losses.reconstruction.magnitude_scale_recon = 1.0
        config.losses.reconstruction.magnitude_scale_structural = 1.0
        config.losses.reconstruction.magnitude_scale_physics = 1.0

        # Required config sections
        config.model = MagicMock()
        config.model.in_channels = 2
        config.model.target_domain = "image"
        config.model.model_type = "disentangled"
        config.model.model_kwargs = {}  # a real mapping: validated by the strategy (#2611)

        config.physics = MagicMock()
        config.physics.kspace.enable_kspace_recon = False
        config.physics.field_strength = 3.0

        config.optimization = MagicMock()
        config.optimization.gradient.clip.enabled = False
        # StandardOptimizerStepper validates gradient_clip_method unconditionally
        # (pitfall #9); mixed_precision.py validates amp_dtype. A bare MagicMock
        # fails both — supply valid values (stale-fixture repair, 2026-07-01).
        config.optimization.gradient.clip.method = "norm"
        config.optimization.gradient.clip.value = 1.0
        config.optimization.precision.dtype = "float32"

        config.logging = MagicMock()
        config.logging.log_gradients = False
        config.logging.log_interval = 10

        config.loss_schedules = MagicMock()

        config.training = MagicMock()
        config.training.training_mode = "disentangled"
        config.training.strategy_class = "disentangled"

        config.losses.gan = MagicMock()
        config.losses.gan.enable_adversarial = False

        config.losses.physics = MagicMock()
        config.losses.physics.lambda_physics_constraint = 0.0  # Explicit value
        config.losses.physics.enable_kspace_consistency = False

        config.losses.diffusion = MagicMock()
        config.losses.diffusion.enable_diffusion = False

        config.losses.ssl = MagicMock()
        config.losses.ssl.enable_ssl = False

        return config

    @pytest.fixture
    def mock_state(self, mock_config):
        """Create a minimal mock state."""
        state = MagicMock()
        state.config = mock_config
        state.device = torch.device("cpu")
        state.losses = {}  # Empty losses dict
        state.generator = MagicMock()
        state.optimizers = {"main": MagicMock()}

        return state

    @pytest.fixture
    def mock_env(self, mock_config):
        """Create a minimal mock environment."""
        env = MagicMock()
        env.config = mock_config
        env.device = torch.device("cpu")
        env.losses = {}

        return env

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_loss_weights_property_returns_dict(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """Test that loss_weights property returns a dictionary."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)

        weights = strategy.loss_weights

        assert isinstance(weights, dict), "loss_weights should return a dict"

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_loss_weights_includes_configured_losses(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """Test that loss_weights includes all configured losses with non-zero weights."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)

        weights = strategy.loss_weights

        # Should include losses with non-zero weights
        assert "recon" in weights
        assert weights["recon"] == 10.0

        assert "l1" in weights
        assert weights["l1"] == 10.0

        assert "style" in weights
        assert weights["style"] == 1.0

        assert "content" in weights
        assert weights["content"] == 1.0

        assert "kl" in weights
        assert weights["kl"] == 0.01

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_loss_weights_excludes_zero_weights(
        self, mock_logger, mock_bloch, mock_config, mock_env, mock_state
    ):
        """Test that loss_weights excludes losses with zero weights."""
        # Set some weights to zero
        mock_config.losses.reconstruction.lambda_hist = 0.0
        mock_config.losses.reconstruction.lambda_ffl = 0.0

        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)

        weights = strategy.loss_weights

        # Zero-weight losses should not be in the dict
        assert "histogram_consistency" not in weights
        assert "focal_frequency" not in weights

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_loss_weights_is_dynamic(
        self, mock_logger, mock_bloch, mock_config, mock_env, mock_state
    ):
        """Test that loss_weights reads from config dynamically (not cached)."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)

        # Get initial weights
        weights1 = strategy.loss_weights
        initial_recon_weight = weights1.get("recon", 0.0)

        # Modify config
        mock_config.losses.reconstruction.lambda_recon = 20.0

        # Get weights again - should reflect new value
        weights2 = strategy.loss_weights
        new_recon_weight = weights2.get("recon", 0.0)

        assert (
            new_recon_weight == 20.0
        ), "loss_weights should read dynamically from config, not cache values"
        assert new_recon_weight != initial_recon_weight

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_get_loss_weight_helper_reads_from_recon_config(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """The prefixed-weight helper reads from the reconstruction config."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)

        weight = strategy._resolve_prefixed_loss_weight("lambda_bloch", default=0.0)

        assert weight == 5.0, "Should read lambda_bloch from reconstruction config"

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_get_loss_weight_helper_reads_from_latent_config(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """The prefixed-weight helper reads from the latent config."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)

        weight = strategy._resolve_prefixed_loss_weight("lambda_kl", default=0.0)

        assert weight == 0.01, "Should read lambda_kl from latent config"

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_get_loss_weight_helper_returns_default_when_not_found(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """The prefixed-weight helper returns default for non-existent weights."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)

        weight = strategy._resolve_prefixed_loss_weight(
            "lambda_nonexistent", default=99.0
        )

        assert weight == 99.0, "Should return default for non-existent weight"

    def test_base_loss_weight_seam_is_not_shadowed(self):
        """LSP guard (2026-07-01): DisentangledTrainingStrategy must inherit
        ``BaseTrainingStrategy._get_loss_weight`` unshadowed. The old override
        ``_get_loss_weight(loss_name, default=0.0)`` collided with the base
        seam ``(loss_name, epoch=0, **kwargs)`` — a generic caller would have
        bound ``epoch`` to ``default`` and silently received ``float(epoch)``
        as a loss weight, losing the warm-up masking."""
        from spectramr.infrastructure.training.strategies.base import (
            BaseTrainingStrategy,
        )

        assert (
            DisentangledTrainingStrategy._get_loss_weight
            is BaseTrainingStrategy._get_loss_weight
        ), "disentangled must not shadow the base _get_loss_weight seam"
        # The renamed helper exists with its own honest name.
        assert hasattr(DisentangledTrainingStrategy, "_resolve_prefixed_loss_weight")

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_no_lambda_instance_variables_exist(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """Test that lambda_* instance variables have been removed (SSOT compliance)."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)

        # These should NOT exist as instance variables anymore
        forbidden_attrs = [
            "lambda_recon",
            "lambda_style",
            "lambda_content",
            "lambda_kl",
            "lambda_bloch",
            "lambda_anat",
            "lambda_mind_ssc",
            "lambda_hist",
            "lambda_ffl",
            "lambda_perceptual",
            "lambda_lpips",
            "lambda_ssim",
            "lambda_tissue_bounds",
        ]

        for attr in forbidden_attrs:
            assert (
                not hasattr(strategy, attr) or getattr(strategy, attr, None) is None
            ), f"{attr} should not exist as instance variable (SSOT violation)"

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_tissue_bounds_hardcoded_to_zero(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """Test that lambda_tissue_bounds is no longer used in strategy."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)
        assert not hasattr(strategy, "lambda_tissue_bounds")

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_magnitude_scaling_hyperparameters_exist(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """Test that magnitude scaling hyperparameters are no longer in strategy (moved to LossComputer)."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)
        assert not hasattr(strategy, "use_magnitude_scaling")

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_kl_capacity_hyperparameters_exist(
        self, mock_logger, mock_bloch, mock_env, mock_state
    ):
        """Test that KL capacity hyperparameters are no longer in strategy (moved to LossComputer)."""
        strategy = DisentangledTrainingStrategy(env=mock_env, state=mock_state)
        assert not hasattr(strategy, "kl_capacity_target")


class TestDisentangledStrategySSOTIntegration:
    """Integration tests for SSOT-compliant loss weight usage."""

    @pytest.fixture
    def full_mock_config(self):
        """Create a complete mock config for integration testing."""
        config = MagicMock(spec=TrainingSettings)
        config.losses = MagicMock()
        config.losses.reconstruction = MagicMock()
        config.losses.reconstruction.loss_type = "l1"
        config.losses.latent = MagicMock()

        # Set comprehensive weights for integration test
        config.losses.reconstruction.lambda_recon = 10.0
        config.losses.reconstruction.lambda_l1 = 10.0
        config.losses.reconstruction.lambda_style = 1.0
        config.losses.reconstruction.lambda_content = 1.0
        config.losses.reconstruction.lambda_bloch = 5.0
        config.losses.reconstruction.lambda_hist = 2.0
        config.losses.reconstruction.lambda_ffl = 1.5
        config.losses.reconstruction.enable_hist = False
        config.losses.reconstruction.enable_ffl = False
        config.losses.reconstruction.use_magnitude_scaling = False
        config.losses.reconstruction.magnitude_scale_recon = 1.0
        config.losses.reconstruction.magnitude_scale_structural = 1.0
        config.losses.reconstruction.magnitude_scale_physics = 1.0

        config.losses.latent.lambda_kl = 0.01
        config.losses.latent.kl_capacity_target = 20.0
        config.losses.latent.kl_capacity_warmup_steps = 10000
        config.losses.latent.use_capacity_scheduling = False

        config.model = MagicMock()
        config.model.in_channels = 2
        config.model.target_domain = "image"
        config.model.model_type = "disentangled"
        config.model.model_kwargs = {}  # a real mapping: validated by the strategy (#2611)

        config.physics = MagicMock()
        config.physics.kspace.enable_kspace_recon = False
        config.physics.field_strength = 3.0

        config.optimization = MagicMock()
        config.optimization.gradient.clip.enabled = False
        config.optimization.gradient.clip.method = "norm"
        config.optimization.gradient.clip.value = 1.0
        config.optimization.precision.dtype = "float32"

        config.logging = MagicMock()
        config.logging.log_gradients = False
        config.logging.log_interval = 10

        config.loss_schedules = MagicMock()

        config.training = MagicMock()
        config.training.training_mode = "disentangled"
        config.training.strategy_class = "disentangled"

        config.losses.gan = MagicMock()
        config.losses.gan.enable_adversarial = False

        config.losses.physics = MagicMock()
        config.losses.physics.lambda_physics_constraint = 0.0
        config.losses.physics.enable_kspace_consistency = False

        config.losses.diffusion = MagicMock()
        config.losses.diffusion.enable_diffusion = False

        config.losses.ssl = MagicMock()
        config.losses.ssl.enable_ssl = False

        return config

    @patch(
        "spectramr.infrastructure.training.strategies.disentangled_strategy.MultiPhysicsBlochLayer"
    )
    @patch("spectramr.infrastructure.training.strategies.disentangled_strategy.logger")
    def test_ssot_pattern_no_duplication(
        self, mock_logger, mock_bloch, full_mock_config
    ):
        """Integration test: Verify SSOT pattern eliminates duplication."""
        env = MagicMock()
        env.config = full_mock_config
        env.device = torch.device("cpu")
        # [FIX] Populate losses to satisfy fail-fast SSOT checks
        env.losses = {
            "perceptual": MagicMock(),
            "ssim": MagicMock(),
            "lpips": MagicMock(),
            "hfen": MagicMock(),
            "ffl": MagicMock(),
            "hist": MagicMock(),
            "mind_ssc": MagicMock(),
        }
        state = MagicMock()
        state.config = full_mock_config
        state.device = torch.device("cpu")
        state.losses = {}
        state.generator = MagicMock()
        state.optimizers = {"main": MagicMock()}

        strategy = DisentangledTrainingStrategy(env=env, state=state)

        # Get weights via property
        weights_via_property = strategy.loss_weights

        # Get same weights via helper
        hist_via_helper = strategy._resolve_prefixed_loss_weight("lambda_hist", 0.0)
        ffl_via_helper = strategy._resolve_prefixed_loss_weight("lambda_ffl", 0.0)

        # Both methods should give same values (single source of truth)
        assert weights_via_property.get("histogram_consistency") == hist_via_helper
        assert weights_via_property.get("focal_frequency") == ffl_via_helper
