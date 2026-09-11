from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.infrastructure.training.strategies.vae import VQVAETrainingStrategy


class TestVQVAETrainingStrategy:
    @pytest.fixture
    def mock_precision_manager(self):
        with patch(
            "spectramr.infrastructure.training.strategies.vae.VQVAEPrecisionManager"
        ) as mock:
            manager = mock.return_value
            # Setup default behavior for validate_vq_loss_precision
            manager.validate_vq_loss_precision.side_effect = lambda x: x
            yield manager

    @pytest.fixture
    def mock_loss_computer(self):
        with patch(
            "spectramr.infrastructure.training.strategies.vae.UnifiedVQVAELossComputer"
        ) as mock:
            mock_output = MagicMock()
            mock_output.total = torch.tensor(1.0)
            mock_output.components = {
                "reconstruction": torch.tensor(0.5),
                "vq_loss": torch.tensor(0.5),
            }
            mock.return_value.compute.return_value = mock_output
            yield mock.return_value

    @pytest.fixture
    def mock_env(self):
        """Create a mock training environment (NEW API - env only)."""
        env = MagicMock()
        env.device = torch.device("cpu")

        # Mock VQ-VAE model
        mock_model = MagicMock()
        mock_model.return_value = (
            torch.randn(2, 3, 64, 64),
            torch.tensor(0.5, requires_grad=True),
        )
        mock_model.training = True
        mock_model.train = MagicMock()
        env.generator = mock_model
        env.models = {"generator": mock_model}
        env.losses = {}

        # Mock config
        env.config = MagicMock()
        env.config.training.training_mode = "vqvae"
        env.config.optimization.optimizer.learning_rate = 1e-4
        # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
        env.config.optimization.gradient.clip.enabled = False
        env.config.optimization.gradient.clip.method = "norm"
        env.config.optimization.gradient.clip.value = 1.0
        # mixed_precision.py validates amp_dtype against a fixed set; a bare
        # MagicMock fails it (stale-fixture repair, 2026-07-01).
        env.config.optimization.precision.dtype = "float32"
        env.config.model.model_type = "vqvae"
        env.model_type = "vqvae"
        # These are loss-composition tests; training metrics are a separate
        # subject with their own suite. Turn that path off through the switch
        # the schema provides, rather than leaving MagicMocks to decide it
        # (#1922). Two mock reads sit on the way in and neither is benign:
        #   * `_compute_training_metrics` evaluates `train_metric_interval > 0`,
        #     which raises `TypeError: '>' not supported between instances of
        #     'MagicMock' and 'int'` -- the reported failure;
        #   * `VQVAETrainingStrategy._compute_losses_impl` takes the step from
        #     `getattr(self.env, "step", 0)`, so a mock there makes the throttle
        #     evaluate `mock % 100 == 0` -> False. Pinning ONLY the interval
        #     turns the crash into a silent skip: the tests go green having
        #     never entered the path they cross.
        # `enable_tracking=False` is the designed opt-out and returns before
        # either read, so the skip is a decision rather than mock arithmetic.
        # `env.step` is pinned to 0 to match the producer: the real
        # `builders.environment.TrainingEnvironment` declares no `step` field at
        # all (0 assignments to `env.step` under `infrastructure/training/`), so
        # production always takes that `getattr` default. A MagicMock inventing
        # an attribute production does not have is the opposite of a fixture.
        env.config.metrics.enable_tracking = False
        env.step = 0

        return env

    @pytest.fixture
    def strategy(self, mock_env, mock_precision_manager, mock_loss_computer):
        """Initialize strategy with NEW API (env only)."""
        strategy = VQVAETrainingStrategy(env=mock_env)
        # Manually attach missing components often expected by base class or mixins
        strategy.logging_service = MagicMock()

        return strategy

    def test_initialization(self, strategy, mock_precision_manager):
        """Test that VQ-VAE strategy initializes correctly with precision manager."""
        assert isinstance(strategy, VQVAETrainingStrategy)
        assert strategy.precision_manager == mock_precision_manager

    def test_compute_losses_standard_output(self, strategy, mock_env):
        """Test compute_losses with standard tuple output (recon, vq_loss)."""
        # Setup inputs
        input_batch = torch.randn(2, 3, 64, 64)
        target_batch = torch.randn(2, 3, 64, 64)
        epoch = 1

        # Setup model output
        recon = torch.randn(2, 3, 64, 64)
        vq_loss = torch.tensor(0.5, requires_grad=True)
        mock_env.generator.return_value = (recon, vq_loss)

        # Mock loss computer output
        mock_output = MagicMock()
        mock_output.total = torch.tensor(1.0)
        mock_output.components = {
            "reconstruction": torch.tensor(0.5),
            "vq_loss": vq_loss,
        }
        strategy.loss_computer.compute.return_value = mock_output

        # Execute
        losses = strategy._compute_losses_impl(input_batch, target_batch, epoch)

        # Verify
        assert "reconstruction" in losses
        assert "vq_loss" in losses
        assert "g_total_loss" in losses

        # Check values
        assert losses["vq_loss"] == vq_loss

    def test_compute_losses_dict_output(self, strategy):
        """Test compute_losses with dictionary output."""
        # Setup inputs
        input_batch = torch.randn(2, 3, 64, 64)
        target_batch = torch.randn(2, 3, 64, 64)
        epoch = 1

        # Setup model output
        recon = torch.randn(2, 3, 64, 64)
        vq_loss = torch.tensor(0.5, requires_grad=True)
        perplexity = torch.tensor(10.0)

        strategy.env.generator.return_value = {
            "reconstruction": recon,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }

        # Mock loss computer output
        mock_output = MagicMock()
        mock_output.total = torch.tensor(1.0)
        mock_output.components = {
            "reconstruction": torch.tensor(0.5),
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }
        strategy.loss_computer.compute.return_value = mock_output

        # Execute
        losses = strategy._compute_losses_impl(input_batch, target_batch, epoch)

        # Verify
        assert "reconstruction" in losses
        assert "vq_loss" in losses
        assert "perplexity" in losses
        assert losses["perplexity"] == perplexity

    def test_compute_losses_with_env_losses(self, strategy):
        """Test that configured environment losses are used."""
        # Setup environment losses
        mock_l1_loss = MagicMock(return_value=torch.tensor(0.1, requires_grad=True))

        strategy.env.losses = {"l1_loss": mock_l1_loss}

        # Setup inputs
        input_batch = torch.randn(2, 3, 64, 64)
        target_batch = torch.randn(2, 3, 64, 64)
        epoch = 1

        # Setup model output (recon only for simplicity)
        recon = torch.randn(2, 3, 64, 64)
        strategy.env.generator.return_value = (recon, None)

        # Mock loss computer output
        mock_output = MagicMock()
        mock_output.total = torch.tensor(1.0, requires_grad=True)
        mock_output.components = {"l1_loss": torch.tensor(0.1, requires_grad=True)}
        strategy.loss_computer.compute.return_value = mock_output

        # Execute
        losses = strategy._compute_losses_impl(input_batch, target_batch, epoch)

        # Verify
        assert "l1_loss" in losses
        assert losses["g_total_loss"].requires_grad

    def test_validation_step(self, strategy):
        """Test validation step processing."""
        # Setup inputs
        input_batch = torch.randn(2, 3, 64, 64)
        target_batch = torch.randn(2, 3, 64, 64)
        batch = [input_batch, target_batch]

        # Setup model output
        recon = torch.randn(2, 3, 64, 64)
        vq_loss = torch.tensor(0.5)
        strategy.env.generator.return_value = (recon, vq_loss)

        # Mock validation metrics computer
        mock_computer = MagicMock()
        mock_computer.compute.return_value = {"psnr": 30.0}
        strategy._get_validation_metrics_computer = MagicMock(return_value=mock_computer)

        # Execute
        metrics = strategy.validation_step(batch, input_batch, target_batch)

        # Verify
        assert "val_psnr" in metrics
        assert "vq_loss" in metrics
        assert metrics["val_psnr"] == 30.0
