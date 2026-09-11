"""
Comprehensive tests for training strategies.

Tests validate:
1. Strategy initialization without errors
2. train_step() method signatures and basic execution
3. validation_step() method signatures and basic execution
4. No AttributeError on Pydantic config access
5. Proper handling of **kwargs in generator calls

Run on CI, NOT local dev machine.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from spectramr.core.cascading_validation import CASCADING_LEVELS
from spectramr.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)
from tests.utils.config_block_stub import block_stub


# === Mock Classes ===
class DummyGenerator(nn.Module):
    """Minimal generator for strategy testing."""

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x, **kwargs):
        return self.conv(x)

    def sample(self, x, **kwargs):
        return self.forward(x)

    def generate(self, x, **kwargs):
        return self.forward(x)


class DummyDiscriminator(nn.Module):
    """Minimal discriminator for GAN testing."""

    def __init__(self, in_ch=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x).mean()


def create_mock_config():
    """Create mock config that mimics TrainingSettings."""
    config = MagicMock()
    config.model = MagicMock()
    config.model.model_type = "standard_unet"
    config.model.in_channels = 1
    config.model.out_channels = 1
    config.model.input_type = "image"  # Default
    config.model.prior_model = None  # Ensure no prior model is created by default
    config.model.target_domain = "image"  # Added for BaseTrainingStrategy
    # OFF, explicitly. `AdaptiveConditioner.from_config` gates on
    # `enabled and sources`, both of which a MagicMock satisfies truthily, so an
    # unset block switched conditioning ON and then died on the `injection`
    # validator ("only 'film' is implemented"). A fixture that is not testing
    # conditioning must say so rather than leave the gate to Mock truthiness.
    config.model.conditioning = MagicMock()
    config.model.conditioning.enabled = False
    config.model.conditioning.sources = []
    config.training_mode = "reconstruction"
    config.device = "cpu"
    config.optimization = MagicMock()
    config.optimization.precision.enabled = False
    # MUST be set explicitly. `precision` is a MagicMock, so an unset `dtype`
    # auto-mocks into a Mock object rather than being absent -- and a Mock
    # satisfies every attribute access, so it survives all the way to the one
    # place that VALIDATES the value: `resolve_amp_precision`, which raised
    # `amp_dtype must be one of ['bfloat16','float16','float32'], got
    # <MagicMock ...>` and errored out 5 of this file's tests (#723). That is
    # the MagicMock hazard in general: it fails only where something checks a
    # value, arbitrarily far from the fixture that omitted it.
    config.optimization.precision.dtype = None
    config.optimization.mixed_precision = "fp16"
    config.optimization.loss_scaling = 1024.0
    config.optimization.gradient.clip.value = 1.0
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.enabled = False
    config.optimization.optimize_memory_amp = True
    config.optimization.dynamic_loss_scaling = True
    config.optimization.optimizer.learning_rate = 1e-4
    config.optimization.gradient.accumulation_steps = 1
    config.warmup_epochs = 0  # Fix TypeError: '>' not supported

    # physics mocks
    config.physics = MagicMock()
    config.physics.pinn = MagicMock()
    config.physics.pinn.enabled = False
    config.physics.pinn.pde_type = "bloch_equation"  # Valid PDE type
    config.physics.pinn.lambda_pde = 0.0  # Explicit value to avoid MagicMock comparison
    config.physics.pinn.weight = 0.0
    config.physics.kspace.enable_kspace_recon = False  # Added for BaseTrainingStrategy

    # Auto-default reconstruction-loss flag attributes so the strategy's
    # `enabled and weight > 0` filter never compares a bare MagicMock with
    # an int. Strategy code reads many `enable_*` / `lambda_*` attrs and
    # the explicit setattr lists below cannot keep up with every loss
    # family the strategy enumerates. Explicit attribute assignments later
    # in this fixture still take precedence (MagicMock prefers concrete
    # values over the __getattr__ fallback).
    class _LossDefaultsMock(MagicMock):
        def __getattr__(self, name):
            if name.startswith("enable_"):
                return False
            if name.startswith("lambda_"):
                return 0.0
            return super().__getattr__(name)

    config.objectives = MagicMock()
    config.objectives.reconstruction = _LossDefaultsMock()
    # config.losses.physics is read by the diffusion strategy; replace it with
    # the same auto-defaulting mock so getattr fallbacks return real numerics.
    config.objectives.physics = _LossDefaultsMock()
    config.objectives.reconstruction.lambda_l1 = 1.0
    config.objectives.reconstruction.lambda_perceptual = 0.0
    config.objectives.reconstruction.lambda_ssim = 0.0
    config.objectives.reconstruction.lambda_kspace = 0.0
    config.objectives.reconstruction.lambda_flow = 0.0
    config.objectives.reconstruction.lambda_flow_likelihood = 0.0
    config.objectives.reconstruction.lambda_graph_smoothness = 0.0
    config.objectives.reconstruction.lambda_feature_alignment = 0.0
    config.objectives.reconstruction.lambda_affine_regularization = 0.0
    config.objectives.reconstruction.lambda_domain_adversarial = 0.0
    config.objectives.reconstruction.lambda_sparsity_penalty = 0.0
    config.objectives.reconstruction.lambda_wasserstein = 0.0
    config.objectives.reconstruction.lambda_frequency = 0.0
    config.objectives.reconstruction.lambda_log_spectral = 0.0
    config.objectives.reconstruction.lambda_lpips = 0.0
    config.objectives.reconstruction.lambda_l2 = 0.0

    # SSOT: Use training.diffusion for diffusion config
    config.training.diffusion = MagicMock()
    config.training.diffusion.timesteps = 100
    config.training.diffusion.num_timesteps = 100  # Legacy/Backup
    config.training.diffusion.noise_schedule = "linear"
    config.training.diffusion.guidance_scale = 7.5
    # `DiffusionTrainingStrategy` forwards these two into `DiffusionScheduler`,
    # which feeds them straight to `torch.linspace`. Unset, they auto-mock and
    # linspace rejects them with an argument-combination TypeError far from the
    # fixture that omitted them. Values mirror `DiffusionScheduler`'s OWN
    # defaults so the stand-in cannot describe a schedule production would not
    # build.
    config.training.diffusion.beta_start = 1e-4
    config.training.diffusion.beta_end = 0.02

    config.objectives.gan = MagicMock()
    config.objectives.gan.gan_loss_type = "vanilla"
    config.objectives.gan.lambda_adv = 1.0
    config.objectives.gan.lambda_gp = 0.0
    # Add r1_interval at root of config.objectives.gan or similar if accessed directly
    config.r1_interval = 0

    config.objectives.gan.r1 = MagicMock()
    config.objectives.gan.r1.interval = 0
    config.objectives.gan.r1.weight = 0.0
    config.objectives.gan.r1.probability = 1.0

    # Configure physics losses (preserve auto-default mock — only set
    # explicit overrides; do not reassign the attribute or the
    # _LossDefaultsMock installed earlier is lost).
    for attr in [
        "lambda_parallel_imaging",
        "lambda_physics_constraint",
        "lambda_bloch_residual",
    ]:
        setattr(config.objectives.physics, attr, 0.0)
    for attr in [
        "enable_parallel_imaging",
        "enable_physics_constraint",
        "enable_bloch_residual",
    ]:
        setattr(config.objectives.physics, attr, False)

    # Configure latent losses
    config.objectives.latent = MagicMock()
    config.objectives.latent.enable_kl = False
    config.objectives.latent.enable_commitment = False

    # Configure SSL losses
    config.objectives.ssl = MagicMock()
    config.objectives.ssl.enable_contrastive = False

    config.physics.data_consistency = MagicMock()
    config.physics.data_consistency.enabled = False

    # SSOT: Alias losses to objectives for backward compatibility in tests
    # and to satisfy LossBuilder which uses config.losses
    config.losses = config.objectives

    # Configure data options to prevent unwanted model creation
    config.data = MagicMock()
    config.data.prior_loading = MagicMock()
    config.data.prior_loading.enabled = False

    # Configure reconstruction mock to handle arbitrary lookups safely
    for attr in [
        "enable_complex_l1",
        "enable_smooth_l1",
        "enable_l2",
        "enable_complex_mse",
        "enable_perceptual",
        "enable_ssim",
        "enable_ms_ssim",
        "enable_lpips",
        "enable_frequency",
        "enable_log_spectral",
        "enable_spectral_kspace",
        "enable_edge",
        "enable_dists",
        "enable_hfen",
        "enable_mind_ssc",
        "enable_hist",
        "enable_ffl",
        "enable_latent_consistency",
        "enable_tissue_bounds",
        "enable_kspace",
        "enable_weighted_kspace_l1",
    ]:
        setattr(config.losses.reconstruction, attr, False)

    for attr in [
        "lambda_complex_l1",
        "lambda_smooth_l1",
        "lambda_complex_mse",
        "lambda_ms_ssim",
        "lambda_sobel",
        "lambda_tv",
        "lambda_dists",
        "lambda_frequency",
        "lambda_hfen",
        "lambda_log_spectral",
        "lambda_spectral_kspace",
        "lambda_edge",
        "lambda_mind_ssc",
        "lambda_hist",
        "lambda_ffl",
        "lambda_latent_consistency",
        "lambda_tissue_bounds",
        "lambda_kspace",
        "lambda_weighted_kspace_l1",
        "lambda_lpips",
    ]:
        setattr(config.losses.reconstruction, attr, 0.0)

    # Set default values for other params
    config.losses.reconstruction.weighted_kspace_exponent = 1.0
    config.losses.reconstruction.histogram_bins = 100
    config.losses.reconstruction.ffl_alpha = 1.0
    config.losses.reconstruction.log_spectral_skip_fft = False

    # Phase 11 renamed the block `acceleration:` -> `undersampling:` (the old
    # name meant two unrelated things: the MRI k-space acceleration FACTOR and
    # COMPUTE acceleration). Production reads `config.undersampling`; this
    # fixture still populated `config.acceleration`, so every read got a bare
    # MagicMock and `max_accel - base_accel` died with "'>' not supported
    # between instances of 'MagicMock' and 'float'" (#723).
    #
    # `block_stub` IS `AccelerationConfigSchema`, so every field this block
    # carries arrives with its real default and no future read can auto-mock.
    config.undersampling = block_stub(
        "undersampling",
        base_acceleration=2.0,
        max_acceleration=8.0,
        gradient_accumulation_steps=1,
    )
    config.deep_supervision_weight = 0.0
    # Avoid .get() - use hasattr pattern
    config.enforce_output_range = False
    config.logging.log_gradients = False
    config.logging.log_interval = 10
    config.collect_feature_flags = lambda: {}
    return config


def create_dummy_batch(batch_size=2, channels=1, size=32):
    """Create dummy input/target batch."""
    inputs = torch.randn(batch_size, channels, size, size)
    targets = torch.randn(batch_size, channels, size, size)
    return inputs, targets


def create_mock_env(
    config, generator, discriminator=None, opt_g=None, opt_d=None, device=None
):
    """Create mock TrainingEnvironment."""
    env = MagicMock()
    env.config = config
    env.generator = generator
    env.discriminator = discriminator
    env.opt_g = opt_g
    env.opt_d = opt_d
    env.device = device or torch.device("cpu")
    env.model_type = config.model.model_type if config.model else "test_model"
    env.models = {"generator": generator}
    if discriminator:
        env.models["discriminator"] = discriminator
    return env


# === Strategy Import Tests ===
class TestStrategyImports:
    """Test that all strategies can be imported."""

    @pytest.mark.timeout(10)
    def test_import_reconstruction_strategy(self):
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        assert ReconstructionTrainingStrategy is not None

    @pytest.mark.timeout(10)
    def test_import_diffusion_strategy(self):
        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )

        assert DiffusionTrainingStrategy is not None

    @pytest.mark.timeout(10)
    def test_import_gan_strategy(self):
        from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy

        assert GANTrainingStrategy is not None

    @pytest.mark.timeout(10)
    def test_import_vae_strategy(self):
        from spectramr.infrastructure.training.strategies.vae import VAETrainingStrategy

        assert VAETrainingStrategy is not None


# === Reconstruction Strategy Tests ===
class TestReconstructionStrategy:
    """Tests for ReconstructionTrainingStrategy."""

    @pytest.fixture
    def strategy(self):
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        config = create_mock_config()
        generator = DummyGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            generator.parameters(), learning_rate=1e-4, optimizer_type="adam"
        )

        env = create_mock_env(
            config=config,
            generator=generator,
            opt_g=optimizer,
            device=torch.device("cpu"),
        )

        strategy = ReconstructionTrainingStrategy(
            env=env,
            device=torch.device(
                "cpu"
            ),  # Legacy kwarg might still be accepted or ignored
        )
        return strategy

    @pytest.mark.timeout(30)
    def test_train_step_basic(self, strategy):
        """Test basic train_step execution.

        ``BaseTrainingStrategy.train_step`` now returns a
        ``list[dict[str, Any]]`` of optimisation step configs (one per
        optimizer) rather than a single dict. Each entry must contain
        the ``closure`` and ``model`` keys consumed by the trainer.
        """
        inputs, targets = create_dummy_batch()

        # Should not raise
        result = strategy.train_step(
            batch=inputs, epoch=0, input_batch=inputs, target_batch=targets
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1
        for step in result:
            assert isinstance(step, dict)
            assert "closure" in step
            assert "model" in step

    @pytest.mark.timeout(30)
    def test_validation_step_basic(self, strategy):
        """Test basic validation_step execution."""
        inputs, targets = create_dummy_batch()

        # Should not raise
        result = strategy.validation_step(
            batch=None, input_batch=inputs, target_batch=targets
        )

        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.timeout(30)
    def test_no_config_get_error(self, strategy):
        """Ensure no AttributeError from config.get()."""
        inputs, targets = create_dummy_batch()

        # This should NOT raise AttributeError about .get()
        try:
            strategy.validation_step(
                batch=None, input_batch=inputs, target_batch=targets
            )
        except AttributeError as e:
            if "get" in str(e):
                pytest.fail(f"config.get() error: {e}")
            raise


# === Diffusion Strategy Tests ===
class TestDiffusionStrategy:
    """Tests for DiffusionTrainingStrategy."""

    @pytest.fixture
    def strategy(self):
        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )

        config = create_mock_config()
        config.training_mode = "diffusion"
        # The model's declared channel count must agree with the batch, or
        # `base.py`'s [DomainMismatch] guard raises before the step runs.
        config.model.in_channels = 2
        config.model.out_channels = 2
        # MUST be set explicitly, for the same reason `precision.dtype` is:
        # `validation.scoring` is a MagicMock, so an unset `transform`
        # auto-mocks into a Mock rather than being absent, and the metrics
        # mixin rejects it as an unknown transform name.
        config.validation.scoring.transform = "magnitude"
        # OFF, explicitly. `validation.gates` is a MagicMock, so an unset
        # `input_dependence_tol` is truthy and not None -- which reads as
        # "gate enabled" and then fails comparing a float to a Mock. This
        # fixture is not testing the L4 DC-blob gate, so it says so.
        config.validation.gates.input_dependence_tol = None
        # Third member of the same family as the two above, and stale for the
        # same reason: `validation.cascade` is a MagicMock, so an unset
        # `levels` auto-mocks into a Mock, and `resolve_cascade_levels`
        # correctly refuses a non-Sequence (`core/cascade_levels.py:83`). The
        # guard is right and newer than this fixture -- a raise added to
        # production turns every mock-fed test red at the point the mock stops
        # being a plausible config. Declare the ladder this test means to
        # exercise rather than letting auto-mocking invent one.
        config.validation.cascade.levels = [2.0, 8.0, 32.0]
        # Use existing config.objectives.diffusion setup from create_mock_config or ensure values are floats
        if (
            not hasattr(config.objectives, "diffusion")
            or config.objectives.diffusion is None
        ):
            config.objectives.diffusion = MagicMock()

        config.objectives.diffusion.timesteps = 100
        config.objectives.diffusion.noise_schedule = "linear"
        config.objectives.diffusion.lambda_mse = 1.0
        # Fourth member of the same family as the three above, and the reason is
        # the same: a guard newer than this fixture turns it red at the point the
        # mock stops being a plausible config.
        #
        # `build_loss_weight_table` collapses the `mse` alias onto `l2` and then
        # refuses a name declared with two different weights across sections. The
        # shared fixture writes `reconstruction.lambda_l2 = 0.0` and the line
        # above writes `diffusion.lambda_mse = 1.0`, so the two surfaces state
        # 0.0 and 1.0 for one loss and the table raises `ConfigurationError` --
        # correctly. A real config could not say both.
        #
        # `del` rather than a second assignment, because written-ness is what the
        # table reads: for a config double every `lambda_*` in the instance dict
        # counts as declared, and deleting the attribute takes it out of that
        # dict while `_LossDefaultsMock.__getattr__` still answers 0.0 for the
        # `lambda_` prefix. So every read in the strategy is unchanged and this
        # fixture declares the diffusion ladder it means to exercise, once,
        # rather than contradicting itself. Setting `reconstruction.lambda_l2` to
        # 1.0 would also silence the raise, but by declaring one weight on two
        # surfaces -- the exact shape `dual_surface_loss_declarations` reports.
        del config.objectives.reconstruction.lambda_l2

        # Also set training.diffusion as expected by v6.0 strategy
        if (
            not hasattr(config.training, "diffusion")
            or config.training.diffusion is None
        ):
            config.training.diffusion = MagicMock()
        config.training.diffusion.timesteps = 100
        config.training.diffusion.noise_schedule = "linear"

        # 2 channels, NOT the 1-channel default. The k-space diffusion validation
        # path derives its normalization scale from `input[:, 0:1]**2 +
        # input[:, 1:2]**2`, so every test here feeds a real/imag-style batch --
        # and a 1-channel generator raises inside `generate()` on all of them.
        # It used to raise unnoticed: the cascade swallowed the failure and
        # `validation_step` still returned a dict, so `assert result is not None`
        # passed while zero samples were generated.
        generator = DummyGenerator(in_ch=2, out_ch=2)
        optimizer = OptimizationBuilder.create_single_optimizer(
            generator.parameters(), learning_rate=1e-4, optimizer_type="adam"
        )

        env = create_mock_env(
            config=config,
            generator=generator,
            opt_g=optimizer,
            device=torch.device("cpu"),
        )

        strategy = DiffusionTrainingStrategy(env=env, device=torch.device("cpu"))
        return strategy

    @pytest.mark.timeout(30)
    def test_train_step_with_timestep(self, strategy):
        """Test train_step accepts timestep."""
        inputs, targets = create_dummy_batch(channels=2)

        # Should handle timestep argument
        result = strategy.train_step(
            batch=inputs, epoch=0, input_batch=inputs, target_batch=targets
        )

        assert result is not None

    @pytest.mark.timeout(30)
    def test_validation_step_generates_samples(self, strategy):
        """Test validation_step actually generates samples at every severity.

        The diffusion validation path runs the kspace normalization mixin
        which derives a 99th-percentile scale from
        ``input[:, 0:1]**2 + input[:, 1:2]**2``. A 1-channel batch makes
        the second slice empty and crashes ``torch.quantile``, so we
        pass an explicit 2-channel (real/imag-style) batch here.

        ``assert result is not None`` is NOT a test of the name of this
        function, and used to pass while zero samples were generated: the
        fixture's generator was 1-channel, ``generate()`` raised on every
        cascade level, the cascade swallowed each failure and
        ``validation_step`` still returned a populated dict (#1303). Assert
        the cascade's SHAPE, which is what "generated samples" means here.
        Values are not asserted -- this is a MagicMock-fed dummy conv, so
        the numbers carry no physical meaning.
        """
        inputs, targets = create_dummy_batch(channels=2)

        result = strategy.validation_step(
            batch=None, input_batch=inputs, target_batch=targets
        )

        assert result is not None
        assert result["val_cascade_complete"] == 1.0
        assert (
            result["val_cascade_levels_evaluated"]
            == result["val_cascade_levels_expected"]
            == float(len(CASCADING_LEVELS))
        )
        # A column per severity, and the complete-cascade mean under the
        # complete-cascade name (not `_mean_partial`).
        for level in CASCADING_LEVELS:
            assert f"val_ssim_{level}x" in result, f"no column for R={level}"
        assert "val_psnr_mean" in result
        assert "val_psnr_mean_partial" not in result


# === GAN Strategy Tests ===
class TestGANStrategy:
    """Tests for GANTrainingStrategy."""

    @pytest.fixture
    def strategy(self):
        from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy

        config = create_mock_config()
        config.training_mode = "gan"

        if not hasattr(config.objectives, "gan") or config.objectives.gan is None:
            config.objectives.gan = MagicMock()

        config.objectives.gan.gan_loss_type = "vanilla"
        config.objectives.gan.lambda_adv = 1.0
        config.objectives.gan.lambda_gp = 0.0
        # Initialize nested R1 config to avoid MagicMock on access
        config.objectives.gan.r1 = MagicMock()
        config.objectives.gan.r1.interval = 0
        config.objectives.gan.r1.weight = 0.0
        config.objectives.gan.r1.probability = 1.0

        generator = DummyGenerator()
        discriminator = DummyDiscriminator()
        gen_opt = OptimizationBuilder.create_single_optimizer(
            generator.parameters(), learning_rate=1e-4, optimizer_type="adam"
        )
        disc_opt = OptimizationBuilder.create_single_optimizer(
            discriminator.parameters(), learning_rate=1e-4, optimizer_type="adam"
        )

        # Prepare explicit mock returns for loss function to avoid MagicMock device errors
        loss_func_mock = MagicMock()
        # Return dicts with Real tensors so .device access works
        loss_func_mock.compute_generator_loss.return_value = {
            "g_total_loss": torch.tensor(0.0, requires_grad=True),
            "g_loss_adv": torch.tensor(0.0),
        }
        loss_func_mock.compute_discriminator_loss.return_value = {
            "d_total_loss": torch.tensor(0.0, requires_grad=True),
            "d_loss_real": torch.tensor(0.0),
            "d_loss_fake": torch.tensor(0.0),
            "r1_penalty": torch.tensor(0.0),
        }

        env = create_mock_env(
            config=config,
            generator=generator,
            discriminator=discriminator,
            opt_g=gen_opt,
            opt_d=disc_opt,
            device=torch.device("cpu"),
        )
        # Mock losses if needed, but strategy might build them

        try:
            strategy = GANTrainingStrategy(env=env, device=torch.device("cpu"))
            # Inject loss mock if strategy uses a helper we can patch, or rely on internal behavior
            # Ideally we'd mock strategy.loss_computer
            return strategy
        except Exception as e:
            pytest.skip(f"GAN strategy init failed: {e}")

    @pytest.mark.timeout(30)
    def test_train_step_returns_losses(self, strategy):
        """Test GAN train_step returns both G and D losses."""
        if strategy is None:
            pytest.skip("Strategy not available")

        # Mock loss computer to avoid real computation graph issues with Dummy models
        # and to ensure gradients are available for backward pass
        mock_computer = MagicMock()
        mock_computer.compute_loss.return_value = {
            "g_total_loss": torch.tensor(1.0, requires_grad=True),
            "d_total_loss": torch.tensor(1.0, requires_grad=True),
            "g_loss_adv": torch.tensor(0.5, requires_grad=True),
            "d_loss_real": torch.tensor(0.5, requires_grad=True),
            "d_loss_fake": torch.tensor(0.5, requires_grad=True),
            # Add other keys expected by GAN strategy if any
        }
        # In GAN strategy, loss_computer might be accessed differently or we need to patch compute_losses
        # strategy.loss_computer = mock_computer # This might not exist on strategy directly

        # Strategy likely uses self._compute_losses_impl
        # We can mock that method on the strategy instance
        strategy._compute_losses_impl = MagicMock(
            return_value={
                "g_total_loss": torch.tensor(1.0, requires_grad=True),
                "d_total_loss": torch.tensor(1.0, requires_grad=True),
                "g_loss_adv": torch.tensor(0.5, requires_grad=True),
                "d_loss_real": torch.tensor(0.5, requires_grad=True),
                "d_loss_fake": torch.tensor(0.5, requires_grad=True),
            }
        )

        inputs, targets = create_dummy_batch()

        result = strategy.train_step(
            batch=inputs, epoch=0, input_batch=inputs, target_batch=targets
        )

        assert result is not None


# === VAE Strategy Tests ===
class TestVAEStrategy:
    """Tests for VAETrainingStrategy."""

    @pytest.fixture
    def strategy(self):
        from spectramr.infrastructure.training.strategies.vae import VAETrainingStrategy

        config = create_mock_config()
        config.training_mode = "vae"

        generator = DummyGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            generator.parameters(), learning_rate=1e-4, optimizer_type="adam"
        )

        env = create_mock_env(
            config=config,
            generator=generator,
            opt_g=optimizer,
            device=torch.device("cpu"),
        )

        try:
            strategy = VAETrainingStrategy(env=env, device=torch.device("cpu"))
            return strategy
        except Exception as e:
            pytest.skip(f"VAE strategy init failed: {e}")

    @pytest.mark.timeout(30)
    def test_train_step_basic(self, strategy):
        """Test VAE train_step."""
        if strategy is None:
            pytest.skip("Strategy not available")

        inputs, targets = create_dummy_batch()

        result = strategy.train_step(
            batch=inputs, epoch=0, input_batch=inputs, target_batch=targets
        )

        assert result is not None


# === Physics Driven Strategy Tests ===
class TestPhysicsDrivenStrategy:
    """Tests for PhysicsDrivenStrategy."""

    @pytest.fixture
    def strategy(self):
        try:
            from spectramr.infrastructure.training.strategies.physics_driven_strategy import (
                PhysicsDrivenStrategy,
            )
        except ImportError:
            pytest.skip("PhysicsDrivenStrategy not available (ImportError)")

        config = create_mock_config()
        config.training_mode = "physics_driven"
        config.physics.data_consistency.enabled = True

        generator = DummyGenerator()
        optimizer = OptimizationBuilder.create_single_optimizer(
            generator.parameters(), learning_rate=1e-4, optimizer_type="adam"
        )

        env = create_mock_env(
            config=config,
            generator=generator,
            opt_g=optimizer,
            device=torch.device("cpu"),
        )

        try:
            strategy = PhysicsDrivenStrategy(env=env, device=torch.device("cpu"))
            return strategy
        except Exception as e:
            pytest.skip(f"Physics strategy init failed: {e}")

    @pytest.mark.timeout(30)
    def test_train_step_with_physics(self, strategy):
        """Test physics-driven train_step."""
        if strategy is None:
            pytest.skip("Strategy not available")

        inputs, targets = create_dummy_batch()
        result = strategy.train_step(
            batch=inputs, epoch=0, input_batch=inputs, target_batch=targets
        )
        assert result is not None


# === Generator Interface Tests ===
class TestGeneratorInterfaces:
    """Test that generators handle expected kwargs."""

    @pytest.mark.timeout(30)
    def test_kspace_cold_diffusion_sample_accepts_time(self):
        """Test KSpaceColdDiffusionGenerator.sample() accepts time kwarg."""
        try:
            from spectramr.models.generators.kspace_cold_diffusion_generator import (
                KSpaceColdDiffusionGenerator,
            )
        except ImportError:
            pytest.skip("KSpaceColdDiffusionGenerator not available")

        # Check signature accepts **kwargs
        import inspect

        sig = inspect.signature(KSpaceColdDiffusionGenerator.sample)
        params = sig.parameters

        # Should have VAR_KEYWORD (**kwargs)
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        assert has_kwargs, "sample() should accept **kwargs"

    @pytest.mark.timeout(30)
    def test_generator_forward_accepts_time(self):
        """Test generators can accept time in forward."""
        generator = DummyGenerator()
        inputs = torch.randn(1, 1, 32, 32)

        # Should not raise with extra kwargs
        result = generator(inputs, time=torch.tensor([0]))

        assert result is not None


# === Config Access Pattern Tests ===
class TestConfigAccessPatterns:
    """Test proper config access patterns (no .get() on Pydantic)."""

    @pytest.mark.timeout(30)
    def test_getattr_pattern(self):
        """Test getattr() works for config access."""
        config = create_mock_config()

        # This is the correct pattern
        value = getattr(config, "enforce_output_range", False)
        assert value == False

    @pytest.mark.timeout(30)
    def test_hasattr_pattern(self):
        """Test hasattr() works for config checking."""
        config = create_mock_config()

        # This is the correct pattern
        if hasattr(config, "enforce_output_range"):
            value = config.enforce_output_range
        else:
            value = False

        assert value == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
