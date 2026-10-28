from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy
from spectramr.infrastructure.training.strategies.vae import VAETrainingStrategy

from tests.utils.config_block_stub import block_stub
from tests.utils.data_config_stub import DataConfigStub


# --- Helper Classes for Config ---
class MockReconConfig:
    def __init__(self):
        self.enable_l1 = False
        self.lambda_l1 = 0.0
        self.enable_complex_l1 = False
        self.lambda_complex_l1 = 0.0
        self.enable_smooth_l1 = False
        self.lambda_smooth_l1 = 0.0
        # ``l2`` and ``mse`` are the same loss under alias collapse, so the two
        # declarations must agree: the weight SSOT compares the raw declared values
        # and rejects 0.0-beside-1.0 regardless of the ``enable_*`` flags.
        self.enable_l2 = True
        self.lambda_l2 = 1.0
        self.enable_complex_mse = False
        self.lambda_complex_mse = 0.0
        self.enable_weighted_kspace_l1 = False
        self.lambda_weighted_kspace_l1 = 0.0
        self.enable_frequency = False
        self.lambda_frequency = 0.0
        self.enable_mse = True
        self.lambda_mse = 1.0
        self.enable_lpips = False
        self.lambda_lpips = 0.0
        self.enable_perceptual = False
        self.lambda_perceptual = 0.0
        self.enable_ssim = False
        self.lambda_ssim = 0.0
        self.enable_ms_ssim = False
        self.lambda_ms_ssim = 0.0
        self.enable_dists = False
        self.lambda_dists = 0.0
        self.enable_hfen = False
        self.lambda_hfen = 0.0
        self.enable_log_spectral = False
        self.lambda_log_spectral = 0.0
        self.enable_spectral_kspace = False
        self.lambda_spectral_kspace = 0.0
        self.enable_edge = False
        self.lambda_edge = 0.0
        self.lambda_sobel = 0.0
        self.enable_mind_ssc = False
        self.lambda_mind_ssc = 0.0
        self.enable_hist = False
        self.lambda_hist = 0.0
        self.enable_ffl = False
        self.lambda_ffl = 0.0
        self.enable_latent_consistency = False
        self.lambda_latent_consistency = 0.0
        self.enable_tissue_bounds = False
        self.lambda_tissue_bounds = 0.0
        self.enable_gradient_penalty = False
        self.lambda_gradient_penalty = 0.0
        self.lambda_tv = 0.0
        self.enable_kspace = False
        self.lambda_kspace = 0.0

        # Added for diffusion strategy validation
        self.frequency_weighted_l1_kspace_alpha = 1.0
        self.enable_frequency_weighted_l1_kspace = False
        self.lambda_frequency_weighted_l1_kspace = 0.0
        self.enable_background_suppression = False
        self.lambda_background_suppression = 0.0
        self.enable_energy_conservation = False
        self.lambda_energy_conservation = 0.0
        self.enable_frequency_domain = False
        self.lambda_frequency_domain = 0.0


class MockGanConfig:
    def __init__(self):
        self.enable_adversarial = False
        self.enable_r1 = False
        self.lambda_gp = 0.0


class MockPhysicsConfig:
    def __init__(self):
        self.enable_physics_constraint = False
        self.lambda_physics_constraint = 0.0
        self.enable_bloch_residual = False
        self.lambda_bloch_residual = 0.0


class MockKspaceConfig:
    def __init__(self):
        self.enable_kspace_recon = False
        self.enforce_hermitian_symmetry = False


class MockPhysicsTopConfig:
    def __init__(self):
        self.kspace = MockKspaceConfig()
        self.data_consistency = MagicMock()
        self.data_consistency.enabled = False
        # Real PhysicsConfigSchema (src/config/schemas/physics.py:1005) populates
        # these via default_factory, so they're always present at runtime.
        # _log_config_features in base.py reads them — keep the mock surface
        # in sync or AttributeError fires before the `enabled` short-circuit.
        self.compressed_sensing = MagicMock()
        self.compressed_sensing.enabled = False
        self.pinn = MagicMock()
        self.pinn.enabled = False
        self.digital_twin = MagicMock()
        self.digital_twin.enabled = False


class MockLossesConfig:
    def __init__(self):
        self.reconstruction = MockReconConfig()
        self.gan = MockGanConfig()
        self.physics = MockPhysicsConfig()
        self.adversarial = MagicMock()
        self.adversarial.enable = False
        self.regularization = MagicMock()
        self.regularization.enable_kl = False
        self.diffusion = MagicMock()
        self.diffusion.enable = False
        self.latent = MagicMock()
        self.latent.enable = False
        self.ssl = MagicMock()
        self.ssl.enable = False
        self.self_supervised = MagicMock()
        self.self_supervised.enable = False


def MockOptimizationConfig():  # noqa: N802 -- kept as a factory so call sites are unchanged
    """The real `optimization:` schema with the legacy flat knobs routed home.

    Was a hand-rolled class carrying `gradient_clip_value` / `clip_method` /
    `enable_gradient_clipping` flat. Production reads
    `optimization.gradient.clip.{value,method,enabled}`, so the reader saw none
    of them and fell back to schema defaults -- the test then pinned values it
    never chose. All five kwargs below were checked against
    `flat_to_canonical("optimization")`; an unrouted one becomes a stray
    attribute and silently reintroduces exactly that failure.
    """
    return block_stub(
        "optimization",
        gradient_clip_value=1.0,
        enable_gradient_clipping=True,
        gradient_clip_method="norm",
        use_amp=False,
        gradient_accumulation_steps=1,
    )


def MockLoggingConfig():  # noqa: N802 -- factory, so call sites are unchanged
    """The real `logging:` schema; the flat double was missing `intervals`.

    `strategies/base.py` reads `logging.intervals.*`, which a two-attribute
    stand-in cannot provide. Kept as a factory so the two call sites read the
    same as before.
    """
    return block_stub("logging", log_gradients=False, log_interval=100)


class MockMetricsConfig:
    def __init__(self):
        self.compute_psnr = True
        self.domain = "image"
        self.transform = None


class MockModelConfig:
    """The `model:` block readers walk; the flat double was missing the critic.

    Same defect as `MockDataConfig` below: `discriminator_component` is a
    declared field on `ModelConfigSchema`, so Pydantic materializes it on every
    real config -- `None` when the arm configures no critic. Omitting it here
    made this double assert a shape the schema does not have, and the elected
    critic-name owner raises on that by design rather than reporting "this arm
    has no critic" (#1921, non-negotiable 3).
    """

    def __init__(self):
        self.domain = "image"
        self.target_domain = "image"
        self.in_channels = 1
        self.out_channels = 1
        self.model_type = "gan"
        self.discriminator_component = None


def MockDataConfig():  # noqa: N802 -- factory, so call sites are unchanged
    """The real `data:` schema; the flat double was missing `processing`.

    Readers walk `data.processing.*` after the decomposition, which a
    three-attribute stand-in cannot provide. `prior_loading` is left as a
    MagicMock override because the test only needs its `.enabled` gate.
    """
    stub = DataConfigStub(normalize_kspace=False, normalization_kwargs={})
    stub.prior_loading = MagicMock()
    stub.prior_loading.enabled = False
    return stub


class MockDiffusionConfig:
    def __init__(self):
        from spectramr.config.schemas.training.diffusion import TrainingConfigDiffusion

        self.timesteps = 1000
        self.num_timesteps = 1000
        self.noise_schedule = "linear"
        # `DiffusionTrainingStrategy.__init__` forwards `beta_start`/`beta_end`
        # UNCONDITIONALLY (issue #799). A hand-rolled double that omits them is a
        # second implementation of the schema missing fields the reader requires.
        # Sourced from the schema rather than restated: a third copy of these
        # numbers is how the defaults drift apart in the first place.
        fields = TrainingConfigDiffusion.model_fields
        self.beta_start = fields["beta_start"].default
        self.beta_end = fields["beta_end"].default


class MockTrainingConfig:
    def __init__(self):
        self.losses = MockLossesConfig()
        self.optimization = MockOptimizationConfig()
        self.logging = MockLoggingConfig()
        self.metrics = MockMetricsConfig()
        self.model = MockModelConfig()
        self.physics = MockPhysicsTopConfig()
        self.data = MockDataConfig()
        self.training = MagicMock()
        self.training.diffusion = MockDiffusionConfig()
        self.training.enforce_output_range = False
        self.deep_supervision_weight = 0.0
        self.acceleration = {}


# --- Fixtures ---


@pytest.fixture
def mock_loss_computer():
    with patch("spectramr.models.losses.computers.UnifiedGANLossComputer") as mock:
        yield mock


@pytest.fixture
def mock_diffusion_loss_computer():
    with patch("spectramr.models.losses.computers.UnifiedDiffusionLossComputer") as mock:
        yield mock


@pytest.fixture
def mock_vae_loss_computer():
    with patch("spectramr.models.losses.computers.UnifiedVAELossComputer") as mock:
        yield mock


@pytest.fixture
def mock_env():
    # We use a simple MagicMock and assign concrete config
    env = MagicMock()
    env.config = MockTrainingConfig()
    env.device = torch.device("cpu")

    # Mock generator/discriminator
    env.generator = MagicMock(spec=nn.Module)
    env.generator.parameters.return_value = iter([])
    env.generator.training = True

    env.discriminator = MagicMock(spec=nn.Module)
    env.discriminator.parameters.return_value = iter([])
    env.discriminator.training = True

    env.opt_g = MagicMock()
    env.opt_d = MagicMock()

    return env


@pytest.fixture
def mock_state(mock_env):
    state = MagicMock(spec=TrainingEnvironment)
    state.config = mock_env.config
    state.device = mock_env.device
    state.generator = mock_env.generator
    state.discriminator = mock_env.discriminator
    state.opt_g = mock_env.opt_g
    state.opt_d = mock_env.opt_d
    state.step = 0
    state.epoch = 0

    # Critical: Ensure attributes are accessible
    state.model_type = "gan"
    return state


@patch("spectramr.infrastructure.di.di_container.resolve_service")
def test_gan_strategy_init(mock_resolve, mock_env, mock_state, mock_loss_computer):
    mock_resolve.return_value = MagicMock()  # LoggingService

    # GANTrainingStrategy init
    strategy = GANTrainingStrategy(env=mock_env, device="cpu", state=mock_state)

    assert strategy is not None
    assert hasattr(strategy, "loss_computer")
    # Check mixin methods
    assert hasattr(strategy, "setup_adversarial")
    assert hasattr(strategy, "train_step_adversarial")
    # Check inheritance
    assert isinstance(strategy, BaseTrainingStrategy)


@patch("spectramr.infrastructure.di.di_container.resolve_service")
def test_diffusion_strategy_init(
    mock_resolve, mock_env, mock_state, mock_diffusion_loss_computer
):
    mock_resolve.return_value = MagicMock()

    # Update config for diffusion
    mock_state.config.model.model_type = "diffusion"

    strategy = DiffusionTrainingStrategy(env=mock_env, device="cpu", state=mock_state)

    assert strategy is not None
    assert hasattr(strategy, "loss_computer")
    # Check KspaceMixin methods
    assert hasattr(strategy, "setup_kspace_components")
    # Check inheritance
    assert isinstance(strategy, BaseTrainingStrategy)


@patch("spectramr.infrastructure.di.di_container.resolve_service")
def test_vae_strategy_init(mock_resolve, mock_env, mock_state, mock_vae_loss_computer):
    mock_resolve.return_value = MagicMock()

    mock_state.config.model.model_type = "vae"

    # VAETrainingStrategy(env, device, state, **kwargs)
    strategy = VAETrainingStrategy(env=mock_env, device="cpu", state=mock_state)

    assert strategy is not None
    assert hasattr(strategy, "loss_computer")
    # Check PerceptualMixin methods via Base
    assert hasattr(strategy, "_compute_training_metrics")
    assert hasattr(strategy, "validation_step")
