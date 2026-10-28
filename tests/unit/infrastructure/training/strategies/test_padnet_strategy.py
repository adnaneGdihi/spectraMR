"""Unit tests for PaDNetTrainingStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.padnet_strategy import (
    PaDNetTrainingStrategy,
)
from tests.utils.config_block_stub import block_stub


class MockConfig:
    """Mock config that returns numeric defaults and supports nested access."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        # For lambda_* return 0.0 (loss weights)
        if name.startswith("lambda_"):
            return 0.0
        # For enable_* return False (feature flags)
        if name.startswith("enable_"):
            return False
        # For *_weight return 0.0 (regularization weights)
        if name.endswith("_weight"):
            return 0.0
        # For *_steps return 1 (gradient accumulation, etc.)
        if "steps" in name:
            return 1
        # For everything else, return a new MockConfig (supports infinite nesting)
        return MockConfig()

    def __gt__(self, other):
        """Support > comparison (always False for weight comparisons)."""
        return False

    def __lt__(self, other):
        """Support < comparison."""
        return False

    def __ge__(self, other):
        """Support >= comparison."""
        return False

    def __le__(self, other):
        """Support <= comparison."""
        return False

    def __eq__(self, other):
        """Support == comparison."""
        return other == 0 or other is False or other is None

    def __bool__(self):
        """Support boolean evaluation (False for disabled features)."""
        return False

    def __int__(self):
        """Support int() conversion (for gradient_accumulation_steps, etc.)."""
        return 1

    def __float__(self):
        """Support float() conversion."""
        return 0.0

    def __index__(self):
        """Support indexing operations."""
        return 1


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Mock generator with predict_q_maps
    gen = MagicMock()
    gen.predict_q_maps = MagicMock(return_value=torch.randn(2, 3, 64, 64))  # T1, T2, PD
    gen.bloch_solver = MagicMock()
    gen.bloch_solver.return_value = torch.randn(2, 64, 64, 2)  # Signal [B, H, W, 2]
    gen.training = True
    gen.train = MagicMock()
    gen.eval = MagicMock()
    env.generator = gen
    env.models = {"generator": gen}

    # Config using MockConfig
    config = MockConfig()
    # The SHARED stub. Production reads `optimization.gradient.clip.method`
    # (standard_optimizer_stepper.py:30, `method not in _valid_clip`), and a
    # bare MockConfig fabricates a fresh MockConfig for `gradient` -- which is
    # unhashable, so the membership test raises rather than the value being
    # wrong. block_stub routes the flat legacy kwargs from RENAMES.
    config.optimization = block_stub(
        "optimization",
        learning_rate=1e-4,
        use_amp=False,
        enable_gradient_clipping=False,
        gradient_clip_method="norm",
        gradient_clip_value=1.0,
    )
    config.model = MockConfig(model_type="padnet", domain="image", in_channels=2, out_channels=2)
    config.losses = MockConfig()
    config.losses.reconstruction = MockConfig(enable_l1=False, lambda_l1=0.0)
    config.physics = MockConfig()
    config.physics.kspace = MockConfig(enable_kspace_recon=False)
    # `logging.log_interval` folded to `logging.intervals.log`, which the
    # training loop reads in a modulo -- a fabricated MockConfig there is a
    # TypeError, not a wrong interval.
    config.logging = block_stub("logging", log_interval=10)
    config.data = MockConfig(normalize_kspace=False)

    # Diffusion-specific config (required by DiffusionTrainingStrategy parent)
    config.training = MockConfig(
        training_mode="padnet",
        diffusion=MockConfig(
            timesteps=1000, noise_schedule="linear", beta_start=1e-4, beta_end=0.02
        ),
        enforce_output_range=False,
    )

    env.config = config
    env.model_type = "padnet"

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate PaDNetTrainingStrategy (NEW API - env only)."""
    # Patch PhysicsInformedLoss to avoid complex init
    with patch(
        "spectramr.infrastructure.training.strategies.padnet_strategy.PhysicsInformedLoss"
    ) as MockLoss:
        MockLoss.return_value.return_value = {
            "total": torch.tensor(1.0),
            "l2_image": torch.tensor(0.8),
            "regularization": torch.tensor(0.2),
        }
        strategy = PaDNetTrainingStrategy(env=mock_env)
        return strategy


def test_initialization(strategy):
    """Test initialization."""
    assert strategy is not None
    assert strategy.physics_loss is not None


def test_compute_losses_impl_padnet(strategy, mock_env):
    """Test PaDNet specific loss computation."""
    input_batch = torch.randn(2, 1, 64, 64)  # T1w
    target_batch = torch.randn(2, 1, 64, 64)  # T2w

    # Batch metadata via kwargs
    batch_kwargs = {"batch": {"tr": 500.0, "te": 10.0}}

    losses = strategy._compute_losses_impl(input_batch, target_batch, epoch=0, **batch_kwargs)

    assert "g_total_loss" in losses
    assert "g_loss_mse" in losses
    assert "g_loss_reg" in losses

    # Verify q-map prediction
    mock_env.generator.predict_q_maps.assert_called_with(input_batch)

    # Verify bloch solver usage
    mock_env.generator.bloch_solver.assert_called()


def test_compute_losses_impl_missing_predict_q_maps(strategy, mock_env):
    """Test error when predict_q_maps is missing."""
    # Create a new generator without predict_q_maps
    gen_without_method = MagicMock(spec=nn.Module)
    # Ensure it doesn't have the method by deleting it after spec
    if hasattr(gen_without_method, "predict_q_maps"):
        del gen_without_method.predict_q_maps

    mock_env.generator = gen_without_method

    # Supply tr/te so the (correct) sequence-param guard passes and execution
    # reaches the predict_q_maps check this test is actually asserting on.
    with pytest.raises(NotImplementedError, match="must implement predict_q_maps"):
        strategy._compute_losses_impl(
            torch.randn(1, 1, 32, 32),
            torch.randn(1, 1, 32, 32),
            0,
            batch={"tr": 500.0, "te": 10.0},
        )


class TestModelInputContract:
    """Non-negotiable 14, wrong in the OPPOSITE direction to the family.

    PaDNet subclasses ``DiffusionTrainingStrategy`` and so inherited
    ``snapshot_prepared_is_model_input = False`` + ``snapshot_model_input_tag =
    "diffusion_step"`` — the cold-diffusion carve-out for a forward process it
    does not run. Every PaDNet artifact therefore *under-claimed* an honest
    snapshot and pointed readers at a ``diffusion_step`` that neither exists nor
    should. The six sibling fixes add a declaration; this one removes a lie.
    """

    def test_declares_prepared_is_the_model_input(self):
        assert PaDNetTrainingStrategy.snapshot_prepared_is_model_input is True
        assert PaDNetTrainingStrategy.snapshot_model_input_tag is None

    def test_the_network_receives_input_batch_verbatim(self, strategy, mock_env):
        """The claim above is only honest because nothing degrades the input.

        This is the assertion that earns the ``True``: ``input_batch`` reaches
        ``predict_q_maps`` unchanged, and the Bloch physics happen downstream on
        the PREDICTED q-maps. Asserting the flag without asserting this would
        pin the declaration to itself.
        """
        seen = {}

        def _predict_q_maps(x):
            seen["x"] = x
            return torch.randn(1, 3, 32, 32, requires_grad=True)

        mock_env.generator.predict_q_maps = _predict_q_maps

        input_batch = torch.randn(1, 1, 32, 32)
        strategy._compute_losses_impl(
            input_batch,
            torch.randn(1, 1, 32, 32),
            0,
            batch={"tr": 500.0, "te": 10.0},
        )

        assert seen["x"] is input_batch, (
            "PaDNet must feed input_batch verbatim; anything else would make "
            "snapshot_prepared_is_model_input=True a false claim"
        )

    def test_no_declaration_is_stashed(self, strategy, mock_env):
        """`True` and a stashed declaration would contradict each other.

        The wrapper returns early on ``snapshot_prepared_is_model_input`` and
        would never emit a stashed tensor, so a declaration here would be dead
        weight that silently pinned one step's activations.
        """
        mock_env.generator.predict_q_maps = lambda x: torch.randn(1, 3, 32, 32, requires_grad=True)
        strategy._declared_model_input = None

        strategy._compute_losses_impl(
            torch.randn(1, 1, 32, 32),
            torch.randn(1, 1, 32, 32),
            0,
            batch={"tr": 500.0, "te": 10.0},
        )

        assert strategy._declared_model_input is None


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: only the physics module's l2_image runs; nothing else reaches the objective.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert PaDNetTrainingStrategy.__dict__["folds_image_losses"] is False
    assert PaDNetTrainingStrategy.__dict__["inline_losses"] == frozenset({"l2"})
