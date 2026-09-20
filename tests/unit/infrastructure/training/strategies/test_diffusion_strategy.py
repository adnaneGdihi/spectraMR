"""Unit tests for diffusion training strategy."""

import inspect
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.core.cascading_validation import CASCADING_LEVELS
from spectramr.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)
from spectramr.infrastructure.training.strategies import diffusion as diffusion_mod
from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)
from tests.utils.mock_environment import create_mock_training_env


def test_curriculum_reads_loop_state_not_frozen_step_or_dead_getattr() -> None:
    """The R-curriculum reads the live iteration from the ``loop_state`` seam.

    Lineage of this read (each step removed a layer of inert-ness):
      1. ``getattr(self.trainer, "global_step", current_step)`` — the per-step
         executor never stored ``global_step``, so the getattr ALWAYS took its
         fallback (dead read).
      2. WS-1 minimal fix: ``current_iter = current_step`` — but that param was
         overwritten inside ``_compute_and_log_debug_info`` by
         ``current_step = self.env.step``, and ``TrainingEnvironment`` is frozen
         so ``env.step`` is a constant 0 (still inert — pinned to the 2x stage).
      3. WS-3 PR-3: ``current_iter = self.loop_state.iteration`` — the training
         loop advances this each step, so the diagnostic finally tracks real
         progress. This pins (3) and forbids any regression to (1) or (2).
    """
    assert 'getattr(self.trainer, "global_step"' not in inspect.getsource(diffusion_mod)
    # Scope the seam assertions to the method that owns the curriculum read, so
    # other methods' (separate) env.step reads don't confound this regression.
    method_src = inspect.getsource(
        DiffusionTrainingStrategy._compute_and_log_debug_info
    )
    assert "current_iter = self.loop_state.iteration" in method_src
    # The frozen-env.step *reads* inside this debug/curriculum method are gone
    # (ignore comment lines, which legitimately reference the old inert read).
    code_lines = [
        ln for ln in method_src.splitlines() if not ln.lstrip().startswith("#")
    ]
    assert not any("self.env.step" in ln for ln in code_lines)


def test_metric_and_tb_steps_read_loop_state_not_frozen_env_step() -> None:
    """WS-3 follow-up: the two remaining diffusion ``env.step`` inert reads are
    migrated to the live ``loop_state`` seam.

    * ``_compute_losses_impl`` feeds ``current_step`` to the
      ``current_step % train_metric_interval`` throttle — with the frozen
      ``env.step`` 0 it fired EVERY step (a hidden per-step metric cost).
    * ``_log_validation_images_to_tensorboard`` labels TB images with the step;
      0 mislabelled them.

    Both now use ``resolve_loop_iteration(self)``; the active (non-comment) code
    no longer reads ``self.env.step``.

    Since #585 ``_log_validation_images_to_tensorboard`` reaches the seam through
    ``_validation_image_step()``. That indirection exists so the step label is
    unit-testable (``test_validation_image_step_label.py``), so the guard accepts the
    seam directly OR the helper that owns it, and pins the helper to the seam
    separately. A regression to ``env.step`` -- or back to the per-cascade-level
    ``validation_step_count`` -- still fails here.
    """
    seam = "resolve_loop_iteration(self)"
    accepted = {
        "_compute_losses_impl": (seam,),
        "_log_validation_images_to_tensorboard": (seam, "_validation_image_step()"),
        "_validation_image_step": (seam,),
    }
    for method in (
        DiffusionTrainingStrategy._compute_losses_impl,
        DiffusionTrainingStrategy._log_validation_images_to_tensorboard,
        DiffusionTrainingStrategy._validation_image_step,
    ):
        code = "\n".join(
            ln
            for ln in inspect.getsource(method).splitlines()
            if not ln.lstrip().startswith("#")
        )
        assert "self.env.step" not in code, f"{method.__name__} still reads env.step"
        assert any(
            token in code for token in accepted[method.__name__]
        ), f"{method.__name__} should reach the loop_state seam"

    # The label is derived in exactly one place, and that place must never read the
    # per-cascade-level counter (#585). The counter is still bumped inside
    # ``_log_validation_images_to_tensorboard`` -- that is its real job -- so the
    # prohibition is scoped to the helper that owns the label. Compare EXECUTABLE code
    # only: the helper's docstring names the counter to explain what it must not do,
    # which a plain substring check would read as a violation.
    import ast
    import textwrap

    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(DiffusionTrainingStrategy._validation_image_step)
        )
    )
    fn = tree.body[0]
    body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
    executable = "\n".join(ast.unparse(node) for node in body)
    assert "validation_step_count" not in executable, (
        "_validation_image_step must not derive the step label from the "
        "per-cascade-level validation counter (#585)"
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 1, kernel_size=3, padding=1),
    )


@pytest.fixture
def mock_diffusion_config():
    """Create mock diffusion configuration."""
    config = MagicMock()
    config.training_mode = "diffusion"
    config.optimization = MagicMock()
    config.optimization.optimizer.learning_rate = 1e-3
    config.optimization.precision.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    config.optimization.precision.dtype = "float32"
    # A real config always carries these; without them the auto-mock value of
    # gradient_clip_method fails StandardOptimizerStepper's raise-on-unknown
    # validation (pitfall #9) at strategy construction.
    config.optimization.gradient.clip.enabled = False
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.value = 1.0
    config.objectives = MagicMock()
    config.training.diffusion = MagicMock()
    config.training.diffusion.num_timesteps = 1000
    config.training.diffusion.noise_schedule = "linear"
    config.training.diffusion.beta_start = 1e-4
    config.training.diffusion.beta_end = 0.02
    config.training.diffusion.beta_start = 0.0001
    config.device = "cpu"

    # Provide defaults to avoid attribution errors
    config.acceleration = None
    config.deep_supervision_weight = 0.0

    config.data = MagicMock()
    config.data.prior_loading = MagicMock()
    config.data.prior_loading.enabled = False

    config.physics = (
        None  # Explicitly None to prevent MagicMock auto-creation for phys_cfg
    )

    config.model = MagicMock()
    config.model.model_type = "diffusion"
    config.model.in_channels = 1
    config.model.out_channels = 1

    # Configure losses with explicit object to prevent MagicMock auto-creation
    class MockLosses:
        def __init__(self):
            self.reconstruction = MagicMock()
            self.reconstruction.frequency_weighted_l1_kspace_alpha = 0.0
            self.reconstruction.enable_complex_l1 = False
            self.reconstruction.lambda_complex_l1 = 0.0
            self.reconstruction.enable_log_spectral = False
            self.reconstruction.lambda_log_spectral = 0.0
            self.reconstruction.enable_frequency_weighted_l1_kspace = False
            self.reconstruction.lambda_frequency_weighted_l1_kspace = 0.0
            self.reconstruction.enable_background_suppression = False
            self.reconstruction.lambda_background_suppression = 0.0
            self.reconstruction.enable_l1 = False
            self.reconstruction.lambda_l1 = 0.0
            self.reconstruction.enable_l2 = False
            self.reconstruction.lambda_l2 = 0.0
            self.reconstruction.enable_energy_conservation = False
            self.reconstruction.lambda_energy_conservation = 0.0
            self.reconstruction.enable_frequency_domain = False
            self.reconstruction.lambda_frequency_domain = 0.0
            self.reconstruction.enable_hfen = False
            self.reconstruction.lambda_hfen = 0.0
            self.reconstruction.enable_rician_consistency = False
            self.reconstruction.lambda_rician_consistency = 0.0
            self.gan = None
            self.physics = None
            self.latent = None
            self.ssl = None
            self.diffusion = config.objectives.diffusion

    config.losses = MockLosses()

    # [FIX] Explicitly disable new config paths to allow testing of fallback logic
    config.training = MagicMock()
    config.training.diffusion = MagicMock()
    config.training.diffusion.timesteps = 1000
    config.training.diffusion.num_timesteps = 1000
    config.training.diffusion.noise_schedule = "linear"
    config.training.diffusion.beta_start = 1e-4
    config.training.diffusion.beta_end = 0.02
    config.diffusion = None

    return config


@pytest.fixture
def training_env(simple_model, mock_diffusion_config):
    """Create training environment."""
    opt = OptimizationBuilder.create_single_optimizer(
        simple_model.parameters(), learning_rate=1e-4, optimizer_type="adam"
    )

    env = create_mock_training_env(
        config=mock_diffusion_config,
        device="cpu",
        generator=simple_model,
        opt_g=opt,
        model_type="diffusion",
    )
    return env


@pytest.fixture(autouse=True)
def mock_resolve_service():
    """Mock resolve_service to avoid DI errors."""
    with patch(
        "spectramr.infrastructure.di.di_container.resolve_service"
    ) as mock_resolve:
        mock_resolve.return_value = MagicMock()
        yield mock_resolve


class TestDiffusionStrategyInitialization:
    """Test DiffusionTrainingStrategy initialization."""

    def test_init_default_parameters(self, mock_diffusion_config, training_env):
        """Test initialization with default parameters."""
        # Need to provide state with optimizer for initialization validation
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert strategy is not None

    def test_init_with_device_string(self, mock_diffusion_config, training_env):
        """Test initialization with device string."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert strategy is not None

    def test_init_with_training_env(self, mock_diffusion_config, training_env):
        """Test initialization with provided training environment."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert strategy is not None


class TestDiffusionStrategyParameters:
    """Test diffusion parameter initialization."""

    def test_initialize_diffusion_parameters(self, training_env):
        """Test diffusion parameters are initialized."""
        with patch.object(
            DiffusionTrainingStrategy, "initialize_diffusion_parameters"
        ) as mock_init:
            # We mock the method, but we must call super().__init__ which calls it
            # But wait, super().__init__ calls initialize_diffusion_parameters?
            # No, DiffusionTrainingStrategy.__init__ calls it.

            # Since we patch it on the class, the instance method is replaced.
            strategy = DiffusionTrainingStrategy(env=training_env)
            mock_init.assert_called()

    def test_timesteps_default(self, mock_diffusion_config, training_env):
        """Test default timesteps value."""
        mock_diffusion_config.training.diffusion.num_timesteps = 1000

        strategy = DiffusionTrainingStrategy(env=training_env)
        # Check if parameters were initialized correctly (via internal attribute check if possible)
        # Assuming internal storage in self.num_timesteps (mixin behavior)
        if hasattr(strategy, "num_timesteps"):
            assert strategy.num_timesteps == 1000

    def test_timesteps_custom(self, mock_diffusion_config, training_env):
        """Test custom timesteps value."""
        mock_diffusion_config.training.diffusion.timesteps = 500

        strategy = DiffusionTrainingStrategy(env=training_env)
        if hasattr(strategy, "num_timesteps"):
            assert strategy.num_timesteps == 500


class TestDiffusionStrategyComponents:
    """Test strategy component initialization."""

    def test_loss_computer_initialization(self, training_env):
        """Test loss computer initialization."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert hasattr(strategy, "loss_computer")

    def test_mask_generator_is_taken_from_the_models_process(
        self, mock_diffusion_config, training_env
    ):
        """The strategy borrows; it no longer builds a second generator (#2056).

        The generator carries the process in production -- every ``model_type``
        the guard matches resolves to ``KSpaceColdDiffusionGenerator``, which
        builds one unconditionally -- so attaching one here is what makes this
        fixture the shape the strategy actually meets.
        """
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        mock_diffusion_config.training_mode = "kspace_cold_diffusion"
        training_env.model_type = "kspace_cold_diffusion"
        mock_diffusion_config.model.model_type = "kspace_cold_diffusion"
        process = KSpaceUndersamplingProcess(num_timesteps=28)
        training_env.generator.kspace_process = process

        strategy = DiffusionTrainingStrategy(env=training_env)

        assert strategy.mask_generator is process.mask_generator

    def test_a_model_without_a_process_fails_setup_rather_than_masking_wrong(
        self, mock_diffusion_config, training_env
    ):
        """Anti-vacuity for the test above, and non-negotiable 3.

        Declaring cold diffusion while the generator cannot degrade k-space is
        a mis-wiring. Building a strategy-side generator instead -- what this
        path did until #2056 -- produced masks from a second resolution of the
        same block that nothing compared against.
        """
        mock_diffusion_config.training_mode = "kspace_cold_diffusion"
        training_env.model_type = "kspace_cold_diffusion"
        mock_diffusion_config.model.model_type = "kspace_cold_diffusion"

        with pytest.raises(ValueError, match="kspace_process"):
            DiffusionTrainingStrategy(env=training_env)


class TestDiffusionStrategyConfiguration:
    """Test diffusion strategy configuration."""

    def test_verify_strategy_config_diffusion(self, training_env):
        """Test config verification for diffusion mode."""
        # Should not raise
        DiffusionTrainingStrategy(env=training_env)

    def test_verify_strategy_config_kspace(self, mock_diffusion_config, training_env):
        """Test config verification for k-space cold diffusion."""
        mock_diffusion_config.training_mode = "kspace_cold_diffusion"
        training_env.model_type = "kspace_cold_diffusion"

        # Should not raise
        DiffusionTrainingStrategy(env=training_env)


class TestDiffusionStrategyTrainingLoop:
    """Test training loop methods."""

    def test_train_step_method_exists(self, training_env):
        """Test that train_step method exists."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert hasattr(strategy, "train_step")

    def test_validation_step_method_exists(self, training_env):
        """Test that validation_step method exists."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert hasattr(strategy, "validation_step")


class TestDiffusionStrategyLossComputation:
    """Test loss computation for diffusion."""

    def test_compute_losses_method_exists(self, training_env):
        """Test that _compute_losses method exists."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert hasattr(strategy, "_compute_losses_impl")

    def test_compute_losses_delegates_to_computer(self, training_env):
        """Test that _compute_losses_impl delegates to loss computer."""
        strategy = DiffusionTrainingStrategy(env=training_env)

        # Mock dependencies
        strategy.loss_computer = MagicMock()
        mock_output = MagicMock()
        mock_output.components = {"loss": torch.tensor(1.0)}
        mock_output.total = torch.tensor(1.0)
        strategy.loss_computer.compute.return_value = mock_output

        mock_generator = MagicMock()
        tensor_output = torch.randn(2, 1, 32, 32)
        mock_generator.return_value = tensor_output
        # MagicMock has 'model' attribute by default, so BaseTrainingStrategy accesses it.
        # We must configure it too, or effectively make it the same as the generator.
        mock_generator.model.return_value = tensor_output
        mock_generator.model.forward.return_value = tensor_output
        # Mock the generator property accessor
        strategy.env.models["generator"] = mock_generator
        strategy.sample_timesteps = MagicMock(return_value=torch.tensor([1, 1]))

        # Disable logging for cleaner output
        strategy.log_batch_info = MagicMock()

        input_batch = torch.randn(2, 1, 32, 32)
        target_batch = torch.randn(2, 1, 32, 32)

        losses = strategy._compute_losses_impl(input_batch, target_batch, epoch=0)

        assert strategy.loss_computer.compute.called
        assert "g_total_loss" in losses
        assert losses["g_total_loss"] == 1.0


class TestDiffusionStrategyMixinIntegration:
    """Test DiffusionStrategyMixin integration."""

    def test_mixin_methods_available(self, training_env):
        """Test that mixin methods are available."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert hasattr(strategy, "initialize_diffusion_parameters")


class TestDiffusionStrategyLogging:
    """Test logging in diffusion strategy."""

    # These tests are hard to verify without mocking logger deeply,
    # but instantiation implies logging calls were made.

    def test_logs_model_info(self, training_env):
        strategy = DiffusionTrainingStrategy(env=training_env)


class TestDiffusionStrategyDeviceManagement:
    """Test device management."""

    def test_device_assignment(self, training_env):
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert strategy.device is not None


class TestDiffusionStrategyErrorHandling:
    """Test error handling in diffusion strategy."""

    def test_invalid_timesteps_handling(self, mock_diffusion_config, training_env):
        """Test handling of invalid timesteps."""
        mock_diffusion_config.training.diffusion.timesteps = -1

        # Should raise ConfigurationError
        from spectramr.domain.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Invalid diffusion timesteps"):
            DiffusionTrainingStrategy(env=training_env)

    def test_invalid_noise_schedule_handling(self, mock_diffusion_config, training_env):
        """An unknown noise schedule must RAISE, not silently coerce to linear.

        Updated for the 2026-06 ``diffusion_scheduler`` hardening (pitfall #9):
        the silent fallback to ``linear`` was removed. See the dedicated
        ``training/test_diffusion_scheduler_raise_2026_06.py``.
        """
        mock_diffusion_config.training.diffusion.noise_schedule = "invalid"

        with pytest.raises(ValueError, match="[Uu]nknown beta schedule"):
            DiffusionTrainingStrategy(env=training_env)


class TestDiffusionStrategyIntegration:
    """Integration tests for diffusion strategy."""

    def test_full_strategy_initialization(self, training_env):
        """Test full initialization of diffusion strategy."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert strategy is not None

    def test_diffusion_vs_kspace_variants(self, mock_diffusion_config, training_env):
        """Test both diffusion mode variants."""
        # Test regular diffusion
        mock_diffusion_config.training_mode = "diffusion"
        training_env.model_type = "diffusion"
        strategy1 = DiffusionTrainingStrategy(env=training_env)

        # Test k-space cold diffusion. The generator carries the undersampling
        # process because every real one does, and the strategy takes its mask
        # generator from there rather than building a second (#2056).
        from spectramr.models.diffusion.kspace_process import KSpaceUndersamplingProcess

        mock_diffusion_config.training_mode = "kspace_cold_diffusion"
        training_env.model_type = "kspace_cold_diffusion"
        mock_diffusion_config.model.model_type = "kspace_cold_diffusion"
        training_env.generator.kspace_process = KSpaceUndersamplingProcess(num_timesteps=1000)
        strategy2 = DiffusionTrainingStrategy(env=training_env)

        # Both should be initialized
        assert strategy1 is not None
        assert strategy2 is not None


class TestInputDependenceGate:
    """L4 measurement-independence (DC-blob) gate in validation_step."""

    @staticmethod
    def _make_strategy(training_env, mock_diffusion_config, tol):
        from tests.utils.config_block_stub import block_stub

        strategy = DiffusionTrainingStrategy(env=training_env)
        # The knob moved to `validation.gates.input_dependence_tol` in the block
        # decomposition. A hand-rolled `SimpleNamespace(input_dependence_tol=tol)`
        # kept the flat spelling, so the reader raised `AttributeError:
        # 'SimpleNamespace' object has no attribute 'gates'` (#723). `block_stub`
        # routes the legacy name to its canonical home via RENAMES, so the call
        # site below still reads `input_dependence_tol=` and cannot drift again.
        mock_diffusion_config.validation = block_stub(
            "validation", input_dependence_tol=tol
        )
        strategy.config = mock_diffusion_config
        strategy.logging_service = MagicMock()
        return strategy

    def test_flags_structural_collapse(self, mock_diffusion_config, training_env):
        """Constant prediction across the cascade -> collapse flagged + warned."""
        strategy = self._make_strategy(training_env, mock_diffusion_config, tol=0.01)
        blob = torch.full((1, 1, 8, 8), 0.137)
        all_metrics = {"val_psnr_2x": 20.0}
        strategy._apply_input_dependence_gate(
            all_metrics,
            [blob, blob.clone(), blob.clone()],
            [2, 8, 32],
        )
        assert all_metrics["val_measurement_collapse"] == 1.0
        assert all_metrics["val_input_dependence"] < 0.01
        strategy.logging_service.log_warning.assert_called_once()

    def test_no_flag_when_output_varies(self, mock_diffusion_config, training_env):
        """Distinct predictions per level -> not flagged, no warning."""
        torch.manual_seed(1)
        strategy = self._make_strategy(training_env, mock_diffusion_config, tol=0.01)
        preds = [torch.rand(1, 1, 8, 8) for _ in range(3)]
        all_metrics: dict[str, float] = {}
        strategy._apply_input_dependence_gate(all_metrics, preds, [2, 8, 32])
        assert all_metrics["val_measurement_collapse"] == 0.0
        assert all_metrics["val_input_dependence"] > 0.1
        strategy.logging_service.log_warning.assert_not_called()

    def test_scalar_mean_fallback_flags_collapse(
        self, mock_diffusion_config, training_env
    ):
        """With no captured tensors, the val_pred_mean_<R>x scalars are used."""
        strategy = self._make_strategy(training_env, mock_diffusion_config, tol=0.01)
        all_metrics = {
            "val_pred_mean_2x": 0.1369,
            "val_pred_mean_8x": 0.1366,
            "val_pred_mean_32x": 0.1364,
        }
        strategy._apply_input_dependence_gate(all_metrics, [], [2, 8, 32])
        assert all_metrics["val_measurement_collapse"] == 1.0

    def test_gate_disabled_when_tol_none(self, mock_diffusion_config, training_env):
        """tol=None disables the gate: no keys stamped, no warning."""
        strategy = self._make_strategy(training_env, mock_diffusion_config, tol=None)
        blob = torch.full((1, 1, 8, 8), 0.137)
        all_metrics: dict[str, float] = {"val_psnr_2x": 20.0}
        strategy._apply_input_dependence_gate(
            all_metrics, [blob, blob.clone()], [2, 8, 32]
        )
        assert "val_measurement_collapse" not in all_metrics
        assert "val_input_dependence" not in all_metrics
        strategy.logging_service.log_warning.assert_not_called()

    def test_cascade_image_iffts_kspace_to_image(
        self, mock_diffusion_config, training_env
    ):
        """The gate measures the IMAGE, not k-space: a k-space prediction is
        iFFT'd to a single-channel RSS image, so a centred DC delta (pure
        low-frequency) becomes a ~uniform image."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        strategy.config = mock_diffusion_config
        B, H, W = 1, 8, 8
        k = torch.zeros(B, 2, H, W)
        k[:, 0, H // 2, W // 2] = 1.0  # DC (real) delta at k-space centre
        img = strategy._cascade_prediction_image(k, needs_ifft=True)
        assert img.shape == (B, 1, H, W)
        # iFFT of a centred DC delta is a ~uniform-magnitude image.
        assert img.std().item() < 1e-3

    def test_cascade_image_passthrough_when_already_image(
        self, mock_diffusion_config, training_env
    ):
        """needs_ifft=False -> no transform, just RSS magnitude to 1 channel."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        strategy.config = mock_diffusion_config
        x = torch.randn(2, 4, 8, 8)
        out = strategy._cascade_prediction_image(x, needs_ifft=False)
        assert out.shape == (2, 1, 8, 8)
        assert torch.all(out >= 0)  # magnitude is non-negative


class TestPreDcFidelity:
    """OPT-IN pre-DC fidelity supervision (DC-blob L1+)."""

    @staticmethod
    def _strategy(training_env, mock_diffusion_config, lam):
        strategy = DiffusionTrainingStrategy(env=training_env)
        mock_diffusion_config.losses.reconstruction.lambda_pre_dc_kspace = lam
        strategy.config = mock_diffusion_config
        strategy._loss_dict_reuse = {}
        return strategy

    def test_adds_term_when_enabled(self, mock_diffusion_config, training_env):
        """lambda>0 + a (post_dc, pre_dc) tuple -> total += lam*mean|pre-tgt|,
        and the term is stamped into the loss dict for provenance."""
        strategy = self._strategy(training_env, mock_diffusion_config, lam=0.5)
        total = torch.tensor(1.0)
        target = torch.zeros(1, 2, 8, 8)
        pre_dc = torch.ones(1, 2, 8, 8)  # mean|pre_dc - target| = 1.0
        out = strategy._add_pre_dc_fidelity(
            total, (torch.zeros(1, 2, 8, 8), pre_dc), target
        )
        assert out.item() == pytest.approx(1.5)  # 1.0 + 0.5 * 1.0
        assert strategy._loss_dict_reuse["pre_dc_kspace_l1"].item() == pytest.approx(
            1.0
        )

    def test_noop_when_weight_zero(self, mock_diffusion_config, training_env):
        """Default weight 0.0 -> exact no-op (same object, nothing stamped)."""
        strategy = self._strategy(training_env, mock_diffusion_config, lam=0.0)
        total = torch.tensor(1.0)
        out = strategy._add_pre_dc_fidelity(
            total,
            (torch.zeros(1, 2, 8, 8), torch.ones(1, 2, 8, 8)),
            torch.zeros(1, 2, 8, 8),
        )
        assert out is total
        assert "pre_dc_kspace_l1" not in strategy._loss_dict_reuse

    def test_noop_without_pre_dc(self, mock_diffusion_config, training_env):
        """No 2nd tuple element (or legacy None) -> no-op, never crashes."""
        strategy = self._strategy(training_env, mock_diffusion_config, lam=0.5)
        total = torch.tensor(1.0)
        tgt = torch.zeros(1, 2, 8, 8)
        # bare tensor (no tuple) and the legacy (x_out, None) shape
        assert (
            strategy._add_pre_dc_fidelity(total, torch.zeros(1, 2, 8, 8), tgt) is total
        )
        assert (
            strategy._add_pre_dc_fidelity(total, (torch.zeros(1, 2, 8, 8), None), tgt)
            is total
        )

    def test_inactive_stamps_zero_for_visibility(
        self, mock_diffusion_config, training_env
    ):
        """lambda>0 but no pre-DC prediction -> term is INACTIVE, but stamps an
        explicit 0.0 (not absent) so the silent no-op is visible in the CSV
        (pitfall #9/#15)."""
        strategy = self._strategy(training_env, mock_diffusion_config, lam=0.5)
        out = strategy._add_pre_dc_fidelity(
            torch.tensor(1.0), torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8)
        )
        assert out.item() == pytest.approx(1.0)  # total unchanged
        assert "pre_dc_kspace_l1" in strategy._loss_dict_reuse
        assert strategy._loss_dict_reuse["pre_dc_kspace_l1"].item() == 0.0

    def test_unsampled_mask_weighting_ignores_sampled_bins(
        self, mock_diffusion_config, training_env
    ):
        """The L1 must weight by (1-mask): a pre-DC error confined to SAMPLED
        bins (which DC fixes anyway) contributes ~0; the same error on UNSAMPLED
        bins contributes fully. Proves the gradient targets the unmeasured HF."""
        strategy = self._strategy(training_env, mock_diffusion_config, lam=1.0)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :, :4] = 1.0  # left half sampled, right half unsampled
        tgt = torch.zeros(1, 2, 8, 8)

        # error only on SAMPLED (left) bins -> weighted term ≈ 0
        pre_sampled_err = torch.zeros(1, 2, 8, 8)
        pre_sampled_err[..., :, :4] = 5.0
        strategy._add_pre_dc_fidelity(
            torch.tensor(0.0), (tgt.clone(), pre_sampled_err), tgt, mask
        )
        assert strategy._loss_dict_reuse["pre_dc_kspace_l1"].item() == pytest.approx(
            0.0
        )

        # error only on UNSAMPLED (right) bins -> weighted term > 0
        pre_unsampled_err = torch.zeros(1, 2, 8, 8)
        pre_unsampled_err[..., :, 4:] = 5.0
        strategy._add_pre_dc_fidelity(
            torch.tensor(0.0), (tgt.clone(), pre_unsampled_err), tgt, mask
        )
        assert strategy._loss_dict_reuse["pre_dc_kspace_l1"].item() == pytest.approx(
            5.0
        )

    def test_unsampled_weight_none_when_fully_sampled(self):
        """No unmeasured bins -> _unsampled_weight returns None (uniform L1)."""
        ref = torch.zeros(1, 2, 8, 8)
        assert DiffusionTrainingStrategy._unsampled_weight(None, ref) is None
        assert (
            DiffusionTrainingStrategy._unsampled_weight(torch.ones(1, 1, 8, 8), ref)
            is None
        )
        w = DiffusionTrainingStrategy._unsampled_weight(torch.zeros(1, 1, 8, 8), ref)
        assert w is not None and float(w.sum()) == pytest.approx(8 * 8)


class TestAccelPsnrGap:
    """Across-acceleration PSNR gap = the DC-blob signal (first-class metric)."""

    def test_stamps_gap_low_minus_high(self):
        m = {
            "val_psnr_2x": 14.0,
            "val_psnr_32x": 1.5,
            "val_robust_mri_psnr_2x": 10.0,
            "val_robust_mri_psnr_32x": -4.0,
        }
        DiffusionTrainingStrategy._stamp_accel_psnr_gap(m, [2, 8, 32])
        assert m["val_psnr_accel_gap"] == pytest.approx(12.5)
        assert m["val_robust_mri_psnr_accel_gap"] == pytest.approx(14.0)

    def test_skips_when_endpoint_missing(self):
        m = {"val_psnr_2x": 14.0}  # no 32x endpoint
        DiffusionTrainingStrategy._stamp_accel_psnr_gap(m, [2, 8, 32])
        assert "val_psnr_accel_gap" not in m

    def test_noop_with_single_level(self):
        m = {"val_psnr_2x": 14.0}
        DiffusionTrainingStrategy._stamp_accel_psnr_gap(m, [2])
        assert "val_psnr_accel_gap" not in m


class TestOutputSnapshot:
    """Model-output training snapshot (DC-blob diagnostic instrument)."""

    def test_includes_pre_and_post_dc(self, mock_diffusion_config, training_env):
        """The snapshot dict carries the LIVE model output — post-DC always, and
        pre-DC when the generator exposed it — plus input/target/mask."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        strategy.config = mock_diffusion_config
        post = torch.randn(1, 2, 8, 8)
        pre = torch.randn(1, 2, 8, 8)
        snap = strategy._build_output_snapshot(
            post,
            (post, pre),  # predicted_output tuple; [1] = pre-DC
            torch.randn(1, 2, 8, 8),  # target
            torch.randn(1, 2, 8, 8),  # input
            torch.ones(1, 1, 8, 8),  # mask
        )
        assert torch.equal(snap["model_output_post_dc"], post)
        assert torch.equal(snap["model_output_pre_dc"], pre)
        assert "target" in snap and "input" in snap and "mask" in snap

    def test_no_pre_dc_key_when_absent(self, mock_diffusion_config, training_env):
        """A bare-tensor predicted_output (or (post, None)) -> no pre-DC key;
        mask=None -> no mask key. Output is always present."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        strategy.config = mock_diffusion_config
        post = torch.randn(1, 2, 8, 8)
        snap = strategy._build_output_snapshot(
            post, post, torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8), None
        )
        assert "model_output_post_dc" in snap
        assert "model_output_pre_dc" not in snap
        assert "mask" not in snap


class TestGeneratorAcceptsTime:
    """Finding #2: time-conditioning detection via signature introspection.

    The velocity-field subclasses (flow-matching, stochastic-interpolants,
    Schroedinger-bridge) dispatch 1-arg vs 2-arg forward through this helper
    instead of `try: gen(x, t) except TypeError: gen(x)`, so a genuine
    in-forward TypeError is NOT swallowed (pitfall #9).
    """

    def test_two_positional_args_is_time_conditioned(self):
        class _TimeGen:
            def forward(self, x, t):
                return x

        assert DiffusionTrainingStrategy._generator_accepts_time(_TimeGen()) is True

    def test_single_positional_arg_is_not_time_conditioned(self):
        class _NoTimeGen:
            def forward(self, x):
                return x

        assert DiffusionTrainingStrategy._generator_accepts_time(_NoTimeGen()) is False

    def test_uninspectable_callable_assumed_time_conditioned(self):
        import torch as _torch

        # nn.Sequential.forward is introspectable with one positional arg, but a
        # builtin / C-callable is not; assume time-conditioned for the latter.
        assert DiffusionTrainingStrategy._generator_accepts_time(_torch.relu) is True


class TestXDiffusionIterationThreadsToWarmupGate:
    """Regression: XDiffusion must thread ``iteration`` into ``_get_loss_weight``.

    Pre-fix, ``XDiffusionTrainingStrategy._compute_losses_impl`` called
    ``self._get_loss_weight(loss_name, epoch=epoch)`` WITHOUT ``iteration``.
    ``BaseTrainingStrategy._get_loss_weight`` then defaulted ``iteration`` to
    ``1_000_000`` (``kwargs.get("iteration", 1000000)``), so the spatial-loss
    warm-up gate (``iteration < warmup_iterations`` -> weight 0.0) was
    permanently bypassed for every XDiffusion step.
    """

    def _make_strategy(self, training_env, mock_diffusion_config, warmup):
        from spectramr.infrastructure.training.strategies.diffusion import (
            XDiffusionTrainingStrategy,
        )

        # ``l1`` is one of base.SPATIAL_LOSSES; the warm-up gate applies to it.
        mock_diffusion_config.losses.reconstruction.warmup_iterations = warmup
        mock_diffusion_config.losses.reconstruction.lambda_l1 = 1.0
        # Disable cross-modal so the plain denoiser path is exercised.
        mock_diffusion_config.training.diffusion.cross_modal = None
        strategy = XDiffusionTrainingStrategy(env=training_env)
        # Wipe any cached weights so the gate is recomputed per call.
        strategy._loss_weight_cache = {}
        return strategy

    def test_iteration_kwarg_reaches_get_loss_weight(
        self, mock_diffusion_config, training_env
    ):
        """The real training ``iteration`` is forwarded to ``_get_loss_weight``."""
        strategy = self._make_strategy(training_env, mock_diffusion_config, 1000)
        # Provide a single env loss so the X-Diffusion weight-lookup branch runs.
        strategy.env.losses = {"l1": lambda p, t: torch.nn.functional.l1_loss(p, t)}
        calls: list[tuple[str, object]] = []
        orig = strategy._get_loss_weight

        def _spy(loss_name, **kwargs):
            calls.append((loss_name, kwargs.get("iteration", "__missing__")))
            return orig(loss_name, **kwargs)

        strategy._get_loss_weight = _spy
        strategy._compute_losses_impl(
            torch.randn(2, 1, 16, 16),
            torch.randn(2, 1, 16, 16),
            epoch=0,
            iteration=42,
        )
        # The X-Diffusion env-loss loop must look up the "l1" weight WITH the
        # real iteration. Pre-fix that call passed only ``epoch`` -> the spy
        # records ("l1", "__missing__"); post-fix it records ("l1", 42).
        l1_iters = [it for (name, it) in calls if name == "l1"]
        assert l1_iters, f"expected an l1 weight lookup, got calls={calls!r}"
        assert all(
            it == 42 for it in l1_iters
        ), f"X-Diffusion l1 weight lookup was iteration-blind: calls={calls!r}"

    def test_spatial_loss_weight_zero_before_warmup_nonzero_after(
        self, mock_diffusion_config, training_env
    ):
        """``l1`` weight is gated 0.0 before warm-up and active afterwards."""
        warmup = 1000
        strategy = self._make_strategy(training_env, mock_diffusion_config, warmup)

        # Before warm-up: weight must be gated to 0.0 (iteration < warmup).
        strategy._loss_weight_cache = {}
        w_before = strategy._get_loss_weight("l1", epoch=0, iteration=0)
        assert w_before == 0.0

        # After warm-up: weight must be the real (nonzero) configured value.
        strategy._loss_weight_cache = {}
        w_after = strategy._get_loss_weight("l1", epoch=0, iteration=warmup + 1)
        assert w_after != 0.0

    def test_pre_fix_default_would_bypass_gate(
        self, mock_diffusion_config, training_env
    ):
        """Documents the bug: the 1_000_000 default skips the gate entirely.

        With ``iteration`` omitted, ``_get_loss_weight`` defaults to 1_000_000,
        which is never ``< warmup_iterations`` for any realistic warm-up, so a
        spatial loss is (wrongly) active from step 0. Threading the real
        iteration (the fix) is what makes the gate observable above.
        """
        strategy = self._make_strategy(training_env, mock_diffusion_config, 1000)
        strategy._loss_weight_cache = {}
        w_no_iter = strategy._get_loss_weight("l1", epoch=0)
        assert w_no_iter != 0.0  # the bypassed-gate behaviour of the base default


# ---------------------------------------------------------------------------
# Regression: X-Diffusion cross-modal source conditioning (pitfalls #9, #15).
#
# Pre-fix, ``XDiffusionTrainingStrategy`` advertised cross-modal knobs
# (source_modality / target_modalities / condition_embedding_dim /
# condition_encoder / num_contrasts) that were NEVER read, validated or
# stamped, and ``_compute_losses_impl`` ignored ``input_batch`` entirely — the
# prediction depended only on ``target_batch``, so the advertised cross-modal
# paradigm had no paradigm-specific math. These tests would FAIL on the
# pre-fix code: there was no condition encoder to receive gradient,
# ``input_batch`` did not influence the prediction, and an unknown encoder type
# silently did nothing instead of raising.
# ---------------------------------------------------------------------------
import types as _types  # noqa: E402

from spectramr.domain.exceptions import (  # noqa: E402
    ConfigurationError as _ConfigurationError,
)
from spectramr.infrastructure.training.strategies.diffusion import (  # noqa: E402
    XDiffusionTrainingStrategy as _XDiffusionTrainingStrategy,
)


class _XModalCondGenerator(torch.nn.Module):
    """x_0-predictor whose forward accepts a ``cond`` conditioning kwarg."""

    def __init__(self, channels: int = 2, embedding_dim: int = 4) -> None:
        super().__init__()
        self.body = torch.nn.Conv2d(channels, channels, kernel_size=1)
        self.cond_proj = torch.nn.Conv2d(embedding_dim, channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.body(x)
        if cond is not None:
            out = out + self.cond_proj(cond)
        return out


class _XModalPlainGenerator(torch.nn.Module):
    """x_0-predictor whose forward accepts NO conditioning kwarg."""

    def __init__(self, channels: int = 2) -> None:
        super().__init__()
        self.body = torch.nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _make_xmodal_strategy(generator, cross_modal):
    """Bare X-Diffusion strategy with an explicit cross_modal config block."""
    s = object.__new__(_XDiffusionTrainingStrategy)
    opt_g = torch.optim.SGD(list(generator.parameters()), lr=0.1)
    s.env = _types.SimpleNamespace(generator=generator, losses={}, opt_g=opt_g)
    s.config = _types.SimpleNamespace(
        training=_types.SimpleNamespace(
            diffusion=_types.SimpleNamespace(timesteps=50, cross_modal=cross_modal)
        ),
        model=_types.SimpleNamespace(model_type="x_diffusion"),
    )
    s.device = torch.device("cpu")
    s.logging_service = _types.SimpleNamespace(log_info=lambda *a, **k: None)
    s._loss_dict_reuse = {}
    # Stub base verification / logging helpers (object.__new__ skips __init__).
    s._verify_strategy_config = lambda **kw: None
    s._log_config_features = lambda *a, **k: None
    s._setup_strategy_specific_components()
    return s


class TestXDiffusionCrossModalConditioning:
    def _block(self, **overrides):
        block = {
            "enabled": True,
            "condition_encoder": "conv",
            "condition_embedding_dim": 4,
            "source_modality": "t1",
            "target_modalities": ["t2", "flair"],
            "num_contrasts": 2,
        }
        block.update(overrides)
        return block

    def test_knobs_are_read_validated_and_stamped(self):
        gen = _XModalCondGenerator(channels=2, embedding_dim=4)
        s = _make_xmodal_strategy(gen, self._block())
        prov = s.cross_modal_provenance
        assert prov["enabled"] is True
        assert prov["condition_encoder"] == "conv"
        assert prov["condition_embedding_dim"] == 4
        assert prov["source_modality"] == "t1"
        assert prov["target_modalities"] == ["t2", "flair"]
        assert prov["num_contrasts"] == 2
        assert s.condition_encoder is not None

    def test_unknown_condition_encoder_raises(self):
        gen = _XModalCondGenerator(channels=2, embedding_dim=4)
        with pytest.raises(_ConfigurationError):
            _make_xmodal_strategy(gen, self._block(condition_encoder="transformer"))

    def test_nonpositive_embedding_dim_raises(self):
        gen = _XModalCondGenerator(channels=2, embedding_dim=4)
        with pytest.raises(_ConfigurationError):
            _make_xmodal_strategy(gen, self._block(condition_embedding_dim=0))

    def test_enabled_but_generator_cannot_consume_condition_raises(self):
        # Cross-modal enabled, but the generator forward has no cond/context/
        # condition kwarg => must RAISE rather than silently drop the embedding.
        gen = _XModalPlainGenerator(channels=2)
        s = _make_xmodal_strategy(gen, self._block())
        x = torch.randn(2, 2, 8, 8)
        y = torch.randn(2, 2, 8, 8)
        with pytest.raises(_ConfigurationError):
            s._compute_losses_impl(input_batch=x, target_batch=y, epoch=1)

    def test_condition_encoder_params_receive_gradient(self):
        gen = _XModalCondGenerator(channels=2, embedding_dim=4)
        s = _make_xmodal_strategy(gen, self._block())

        x = torch.randn(2, 2, 8, 8)
        y = torch.randn(2, 2, 8, 8)
        out = s._compute_losses_impl(input_batch=x, target_batch=y, epoch=1)
        total = out["g_total_loss"]
        assert total.requires_grad

        enc_params = list(s.condition_encoder.parameters())
        assert enc_params, "condition encoder has no parameters"
        grads = torch.autograd.grad(
            total, enc_params, retain_graph=True, allow_unused=True
        )
        norms = [torch.linalg.vector_norm(g).item() for g in grads if g is not None]
        assert norms, "no encoder parameter received gradient"
        assert max(norms) > 0.0, "encoder gradient is zero — input_batch ignored"

    def test_input_batch_influences_prediction(self):
        # Two DIFFERENT source modalities must yield DIFFERENT losses with the
        # SAME noise/timestep draw. Pre-fix, input_batch was ignored, so the
        # two would be identical.
        gen = _XModalCondGenerator(channels=2, embedding_dim=4)
        s = _make_xmodal_strategy(gen, self._block())

        y = torch.randn(2, 2, 8, 8)
        x_a = torch.randn(2, 2, 8, 8)
        x_b = torch.randn(2, 2, 8, 8) * 5.0 + 3.0

        torch.manual_seed(1234)
        out_a = s._compute_losses_impl(input_batch=x_a, target_batch=y, epoch=1)
        loss_a = out_a["g_total_loss"].detach().clone()

        torch.manual_seed(1234)
        out_b = s._compute_losses_impl(input_batch=x_b, target_batch=y, epoch=1)
        loss_b = out_b["g_total_loss"].detach().clone()

        assert not torch.allclose(
            loss_a, loss_b
        ), "prediction does not depend on input_batch (source modality)"

    def test_absent_cross_modal_block_stays_single_modality(self):
        # No cross_modal block => no encoder, no provenance, plain denoiser.
        gen = _XModalPlainGenerator(channels=2)
        s = object.__new__(_XDiffusionTrainingStrategy)
        s.env = _types.SimpleNamespace(generator=gen, losses={}, opt_g=None)
        s.config = _types.SimpleNamespace(
            training=_types.SimpleNamespace(
                diffusion=_types.SimpleNamespace(timesteps=50)
            ),
            model=_types.SimpleNamespace(model_type="x_diffusion"),
        )
        s.device = torch.device("cpu")
        s.logging_service = _types.SimpleNamespace(log_info=lambda *a, **k: None)
        s._loss_dict_reuse = {}
        s._verify_strategy_config = lambda **kw: None
        s._log_config_features = lambda *a, **k: None
        s._setup_strategy_specific_components()

        assert s.condition_encoder is None
        assert s.cross_modal_provenance == {}
        x = torch.randn(2, 2, 8, 8)
        y = torch.randn(2, 2, 8, 8)
        out = s._compute_losses_impl(input_batch=x, target_batch=y, epoch=1)
        assert out["g_total_loss"].requires_grad


class TestSmapsFallbackHonorsConfig:
    """The runtime smaps fallback must dispatch on the configured estimation
    method, not a hardcoded ``power_iter`` (CLAUDE.md pitfall #15)."""

    def test_default_when_no_config(self) -> None:
        # No config attribute at all (bare __new__): falls back to the default.
        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        assert s._configured_estimation_method() == "power_iter"
        assert s._configured_estimation_method(default="espirit") == "espirit"

    def test_reads_configured_method(self) -> None:
        import types as _types

        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        s.config = _types.SimpleNamespace(
            physics=_types.SimpleNamespace(
                coil_processing=_types.SimpleNamespace(
                    estimation=_types.SimpleNamespace(method="espirit")
                )
            )
        )
        assert s._configured_estimation_method() == "espirit"

    def test_none_method_maps_to_default(self) -> None:
        # estimation.method == "none" in a maps-required branch → default,
        # never None (which would break the downstream DC step).
        import types as _types

        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        s.config = _types.SimpleNamespace(
            physics=_types.SimpleNamespace(
                coil_processing=_types.SimpleNamespace(
                    estimation=_types.SimpleNamespace(method="none")
                )
            )
        )
        assert s._configured_estimation_method() == "power_iter"

    def test_real_config_chain_matches_schema(self) -> None:
        # Guard the attribute path against schema drift: the chain
        # config.physics.coil_processing.estimation.method must exist on a real
        # TrainingSettings.
        from spectramr.config.settings import TrainingSettings

        config = TrainingSettings(
            data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 1},
            model={"model_type": "unet"},
            optimization={"learning_rate": 1e-4},
            logging={},
            physics={"coil_processing": {"estimation": {"method": "rss"}}},
        )
        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        s.config = config
        assert s._configured_estimation_method() == "rss"


class TestEstimationKwargsHonorConfig:
    """The smaps fallback must thread estimation sub-knobs (kernel_size/acs_size/
    eigen_threshold/maps_path) from config, not hardcode them (pitfall #15)."""

    def test_empty_when_no_config(self) -> None:
        # Bare __new__ (no config) → {} so estimate_smaps uses its own defaults
        # (kernel_size=6) → bit-identical to the old hardcoded path.
        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        assert s._configured_estimation_kwargs() == {}

    def test_reads_sub_knobs_from_config(self) -> None:
        from spectramr.config.settings import TrainingSettings

        config = TrainingSettings(
            data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 1},
            model={"model_type": "unet"},
            optimization={"learning_rate": 1e-4},
            logging={},
            physics={
                "coil_processing": {
                    "estimation": {
                        "method": "espirit",
                        "kernel_size": 8,
                        "acs_size": 32,
                        "eigen_threshold": 0.9,
                    }
                }
            },
        )
        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        s.config = config
        kw = s._configured_estimation_kwargs()
        assert kw["kernel_size"] == 8
        assert kw["acs_size"] == 32
        assert kw["eigen_threshold"] == 0.9
        # maps_path defaults to None → omitted so estimate_smaps default applies.
        assert "maps_path" not in kw

    def test_default_config_reproduces_kernel_size_6(self) -> None:
        # A config with the default estimation block must yield kernel_size=6
        # (parity with the previous hardcoded call site).
        from spectramr.config.settings import TrainingSettings

        config = TrainingSettings(
            data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 1},
            model={"model_type": "unet"},
            optimization={"learning_rate": 1e-4},
            logging={},
            physics={"coil_processing": {"estimation": {"method": "power_iter"}}},
        )
        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        s.config = config
        assert s._configured_estimation_kwargs()["kernel_size"] == 6

    def test_maps_path_included_when_set(self) -> None:
        from spectramr.config.settings import TrainingSettings

        config = TrainingSettings(
            data={"train_path": "/tmp/t", "val_path": "/tmp/v", "batch_size": 1},
            model={"model_type": "unet"},
            optimization={"learning_rate": 1e-4},
            logging={},
            physics={
                "coil_processing": {
                    "estimation": {"method": "file", "maps_path": "/tmp/maps.pt"}
                }
            },
        )
        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        s.config = config
        assert s._configured_estimation_kwargs()["maps_path"] == "/tmp/maps.pt"


class TestTimestepSamplingStrategyRaisesOnUnknown:
    """SD-001: an unknown timestep_sampling_strategy must RAISE (pitfall #9),
    never silently degrade to the 'uniform' default."""

    def _strategy_with_strategy_value(self, training_env, mock_diffusion_config, value):
        strategy = DiffusionTrainingStrategy(env=training_env)
        # Short-run bypass so the curriculum arithmetic (MagicMock attrs) is
        # skipped and ``high = num_timesteps`` — the dispatch is reached cleanly.
        mock_diffusion_config.training.max_iterations = 1000
        mock_diffusion_config.training.timestep_sampling_strategy = value
        strategy.config = mock_diffusion_config
        # generator_model.training must be True to enter the strategy branch.
        strategy.generator_model.train()
        return strategy

    def test_unknown_timestep_sampling_strategy_raises(
        self, mock_diffusion_config, training_env
    ):
        from spectramr.domain.exceptions import ConfigurationError

        strategy = self._strategy_with_strategy_value(
            training_env, mock_diffusion_config, "cosine_decay"
        )
        with pytest.raises(ConfigurationError, match="cosine_decay"):
            strategy.sample_timesteps(2, iteration=5000)

    def test_uniform_strategy_does_not_raise(self, mock_diffusion_config, training_env):
        strategy = self._strategy_with_strategy_value(
            training_env, mock_diffusion_config, "uniform"
        )
        t = strategy.sample_timesteps(2, iteration=5000)
        assert t.shape == (2,)
        assert t.dtype == torch.long

    def test_importance_strategy_does_not_raise(
        self, mock_diffusion_config, training_env
    ):
        strategy = self._strategy_with_strategy_value(
            training_env, mock_diffusion_config, "importance"
        )
        t = strategy.sample_timesteps(2, iteration=5000)
        assert t.shape == (2,)


class TestUnsampledWeightNoSync:
    """SD-002: ``_unsampled_weight`` must use ``w.any()`` (stays on device),
    not ``float(w.sum())`` which forces a GPU->host sync every step."""

    def test_fully_sampled_returns_none(self):
        """All-ones mask -> no unmeasured bins -> None (uniform L1 fallback)."""
        ref = torch.zeros(1, 2, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        assert DiffusionTrainingStrategy._unsampled_weight(mask, ref) is None

    def test_partial_mask_returns_tensor(self):
        """Half-ones mask -> a weight tensor over the unsampled bins."""
        ref = torch.zeros(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4, :] = 1.0  # top half sampled, bottom half unsampled
        w = DiffusionTrainingStrategy._unsampled_weight(mask, ref)
        assert w is not None
        assert torch.is_tensor(w)
        assert w.dtype == ref.dtype
        # The complement is 1 on the unsampled bins, 0 on the sampled ones.
        assert float(w.sum()) == pytest.approx(8 * 4)

    def test_uses_any_not_sum_for_emptiness_check(self):
        """Regression guard: the source no longer calls ``float(w.sum())`` in the
        emptiness check (that would D2H-sync on every step)."""
        import inspect

        src = inspect.getsource(DiffusionTrainingStrategy._unsampled_weight)
        assert "float(w.sum())" not in src
        assert "w.any()" in src


class TestCachedSmapsNotWrittenByValidation:
    """SD-005: ``_generate_validation_prediction`` must NOT write
    ``self._cached_smaps`` — that persisted a validation-batch smaps across the
    validation->training boundary and crashed ``sense_adjoint`` at iter 1001
    (``tensor a (2) vs b (36)``)."""

    def test_generate_validation_prediction_does_not_write_cached_smaps(self):
        """Static guard: the redundant ``self._cached_smaps = smaps`` write is
        gone from the validation-prediction smaps-persist site."""
        import inspect

        src = inspect.getsource(DiffusionTrainingStrategy)
        # ``_current_smaps`` is still persisted (downstream consumers need it)...
        assert "self._current_smaps = smaps" in src
        # ...but ``_cached_smaps`` is never assigned anywhere in the strategy.
        assert "self._cached_smaps = smaps" not in src
        assert "self._cached_smaps =" not in src

    def test_cached_smaps_defaults_to_none(self, training_env):
        """A freshly constructed strategy has no ``_cached_smaps`` attribute, so
        any reader using ``getattr(self, '_cached_smaps', None)`` sees None."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        assert getattr(strategy, "_cached_smaps", None) is None


class TestStampAccelMean:
    """Cascade-MEAN selection target (2026-06-08 negative-transfer fix).

    ``_stamp_accel_mean`` stamps ``val_<metric>_mean`` over the in-distribution
    ``_<R>x`` columns so checkpoint selection / early-stopping can track a
    non-pathological target (not the monotone-degrading ``_8x`` that made the
    2026-06-06 fix self-defeating, nor the gameable ``_2x``)."""

    def test_stamps_mean_over_cascade_levels(self):
        m = {
            "val_robust_mri_psnr_2x": 10.0,
            "val_robust_mri_psnr_8x": -2.0,
            "val_robust_mri_psnr_32x": -5.0,
            "val_psnr_2x": 16.0,
            "val_psnr_8x": 4.0,
            "val_psnr_32x": 1.0,
        }
        DiffusionTrainingStrategy._stamp_accel_mean(m, [2, 8, 32])
        assert m["val_robust_mri_psnr_mean"] == pytest.approx((10.0 - 2.0 - 5.0) / 3)
        assert m["val_psnr_mean"] == pytest.approx((16.0 + 4.0 + 1.0) / 3)

    def test_mean_not_gamed_by_collapsing_high_r(self):
        """A run that wins at 2x but collapses at 8x/32x scores WORSE on the mean
        than a balanced run — the property that fixes the selection bug."""
        collapsed = {
            "val_robust_mri_psnr_2x": 11.0,
            "val_robust_mri_psnr_8x": -4.5,
            "val_robust_mri_psnr_32x": -5.0,
        }
        balanced = {
            "val_robust_mri_psnr_2x": 9.0,
            "val_robust_mri_psnr_8x": 6.0,
            "val_robust_mri_psnr_32x": 2.0,
        }
        DiffusionTrainingStrategy._stamp_accel_mean(collapsed, [2, 8, 32])
        DiffusionTrainingStrategy._stamp_accel_mean(balanced, [2, 8, 32])
        assert (
            balanced["val_robust_mri_psnr_mean"] > collapsed["val_robust_mri_psnr_mean"]
        )

    def test_partial_levels_present(self):
        m = {"val_robust_mri_psnr_2x": 8.0, "val_robust_mri_psnr_8x": 2.0}
        DiffusionTrainingStrategy._stamp_accel_mean(m, [2, 8, 32])
        # 32x absent -> mean over the two present levels, no KeyError.
        assert m["val_robust_mri_psnr_mean"] == pytest.approx(5.0)

    def test_empty_levels_is_noop(self):
        m = {"val_robust_mri_psnr_2x": 8.0}
        DiffusionTrainingStrategy._stamp_accel_mean(m, [])
        assert "val_robust_mri_psnr_mean" not in m

    def test_configured_metric_gets_a_cascade_mean(self):
        """A metric declared in ``validation.metrics`` must get a ``_mean`` too (#18).

        The set was hardcoded to psnr/robust_mri_psnr, so adding ``hfen`` produced the
        per-level ``val_hfen_<R>x`` columns but NO ``val_hfen_mean`` — and an arm
        pointing ``best_metric_name`` at it selected on a key the run never computes.
        """
        m = {
            "val_hfen_2x": 0.30,
            "val_hfen_8x": 0.50,
            "val_hfen_32x": 0.70,
        }
        DiffusionTrainingStrategy._stamp_accel_mean(m, [2, 8, 32], ["hfen"])
        assert m["val_hfen_mean"] == pytest.approx(0.5)

    def test_hardcoded_defaults_survive_a_configured_set(self):
        """Passing a metric list must ADD to psnr/robust_mri_psnr, never replace them.

        Every existing arm selects on ``val_robust_mri_psnr_mean``; dropping it for
        arms whose ``validation.metrics`` happens to omit the name would silently
        break checkpoint selection cohort-wide.
        """
        m = {
            "val_robust_mri_psnr_2x": 10.0,
            "val_robust_mri_psnr_8x": 6.0,
            "val_hfen_2x": 0.2,
            "val_hfen_8x": 0.4,
        }
        DiffusionTrainingStrategy._stamp_accel_mean(m, [2, 8], ["hfen"])
        assert m["val_robust_mri_psnr_mean"] == pytest.approx(8.0)
        assert m["val_hfen_mean"] == pytest.approx(0.3)

    def test_no_metric_names_preserves_legacy_behaviour(self):
        """Callers that pass nothing get exactly the pre-2026-07-29 two-metric set."""
        m = {
            "val_psnr_2x": 4.0,
            "val_psnr_8x": 2.0,
            "val_hfen_2x": 0.2,
            "val_hfen_8x": 0.4,
        }
        DiffusionTrainingStrategy._stamp_accel_mean(m, [2, 8])
        assert m["val_psnr_mean"] == pytest.approx(3.0)
        assert "val_hfen_mean" not in m

    def test_configured_metrics_read_from_validation_block(self, training_env):
        """``_configured_validation_metrics`` reads the live schema, incl. enum entries."""
        strategy = DiffusionTrainingStrategy(env=training_env)
        names = strategy._configured_validation_metrics()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    # The tests below deliberately bypass ``training_env``: its config is a
    # MagicMock, so ``config.validation.metrics`` auto-mocks and MagicMock's
    # default ``__iter__`` yields nothing — every assertion here would pass
    # vacuously without ever touching ValidationConfigSchema. ``_configured_
    # validation_metrics`` reads only ``self.config``, so a stand-in carrying a
    # REAL frozen TrainingSettings exercises the actual schema.
    @staticmethod
    def _settings_with_metrics(metrics):
        """Real settings, with `metrics` landing where production reads it.

        This used to build the block with
        ``ValidationConfigSchema.model_construct(metrics=metrics)``.
        ``model_construct`` skips validation AND the fold, so the legacy
        ``metrics=`` spelling was stored verbatim and never routed to
        ``validation.scoring.compute`` -- the path
        ``_configured_validation_metrics`` actually reads. The resolver
        therefore returned ``[]`` and the tests compared it against
        ``['psnr', 'ssim']`` (#723).

        Passing ``validation={"metrics": ...}`` to ``TrainingSettings`` lets the
        real loader fold it, so the fixture exercises the same path a YAML does.
        """
        from spectramr.config.settings import TrainingSettings

        return TrainingSettings(
            model={
                "model_type": "standard_unet",
                "in_channels": 1,
                "out_channels": 1,
            },
            training={"training_mode": "reconstruction"},
            data={"batch_size": 2},
            optimization={},
            logging={},
            validation=(None if metrics is None else {"metrics": metrics}),
        )

    def _names_for(self, metrics):
        from types import SimpleNamespace

        return DiffusionTrainingStrategy._configured_validation_metrics(
            SimpleNamespace(config=self._settings_with_metrics(metrics))
        )

    def test_configured_metrics_tolerates_absent_validation_block(self):
        """``config.validation`` is ``ValidationConfigSchema | None``; None means
        no declared metrics, not a crash. Pins the guarded direct read that
        replaced ``getattr(self.config, "validation", None)``."""
        assert self._names_for(None) == []

    def test_configured_metrics_reads_a_declared_list(self):
        assert self._names_for(["psnr", "ssim"]) == ["psnr", "ssim"]

    def test_configured_metrics_honours_disabled_entries_in_a_mapping(self):
        """``validation.metrics`` is ``list[str] | dict[str, bool] | None``.
        Iterating the raw mapping yielded its KEYS, so a metric explicitly
        switched off still entered the cascade-mean set and had a mean stamped
        for values the run never produced."""
        assert self._names_for({"psnr": True, "ssim": False}) == ["psnr"]


class TestMultiStepColdSamplingWiring:
    """2026-06-08 opt-in true multi-step cold sampling. The cold-diffusion
    validation branch must, when ``validation.multistep_cold_sampling`` is set,
    route to the generator's multi-step ``sample()`` using the PRE-smap-concat
    measurement — while preserving the single-forward default. Static-source
    guard (mirrors test_generate_validation_prediction_does_not_write_cached_smaps)
    so the wiring can't silently regress to a one-shot regressor."""

    def test_multistep_branch_is_wired(self):
        import inspect

        src = inspect.getsource(
            DiffusionTrainingStrategy._generate_validation_prediction
        )
        # The knob is read as a DECLARED nested field. Commit 5c8503561 replaced
        # the old getattr(..., "multistep_cold_sampling", False) spelling with a
        # direct read now that ValidationConfigSchema supplies the default
        # (CLAUDE.md non-negotiable #1); assert the read itself, so this tracks
        # the config SSOT rather than a syntax form that is free to change.
        #
        # The leaf then MOVED: the block decomposition folded
        # `validation.multistep_cold_sampling` into
        # `validation.sampling.enable_multistep_cold`. Production followed it;
        # this assertion did not, so it failed against fully-wired code (#723) --
        # the most expensive kind of red, because it reads as a mechanism
        # regression. Resolve the destination through RENAMES instead of
        # restating it, so the next move updates this test by construction.
        from spectramr.config.schemas.renames import RENAMES

        canonical = RENAMES["validation.multistep_cold_sampling"].canonical
        assert f"self.config.{canonical}" in src
        # When enabled, the multistep branch routes the PRE-smap-concat measurement
        # (masked_input) into the genuine multi-step sampler. The OOM fix (46fb9e989)
        # moved the gen.sample() reverse loop into _sample_multistep_chunked, so the
        # call site is now the chunked wrapper fed masked_input — assert that contract.
        assert "_sample_multistep_chunked(" in src
        assert "masked_input" in src
        # ...and the chunked wrapper still invokes the real generator reverse-sampler on
        # the measurement (guards against silent regression to a one-shot regressor).
        chunked_src = inspect.getsource(
            DiffusionTrainingStrategy._sample_multistep_chunked
        )
        assert "gen.sample(" in chunked_src
        assert "measurement=" in chunked_src
        # The single deterministic forward remains the fallback (default path).
        assert "_forward_chunked(" in src

    def test_multistep_default_is_single_forward(self):
        """Absent the knob, the single deterministic forward must still be
        selected — the default behaviour of every existing arm is unchanged.

        The default now lives in ValidationConfigSchema rather than in a
        getattr fallback, so what this guards on the STRATEGY side is that a
        missing validation block still resolves to False instead of raising
        (the schema default itself is pinned by
        tests/unit/config/schemas/test_validation.py)."""
        import inspect

        src = inspect.getsource(
            DiffusionTrainingStrategy._generate_validation_prediction
        )
        assert "if self.config.validation is not None" in src
        assert "else False" in src


def test_diffusion_override_feeds_report_case_recorder():
    """The diffusion strategy overrides _log_validation_images_to_tensorboard
    without super(); the report-case feed seam must be duplicated there or the
    recorder stays unfed (a facade) for the cold-diffusion paradigm."""
    import inspect

    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    src = inspect.getsource(
        DiffusionTrainingStrategy._log_validation_images_to_tensorboard
    )
    assert "feed_report_case_recorder" in src
    assert "_report_case_recorder" in src


class TestSmapsContentCache:
    """ESPIRiT smaps are memoized on the ACS content (2026-06-12 compute audit).

    The estimate is an SVD-class op that was re-run every step / every
    reverse-sampling iteration; the content key makes it run once per distinct
    ACS input while being leak-proof (a content key cannot alias across
    train/val contexts).
    """

    @staticmethod
    def _fake_estimate(monkeypatch):
        calls = {"n": 0}

        def fake_estimate_smaps(acs, **kwargs):
            calls["n"] += 1
            return torch.ones(acs.shape, dtype=torch.complex64)

        monkeypatch.setattr(
            "spectramr.infrastructure.training.strategies.diffusion.estimate_smaps",
            fake_estimate_smaps,
        )
        return calls

    def test_same_acs_estimates_once(self, monkeypatch) -> None:
        calls = self._fake_estimate(monkeypatch)
        strat = object.__new__(DiffusionTrainingStrategy)
        acs = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))
        s1 = strat._estimate_smaps_cached(acs, 16, 16)
        s2 = strat._estimate_smaps_cached(acs, 16, 16)
        assert calls["n"] == 1  # second call hit the cache
        assert torch.equal(s1, s2)

    def test_different_acs_reestimates(self, monkeypatch) -> None:
        calls = self._fake_estimate(monkeypatch)
        strat = object.__new__(DiffusionTrainingStrategy)
        torch.manual_seed(0)
        acs_a = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))
        acs_b = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))
        strat._estimate_smaps_cached(acs_a, 16, 16)
        strat._estimate_smaps_cached(acs_b, 16, 16)
        assert calls["n"] == 2  # distinct content → re-estimated

    def test_cache_is_bounded(self, monkeypatch) -> None:
        self._fake_estimate(monkeypatch)
        strat = object.__new__(DiffusionTrainingStrategy)
        torch.manual_seed(1)
        for _ in range(DiffusionTrainingStrategy._SMAPS_CACHE_MAX + 5):
            acs = torch.complex(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))
            strat._estimate_smaps_cached(acs, 8, 8)
        assert len(strat._smaps_cache) <= DiffusionTrainingStrategy._SMAPS_CACHE_MAX


class TestCurriculumRMax:
    """WS-3 PR-3 regression: the acceleration-curriculum R_max is a pure
    function of the *live* loop iteration (read from ``self.loop_state`` by
    ``_compute_and_log_debug_info``).

    Before the loop_state seam, the caller read the frozen ``env.step`` — a
    constant 0 — so this diagnostic was permanently pinned to the 2x stage and
    its "MASK NOT APPLIED" threshold was mis-calibrated for the whole run. These
    tests pin the iteration->R_max schedule so the inert read cannot silently
    return."""

    _f = staticmethod(DiffusionTrainingStrategy._curriculum_r_max)

    @pytest.mark.unit
    def test_frozen_step_zero_was_always_the_2x_stage(self) -> None:
        # iteration 0 is exactly the value the inert env.step read produced;
        # it legitimately maps to the 2x warm-up stage.
        assert self._f(0, 32.0, 500_000) == 2.0

    @pytest.mark.unit
    def test_warmup_holds_at_2x_through_5000(self) -> None:
        assert self._f(1, 32.0, 500_000) == 2.0
        assert self._f(5000, 32.0, 500_000) == 2.0

    @pytest.mark.unit
    def test_saturates_at_r_max_from_50000(self) -> None:
        assert self._f(50_000, 32.0, 500_000) == 32.0
        assert self._f(120_000, 32.0, 500_000) == 32.0

    @pytest.mark.unit
    def test_linear_ramp_midpoint(self) -> None:
        # Halfway between iter 5000 and 50000 (=27500) is halfway between 2x and
        # 32x = 17x. This is the value the diagnostic could NEVER reach while
        # pinned to the frozen step 0 — the headline regression.
        assert self._f(27_500, 32.0, 500_000) == pytest.approx(17.0)
        assert self._f(27_500, 32.0, 500_000) != self._f(0, 32.0, 500_000)

    @pytest.mark.unit
    def test_short_debug_run_skips_the_cap(self) -> None:
        # max_iters <= 5000: use full R_max immediately so short diagnostics are
        # meaningful (independent of where in the run we are).
        assert self._f(0, 32.0, 5000) == 32.0
        assert self._f(4000, 16.0, 5000) == 16.0


class TestConditionOnInput:
    """`_maybe_condition_on_input` — measurement conditioning for diffusion (#20).

    Standard image diffusion denoises noisy(target) with no measurement input,
    which admits a measurement-independent solution. When
    ``training.diffusion.condition_on_input`` is set, the strategy concatenates
    the (LR/ULF) input onto the noised target along the channel axis. Gated so
    every existing arm (flag default False) is byte-identical.
    """

    @staticmethod
    def _strategy(flag: bool) -> DiffusionTrainingStrategy:
        from types import SimpleNamespace

        s = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
        s.config = SimpleNamespace(
            training=SimpleNamespace(diffusion=SimpleNamespace(condition_on_input=flag))
        )
        return s

    @pytest.mark.unit
    def test_flag_on_standard_path_concats_input(self) -> None:
        s = self._strategy(True)
        noisy = torch.randn(2, 1, 8, 16, 16)
        cond = torch.randn(2, 1, 8, 16, 16)
        out = s._maybe_condition_on_input(
            noisy,
            input_batch=cond,
            noisy_images=noisy,
            is_cold_diffusion=False,
            is_latent_diffusion=False,
        )
        assert out.shape[1] == 2  # [noisy || condition]
        assert torch.equal(out[:, :1], noisy)
        assert torch.equal(out[:, 1:], cond)

    @pytest.mark.unit
    def test_flag_off_is_passthrough(self) -> None:
        s = self._strategy(False)
        noisy = torch.randn(2, 1, 8, 16, 16)
        out = s._maybe_condition_on_input(
            noisy,
            input_batch=torch.randn_like(noisy),
            noisy_images=noisy,
            is_cold_diffusion=False,
            is_latent_diffusion=False,
        )
        assert out is noisy

    @pytest.mark.unit
    def test_cold_and_latent_are_passthrough(self) -> None:
        s = self._strategy(True)
        noisy = torch.randn(1, 1, 4, 8, 8)
        cond = torch.randn_like(noisy)
        for cold, latent in ((True, False), (False, True)):
            out = s._maybe_condition_on_input(
                noisy,
                input_batch=cond,
                noisy_images=noisy,
                is_cold_diffusion=cold,
                is_latent_diffusion=latent,
            )
            assert out is noisy

    @pytest.mark.unit
    def test_already_concatenated_is_not_double_conditioned(self) -> None:
        """When another step (e.g. smaps) already grew the channel axis
        (``model_input is not noisy_images``), do not concat again."""
        s = self._strategy(True)
        noisy = torch.randn(1, 1, 4, 8, 8)
        already = torch.cat([noisy, torch.randn_like(noisy)], dim=1)
        out = s._maybe_condition_on_input(
            already,
            input_batch=torch.randn_like(noisy),
            noisy_images=noisy,
            is_cold_diffusion=False,
            is_latent_diffusion=False,
        )
        assert out is already

    @pytest.mark.unit
    def test_mismatched_spatial_condition_is_resized(self) -> None:
        s = self._strategy(True)
        noisy = torch.randn(2, 1, 8, 16, 16)
        cond_big = torch.randn(2, 1, 8, 32, 32)
        out = s._maybe_condition_on_input(
            noisy,
            input_batch=cond_big,
            noisy_images=noisy,
            is_cold_diffusion=False,
            is_latent_diffusion=False,
        )
        assert tuple(out.shape) == (2, 2, 8, 16, 16)


def _fake_smaps_self():
    return _types.SimpleNamespace(
        _smaps_cache=None,
        _SMAPS_CACHE_MAX=8,
        _configured_estimation_method=lambda: "power_iter",
        _configured_estimation_kwargs=lambda: {"kernel_size": 7, "acs_size": 16},
    )


class TestSmapsCalibrationFromFullySampled:
    """Coil maps must calibrate from the dense fully-sampled ACS, never the
    undersampled input or the diffusion-degraded tensor (g-factor overlap fix)."""

    @pytest.mark.unit
    def test_estimate_smaps_cached_crops_to_acs(self) -> None:
        # acs_only is wired into the cached path → out-of-ACS corruption of the
        # calibration source cannot change the maps.
        torch.manual_seed(0)
        k = torch.randn(1, 4, 64, 64, dtype=torch.complex64)
        torch.manual_seed(1)
        clean = DiffusionTrainingStrategy._estimate_smaps_cached(
            _fake_smaps_self(), k, 64, 64
        )
        corrupt = k.clone()
        outside = torch.ones(64, 64, dtype=torch.bool)
        outside[24:40, 24:40] = False
        corrupt[:, :, outside] = corrupt[:, :, outside] + 100.0
        torch.manual_seed(1)
        after = DiffusionTrainingStrategy._estimate_smaps_cached(
            _fake_smaps_self(), corrupt, 64, 64
        )
        torch.testing.assert_close(clean, after)

    @pytest.mark.unit
    def test_cached_estimate_passes_acs_only(self) -> None:
        src = inspect.getsource(DiffusionTrainingStrategy._estimate_smaps_cached)
        assert "acs_only=True" in src

    @pytest.mark.unit
    def test_training_calibration_uses_target_not_noisy_images(self) -> None:
        src = inspect.getsource(DiffusionTrainingStrategy._prepare_diffusion_inputs)
        assert "acs_kspace = target_batch" in src
        assert "acs_kspace = noisy_images" not in src  # degraded fallback removed
        assert "CLAUDE.md #9/#16" in src  # raises instead of silently degrading

    @pytest.mark.unit
    def test_validation_calibration_uses_ref_not_input_batch(self) -> None:
        src = inspect.getsource(
            DiffusionTrainingStrategy._generate_validation_prediction
        )
        assert "acs_kspace_t = input_batch" not in src  # undersampled source removed
        assert "acs_kspace = target_batch" in src
        assert "acs_only=True" in src


class TestCascadeCompleteness:
    """#1303 — a cascade that loses a severity level must not pass as complete.

    Before this, ``validation_step`` skipped an unevaluable level with a warning
    and returned the remaining columns. Everything downstream then averaged over
    "whatever survived" under the column name a complete cascade uses, so a run
    that lost the hardest rung scored HIGHER than one that evaluated all three —
    and ``restore_best_weights`` preferred the broken one.
    """

    # ---- the two ValueErrors are told apart by STATE, not by message text ----

    @staticmethod
    def _accelerator(num_timesteps, acceleration_range):
        """A REAL accelerator, built the way the cascade builds it.

        Deliberately not a mock: the discrimination this class pins is a property
        of ``KSpaceAccelerator``'s step-schedule inverse, and a mock would let the
        test agree with a wrong understanding of it (which is how the first
        version of the corpus scan behind this PR reported a vacuous "0 arms").
        """
        from spectramr.infrastructure.training.utils.kspace_masks import (
            KSpaceMaskGenerator,
        )

        gen = KSpaceMaskGenerator(
            num_timesteps=num_timesteps,
            default_pattern="random",
            accelerator_kwargs={
                "acceleration_schedule": "step",
                "acceleration_range": list(acceleration_range),
                "base_acceleration": min(acceleration_range),
                "max_acceleration": max(acceleration_range),
            },
        )
        return gen._get_accelerator(None)

    @pytest.mark.unit
    def test_level_absent_from_range_is_not_declared(self) -> None:
        """R=8 is simply not a rung this ladder has: a config LIMITATION.

        ``timestep_for_acceleration`` raises, and skipping is the honest answer —
        so ``_level_is_declared`` must say False and let the cascade continue.
        """
        accel = self._accelerator(10, [2.0, 4.0, 16.0, 32.0])
        with pytest.raises(ValueError):
            accel.timestep_for_acceleration(8.0)
        assert DiffusionTrainingStrategy._level_is_declared(accel, 8) is False
        # ...and the rungs it does declare are still declared, so the predicate
        # is not simply returning False for everything.
        assert DiffusionTrainingStrategy._level_is_declared(accel, 2) is True
        assert DiffusionTrainingStrategy._level_is_declared(accel, 32) is True

    @pytest.mark.unit
    def test_declared_but_unrealised_level_is_declared(self) -> None:
        """#1171: 8 rungs over 2 timesteps — R=8 is declared, realised at no ``t``.

        The forward index is ``min(int(ratio * 8), 7)`` and can take only two
        values, so six of the eight declared rungs are unreachable. That is a
        config DEFECT, and the guard in ``timestep_for_acceleration`` exists to
        catch it — the cascade must let it propagate, not convert it back into a
        silent skip.
        """
        accel = self._accelerator(2, [2.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0])
        with pytest.raises(ValueError, match="realised at NO"):
            accel.timestep_for_acceleration(8.0)
        assert DiffusionTrainingStrategy._level_is_declared(accel, 8) is True

    @pytest.mark.unit
    def test_both_subcases_raise_the_same_exception_type(self) -> None:
        """The reason the split cannot be done in an ``except`` clause.

        Both sub-cases raise a bare ``ValueError``. Discriminating on the message
        would silently flip every arm to the wrong branch the first time the
        wording changed, so the state predicate is the only safe test.
        """
        not_declared = self._accelerator(10, [2.0, 4.0, 16.0, 32.0])
        unrealised = self._accelerator(2, [2.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0])
        excs = []
        for accel in (not_declared, unrealised):
            with pytest.raises(ValueError) as exc_info:
                accel.timestep_for_acceleration(8.0)
            excs.append(exc_info.value)
        assert type(excs[0]) is type(excs[1]) is ValueError

    @pytest.mark.unit
    def test_a_full_ladder_declares_every_cascade_level(self) -> None:
        """The corpus's actual shape: a 28-rung ladder over 28 timesteps.

        All 63 step-scheduled ``experiments/inprogress/`` arms declare a ladder
        covering 2/8/32, which is why the corpus is unaffected by making #1171
        fatal. This pins that the predicate agrees on such a ladder.
        """
        rungs = [2.0, 4.0, 8.0, 16.0, 32.0]
        accel = self._accelerator(len(rungs), rungs)
        for level in CASCADING_LEVELS:
            assert DiffusionTrainingStrategy._level_is_declared(accel, level) is True
            accel.timestep_for_acceleration(float(level))  # must not raise

    @pytest.mark.unit
    def test_predicate_reads_through_the_wrapper_hop(self) -> None:
        """``ColdDiffusionAccelerator`` is a wrapper; the schedule state is inside.

        ``timestep_for_acceleration`` delegates to ``self.accelerator`` while
        ``acceleration_range`` lives on the wrapped instance. Reading it off the
        wrapper returns nothing and the predicate would answer False for every
        arm — i.e. it would route #1171 into the skip branch, restoring exactly
        the bug this PR removes.
        """
        accel = self._accelerator(4, [2.0, 8.0, 16.0, 32.0])
        assert getattr(accel, "acceleration_range", None) is None
        assert accel.accelerator.acceleration_range == [2.0, 8.0, 16.0, 32.0]
        assert DiffusionTrainingStrategy._level_is_declared(accel, 8) is True

    @pytest.mark.unit
    def test_no_explicit_range_means_nothing_is_declared(self) -> None:
        """Without an explicit ladder there is no grid, so no rung is "declared".

        The binary step fallback answers every request without raising, so this
        branch is unreachable in practice — pinned so a future refactor cannot
        make an absent range read as "everything is declared" and turn a skip
        into a spurious fatal error.
        """
        from spectramr.infrastructure.training.utils.kspace_masks import (
            KSpaceMaskGenerator,
        )

        gen = KSpaceMaskGenerator(
            num_timesteps=8,
            default_pattern="random",
            accelerator_kwargs={"acceleration_schedule": "step", "max_acceleration": 32.0},
        )
        accel = gen._get_accelerator(None)
        assert DiffusionTrainingStrategy._level_is_declared(accel, 8) is False

    # ---- a partial mean may not occupy the complete mean's column name ------

    @pytest.mark.unit
    def test_complete_cascade_keeps_the_mean_column(self) -> None:
        """The unchanged path: every arm selecting on ``val_*_mean`` is unaffected."""
        m = {"val_psnr_2x": 30.0, "val_psnr_8x": 24.0, "val_psnr_32x": 18.0}
        DiffusionTrainingStrategy._stamp_accel_mean(m, [2, 8, 32], ["psnr"])
        assert m["val_psnr_mean"] == pytest.approx(24.0)
        assert "val_psnr_mean_partial" not in m

    @pytest.mark.unit
    def test_partial_cascade_is_renamed_not_silently_averaged(self) -> None:
        """An incomplete ladder publishes ``_mean_partial`` and NO ``_mean``.

        An arm pointing ``early_stopping.metric`` at ``val_psnr_mean`` then finds
        the key missing — loud — instead of a number that is not comparable with
        the other epochs' values.
        """
        m = {"val_psnr_2x": 30.0, "val_psnr_8x": 24.0}  # 32x lost
        DiffusionTrainingStrategy._stamp_accel_mean(m, [2, 8, 32], ["psnr"], complete=False)
        assert "val_psnr_mean" not in m
        assert m["val_psnr_mean_partial"] == pytest.approx(27.0)

    @pytest.mark.unit
    def test_losing_the_hardest_level_inflates_the_average(self) -> None:
        """The harm, as a number: this is why the two may not share a name.

        The ladder is monotone in difficulty, so dropping the most-accelerated
        rung RAISES the mean. A run that failed at 32x (27.0) outranks a healthy
        run that evaluated all three (24.0) — the selection target rewards the
        failure.
        """
        healthy = {"val_psnr_2x": 30.0, "val_psnr_8x": 24.0, "val_psnr_32x": 18.0}
        degraded = {"val_psnr_2x": 30.0, "val_psnr_8x": 24.0}
        DiffusionTrainingStrategy._stamp_accel_mean(healthy, [2, 8, 32], ["psnr"])
        DiffusionTrainingStrategy._stamp_accel_mean(degraded, [2, 8, 32], ["psnr"])
        # Pre-fix behaviour, reproduced by asking for the same column name:
        assert degraded["val_psnr_mean"] > healthy["val_psnr_mean"]
        # ...which is precisely what `complete=False` refuses to publish.
        degraded_fixed = {"val_psnr_2x": 30.0, "val_psnr_8x": 24.0}
        DiffusionTrainingStrategy._stamp_accel_mean(
            degraded_fixed, [2, 8, 32], ["psnr"], complete=False
        )
        assert "val_psnr_mean" not in degraded_fixed

    # ---- the two silent no-ops now record themselves -----------------------

    @pytest.mark.unit
    def test_accel_gap_records_whether_it_ran(self) -> None:
        """An absent gap column used to be indistinguishable from an unasked one."""
        present = {"val_psnr_2x": 30.0, "val_psnr_32x": 18.0}
        DiffusionTrainingStrategy._stamp_accel_psnr_gap(present, [2, 8, 32])
        assert present["val_psnr_accel_gap"] == pytest.approx(12.0)
        assert present["val_accel_gap_unavailable"] == 0.0

        missing_endpoint = {"val_psnr_2x": 30.0, "val_psnr_8x": 24.0}  # no 32x
        DiffusionTrainingStrategy._stamp_accel_psnr_gap(missing_endpoint, [2, 8, 32])
        assert "val_psnr_accel_gap" not in missing_endpoint
        assert missing_endpoint["val_accel_gap_unavailable"] == 1.0

    @pytest.mark.unit
    def test_l4_gate_records_that_it_did_not_run(self, mock_diffusion_config, training_env) -> None:
        """Gate ENABLED but under-fed: an un-run DC-blob check must not read as a pass."""
        strategy = TestInputDependenceGate._make_strategy(
            training_env, mock_diffusion_config, tol=0.01
        )
        all_metrics: dict[str, float] = {"val_psnr_2x": 20.0}
        strategy._apply_input_dependence_gate(all_metrics, [], [2, 8, 32])
        assert all_metrics["val_input_dependence_skipped"] == 1.0
        assert "val_measurement_collapse" not in all_metrics
        strategy.logging_service.log_warning.assert_called_once()

    @pytest.mark.unit
    def test_l4_gate_records_that_it_did_run(self, mock_diffusion_config, training_env) -> None:
        """Control for the above: a gate that ran stamps 0.0, not nothing."""
        torch.manual_seed(1)
        strategy = TestInputDependenceGate._make_strategy(
            training_env, mock_diffusion_config, tol=0.01
        )
        preds = [torch.rand(1, 1, 8, 8) for _ in range(3)]
        all_metrics: dict[str, float] = {}
        strategy._apply_input_dependence_gate(all_metrics, preds, [2, 8, 32])
        assert all_metrics["val_input_dependence_skipped"] == 0.0
        assert "val_measurement_collapse" in all_metrics

    @pytest.mark.unit
    def test_disabled_gate_stamps_nothing_at_all(self, mock_diffusion_config, training_env) -> None:
        """``tol=None`` is a deliberate opt-out, not a skip.

        Stamping ``val_input_dependence_skipped`` here would make every arm that
        turned the gate off look degraded in the CSV.
        """
        strategy = TestInputDependenceGate._make_strategy(
            training_env, mock_diffusion_config, tol=None
        )
        all_metrics: dict[str, float] = {"val_psnr_2x": 20.0}
        strategy._apply_input_dependence_gate(all_metrics, [], [2, 8, 32])
        assert "val_input_dependence_skipped" not in all_metrics

    # ---- the source-level contract the loop must keep ----------------------

    @pytest.mark.unit
    def test_in_distribution_prediction_failure_raises_not_continues(self) -> None:
        """``hr_fakes is None`` on an in-distribution rung must abort the batch.

        ``_run_validation`` catches per batch, so raising makes that batch
        contribute NO keys rather than a partial dict — which also keeps it clear
        of the per-key/global-count averaging asymmetry filed as #1323. If the
        failure is systematic every batch raises and the F36 guard turns it fatal.
        """
        src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
        assert "_levels_evaluated" in src
        assert "_levels_skipped" in src
        assert "val_cascade_complete" in src
        # The held-out branch keeps its skip; the in-distribution one raises.
        assert "raise RuntimeError(" in src
        assert "_level_is_declared" in src

    @pytest.mark.unit
    def test_completeness_stamps_are_all_three(self) -> None:
        """Expected/evaluated/complete travel together — one alone cannot be read."""
        src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
        for key in (
            "val_cascade_levels_expected",
            "val_cascade_levels_evaluated",
            "val_cascade_complete",
        ):
            assert key in src, f"{key} is not stamped"


def test_single_rung_ladder_records_the_gap_as_unavailable():
    """`validation.cascade.levels: [8]` has no accel gap — that is DATA.

    The `len < 2` early return was unreachable while the ladder was a 3-element
    module constant. A declared one-rung ladder makes it live, and returning
    silently would leave both `val_psnr_accel_gap` and
    `val_accel_gap_unavailable` absent — restoring the indistinguishable
    absence the flag was added (#1303, pitfall #16) to remove.
    """
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    metrics = {"val_psnr_8x": 18.9}
    DiffusionTrainingStrategy._stamp_accel_psnr_gap(metrics, [8])
    assert metrics["val_accel_gap_unavailable"] == 1.0
    assert "val_psnr_accel_gap" not in metrics


def test_an_empty_ladder_also_records_the_gap_as_unavailable():
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )

    metrics: dict[str, float] = {}
    DiffusionTrainingStrategy._stamp_accel_psnr_gap(metrics, [])
    assert metrics["val_accel_gap_unavailable"] == 1.0


def test_realized_acceleration_reaches_the_per_case_row():
    """Both readouts of the realized R are fed from ONE variable.

    The tall row has carried ``acceleration_realized`` since #1295; the per-case
    CSV declared the column but was never handed the value, so a column the docs
    advertise could not appear. Forwarding the SAME local -- rather than
    re-inverting the schedule at the second call site -- is what keeps the two
    surfaces from disagreeing about the same rung (non-negotiable 17).

    Pinned at the call site, not at the context dict: a producer that names the
    key and always receives ``None`` satisfies the declaration check in
    ``test_metric_sink.py`` while the sink, which omits ``None``, still writes
    no column.
    """
    src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
    assert src.count("acceleration_realized=acceleration_realized") == 2, (
        "expected both the per-case context and the tall row to be fed from the "
        "same `acceleration_realized` local"
    )
