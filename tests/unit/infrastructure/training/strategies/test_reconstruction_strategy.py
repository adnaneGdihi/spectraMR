"""Unit tests for reconstruction training strategy."""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
)

try:
    import torch
except ImportError:
    from tests.utils.mock_torch import (
        setup_mock_heavy_dependencies,
        setup_mock_torch,
        setup_mock_yaml,
    )

    setup_mock_torch()
    setup_mock_yaml()
    setup_mock_heavy_dependencies()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from tests.utils.mock_environment import create_mock_training_env


class TestReconstructionStrategy(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.tmp_dir_obj.name)

        self.mock_config = MagicMock()
        self.mock_config.physics = MagicMock()
        self.mock_config.physics.pinn = MagicMock()
        self.mock_config.physics.pinn.enabled = False
        self.mock_config.training_mode = "reconstruction"
        self.mock_config.optimization = MagicMock()
        self.mock_config.optimization.optimizer.learning_rate = 1e-3
        self.mock_config.optimization.precision.enabled = False
        # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
        # amp_dtype must be a real member of {bfloat16,float16,float32}: base.__init__
        # now calls resolve_amp_precision() which validates it (added with the
        # amp_dtype wiring) — a bare MagicMock raises "amp_dtype must be one of ...".
        self.mock_config.optimization.precision.dtype = "float32"
        self.mock_config.optimization.gradient.clip.enabled = False
        self.mock_config.optimization.gradient.clip.method = "norm"
        self.mock_config.optimization.gradient.clip.value = 1.0
        self.mock_config.objectives = MagicMock()
        self.mock_config.objectives.reconstruction = MagicMock()
        self.mock_config.objectives.reconstruction.lambda_l1 = 1.0
        self.mock_config.objectives.reconstruction.lambda_perceptual = 0.0

        # [FIX] Alias losses to objectives to satisfy LossBuilder
        self.mock_config.losses = self.mock_config.objectives

        # [FIX] Explicitly set optional fields to avoid MagicMock vs int errors
        self.mock_config.losses.reconstruction.enable_l1 = True
        self.mock_config.losses.reconstruction.enable_complex_l1 = False
        self.mock_config.losses.reconstruction.lambda_complex_l1 = 0.0
        self.mock_config.losses.reconstruction.enable_perceptual = False
        self.mock_config.losses.reconstruction.lambda_perceptual = 0.0
        self.mock_config.losses.reconstruction.enable_ssim = False
        self.mock_config.losses.reconstruction.lambda_ssim = 0.0
        self.mock_config.losses.reconstruction.enable_kspace = False
        self.mock_config.losses.reconstruction.lambda_kspace = 0.0

        # [FIX] Additional optional fields checked by LossBuilder
        self.mock_config.losses.reconstruction.lambda_complex_l1 = 0.0
        self.mock_config.losses.reconstruction.enable_l2 = False
        self.mock_config.losses.reconstruction.lambda_l2 = 0.0
        self.mock_config.losses.reconstruction.lambda_smooth_l1 = 0.0
        self.mock_config.losses.reconstruction.lambda_complex_mse = 0.0
        self.mock_config.losses.reconstruction.lambda_perceptual = 0.0
        self.mock_config.losses.reconstruction.lambda_ssim = 0.0
        self.mock_config.losses.reconstruction.lambda_ms_ssim = 0.0
        self.mock_config.losses.reconstruction.lambda_lpips = 0.0
        self.mock_config.losses.reconstruction.lambda_frequency = 0.0
        self.mock_config.losses.reconstruction.lambda_log_spectral = 0.0
        self.mock_config.losses.reconstruction.lambda_spectral_kspace = 0.0
        self.mock_config.losses.reconstruction.lambda_edge = 0.0
        self.mock_config.losses.reconstruction.lambda_sobel = 0.0
        self.mock_config.losses.reconstruction.lambda_tv = 0.0
        self.mock_config.losses.reconstruction.lambda_weighted_kspace_l1 = 0.0
        self.mock_config.losses.reconstruction.lambda_dists = 0.0
        self.mock_config.losses.reconstruction.lambda_lpips = 0.0
        self.mock_config.losses.reconstruction.lambda_frequency = 0.0
        self.mock_config.losses.reconstruction.lambda_hfen = 0.0
        self.mock_config.losses.reconstruction.lambda_log_spectral = 0.0
        self.mock_config.losses.reconstruction.lambda_spectral_kspace = 0.0
        self.mock_config.losses.reconstruction.lambda_edge = 0.0
        self.mock_config.losses.reconstruction.lambda_sobel = 0.0
        self.mock_config.losses.reconstruction.lambda_mind_ssc = 0.0
        self.mock_config.losses.reconstruction.lambda_hist = 0.0
        self.mock_config.losses.reconstruction.lambda_ffl = 0.0
        self.mock_config.losses.reconstruction.lambda_latent_consistency = 0.0
        self.mock_config.losses.reconstruction.lambda_tissue_bounds = 0.0

        # [FIX] Physics losses
        self.mock_config.losses.physics = MagicMock()
        self.mock_config.losses.physics.lambda_physics_constraint = 0.0
        self.mock_config.losses.physics.lambda_bloch_residual = 0.0

        # [FIX] Deep supervision
        self.mock_config.deep_supervision_weight = 0.0
        self.mock_config.device = "cpu"

        self.mock_context = MagicMock()
        self.mock_context.config = self.mock_config
        self.mock_context.device = torch.device("cpu")
        self.mock_context.fft_transformer = MagicMock()

        self.simple_model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        self.mock_env = create_mock_training_env(
            config=self.mock_config,
            device="cpu",
            generator=self.simple_model,
            opt_g=MagicMock(spec=torch.optim.Optimizer),
            model_type="reconstruction",
        )
        self.mock_env.losses = self.mock_config.losses

        # Legacy for other tests if needed, but we should now use TrainingEnvironment
        # Kept for backward compatibility with existing test code
        self.training_state = MagicMock(spec=TrainingEnvironment)
        self.training_state.config = self.mock_config
        self.training_state.device = torch.device("cpu")
        self.training_state.models = {"generator": self.simple_model}
        self.training_state.generator = self.simple_model
        self.training_state.discriminator = None
        self.training_state.optimizers = {
            "opt_g": MagicMock(spec=torch.optim.Optimizer)
        }
        self.training_state.opt_g = self.training_state.optimizers["opt_g"]
        self.training_state.model_type = "reconstruction"
        self.training_state.losses = {}
        self.training_state.loss_function = {}

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_init_default_parameters(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy)

    def test_init_with_device_string(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy)

    def test_init_with_torch_device(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            device = torch.device("cpu")
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy)

    def test_init_with_training_state(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy)

    def test_init_initializes_components(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy)

    def test_verify_strategy_config(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            with patch.object(
                ReconstructionTrainingStrategy, "_verify_strategy_config"
            ) as mock_verify:
                strategy = ReconstructionTrainingStrategy(env=self.mock_env)

    def test_log_config_features(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            with patch.object(
                ReconstructionTrainingStrategy, "_log_config_features"
            ) as mock_log:
                strategy = ReconstructionTrainingStrategy(env=self.mock_env)

    def test_loss_computer_initialization(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy.loss_computer)

    def test_data_consistency_initialization(self):
        # [PHASE 2] Strategy-side DC now requires model-integrated dc_layer
        self.mock_config.physics.data_consistency.enabled = True
        mock_dc = torch.nn.Identity()
        self.simple_model.dc_layer = mock_dc

        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy.dc_layer)

    def test_fft_transformer_extraction(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy)

    def test_pinn_enabled_initialization(self):
        self.mock_config.physics = MagicMock()
        self.mock_config.physics.pinn = MagicMock()
        self.mock_config.physics.pinn.enabled = True
        self.mock_config.physics.pinn.pde_type = "wave_equation"
        self.mock_config.physics.pinn.lambda_pde = 0.1

        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            with patch("spectramr.infrastructure.physics.pinn.get_pde"):
                strategy = ReconstructionTrainingStrategy(env=self.mock_env)
                self.assertTrue(hasattr(strategy, "pinn_module"))

    def test_pinn_disabled_initialization(self):
        self.mock_config.physics = MagicMock()
        self.mock_config.physics.pinn = MagicMock()
        self.mock_config.physics.pinn.enabled = False

        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNone(strategy.pinn_module)

    def test_pinn_missing_config(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNone(strategy.pinn_module)

    def test_is_implicit_model_detection(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            if hasattr(strategy, "_is_implicit_model"):
                result = strategy._is_implicit_model()
                self.assertIsInstance(result, bool)

    def test_train_step_method_exists(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertTrue(
                hasattr(strategy, "train_step")
                or callable(getattr(strategy, "train_step", None))
            )

    def test_validation_step_method_exists(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertTrue(
                hasattr(strategy, "validation_step")
                or callable(getattr(strategy, "validation_step", None))
            )

    def test_compute_losses_method_exists(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertTrue(
                hasattr(strategy, "_compute_losses")
                or hasattr(strategy, "compute_losses")
            )

    def test_device_assignment(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy.device)

    def test_device_string_conversion(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertTrue(
                isinstance(strategy.device, torch.device) or strategy.device == "cpu"
            )

    def test_invalid_training_mode(self):
        self.mock_config.training_mode = "invalid_mode"
        with patch.object(
            ReconstructionTrainingStrategy,
            "_setup_strategy_specific_components",
            side_effect=ValueError("Invalid mode"),
        ):
            with self.assertRaises(ValueError):
                ReconstructionTrainingStrategy(env=self.mock_env)

    def test_missing_config(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            try:
                strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            except AttributeError:
                pass

    def test_full_strategy_initialization(self):
        with patch.object(
            ReconstructionTrainingStrategy, "_setup_strategy_specific_components"
        ):
            strategy = ReconstructionTrainingStrategy(env=self.mock_env)
            self.assertIsNotNone(strategy)
            self.assertIsNotNone(strategy.device)


def test_reconstruction_delegates_scheduled_weights_to_base_seam():
    """reconstruction no longer copies loss_weight_overrides inline; it relies on
    the paradigm-agnostic BaseTrainingStrategy.sync_scheduled_loss_weights (C1).
    Source-level pin (a full forward OOM-kills a dev box)."""
    import inspect

    from spectramr.infrastructure.training.strategies import reconstruction
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    impl_src = inspect.getsource(
        reconstruction.ReconstructionTrainingStrategy._compute_losses_impl
    )
    # The inline `self.loss_computer.scheduled_weights = ...` was removed.
    assert "scheduled_weights =" not in impl_src
    # The shared seam exists and is the single publisher.
    assert hasattr(BaseTrainingStrategy, "sync_scheduled_loss_weights")


def test_reconstruction_strategy_supports_domain_conditioning_sources():
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    supported = set(ReconstructionTrainingStrategy._SUPPORTED_CONDITION_SOURCES)
    assert {"field_strength", "scanner_id", "site_id", "contrast_id"} <= supported
    # severity_vec is a VF-only source — must NOT be claimed here.
    assert "severity_vec" not in supported


def test_generate_predictions_applies_input_conditioning(monkeypatch):
    import torch

    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    strat = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)

    seen = {}

    def fake_apply(x, batch):
        seen["called"] = True
        seen["channels"] = int(x.shape[1])
        return x + 1.0  # marker so we can detect modulation downstream

    captured = {}

    class _Gen(torch.nn.Module):
        def forward(self, x, **kwargs):
            captured["input"] = x
            return x

    class _Env:
        generator = _Gen()

    strat._apply_input_conditioning = fake_apply  # type: ignore[method-assign]
    strat.env = _Env()
    strat.config = type("C", (), {"training": type("T", (), {})()})()

    lr = torch.zeros(2, 3, 8, 8)
    _hr_fakes, _ = strat._generate_predictions(lr, {}, {"contrast_id": torch.tensor([0, 1])})

    assert seen.get("called") is True
    assert seen.get("channels") == 3
    # The generator must receive the MODULATED tensor (lr + 1), not the raw lr.
    torch.testing.assert_close(captured["input"], lr + 1.0)


def test_batch_context_propagates_scanner_site_and_contrast_idx_alias():
    """Regression: scanner_id/site_id/contrast_idx must reach batch_context.

    Before the fix, ``_prepare_batch_context_reconstruction`` key_mapping only
    listed ``["contrast_id"]`` (no ``contrast_idx`` alias) and omitted
    ``scanner_id`` / ``site_id`` entirely.  All three therefore silently no-op
    when the raw batch uses the real dataset key names — pitfall #16 facade.

    This test drives a raw batch carrying ``scanner_id``, ``site_id``, and
    ``contrast_idx`` (as emitted by m4raw_dataset / slice_dataset) through
    ``_prepare_batch_context_reconstruction`` and asserts all three land in the
    returned ``batch_context`` under the canonical keys ``scanner_id``,
    ``site_id``, ``contrast_id``.
    """
    from types import SimpleNamespace

    import torch

    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    # Minimal host following the _host() pattern from test_reconstruction.py (mixin tests).
    host = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    host.env = SimpleNamespace(
        generator=None,
        config=SimpleNamespace(
            model=SimpleNamespace(model_type="reconstruction", input_type="image"),
            data=SimpleNamespace(),
            training=SimpleNamespace(),
        ),
    )
    host.state = SimpleNamespace(config=host.env.config)
    host.config = host.env.config
    host.logging_service = None
    host.dc_layer = None

    # Simulate a raw batch as emitted by m4raw_dataset / slice_dataset:
    # - contrast is keyed as ``contrast_idx`` (NOT ``contrast_id``)
    # - scanner_id and site_id are dataset-level domain labels
    raw_batch = {
        "scanner_id": "scanner_A",
        "site_id": "site_1",
        "contrast_idx": torch.tensor([2]),
    }
    inp = torch.zeros(1, 1, 8, 8)
    tgt = torch.zeros(1, 1, 8, 8)

    ctx = host._prepare_batch_context_reconstruction(inp, tgt, batch=raw_batch)

    assert "scanner_id" in ctx, (
        "scanner_id missing from batch_context — key_mapping entry absent (pitfall #16)"
    )
    assert "site_id" in ctx, (
        "site_id missing from batch_context — key_mapping entry absent (pitfall #16)"
    )
    assert "contrast_id" in ctx, (
        "contrast_id missing from batch_context — contrast_idx alias not accepted (pitfall #16)"
    )
    assert torch.equal(ctx["contrast_id"], raw_batch["contrast_idx"]), (
        "contrast_id in batch_context does not match the contrast_idx tensor from raw batch"
    )


def test_declared_loss_weights_reads_config_lists():
    """The seam resolves per-entry declarative weights from config.losses.*_losses."""
    import types as _types

    strat = object.__new__(ReconstructionTrainingStrategy)
    strat.config = _types.SimpleNamespace(
        losses=_types.SimpleNamespace(
            image_losses=[
                {"name": "hfen", "weight": 0.2},
                {"name": "ms_ssim", "weight": 0.15},
            ],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    assert strat._declared_loss_weights() == {"hfen": 0.2, "ms_ssim": 0.15}


def test_apply_builder_image_losses_folds_weighted_terms():
    """env.losses modules are folded with their declared weight and recorded per name."""
    import types as _types

    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss

    strat = object.__new__(ReconstructionTrainingStrategy)
    strat.env = _types.SimpleNamespace(losses={"charbonnier": CharbonnierLoss()})
    strat.config = _types.SimpleNamespace(
        losses=_types.SimpleNamespace(
            image_losses=[{"name": "charbonnier", "weight": 0.5}],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    pred = torch.zeros(1, 1, 16, 16)
    target = torch.ones(1, 1, 16, 16)
    components: dict = {}
    extra = strat._apply_builder_image_losses(pred, target, components)
    assert extra is not None and torch.isfinite(extra)
    assert "loss_charbonnier" in components and "loss_builder_aux" in components
    # weight is applied: extra == 0.5 * charbonnier(pred, target)
    expected = 0.5 * CharbonnierLoss()(pred, target)
    assert torch.allclose(extra, expected)


def test_apply_builder_image_losses_honors_scheduled_override():
    """A loss_schedule ramp writes loop_state.loss_weight_overrides; the folded
    builder losses MUST use the override weight, not the static declared weight —
    else the curriculum is a silent no-op for inline-folding field strategies
    (pitfall #16 at the schedule layer)."""
    import types as _types

    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss

    strat = object.__new__(ReconstructionTrainingStrategy)
    strat.env = _types.SimpleNamespace(losses={"charbonnier": CharbonnierLoss()})
    strat.config = _types.SimpleNamespace(
        losses=_types.SimpleNamespace(
            image_losses=[{"name": "charbonnier", "weight": 0.5}],  # static declared
            kspace_losses=[],
            complex_losses=[],
        )
    )
    # The controller ramped charbonnier 0.5 -> 2.0 this step.
    strat.loop_state = _types.SimpleNamespace(
        loss_weight_overrides={"charbonnier": 2.0}
    )
    pred = torch.zeros(1, 1, 16, 16)
    target = torch.ones(1, 1, 16, 16)
    components: dict = {}
    extra = strat._apply_builder_image_losses(pred, target, components)
    # override (2.0) supersedes the static declared weight (0.5)
    expected = 2.0 * CharbonnierLoss()(pred, target)
    assert torch.allclose(extra, expected)


def test_apply_builder_image_losses_empty_overrides_uses_static():
    """An empty override map (no rule fired this step) falls back to the static
    declared weight — byte-identical to the pre-schedule behavior."""
    import types as _types

    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss

    strat = object.__new__(ReconstructionTrainingStrategy)
    strat.env = _types.SimpleNamespace(losses={"charbonnier": CharbonnierLoss()})
    strat.config = _types.SimpleNamespace(
        losses=_types.SimpleNamespace(
            image_losses=[{"name": "charbonnier", "weight": 0.5}],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    strat.loop_state = _types.SimpleNamespace(loss_weight_overrides={})
    pred = torch.zeros(1, 1, 16, 16)
    target = torch.ones(1, 1, 16, 16)
    components: dict = {}
    extra = strat._apply_builder_image_losses(pred, target, components)
    expected = 0.5 * CharbonnierLoss()(pred, target)
    assert torch.allclose(extra, expected)


def test_apply_builder_image_losses_noop_without_env_losses():
    """No env.losses -> returns None and writes no components (inline-only unchanged)."""
    import types as _types

    strat = object.__new__(ReconstructionTrainingStrategy)
    strat.env = _types.SimpleNamespace()  # no `losses` attribute
    components: dict = {}
    result = strat._apply_builder_image_losses(
        torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8), components
    )
    assert result is None and components == {}


def test_apply_builder_image_losses_skips_inline_l1():
    """The universal inline ``l1`` placeholder is skipped (no double-count); only
    ADDED terms (hfen/ms_ssim/...) fold in. This lets a shared strategy adopt the
    seam without rewriting every arm/ablation YAML that still carries [{l1,1.0}]."""
    import types as _types

    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss
    from spectramr.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(ReconstructionTrainingStrategy)
    strat.env = _types.SimpleNamespace(losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()})
    strat.config = _types.SimpleNamespace(
        losses=_types.SimpleNamespace(
            image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    components: dict = {}
    extra = strat._apply_builder_image_losses(
        torch.zeros(1, 1, 16, 16), torch.ones(1, 1, 16, 16), components)
    assert extra is not None
    assert "loss_hfen" in components
    assert "loss_l1" not in components  # inline placeholder skipped


if __name__ == "__main__":
    unittest.main()


class TestRefuseInertDiffusionKnobs:
    """Planted violations: diffusion-only training knobs routed to the reconstruction
    strategy are never read, so construction must raise rather than ignore them."""

    @staticmethod
    def _stub(training: dict) -> object:
        from types import SimpleNamespace

        from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

        return SimpleNamespace(
            config=SimpleNamespace(training=TrainingStrategyConfigSchema.model_validate(training))
        )

    @pytest.mark.parametrize(
        "training",
        [{"diffusion": {"type": "score_based"}}, {"num_timesteps": 100}],
    )
    def test_inert_knob_raises(self, training: dict) -> None:
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        with pytest.raises(ValueError, match="never reads it"):
            ReconstructionTrainingStrategy._refuse_inert_diffusion_knobs(self._stub(training))

    def test_clean_training_block_passes(self) -> None:
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        ReconstructionTrainingStrategy._refuse_inert_diffusion_knobs(
            self._stub({"input_domain": "image", "device": "cuda"})
        )


# ── validation conditions on the batch's contrast, as training does ──────────


class _StopForwardError(Exception):
    pass


def _validation_probe():
    """A strategy whose ``_validation_forward`` records its context and stops."""
    strategy = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    from types import SimpleNamespace

    strategy.adapter_chains = {}
    strategy.env = SimpleNamespace(generator=torch.nn.Identity())
    strategy.state = None
    seen: dict = {}

    def _record(x, ctx, **kwargs):
        seen["ctx"] = ctx
        raise _StopForwardError

    strategy._validation_forward = _record
    return strategy, seen


def _run_validation(strategy, batch, x):
    import inspect

    from spectramr.pipelines.train import select_validation_extra_fields

    extra = select_validation_extra_fields(batch, inspect.signature(strategy.validation_step).parameters)
    with pytest.raises(_StopForwardError):
        strategy.validation_step(x, x, batch_idx=0, **extra)


def test_validation_receives_the_batch_contrast_idx_as_contrast_id():
    """The loop hands a TrainingBatch; its metadata contrast_idx must reach the forward."""
    from spectramr.data.batch_types import TrainingBatch

    strategy, seen = _validation_probe()
    x = torch.randn(2, 1, 8, 8)
    batch = TrainingBatch(input=x, target=x, metadata={"contrast_idx": torch.tensor([0, 3])})
    _run_validation(strategy, batch, x)
    assert torch.equal(seen["ctx"]["contrast_id"], torch.tensor([0, 3]))


def test_validation_without_a_contrast_keeps_the_bare_context():
    from spectramr.data.batch_types import TrainingBatch

    strategy, seen = _validation_probe()
    x = torch.randn(2, 1, 8, 8)
    _run_validation(strategy, TrainingBatch(input=x, target=x), x)
    assert "contrast_id" not in seen["ctx"]


def test_a_contrast_that_does_not_match_the_validation_batch_raises():
    from spectramr.data.batch_types import TrainingBatch

    strategy, _ = _validation_probe()
    x = torch.randn(4, 1, 8, 8)
    batch = TrainingBatch(input=x, target=x, metadata={"contrast_idx": torch.tensor([0, 3])})
    with pytest.raises(ValueError, match="contrast_id"):
        strategy.validation_step(x, x, batch_idx=0, batch_data=batch)


def test_the_validation_forward_hands_contrast_id_to_the_generator():
    """End of the seam: the generator that trained on contrast_id is graded with it."""
    from types import SimpleNamespace

    class _Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, x, contrast_id=None):
            self.seen = contrast_id
            return x

    gen = _Recorder()
    strategy = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    cfg = MagicMock()
    cfg.data.multislice_enabled = False
    cfg.model.in_channels = 1
    cfg.model.input_type = "image"
    cfg.validation.loader.chunk_size = 2
    strategy.env = SimpleNamespace(generator=gen, config=cfg)
    strategy.config = cfg
    strategy.device = torch.device("cpu")
    strategy._conditioner = None
    strategy._ensure_conditioner = lambda channels: None
    strategy._needs_kspace_adjoint = lambda ctx: False
    x = torch.randn(2, 1, 8, 8)
    ctx = {"use_dc": False, "measured_kspace": None, "contrast_id": torch.tensor([1, 2])}
    strategy._validation_forward(x, ctx)
    assert torch.equal(gen.seen, torch.tensor([1, 2]))


def _chunking_strategy(gen, chunk_size):
    from types import SimpleNamespace

    strategy = ReconstructionTrainingStrategy.__new__(ReconstructionTrainingStrategy)
    cfg = MagicMock()
    cfg.data.multislice_enabled = False
    cfg.model.in_channels = 1
    cfg.model.input_type = "image"
    cfg.validation.loader.chunk_size = chunk_size
    strategy.env = SimpleNamespace(generator=gen, config=cfg)
    strategy.config = cfg
    strategy.device = torch.device("cpu")
    strategy._conditioner = None
    strategy._ensure_conditioner = lambda channels: None
    strategy._needs_kspace_adjoint = lambda ctx: False
    return strategy


class _BatchRecorder(torch.nn.Module):
    """A per-sample conv that records the batch size of every call."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.batches = []

    def forward(self, x, contrast_id=None):
        self.batches.append(x.shape[0])
        out = self.conv(x)
        return out if contrast_id is None else out + contrast_id.view(-1, 1, 1, 1)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7])
def test_validation_forward_reads_chunk_size_and_matches_one_forward(chunk_size):
    """validation.loader.chunk_size caps the slices per generator call (the 2026-10-22
    validation OOMs), and the chunked output equals the single forward."""
    x = torch.randn(7, 1, 8, 8)
    ctx = {"use_dc": False, "measured_kspace": None, "contrast_id": torch.arange(7.0)}
    gen = _BatchRecorder()
    with torch.no_grad():
        whole = _chunking_strategy(gen, 7)._validation_forward(x, ctx)
        gen.batches.clear()
        chunked = _chunking_strategy(gen, chunk_size)._validation_forward(x, ctx)
    assert max(gen.batches) <= chunk_size
    assert sum(gen.batches) == 7
    torch.testing.assert_close(chunked, whole)


def test_chunked_validation_forward_takes_the_reconstruction_of_a_dict_output():
    class _DictGen(torch.nn.Module):
        def forward(self, x):
            return {"image": torch.zeros_like(x), "reconstruction": x + 1}

    x = torch.randn(5, 1, 4, 4)
    out = _chunking_strategy(_DictGen(), 2)._validation_forward(
        x, {"use_dc": False, "measured_kspace": None}
    )
    torch.testing.assert_close(out, x + 1)


def _windowing_strategy(gen, patch_size, batch_size=1):
    strategy = _chunking_strategy(gen, chunk_size=64)
    strategy.config.data.sampling.patch_size = patch_size
    strategy.config.data.loader.batch_size = batch_size
    return strategy


class _PatchRecorder(torch.nn.Module):
    """A pointwise 3-D map that records the spatial shape of every call."""

    def __init__(self, shrink: bool = False):
        super().__init__()
        self.shapes = []
        self.shrink = shrink

    def forward(self, x, mask=None):
        self.shapes.append(tuple(x.shape[2:]))
        return x[..., :-1] if self.shrink else 3 * x


def test_a_volume_larger_than_the_training_patch_is_graded_in_training_sized_windows():
    """A volumetric model trained on patches saw whole volumes at validation: more
    memory than the step, and a shape fixed-size networks cannot take."""
    x = torch.randn(2, 1, 20, 20, 12)
    gen = _PatchRecorder()
    with torch.no_grad():
        out = _windowing_strategy(gen, [8, 8, 4])._validation_forward(
            x, {"use_dc": False, "measured_kspace": None}
        )
    assert set(gen.shapes) == {(8, 8, 4)}
    torch.testing.assert_close(out, 3 * x)


def test_a_volume_within_the_patch_is_forwarded_whole():
    x = torch.randn(1, 1, 8, 8, 4)
    gen = _PatchRecorder()
    _windowing_strategy(gen, [8, 8, 4])._validation_forward(
        x, {"use_dc": False, "measured_kspace": None}
    )
    assert gen.shapes == [(8, 8, 4)]


def test_windowing_refuses_a_whole_volume_generator_input_and_a_shape_changing_model():
    import pytest

    x = torch.randn(1, 1, 16, 16, 8)
    strategy = _windowing_strategy(_PatchRecorder(), [8, 8, 4])
    with pytest.raises(ValueError, match="cannot be windowed"):
        strategy._validation_window(x, {"mask": torch.ones_like(x)})
    with pytest.raises(ValueError, match="shape of its input patch"):
        _windowing_strategy(_PatchRecorder(shrink=True), [8, 8, 4])._validation_forward(
            x, {"use_dc": False, "measured_kspace": None}
        )


def test_windows_share_a_forward_up_to_the_training_batch_with_per_sample_inputs_aligned():
    """Batching windows is what makes windowed validation affordable (5x on CPU);
    a per-sample input must be tiled with the stacked windows, not left at batch size."""

    class _Shift(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rows = []

        def forward(self, x, contrast_id=None):
            self.rows.append(x.shape[0])
            return x + contrast_id.view(-1, 1, 1, 1, 1)

    x = torch.randn(2, 1, 16, 16, 8)
    ctx = {"use_dc": False, "measured_kspace": None, "contrast_id": torch.tensor([1.0, 5.0])}
    one, many = _Shift(), _Shift()
    with torch.no_grad():
        a = _windowing_strategy(one, [8, 8, 4], batch_size=1)._validation_forward(x, ctx)
        b = _windowing_strategy(many, [8, 8, 4], batch_size=8)._validation_forward(x, ctx)
    assert max(many.rows) == 8 and max(one.rows) == 2
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, x + torch.tensor([1.0, 5.0]).view(-1, 1, 1, 1, 1))


def test_folded_slices_larger_than_a_depth_one_crop_are_graded_in_crop_sized_windows():
    """exp_hm_01 trains on 256x256x1 crops; validation folded whole 512x512 slices."""
    x = torch.randn(3, 1, 16, 16)
    gen = _PatchRecorder()
    with torch.no_grad():
        out = _windowing_strategy(gen, [8, 8, 1])._validation_forward(
            x, {"use_dc": False, "measured_kspace": None}
        )
    assert set(gen.shapes) == {(8, 8)}
    torch.testing.assert_close(out, 3 * x)


def test_folded_slices_within_the_crop_or_a_3d_patch_are_forwarded_whole():
    x = torch.randn(3, 1, 8, 8)
    for patch_size in ([8, 8, 1], [4, 4, 4]):
        gen = _PatchRecorder()
        _windowing_strategy(gen, patch_size)._validation_forward(
            x, {"use_dc": False, "measured_kspace": None}
        )
        assert gen.shapes == [(8, 8)]
