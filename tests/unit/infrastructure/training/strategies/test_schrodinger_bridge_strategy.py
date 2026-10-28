"""Unit tests for the Bloch Schrödinger-bridge (I2SB + Bloch-manifold) strategy.

Plan: PR-4 / idea N-A.

The strategy is an I2SB stochastic-interpolant bridge whose endpoints are
DATA (x0 = ULF source, x1 = HF target) plus a Bloch-manifold-consistency
penalty that constrains the predicted clean endpoint to physically
realizable (M0, T1, T2) spin signals via
:class:`spectramr.infrastructure.physics.manifolds.BlochRelaxationManifold`.

These tests are written first (TDD): they fail until
``schrodinger_bridge_strategy.py`` exists with the documented contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)
from spectramr.infrastructure.training.strategies.schrodinger_bridge_strategy import (
    BlochSchrodingerBridgeStrategy,
)
from tests.utils.mock_environment import create_mock_training_env


class _TinyVelocityNet(nn.Module):
    """Tiny generator: (I_t[B,C,H,W], t[B]) -> velocity[B,C,H,W]."""

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)


def _make_config() -> MagicMock:
    """Mock config that lets ``DiffusionTrainingStrategy.__init__`` run.

    Mirrors ``tests/unit/.../test_diffusion_strategy.py::mock_diffusion_config``:
    the diffusion base init iterates reconstruction-loss flags and may build a
    prior model, so those branches must be explicitly disabled rather than left
    as auto-created MagicMocks.
    """
    config = MagicMock()
    config.training_mode = "bloch_schrodinger_bridge"
    config.optimization = MagicMock()
    config.optimization.optimizer.learning_rate = 1e-3
    config.optimization.precision.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    config.optimization.precision.dtype = "float32"
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    config.optimization.gradient.clip.enabled = False
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.value = 1.0
    config.device = "cpu"
    config.acceleration = None
    config.deep_supervision_weight = 0.0
    config.physics = None
    config.objectives = MagicMock()

    config.data = MagicMock()
    config.data.prior_loading = MagicMock()
    config.data.prior_loading.enabled = False

    config.model = MagicMock()
    config.model.model_type = "diffusion"
    config.model.in_channels = 3
    config.model.out_channels = 3

    class MockRecon:
        def __init__(self) -> None:
            for flag in (
                "frequency_weighted_l1_kspace_alpha",
                "lambda_complex_l1",
                "lambda_log_spectral",
                "lambda_frequency_weighted_l1_kspace",
                "lambda_background_suppression",
                "lambda_l1",
                "lambda_l2",
                "lambda_energy_conservation",
                "lambda_frequency_domain",
                "lambda_hfen",
                "lambda_rician_consistency",
            ):
                setattr(self, flag, 0.0)
            for flag in (
                "enable_complex_l1",
                "enable_log_spectral",
                "enable_frequency_weighted_l1_kspace",
                "enable_background_suppression",
                "enable_l1",
                "enable_l2",
                "enable_energy_conservation",
                "enable_frequency_domain",
                "enable_hfen",
                "enable_rician_consistency",
            ):
                setattr(self, flag, False)

    class MockLosses:
        def __init__(self) -> None:
            self.reconstruction = MockRecon()
            self.gan = None
            self.physics = None
            self.latent = None
            self.ssl = None
            self.diffusion = config.objectives.diffusion

    config.losses = MockLosses()

    config.training = MagicMock()
    config.training.diffusion = MagicMock()
    config.training.diffusion.timesteps = 1000
    config.training.diffusion.num_timesteps = 1000
    config.training.diffusion.noise_schedule = "linear"
    # beta_start/beta_end are declared schema fields with numeric defaults;
    # the strategy forwards them to DiffusionScheduler, so a mock that
    # leaves them auto-generated hands linspace() a MagicMock.
    config.training.diffusion.beta_start = 1e-4
    config.training.diffusion.beta_end = 0.02
    # No typed schrodinger_bridge block: a bare MagicMock would auto-vivify and
    # clobber the constructor's lambda_bloch / bloch_cache_resolution knobs (the
    # source treats a present block as the SSOT, failing its int validation).
    config.training.schrodinger_bridge = None
    config.diffusion = None
    return config


def _make_env(channels: int = 3) -> tuple[MagicMock, _TinyVelocityNet]:
    gen = _TinyVelocityNet(channels=channels)
    opt = OptimizationBuilder.create_single_optimizer(
        gen.parameters(), learning_rate=1e-4, optimizer_type="adam"
    )
    env = create_mock_training_env(
        config=_make_config(),
        device="cpu",
        generator=gen,
        opt_g=opt,
        model_type="diffusion",
    )
    return env, gen


@pytest.fixture(autouse=True)
def _mock_resolve_service():
    with patch("spectramr.infrastructure.di.di_container.resolve_service") as mock_resolve:
        mock_resolve.return_value = MagicMock()
        yield mock_resolve


def _batch(b: int = 2, c: int = 3, h: int = 8, w: int = 8) -> dict[str, torch.Tensor]:
    # M0 in (0,1.5), T1 in (50,3500) ms, T2 in (10,500) ms — physical-ish ranges
    # so the manifold geodesic-distance machinery is exercised on realistic values.
    src = torch.stack(
        [
            torch.rand(b, h, w) * 1.0 + 0.2,
            torch.rand(b, h, w) * 1000 + 500,
            torch.rand(b, h, w) * 200 + 50,
        ],
        dim=1,
    )
    tgt = torch.stack(
        [
            torch.rand(b, h, w) * 1.0 + 0.2,
            torch.rand(b, h, w) * 1000 + 500,
            torch.rand(b, h, w) * 200 + 50,
        ],
        dim=1,
    )
    return {"source": src, "target": tgt}


class TestComputeLosses:
    def test_returns_total_and_bloch_keys(self) -> None:
        env, _ = _make_env()
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.1)
        out = strat._compute_losses_impl(input_batch=_batch())
        assert "loss_total" in out
        assert "loss_sb_velocity" in out
        assert "loss_bloch_manifold" in out
        lt = out["loss_total"]
        assert isinstance(lt, torch.Tensor)
        assert lt.dim() == 0  # scalar
        assert torch.isfinite(lt)

    def test_lambda_zero_recovers_pure_bridge(self) -> None:
        """λ=0 ⇒ total == velocity loss (trivial I2SB limit)."""
        env, _ = _make_env()
        torch.manual_seed(0)
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.0)
        out = strat._compute_losses_impl(input_batch=_batch())
        assert torch.allclose(out["loss_total"], out["loss_sb_velocity"])
        # the manifold term is computed but contributes zero
        assert out["loss_bloch_manifold"].item() == pytest.approx(0.0, abs=0.0) or (
            out["loss_total"].item() == pytest.approx(out["loss_sb_velocity"].item(), rel=1e-6)
        )

    def test_lambda_positive_increases_total(self) -> None:
        env, _ = _make_env()
        torch.manual_seed(1)
        strat0 = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.0)
        torch.manual_seed(1)
        out0 = strat0._compute_losses_impl(input_batch=_batch())
        env2, _ = _make_env()
        torch.manual_seed(1)
        strat1 = BlochSchrodingerBridgeStrategy(env=env2, lambda_bloch=1.0)
        torch.manual_seed(1)
        out1 = strat1._compute_losses_impl(input_batch=_batch())
        # bloch term is non-negative; with the same RNG the velocity part matches,
        # so total with λ>0 must be >= total at λ=0.
        assert out1["loss_total"].item() >= out0["loss_total"].item() - 1e-6
        assert out1["loss_bloch_manifold"].item() >= 0.0

    def test_manifold_is_exercised(self) -> None:
        """The bloch term must genuinely call a BlochRelaxationManifold method."""
        env, _ = _make_env()
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.5)
        with patch.object(
            strat.manifold,
            "geodesic_distance",
            wraps=strat.manifold.geodesic_distance,
        ) as spy:
            strat._compute_losses_impl(input_batch=_batch())
        assert spy.call_count >= 1

    def test_non_three_channel_raises_when_penalty_active(self) -> None:
        """Regression (finding #1, silent-skip): with lambda_bloch > 0 a C != 3
        endpoint must RAISE, not silently zero the headline Bloch novelty.

        Pre-fix this returned ``new_zeros(())`` so the advertised regularizer
        was a no-op for every shipped config (out_channels in {1, 2}). The fix
        raises so the audit / smoke pass catches the mismatch at startup
        (pitfall #9/#10).
        """
        env, _ = _make_env(channels=1)
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.5)
        b = _batch(c=1)
        b["source"] = b["source"][:, :1]
        b["target"] = b["target"][:, :1]
        with pytest.raises(ValueError, match=r"3, H, W|lambda_bloch"):
            strat._compute_losses_impl(input_batch=b)

    def test_two_channel_raises_when_penalty_active(self) -> None:
        """The frontier arm ships out_channels=2; that path must raise too."""
        strat = object.__new__(BlochSchrodingerBridgeStrategy)
        strat.lambda_bloch = 0.1
        with pytest.raises(ValueError):
            strat._bloch_manifold_penalty(torch.zeros(1, 2, 4, 4))

    def test_lambda_zero_with_wrong_channels_is_inert(self) -> None:
        """lambda_bloch == 0 disables the penalty entirely: no raise regardless
        of channel count (the trivial I2SB limit)."""
        env, _ = _make_env(channels=1)
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.0)
        b = _batch(c=1)
        b["source"] = b["source"][:, :1]
        b["target"] = b["target"][:, :1]
        out = strat._compute_losses_impl(input_batch=b)
        assert out["loss_bloch_manifold"].item() == pytest.approx(0.0)
        assert torch.allclose(out["loss_total"], out["loss_sb_velocity"])


class TestBlochCacheWiring:
    """Finding #1 (perf): the per-voxel autograd loop must be off the hot path."""

    def test_default_uses_tabulated_metric_cache(self) -> None:
        """By default the manifold is built with cache_resolution > 0 so
        ``metric_tensor`` interpolates a grid instead of running a per-voxel
        autograd loop."""
        env, _ = _make_env()
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.1)
        assert strat.bloch_cache_resolution > 0
        assert strat.manifold.cache_resolution > 0
        assert strat.manifold._cached_grid is not None

    def test_cache_query_does_not_invoke_per_point_fisher(self) -> None:
        """With the cache built, a metric query must NOT call the per-voxel
        ``_fisher_at_single_point`` autograd path (that is the hot-loop the fix
        removes)."""
        env, _ = _make_env()
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.1)
        with patch.object(
            strat.manifold,
            "_fisher_at_single_point",
            wraps=strat.manifold._fisher_at_single_point,
        ) as spy:
            strat._compute_losses_impl(input_batch=_batch())
        assert spy.call_count == 0, (
            "cache path must not fall back to per-voxel autograd in the loop"
        )

    def test_cache_resolution_zero_keeps_exact_path(self) -> None:
        """cache_resolution=0 opts back into the exact (slow) per-point path."""
        env, _ = _make_env()
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.1, bloch_cache_resolution=0)
        assert strat.manifold.cache_resolution == 0
        assert strat.manifold._cached_grid is None

    def test_invalid_cache_resolution_raises(self) -> None:
        """An illegal cache_resolution must fail at construction (pitfall #15:
        no silently-ignored knob)."""
        env, _ = _make_env()
        with pytest.raises(ValueError, match="bloch_cache_resolution"):
            BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.1, bloch_cache_resolution=-1)


class TestForwardTypeErrorNotSwallowed:
    """Finding #2: a forward-time TypeError must propagate, not be retried."""

    def test_in_forward_typeerror_propagates(self) -> None:
        """A genuine TypeError raised INSIDE gen.forward must NOT be swallowed.

        Pre-fix (`try: gen(i_t, t) except TypeError: gen(i_t)`) silently retried
        the 1-arg form, masking the bug. The signature-based dispatch surfaces
        it (pitfall #9).
        """

        class _BuggyTimeGen(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.ones(1))

            def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                raise TypeError("genuine in-forward bug")

        gen = _BuggyTimeGen()
        opt = OptimizationBuilder.create_single_optimizer(
            gen.parameters(), learning_rate=1e-4, optimizer_type="adam"
        )
        env = create_mock_training_env(
            config=_make_config(),
            device="cpu",
            generator=gen,
            opt_g=opt,
            model_type="diffusion",
        )
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.0)
        with pytest.raises(TypeError, match="genuine in-forward bug"):
            strat._compute_losses_impl(input_batch=_batch())

    def test_generator_accepts_time_detection(self) -> None:
        cls = BlochSchrodingerBridgeStrategy

        class _TimeGen:
            def forward(self, x, t):
                return x

        class _NoTimeGen:
            def forward(self, x):
                return x

        assert cls._generator_accepts_time(_TimeGen()) is True
        assert cls._generator_accepts_time(_NoTimeGen()) is False


class TestSample:
    def test_sample_shape_preserved(self) -> None:
        env, gen = _make_env()
        strat = BlochSchrodingerBridgeStrategy(env=env, lambda_bloch=0.1)
        x0 = _batch()["source"]
        out = strat.sample(lambda x, t: gen(x, t), x0, n_steps=5)
        assert out.shape == x0.shape
        assert torch.isfinite(out).all()


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: mse_loss(pred, true_velocity) regresses the bridge drift, not the target.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert BlochSchrodingerBridgeStrategy.__dict__["folds_image_losses"] is False
    assert BlochSchrodingerBridgeStrategy.__dict__["inline_losses"] == frozenset()
