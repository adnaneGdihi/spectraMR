"""Unit tests for the explicit flow-matching (PF-ODE) strategy (PR-9).

Covers :class:`FlowMatchingStrategy` — the genuine Lipman et al. /
rectified-flow velocity-field objective with a probability-flow ODE
sampler, distinct from the ``flow_matching`` alias that merely points at
the generic :class:`DiffusionTrainingStrategy`.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)
from spectramr.infrastructure.training.strategies.flow_matching_strategy import (
    FlowMatchingStrategy,
)
from tests.utils.mock_environment import create_mock_training_env


@pytest.fixture
def simple_model():
    """A trivial UNet-shaped generator (identity-able)."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, kernel_size=3, padding=1),
    )


@pytest.fixture
def mock_fm_config():
    """Minimal config that satisfies DiffusionTrainingStrategy.__init__."""
    config = MagicMock()
    config.training_mode = "flow_matching_pfode"
    config.optimization = MagicMock()
    config.optimization.optimizer.learning_rate = 1e-3
    config.optimization.precision.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    config.optimization.precision.dtype = "float32"
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    config.optimization.gradient.clip.enabled = False
    config.optimization.gradient.clip.method = "norm"
    config.optimization.gradient.clip.value = 1.0
    config.objectives = MagicMock()
    config.device = "cpu"
    config.acceleration = None
    config.deep_supervision_weight = 0.0

    config.data = MagicMock()
    config.data.prior_loading = MagicMock()
    config.data.prior_loading.enabled = False

    config.physics = None  # prevent MagicMock auto-creation for phys_cfg

    config.model = MagicMock()
    config.model.model_type = "flow_matching_pfode"
    config.model.in_channels = 1
    config.model.out_channels = 1

    class MockLosses:
        def __init__(self):
            self.reconstruction = MagicMock()
            for name in (
                "frequency_weighted_l1_kspace",
                "complex_l1",
                "log_spectral",
                "background_suppression",
                "l1",
                "l2",
                "energy_conservation",
                "frequency_domain",
                "hfen",
                "rician_consistency",
            ):
                setattr(self.reconstruction, f"enable_{name}", False)
                setattr(self.reconstruction, f"lambda_{name}", 0.0)
            self.reconstruction.frequency_weighted_l1_kspace_alpha = 0.0
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
    config.diffusion = None

    return config


@pytest.fixture
def training_env(simple_model, mock_fm_config):
    opt = OptimizationBuilder.create_single_optimizer(
        simple_model.parameters(), learning_rate=1e-4, optimizer_type="adam"
    )
    return create_mock_training_env(
        config=mock_fm_config,
        device="cpu",
        generator=simple_model,
        opt_g=opt,
        model_type="flow_matching_pfode",
    )


@pytest.fixture(autouse=True)
def mock_resolve_service():
    with patch("spectramr.infrastructure.di.di_container.resolve_service") as mock_resolve:
        mock_resolve.return_value = MagicMock()
        yield mock_resolve


class TestFlowMatchingLoss:
    """The velocity-regression objective."""

    def test_loss_dict_keys_and_scalars(self, training_env):
        strategy = FlowMatchingStrategy(env=training_env)
        target = torch.randn(2, 1, 16, 16)
        losses = strategy._compute_losses_impl(input_batch={"target": target}, epoch=0)

        assert "loss_total" in losses
        assert "loss_fm_velocity" in losses
        for key in ("loss_total", "loss_fm_velocity"):
            assert isinstance(losses[key], torch.Tensor)
            assert losses[key].dim() == 0  # scalar

    def test_loss_zero_for_exact_velocity_generator(self, training_env):
        """A generator that returns exactly u_t = x1 - x0 yields ~0 loss.

        We monkeypatch the conditional-path construction so the generator
        receives x0/x1 and can emit the true constant velocity.
        """

        class ExactVelocity(torch.nn.Module):
            """Returns the conditional velocity stamped on the strategy."""

            def __init__(self, owner):
                super().__init__()
                self.owner = owner
                # a real parameter so optimizer construction is valid
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, x_t, t):
                return self.owner._true_velocity

        strategy = FlowMatchingStrategy(env=training_env)
        exact = ExactVelocity(strategy)
        strategy.env.models["generator"] = exact
        strategy.env.generator = exact

        target = torch.randn(2, 1, 16, 16)
        losses = strategy._compute_losses_impl(input_batch={"target": target}, epoch=0)
        assert losses["loss_total"].item() < 1e-10

    def test_loss_no_generator_returns_zero(self, training_env):
        strategy = FlowMatchingStrategy(env=training_env)
        strategy.env.models["generator"] = None
        strategy.env.generator = None
        losses = strategy._compute_losses_impl(
            input_batch={"target": torch.randn(2, 1, 8, 8)}, epoch=0
        )
        assert losses["loss_total"].item() == 0.0


class TestProbabilityFlowODESampler:
    """The PF-ODE integrator."""

    def test_sample_shape_preserved(self, training_env):
        strategy = FlowMatchingStrategy(env=training_env)
        x0 = torch.randn(3, 1, 8, 8)

        def v(x, t):
            return torch.ones_like(x)

        out = strategy.sample(v, x0, n_steps=10, method="euler")
        assert out.shape == x0.shape

    def test_euler_constant_velocity_exact(self, training_env):
        """For a constant field v=c, Euler is exact: x(1) = x0 + c."""
        strategy = FlowMatchingStrategy(env=training_env)
        x0 = torch.zeros(2, 1, 4, 4)
        c = 2.5

        def v(x, t):
            return torch.full_like(x, c)

        out = strategy.sample(v, x0, n_steps=4, method="euler")
        assert torch.allclose(out, torch.full_like(x0, c), atol=1e-5)

    def test_euler_error_decreases_with_steps(self, training_env):
        """On a known time-varying field, more Euler steps -> less error.

        Field v(x, t) = a (constant in x, linear-ish in t handled by
        analytic integral). Use v = t so the analytic solution is
        x(1) = x0 + integral_0^1 t dt = x0 + 0.5.
        """
        strategy = FlowMatchingStrategy(env=training_env)
        x0 = torch.zeros(1, 1, 2, 2)

        def v(x, t):
            # t is a per-sample scalar tensor broadcast over x
            return torch.full_like(x, float(t.flatten()[0]))

        analytic = torch.full_like(x0, 0.5)
        err_coarse = (strategy.sample(v, x0, n_steps=2, method="euler") - analytic).abs().max()
        err_fine = (strategy.sample(v, x0, n_steps=64, method="euler") - analytic).abs().max()
        assert err_fine < err_coarse

    def test_heun_more_accurate_than_euler(self, training_env):
        """Heun (2nd order) beats Euler on the same time-varying field."""
        strategy = FlowMatchingStrategy(env=training_env)
        x0 = torch.zeros(1, 1, 2, 2)

        def v(x, t):
            return torch.full_like(x, float(t.flatten()[0]))

        analytic = torch.full_like(x0, 0.5)
        err_euler = (strategy.sample(v, x0, n_steps=4, method="euler") - analytic).abs().max()
        err_heun = (strategy.sample(v, x0, n_steps=4, method="heun") - analytic).abs().max()
        assert err_heun < err_euler

    def test_unknown_method_raises(self, training_env):
        """No silent fallback for an unknown integrator (pitfall #9)."""
        strategy = FlowMatchingStrategy(env=training_env)
        x0 = torch.zeros(1, 1, 2, 2)
        with pytest.raises(ValueError, match="method"):
            strategy.sample(lambda x, t: x, x0, n_steps=2, method="rk4")


class TestTimeConditioningDispatch:
    """Finding #2: dispatch on signature, never except-TypeError to branch."""

    def test_generator_accepts_time_detection(self):
        class _TimeGen:
            def forward(self, x, t):
                return x

        class _NoTimeGen:
            def forward(self, x):
                return x

        assert FlowMatchingStrategy._generator_accepts_time(_TimeGen()) is True
        assert FlowMatchingStrategy._generator_accepts_time(_NoTimeGen()) is False

    def test_in_forward_typeerror_propagates(self, training_env):
        """A TypeError raised INSIDE gen.forward must NOT be swallowed.

        Pre-fix (`try: gen(x_t, t) except TypeError: gen(x_t)`) silently retried
        the 1-arg form, masking a real forward bug (pitfall #9).
        """

        class _BuggyTimeGen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, x_t, t):
                raise TypeError("genuine in-forward bug")

        strategy = FlowMatchingStrategy(env=training_env)
        buggy = _BuggyTimeGen()
        strategy.env.models["generator"] = buggy
        strategy.env.generator = buggy
        with pytest.raises(TypeError, match="genuine in-forward bug"):
            strategy._compute_losses_impl(input_batch={"target": torch.randn(2, 1, 8, 8)}, epoch=0)


# ---------------------------------------------------------------------------
# The model-input contract (non-negotiable 14).
#
# This strategy overrides `_compute_losses_impl`, so it inherited
# `DiffusionTrainingStrategy`'s `snapshot_prepared_is_model_input = False` +
# `snapshot_model_input_tag = "diffusion_step"` while never emitting that tag:
# its artifacts said "the visible input is not the model input, see
# diffusion_step" and no diffusion_step snapshot existed.
# ---------------------------------------------------------------------------


def _bare_fm(recorder: dict) -> FlowMatchingStrategy:
    """`_compute_losses_impl` with only the seams it reads stubbed."""
    s = object.__new__(FlowMatchingStrategy)
    s.device = torch.device("cpu")
    s._resolve_legacy_batch = lambda input_batch, kwargs: input_batch  # type: ignore[method-assign]
    s._declared_model_input = None

    def _gen(x: torch.Tensor) -> torch.Tensor:  # no time arg -> gen(x_t) branch
        recorder["model_saw"] = x
        return torch.zeros_like(x)

    s.env = MagicMock()
    s.env.generator = _gen
    return s


def test_flow_matching_declares_the_interpolant_it_feeds() -> None:
    recorder: dict = {}
    s = _bare_fm(recorder)
    x1 = torch.randn(2, 1, 8, 8)
    x0 = torch.randn(2, 1, 8, 8)

    s._compute_losses_impl(input_batch={"target": x1, "source": x0}, epoch=0)

    assert s._declared_model_input is not None, (
        "flow matching inherits the carve-out, so it owes the wrapper a declaration"
    )
    tensors, extra, in_kspace_keys = s._declared_model_input

    # The contract: what was declared IS what the network received.
    assert torch.equal(tensors["model_input"], recorder["model_saw"])
    # And it is genuinely the interpolant, not either endpoint -- capturing an
    # endpoint is the failure this contract exists to make impossible.
    assert not torch.equal(tensors["model_input"], x0)
    assert not torch.equal(tensors["model_input"], x1)
    assert extra["model_input_key"] == "model_input"
    assert in_kspace_keys == set(), "must be explicit, not None"


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: mse_loss(pred, u_t) regresses the velocity field, not the target.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert FlowMatchingStrategy.__dict__["folds_image_losses"] is False
    assert FlowMatchingStrategy.__dict__["inline_losses"] == frozenset()
