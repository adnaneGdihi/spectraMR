"""Regression test for the 2026-06 strategy-audit finding SAQ-001 on
``stochastic_interpolants_strategy``.

Finding SAQ-001 [error / silent_fallback]:
    ``_compute_losses_impl`` dispatched the time-conditioned forward via
    ``try: gen(I_t, t) except TypeError: gen(I_t)``. A genuine ``TypeError``
    raised *inside* ``gen.forward`` (a shape/dtype/kwarg bug) was silently
    swallowed and retried with the wrong (1-arg) signature, masking the real
    failure (pitfall #9). The fix dispatches via the parent's
    ``_generator_accepts_time`` signature introspection so a real error
    propagates.

The heavy base ``__init__`` is bypassed via ``__new__`` and only the attributes
the tested path reads (``self.env``, ``self.device``) are wired.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.stochastic_interpolants_strategy import (
    StochasticInterpolantsStrategy,
)


class _TimeGenRaisesReal:
    """Time-conditioned generator (2 positional args) with a real internal bug.

    ``forward`` declares ``(x, t)`` so ``_generator_accepts_time`` returns True
    and the strategy calls ``gen(I_t, t)``. The forward then raises a TypeError
    that is NOT a signature mismatch (it is a genuine internal failure), which
    must propagate rather than trigger a silent 1-arg retry.
    """

    def forward(self, x, t):
        raise TypeError("internal forward bug: bad dtype in time embedding")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _TimeGen:
    """Well-behaved time-conditioned generator (records the call arity)."""

    def __init__(self):
        self.last_nargs = None

    def forward(self, x, t):
        self.last_nargs = 2
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _NoTimeGen:
    """Generator that only accepts the state (1 positional arg)."""

    def __init__(self):
        self.last_nargs = None

    def forward(self, x):
        self.last_nargs = 1
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _make_strategy(gen):
    strat = StochasticInterpolantsStrategy.__new__(StochasticInterpolantsStrategy)
    strat.env = SimpleNamespace(generator=gen)
    strat.device = torch.device("cpu")
    return strat


def _batch():
    return {
        "target": torch.randn(2, 1, 8, 8),
        "source": torch.randn(2, 1, 8, 8),
    }


class TestTimeDispatchDoesNotSwallowForwardError:
    def test_real_forward_typeerror_propagates(self):
        # The old try/except would have retried gen(I_t) and hidden this.
        strat = _make_strategy(_TimeGenRaisesReal())
        with pytest.raises(TypeError, match="internal forward bug"):
            strat._compute_losses_impl(input_batch=_batch())

    def test_time_conditioned_gen_called_with_time(self):
        gen = _TimeGen()
        strat = _make_strategy(gen)
        out = strat._compute_losses_impl(input_batch=_batch())
        assert gen.last_nargs == 2
        assert "loss_total" in out

    def test_non_time_gen_called_without_time(self):
        gen = _NoTimeGen()
        strat = _make_strategy(gen)
        out = strat._compute_losses_impl(input_batch=_batch())
        assert gen.last_nargs == 1
        assert "loss_total" in out


class TestSourceUsesIntrospectionNotTryExcept:
    def test_source_dispatches_via_accepts_time(self):
        src = inspect.getsource(StochasticInterpolantsStrategy._compute_losses_impl)
        assert "self._generator_accepts_time(gen)" in src
        # The silent ``except TypeError:`` *fallback statement* is gone. Strip
        # comment lines first so the explanatory comment that legitimately
        # names the old pattern does not trip the assertion.
        code_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
        code_only = "\n".join(code_lines)
        assert "except TypeError" not in code_only


class TestModelInputContract:
    """Non-negotiable 14: the declared tag must name a tensor that exists.

    This strategy overrides ``_compute_losses_impl``, so it inherited
    ``DiffusionTrainingStrategy``'s ``snapshot_prepared_is_model_input = False``
    / ``snapshot_model_input_tag = "diffusion_step"`` without ever emitting that
    tag — the artifacts pointed a reader at a snapshot that did not exist.
    """

    @staticmethod
    def _bare(recorder: dict) -> StochasticInterpolantsStrategy:
        s = object.__new__(StochasticInterpolantsStrategy)
        s.device = torch.device("cpu")
        s._resolve_legacy_batch = lambda input_batch, kwargs: input_batch  # type: ignore[method-assign]

        def _gen(x: torch.Tensor) -> torch.Tensor:  # 1-arg -> gen(I_t) branch
            recorder["model_saw"] = x
            return torch.zeros_like(x)

        s.env = SimpleNamespace(generator=_gen)
        return s

    def test_declares_the_interpolant_it_feeds(self, declared) -> None:
        recorder: dict = {}
        s = self._bare(recorder)
        x1 = torch.randn(2, 1, 8, 8)
        x0 = torch.randn(2, 1, 8, 8)

        s._compute_losses_impl(input_batch={"target": x1, "source": x0}, epoch=0)

        assert declared() is not None, (
            "SI builds its model input inside the step and names it"
        )
        tensors, extra, in_kspace_keys = declared()
        assert torch.equal(tensors["model_input"], recorder["model_saw"])
        assert extra["model_input_key"] == "model_input"
        assert in_kspace_keys == set(), "must be explicit, not None"

    def test_declared_input_carries_the_stochastic_term(self, declared) -> None:
        """What separates SI from flow matching is `sigma(t) * z`.

        Declaring the *deterministic* interpolant would be a plausible-looking
        near-miss: same shape, same scale, wrong tensor. Pin that the declared
        input is not reproducible from x0/x1/t alone.
        """
        recorder: dict = {}
        s = self._bare(recorder)
        torch.manual_seed(0)
        x1 = torch.randn(2, 1, 8, 8)
        x0 = torch.randn(2, 1, 8, 8)

        s._compute_losses_impl(input_batch={"target": x1, "source": x0}, epoch=0)
        tensors, _extra, _keys = declared()
        i_t = tensors["model_input"]

        # Recover t from the deterministic part is impossible without z, so
        # instead assert the weaker-but-decisive thing: no convex combination
        # of the endpoints equals I_t, because sigma(t) * z displaces it.
        for frac in torch.linspace(0.0, 1.0, 21):
            deterministic = (1 - frac) * x0 + frac * x1
            assert not torch.allclose(i_t, deterministic, atol=1e-6), (
                "declared input must be the STOCHASTIC interpolant, not the "
                "deterministic flow-matching path"
            )


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: mse_loss(pred, true_velocity) regresses the interpolant velocity.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert StochasticInterpolantsStrategy.__dict__["folds_image_losses"] is True
    assert StochasticInterpolantsStrategy.__dict__["empty_loss_stack_is_design"] is True
    assert StochasticInterpolantsStrategy.__dict__["inline_losses"] == frozenset()


def _image_cfg(weight: float) -> SimpleNamespace:
    entry = SimpleNamespace(name="l1", weight=weight, enabled=True, role="target")
    return SimpleNamespace(losses=SimpleNamespace(image_losses=[entry]))


class _RecordingLoss:
    """Image loss that records what it was asked to compare."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.calls.append((pred, target))
        return (pred - target).abs().mean()


def test_declared_image_loss_is_folded_onto_the_endpoint_estimate():
    weight = 0.5
    batch = _batch()

    torch.manual_seed(5)
    off = _make_strategy(_NoTimeGen())
    off.config = _image_cfg(weight)
    off.env.losses = {}
    loss_off = off._compute_losses_impl(input_batch=batch)["loss_total"]

    torch.manual_seed(5)
    on = _make_strategy(_NoTimeGen())
    on.config = _image_cfg(weight)
    l1 = _RecordingLoss()
    on.env.losses = {"l1": l1}
    out = on._compute_losses_impl(input_batch=batch)

    assert "loss_l1" in out
    ((pred, target),) = l1.calls
    assert torch.equal(target, batch["target"]), "the endpoint estimate is compared with x1"
    assert torch.allclose(out["loss_total"] - loss_off, weight * (pred - target).abs().mean())


class TestMeasurementConditionedSource:
    """x0 is the measurement, never pure noise: an unconditional interpolant raises."""

    @staticmethod
    def _factory_strategy(gen):
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        cls = TrainingStrategyFactory()._load_strategy_class("stochastic_interpolants")
        strat = cls.__new__(cls)
        strat.env = SimpleNamespace(generator=gen)
        strat.device = torch.device("cpu")
        return strat

    def test_input_becomes_source_on_the_production_calling_convention(self, declared):
        from spectramr.data.batch_types import TrainingBatch

        strat = self._factory_strategy(_NoTimeGen())
        x_in, x1 = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
        batch = TrainingBatch(input=x_in, target=x1)
        strat._compute_losses_impl(input_batch=x_in, target_batch=x1, epoch=0, batch=batch)
        tensors, _extra, _keys = declared()
        assert torch.equal(tensors["source_x0"], x_in)

    def test_published_source_wins_over_input(self, declared):
        strat = self._factory_strategy(_NoTimeGen())
        b = _batch()
        b["input"] = torch.randn(2, 1, 8, 8)
        strat._compute_losses_impl(input_batch=b)
        tensors, _extra, _keys = declared()
        assert torch.equal(tensors["source_x0"], b["source"])

    def test_missing_measurement_raises(self):
        strat = self._factory_strategy(_NoTimeGen())
        with pytest.raises(ValueError, match="neither 'source' nor 'input'"):
            strat._compute_losses_impl(input_batch={"target": torch.randn(2, 1, 8, 8)})

    def test_endpoint_shape_mismatch_raises(self):
        strat = self._factory_strategy(_NoTimeGen())
        batch = {"target": torch.randn(2, 1, 8, 8), "input": torch.randn(2, 4, 8, 8)}
        with pytest.raises(ValueError, match="endpoints disagree"):
            strat._compute_losses_impl(input_batch=batch)


class TestValidationIntegratesTheVelocityField:
    """Validation runs the interpolant ODE from the measurement, not a diffusion sampler."""

    @staticmethod
    def _strategy(gen, steps=None, chunk_size=8):
        strat = StochasticInterpolantsStrategy.__new__(StochasticInterpolantsStrategy)
        strat.env = SimpleNamespace(generator=gen)
        diff = SimpleNamespace(model_fields_set=set(), sampling_steps=100)
        strat.config = SimpleNamespace(
            validation=SimpleNamespace(
                sampling=SimpleNamespace(steps=steps),
                loader=SimpleNamespace(chunk_size=chunk_size),
            ),
            training=SimpleNamespace(diffusion=diff),
            undersampling=None,
        )
        strat._resolve_validation_sampling_steps = lambda: steps
        return strat

    def test_constant_velocity_moves_the_input_by_one(self):
        class _Unit(torch.nn.Module):
            def forward(self, x, t):
                return torch.ones_like(x)

        x_in = torch.randn(2, 1, 8, 8)
        strat = self._strategy(_Unit(), steps=4)
        out = strat._generate_validation_prediction(x_in, torch.zeros_like(x_in), None, None, {})
        assert torch.allclose(out, x_in + 1.0, atol=1e-6)
        assert strat._zf_measurement is None

    def test_declared_steps_reach_the_integrator(self):
        calls = []

        class _Count(torch.nn.Module):
            def forward(self, x):
                calls.append(1)
                return torch.zeros_like(x)

        x_in = torch.randn(1, 1, 4, 4)
        self._strategy(_Count(), steps=7)._generate_validation_prediction(
            x_in, x_in, None, None, {}
        )
        assert len(calls) == 7
        calls.clear()
        self._strategy(_Count(), steps=None)._generate_validation_prediction(
            x_in, x_in, None, None, {}
        )
        assert len(calls) == 50


def test_arm_routes_undersampling_through_cartesian_masking():
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings
    from spectramr.data.builders.torchio_transform_builder import TorchIOTransformConfig

    repo = Path(__file__).resolve().parents[5]
    arm = repo / "experiments/inprogress/promoted/exp_promoted_stochastic_interpolants.yaml"
    cfg = TorchIOTransformConfig.from_settings(TrainingSettings.from_yaml(arm))
    assert cfg.trajectory_type == "cartesian"


def test_validation_integration_honours_chunk_size():
    seen = []

    class _Rec(torch.nn.Module):
        def forward(self, x, t):
            seen.append(x.shape[0])
            return torch.ones_like(x)

    x_in = torch.randn(3, 1, 4, 4)
    strat = TestValidationIntegratesTheVelocityField._strategy(_Rec(), steps=2, chunk_size=1)
    out = strat._generate_validation_prediction(x_in, x_in, None, None, {})
    assert seen == [1] * 6
    assert torch.allclose(out, x_in + 1.0, atol=1e-6)


class TestValidationScoresOnlyTheMeasuredRung:
    """The ODE starts from the loader's measurement, so only its rung may be labelled."""

    @staticmethod
    def _strategy(levels, grid=None):
        strat = StochasticInterpolantsStrategy.__new__(StochasticInterpolantsStrategy)
        strat.config = SimpleNamespace(
            undersampling=SimpleNamespace(base_acceleration=4),
            validation=SimpleNamespace(
                cascade=SimpleNamespace(levels=levels),
                gates=SimpleNamespace(
                    held_out_severity_eval=grid is not None, held_out_severity_grid=grid
                ),
            ),
            model=SimpleNamespace(model_kwargs={}),
        )
        return strat

    def test_default_ladder_raises(self):
        with pytest.raises(ValueError, match="can score only that rung"):
            self._strategy(None)._assert_single_measured_rung()

    def test_other_rung_raises(self):
        with pytest.raises(ValueError, match="can score only that rung"):
            self._strategy([8])._assert_single_measured_rung()

    def test_held_out_grid_raises(self):
        with pytest.raises(ValueError, match="held-out"):
            self._strategy([4], grid=[12.0])._assert_single_measured_rung()

    def test_measured_rung_passes(self):
        self._strategy([4])._assert_single_measured_rung()


def test_arm_declares_its_measured_rung_as_the_whole_ladder():
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings

    repo = Path(__file__).resolve().parents[5]
    arm = repo / "experiments/inprogress/promoted/exp_promoted_stochastic_interpolants.yaml"
    settings = TrainingSettings.from_yaml(arm)
    strat = StochasticInterpolantsStrategy.__new__(StochasticInterpolantsStrategy)
    strat.config = settings
    strat._assert_single_measured_rung()
