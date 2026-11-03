"""Unit tests for RiemannianBlochDiffusionStrategy's manifold reduction.

The strategy models p(M0, T1, T2) on ``BlochRelaxationManifold``. Its target
used to be reduced with ``flatten(start_dim=1)[..., :3]``, which is a no-op on
the ``[B, 3]`` synthetic batches it was written against and *wrong* on the
``[B, 3, *spatial]`` parameter-map stack that ``dataset_type: quantitative``
serves: it returns the first three voxels of the FIRST map and reads them as
(M0, T1, T2). The residual stays finite and falls smoothly, so nothing shows.

It never fired because ``QuantitativeMapDataset`` emitted no ``target`` at all
and every batch was rejected upstream; making that route serve is what turns
this from dormant into live, so the reduction is pinned here.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold
from spectramr.infrastructure.training.strategies.riemannian_bloch_diffusion_strategy import (
    _UNCACHED_POINT_LIMIT,
    RiemannianBlochDiffusionStrategy,
)


def _strategy(*, cache_resolution: int = 4):
    """A stand-in carrying only what the reduction reads.

    Constructing the real strategy needs a full TrainingEnvironment; the
    reduction depends on nothing but the manifold, so bind the real methods to
    a namespace rather than building a training stack around a reshape.
    """
    stub = types.SimpleNamespace(
        manifold=BlochRelaxationManifold(cache_resolution=cache_resolution),
        _logged_point_count=-1,
    )
    for name in ("_to_manifold_points", "_log_coordinate_ranges"):
        setattr(
            stub,
            name,
            types.MethodType(getattr(RiemannianBlochDiffusionStrategy, name), stub),
        )
    return stub


def _map_stack(b=2, x=3, y=3, z=2):
    """[B, 3, X, Y, Z] with each parameter channel a distinct constant."""
    m0 = torch.full((b, 1, x, y, z), 0.8)
    t1 = torch.full((b, 1, x, y, z), 1000.0)
    t2 = torch.full((b, 1, x, y, z), 80.0)
    return torch.cat([m0, t1, t2], dim=1)


class TestManifoldPointReduction:
    def test_flat_batch_passes_through(self):
        points = torch.rand(5, 3)
        assert torch.equal(_strategy()._to_manifold_points(points), points)

    def test_map_stack_gathers_the_channel_axis_not_the_flattened_prefix(self):
        """Every reduced point must be one voxel's (M0, T1, T2).

        Under the old ``flatten(1)[..., :3]`` this returns (0.8, 0.8, 0.8) —
        three M0 voxels — which is finite, in-range, and not a manifold point.
        """
        reduced = _strategy()._to_manifold_points(_map_stack())
        assert reduced.shape[-1] == 3
        expected = torch.tensor([0.8, 1000.0, 80.0])
        assert torch.allclose(reduced, expected.expand_as(reduced))

    def test_reduction_yields_one_point_per_voxel(self):
        reduced = _strategy()._to_manifold_points(_map_stack(b=2, x=3, y=3, z=2))
        assert reduced.shape == (2 * 3 * 3 * 2, 3)

    @pytest.mark.parametrize("n_channels", [1, 2, 4, 5])
    def test_wrong_parameter_channel_count_raises(self, n_channels):
        stack = torch.rand(2, n_channels, 3, 3, 2)
        with pytest.raises(ValueError, match="parameter axis on dim 1"):
            _strategy()._to_manifold_points(stack)

    def test_flat_batch_with_wrong_width_raises(self):
        with pytest.raises(ValueError, match=r"\[B, 3\] manifold points"):
            _strategy()._to_manifold_points(torch.rand(5, 4))

    def test_scalar_target_raises(self):
        with pytest.raises(ValueError, match="at least"):
            _strategy()._to_manifold_points(torch.rand(5))


class TestUncachedMetricGuard:
    """``metric_tensor`` without a cache loops in Python over every point, so a
    per-voxel reduction on a real patch is a hang rather than a slow step."""

    def test_large_point_count_without_a_cache_raises(self):
        stack = torch.rand(4, 3, 64, 64)
        assert _UNCACHED_POINT_LIMIT < 4 * 64 * 64
        with pytest.raises(ValueError, match="metric_cache_resolution"):
            _strategy(cache_resolution=0)._to_manifold_points(stack)

    def test_large_point_count_with_a_cache_is_allowed(self):
        stack = torch.rand(4, 3, 64, 64)
        reduced = _strategy(cache_resolution=4)._to_manifold_points(stack)
        assert reduced.shape == (4 * 64 * 64, 3)

    def test_small_point_count_without_a_cache_is_allowed(self):
        reduced = _strategy(cache_resolution=0)._to_manifold_points(_map_stack())
        assert reduced.shape[-1] == 3


class TestCoordinateRangeReport:
    """``_interpolate_metric_cache`` clamps out-of-bounds coordinates silently,
    so the measured position of each axis is logged rather than guessed at."""

    def test_reports_each_coordinate_against_the_declared_bounds(self, caplog):
        with caplog.at_level("INFO"):
            _strategy()._to_manifold_points(_map_stack())
        text = caplog.text
        assert "manifold points/step" in text
        for name in ("M0", "T1", "T2"):
            assert f"{name}: median=" in text
        assert "declared bounds" in text

    def test_out_of_bounds_fraction_is_reported_not_swallowed(self, caplog):
        """T1 = 9e5 ms is far outside the manifold's declared (50, 3500)."""
        stack = _map_stack()
        stack[:, 1] = 9e5
        with caplog.at_level("INFO"):
            _strategy()._to_manifold_points(stack)
        assert "outside=100.0%" in caplog.text


class TestHeldoutQMapValidation:
    """The held-out pass, reached the way a run reaches it: training_mode ->
    TrainingStrategyFactory -> the strategy's validation_step, scoring with the
    registered ``geodesic_qmap_error`` on a registry-built model."""

    @staticmethod
    def _resolved_strategy(model, *, eval_t=0.25):
        import inspect

        from spectramr.infrastructure.training.strategies.riemannian_bloch_heldout import (
            HeldoutProtocol,
        )
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        cfg = types.SimpleNamespace(
            training=types.SimpleNamespace(
                strategy_class=None, training_mode="riemannian_bloch_diffusion"
            )
        )
        cls = TrainingStrategyFactory().get_strategy_class(cfg)
        assert cls is RiemannianBlochDiffusionStrategy
        # train.py's Pattern-A gate: two named tensors, so batch_idx reaches the seed.
        params = list(inspect.signature(cls.validation_step).parameters)
        assert params[1:3] == ["input_batch", "target_batch"]
        strat = object.__new__(cls)
        strat.manifold = BlochRelaxationManifold(cache_resolution=4)
        strat._logged_point_count = -1
        strat._qmap_metric = None
        strat._heldout = (
            None if eval_t is None else HeldoutProtocol(t_eval=eval_t, t_min=1e-3, step_size=0.05)
        )
        strat.device = torch.device("cpu")
        strat.config = types.SimpleNamespace(run=types.SimpleNamespace(seed=42))
        strat.env = types.SimpleNamespace(generator=model)
        return strat

    @staticmethod
    def _model():
        import spectramr.models.diffusion.riemannian_bloch  # noqa: F401  (registers the model)
        from spectramr.models.registry import get_model_class

        torch.manual_seed(0)
        return get_model_class("riemannian_diffusion_qmap")(time_dim=8, hidden=8, n_layers=1)

    def test_emits_the_held_out_geodesic_error_and_its_noised_baseline(self):
        strat = self._resolved_strategy(self._model())
        maps = _map_stack(b=1, x=4, y=4, z=1).squeeze(-1)
        out = strat.validation_step(maps, maps, batch_idx=0)
        assert set(out) == {"val_geodesic_qmap_error", "val_geodesic_qmap_error_noised"}
        assert all(torch.isfinite(torch.tensor(v)) for v in out.values())
        assert strat._last_visual_pred.shape == maps.shape
        assert strat._qmap_metric.manifold is strat.manifold
        # Same seed and batch index: the same draws, so passes are comparable.
        assert strat.validation_step(maps, maps, batch_idx=0) == out

    def test_zero_background_does_not_change_the_emitted_errors(self):
        model = self._model()
        brain = _map_stack(b=1, x=4, y=4, z=1).squeeze(-1)
        padded = torch.nn.functional.pad(brain, (2, 2, 2, 2))
        brain_out = self._resolved_strategy(model).validation_step(brain, brain, batch_idx=0)
        padded_out = self._resolved_strategy(model).validation_step(padded, padded, batch_idx=0)
        for key, value in brain_out.items():
            assert padded_out[key] == pytest.approx(value, rel=1e-5)

    def test_emitted_keys_are_declared(self):
        emitted = RiemannianBlochDiffusionStrategy.capabilities.emitted_metrics
        assert {"val_geodesic_qmap_error", "val_geodesic_qmap_error_noised"} <= emitted

    def test_undeclared_noise_level_raises(self):
        strat = self._resolved_strategy(self._model(), eval_t=None)
        maps = _map_stack(b=1, x=2, y=2, z=1).squeeze(-1)
        with pytest.raises(ValueError, match="heldout_eval_t"):
            strat.validation_step(maps, maps, batch_idx=0)
