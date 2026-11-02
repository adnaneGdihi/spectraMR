"""Held-out q-map pass: protocol, shapes, and the reverse-step sign."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold
from spectramr.infrastructure.training.strategies.riemannian_bloch_heldout import (
    HeldoutProtocol,
    foreground_mask,
    heldout_prediction,
    maps_to_points,
    points_to_maps,
)

pytestmark = pytest.mark.unit


def _maps(b: int = 1, h: int = 3, w: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    m0 = torch.empty(b, 1, h, w).uniform_(0.6, 1.0, generator=g)
    t1 = torch.empty(b, 1, h, w).uniform_(800.0, 1200.0, generator=g)
    t2 = torch.empty(b, 1, h, w).uniform_(60.0, 120.0, generator=g)
    return torch.cat([m0, t1, t2], dim=1)


def _manifold() -> BlochRelaxationManifold:
    return BlochRelaxationManifold(cache_resolution=4)


def _zero_field(theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(theta)


def test_protocol_step_count_and_validation():
    assert HeldoutProtocol(t_eval=0.25, t_min=1e-3, step_size=0.05).n_steps == 5
    with pytest.raises(ValueError, match="t_min < t_eval"):
        HeldoutProtocol(t_eval=1e-4, t_min=1e-3, step_size=0.05)
    with pytest.raises(ValueError, match="step_size"):
        HeldoutProtocol(t_eval=0.25, t_min=1e-3, step_size=0.0)


def test_points_round_trip_keeps_each_voxel():
    maps = _maps(b=2)
    points = maps_to_points(maps)
    assert points.shape == (2 * 3 * 3, 3)
    assert torch.equal(points_to_maps(points, maps), maps)


def test_wrong_channel_count_raises():
    with pytest.raises(ValueError, match=r"\[B, 3, \*spatial\]"):
        maps_to_points(torch.rand(1, 2, 3, 3))


def test_zero_field_returns_its_noised_input():
    protocol = HeldoutProtocol(t_eval=0.25, t_min=1e-3, step_size=0.05)
    pred, noised, _ = heldout_prediction(_manifold(), _zero_field, _maps(), protocol)
    assert pred.shape == noised.shape == _maps().shape
    assert torch.allclose(pred, noised)


def test_oracle_training_target_moves_toward_the_clean_map():
    """The field regresses ``-log_map(theta_t, theta_0) / t``; integrating that
    oracle must land closer to theta_0 than the noised input, which pins the sign."""
    manifold = _manifold()
    maps = _maps()
    theta0 = maps_to_points(maps)

    def oracle(theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -manifold.log_map(theta, theta0) / t.unsqueeze(-1)

    protocol = HeldoutProtocol(t_eval=0.25, t_min=1e-3, step_size=0.05)
    gen = torch.Generator().manual_seed(1)
    pred, noised, _ = heldout_prediction(manifold, oracle, maps, protocol, gen)
    assert (pred - maps).abs().mean() < (noised - maps).abs().mean()


def test_seeded_generator_reproduces_the_draw():
    protocol = HeldoutProtocol(t_eval=0.25, t_min=1e-3, step_size=0.05)
    a = heldout_prediction(
        _manifold(), _zero_field, _maps(), protocol, torch.Generator().manual_seed(3)
    )
    b = heldout_prediction(
        _manifold(), _zero_field, _maps(), protocol, torch.Generator().manual_seed(3)
    )
    assert torch.equal(a[1], b[1])


def _padded(brain: torch.Tensor, pad: int = 2) -> torch.Tensor:
    """``brain`` centred in a zero (out-of-manifold) background, as in MAP_MASKED maps."""
    return torch.nn.functional.pad(brain, (pad, pad, pad, pad))


def test_zero_background_is_left_out_of_the_pass():
    from spectramr.core.metrics.registry import MetricsRegistry

    manifold = _manifold()
    metric = MetricsRegistry.get("geodesic_qmap_error", manifold=manifold)
    protocol = HeldoutProtocol(t_eval=0.25, t_min=1e-3, step_size=0.05)
    brain = _maps(h=4, w=4)
    padded = _padded(brain)

    def field(theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 0.1 * theta / t.unsqueeze(-1)

    scores = []
    for maps in (brain, padded):
        gen = torch.Generator().manual_seed(5)
        pred, noised, fg = heldout_prediction(manifold, field, maps, protocol, gen)
        scores.append((metric(pred, maps, foreground=fg), metric(noised, maps, foreground=fg)))
    assert scores[1] == pytest.approx(scores[0], rel=1e-5)
    # Background voxels are untouched, so the emitted maps stay renderable.
    _, _, fg = heldout_prediction(manifold, field, padded, protocol)
    assert int(fg.sum()) == 16


def test_map_without_foreground_raises():
    with pytest.raises(ValueError, match="no foreground"):
        foreground_mask(torch.zeros(1, 3, 4, 4))


class TestFromKnobs:
    """The knob read the strategy's constructor makes, on the real schema block."""

    @staticmethod
    def _read(**knobs):
        from spectramr.config.schemas.training.strategy_knobs_2026_08 import (
            RiemannianBlochDiffusionTrainingConfigSchema,
        )

        rbd = RiemannianBlochDiffusionTrainingConfigSchema(**knobs)
        return HeldoutProtocol.from_knobs(
            rbd.heldout_eval_t, t_min=rbd.t_min, t_max=rbd.t_max, step_size=rbd.step_size_init
        )

    def test_unset_knob_reads_none(self):
        assert self._read() is None

    def test_declared_knob_builds_the_protocol(self):
        protocol = self._read(heldout_eval_t=0.25, step_size_init=0.05)
        assert protocol == HeldoutProtocol(t_eval=0.25, t_min=1e-3, step_size=0.05)

    def test_noise_level_above_t_max_raises(self):
        with pytest.raises(ValueError, match="exceeds t_max"):
            self._read(heldout_eval_t=2.0, t_max=1.0)
