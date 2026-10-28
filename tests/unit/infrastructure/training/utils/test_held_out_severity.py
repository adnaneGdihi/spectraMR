"""One owner for the held-out severity grid, and its motion-meta consumer."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.training.strategies.motion_meta_strategy import (
    ConcreteMotionMetaTrainingStrategy,
)
from spectramr.infrastructure.training.utils.held_out_severity import held_out_severity_points


def _cfg(*, enabled=True, grid=None, legacy=None):
    mk = {"digital_twin_kwargs": {"held_out_severity_grid": legacy}} if legacy else {}
    gates = SimpleNamespace(held_out_severity_eval=enabled, held_out_severity_grid=grid)
    return SimpleNamespace(
        validation=SimpleNamespace(gates=gates), model=SimpleNamespace(model_kwargs=mk)
    )


def test_gate_off_returns_nothing() -> None:
    assert held_out_severity_points(_cfg(enabled=False, grid=[2.0])) == []


def test_gate_field_wins_over_the_legacy_location() -> None:
    assert held_out_severity_points(_cfg(grid=[1.5], legacy=[64])) == [1.5]


def test_legacy_location_is_read_when_the_gate_field_is_unset() -> None:
    assert held_out_severity_points(_cfg(legacy=[64, 128])) == [64.0, 128.0]


class _Kin:
    def __init__(self) -> None:
        self.scales: list[float] = []

    def generate_random_motion(
        self, batch_size, max_translation, max_rotation, motion_type, device
    ):
        self.scales.append(max_translation)
        return torch.zeros(batch_size, 3)

    def __call__(self, x, theta):
        return x


def _bare_strategy(grid, monkeypatch):
    s = ConcreteMotionMetaTrainingStrategy.__new__(ConcreteMotionMetaTrainingStrategy)
    s.config = _cfg(grid=grid)
    s.device = torch.device("cpu")
    s.kinematic_op = _Kin()
    s.fiducial = lambda batch_size: torch.zeros(batch_size, 1, 8, 8, dtype=torch.complex64)
    monkeypatch.setattr(
        ConcreteMotionMetaTrainingStrategy,
        "generator_model",
        property(lambda self: lambda x, corrupted_fiducial=None: x),
    )
    s._max_translation, s._max_rotation, s._motion_type = 2.0, 0.1, "smooth"
    s._to_device = lambda t: t
    s._ensure_image_domain_target = lambda t: t
    s.snapshot_source = lambda *_: contextlib.nullcontext()
    s._capture_model_output = lambda **_: contextlib.nullcontext()
    s._compute_losses_impl = lambda **_: {}
    return s


def test_validation_scores_at_each_held_out_magnitude(monkeypatch) -> None:
    monkeypatch.setattr(ConcreteMotionMetaTrainingStrategy, "validation_metrics_computer", None)
    s = _bare_strategy([1.5, 2.0], monkeypatch)
    target = torch.rand(2, 2, 8, 8)
    out = s.validation_step(target, target)

    assert {"val_psnr", "val_psnr_heldout_x1.5", "val_psnr_heldout_x2"} <= out.keys()
    assert s.kinematic_op.scales == [2.0, 3.0, 4.0]  # in-distribution, then 1.5x, 2x


def test_a_grid_inside_the_training_range_is_rejected(monkeypatch) -> None:
    with pytest.raises(ValueError, match="must exceed 1"):
        _bare_strategy([0.5], monkeypatch)._heldout_motion_scales()
