"""Spin-history corruption on the virtual-fiducial twin pass (arm m6)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from spectramr.config.schemas.training.bloch_ode import BlochODEConfig
from spectramr.infrastructure.physics.spin_history import SpinHistoryStage
from spectramr.infrastructure.training.strategies import virtual_fiducial_strategy as vfs

_KW = {
    "dt": 10.0,
    "slice_thickness": 5.0,
    "n_steps": 150,
    "tr": 500.0,
    "t1_ms": 400.0,
    "t2_ms": 100.0,
    "period_ms": 1500.0,
}


def _stage(amplitude: float) -> SpinHistoryStage:
    return SpinHistoryStage(amplitude=amplitude, **_KW)


def test_motion_inside_the_slice_leaves_the_image_unchanged() -> None:
    gain = _stage(amplitude=1.0).gain(torch.rand(2, 1, 32, 32) + 0.5)
    assert torch.allclose(gain, torch.ones_like(gain), atol=1e-4)


def test_motion_beyond_the_slice_modulates_the_image() -> None:
    torch.manual_seed(0)
    gain = _stage(amplitude=8.0).gain(torch.rand(2, 1, 32, 32) + 0.5)
    assert gain.std() > 0.05 and gain.max() > 1.0 and gain.min() < 1.0


def test_a_window_shorter_than_two_tr_is_rejected() -> None:
    with pytest.raises(ValueError, match="2 TR"):
        SpinHistoryStage(amplitude=8.0, **{**_KW, "n_steps": 50})


def test_schema_rejects_the_retired_te_key() -> None:
    with pytest.raises(ValidationError):
        BlochODEConfig(enabled=True, te=20.0)


def test_validation_pass_corrupts_the_twin_input_with_the_gain(monkeypatch) -> None:
    monkeypatch.setattr(vfs.ConcreteVirtualFiducialStrategy, "validation_metrics_computer", None)
    monkeypatch.setattr(
        "spectramr.infrastructure.training.utils.domain_inference.infer_output_domain",
        lambda cfg: "image",
    )
    seen: list[torch.Tensor] = []

    class _Twin:
        marker_mask = torch.ones(1, 1, 16, 16)

        def __call__(self, x, external_b0_field=None):
            seen.append(x)
            return x, torch.zeros_like(x[:1]), x

    s = vfs.ConcreteVirtualFiducialStrategy.__new__(vfs.ConcreteVirtualFiducialStrategy)
    s.simulator, s.device = _Twin(), torch.device("cpu")
    s.config = SimpleNamespace(model=SimpleNamespace(conditioning=SimpleNamespace(sources=[])))
    s._null_space_hook = None
    s._phase_stego_hook = None
    s._spin_history = _stage(amplitude=8.0)
    s._build_conditioning_context = lambda *a, **k: None
    s._forward_model = lambda corrupted_norm, marker_residual, **k: corrupted_norm

    torch.manual_seed(0)
    target = torch.rand(2, 1, 16, 16, dtype=torch.complex64) + 1.0
    scores = s._score_at_current_twin(target, None, cache_visuals=False)

    assert not torch.allclose(seen[0].abs(), target.abs())
    assert scores["val_spin_history_gain_std"] > 0.0
