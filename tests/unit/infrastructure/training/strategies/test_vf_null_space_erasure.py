"""Null-space marker erasure on the virtual-fiducial validation pass (arm m2)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from spectramr.config.schemas.training.null_space_erasure import NullSpaceErasureConfig
from spectramr.infrastructure.physics.null_space_erasure import NullSpaceMarkerEraser
from spectramr.infrastructure.training.strategies import virtual_fiducial_strategy as vfs
from spectramr.infrastructure.training.strategies.vf_null_space import NullSpaceErasureHook

_H = 16


def _marker() -> torch.Tensor:
    m = torch.zeros(1, 1, _H, _H, dtype=torch.complex64)
    m[0, 0, 4:7, 4:7] = 1.0 + 0.5j
    return m


def test_complex_projection_is_idempotent_and_removes_the_marker() -> None:
    eraser = NullSpaceMarkerEraser(_marker(), epsilon=1e-9)
    y = torch.randn(3, 1, _H, _H, dtype=torch.complex64) + 2.0 * _marker()
    once = eraser(y)
    assert torch.allclose(eraser(once), once, atol=1e-4)
    inner = (_marker().reshape(1, -1).conj() * once.reshape(3, -1)).sum(dim=1)
    assert inner.abs().max() < 1e-3


def test_schema_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        NullSpaceErasureConfig(enabled=True, epsilon=1e-6, apply_at_inferenc=True)


def test_hook_scores_report_zero_residual_and_full_preservation() -> None:
    hook = NullSpaceErasureHook(epsilon=1e-9)
    target = torch.randn(2, 1, _H, _H, dtype=torch.complex64)
    erased = hook.erase(target + 3.0 * _marker(), _marker())
    scores = hook.scores(erased, target, _marker())
    assert scores["val_marker_residual_l2"] < 1e-3
    assert scores["val_null_space_preservation_l2"] < 1e-3


def _bare_strategy(hook: NullSpaceErasureHook | None):
    s = vfs.ConcreteVirtualFiducialStrategy.__new__(vfs.ConcreteVirtualFiducialStrategy)
    marker = _marker()

    class _Twin:
        marker_mask = (marker.abs() > 0).float()

        def __call__(self, target, external_b0_field=None):
            return target + 3.0 * marker, marker, target

    s.simulator = _Twin()
    s.device = torch.device("cpu")
    s.config = SimpleNamespace(model=SimpleNamespace(conditioning=SimpleNamespace(sources=[])))
    s._null_space_hook = hook
    s._build_conditioning_context = lambda *a, **k: None
    s._forward_model = lambda corrupted_norm, marker_residual, **k: corrupted_norm
    return s


def test_validation_pass_erases_the_marker_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(vfs.ConcreteVirtualFiducialStrategy, "validation_metrics_computer", None)
    monkeypatch.setattr(
        "spectramr.infrastructure.training.utils.domain_inference.infer_output_domain",
        lambda cfg: "image",
    )
    target = torch.rand(2, 1, _H, _H, dtype=torch.complex64) + 1.0
    on = _bare_strategy(NullSpaceErasureHook(1e-9))._score_at_current_twin(
        target, None, cache_visuals=False
    )
    off = _bare_strategy(None)._score_at_current_twin(target, None, cache_visuals=False)

    assert "val_marker_residual_l2" in on and "val_marker_residual_l2" not in off
    assert on["val_marker_residual_l2"] < 1e-2
    assert on["val_psnr"] > off["val_psnr"]
    assert on["val_psnr_unerased"] == pytest.approx(off["val_psnr"], abs=1e-3)
