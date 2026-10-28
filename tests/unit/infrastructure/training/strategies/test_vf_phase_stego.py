"""Phase-steganographic marker on the virtual-fiducial twin pass (arm m7)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from spectramr.config.schemas.training.phase_steganography import PhaseSteganographyConfig
from spectramr.infrastructure.training.strategies import virtual_fiducial_strategy as vfs
from spectramr.infrastructure.training.strategies.vf_phase_stego import PhaseStegoHook

_H = 16


def _cfg(**kw) -> PhaseSteganographyConfig:
    return PhaseSteganographyConfig(enabled=True, **kw)


def _anatomy(b: int = 2) -> torch.Tensor:
    return torch.rand(b, 1, _H, _H, dtype=torch.complex64) + 1.0


def test_injection_changes_phase_but_not_magnitude() -> None:
    hook = PhaseStegoHook(_cfg(), torch.device("cpu"))
    clean = _anatomy()
    marked = hook.inject(clean)
    assert torch.allclose(marked.abs(), clean.abs(), atol=1e-5)
    assert (marked.angle() - clean.angle()).abs().max() > 0.1


def test_scores_report_an_invisible_marker() -> None:
    hook = PhaseStegoHook(_cfg(), torch.device("cpu"))
    assert hook.scores(_anatomy())["val_marker_residual_after_magnitude"] < 1e-5


def test_learnable_is_rejected_and_unknown_key_forbidden() -> None:
    with pytest.raises(ValueError, match="learnable"):
        PhaseStegoHook(_cfg(learnable=True), torch.device("cpu"))
    with pytest.raises(ValidationError):
        PhaseSteganographyConfig(enabled=True, not_a_knob=0.5)


def test_validation_pass_feeds_the_twin_the_marked_anatomy(monkeypatch) -> None:
    monkeypatch.setattr(vfs.ConcreteVirtualFiducialStrategy, "validation_metrics_computer", None)
    monkeypatch.setattr(
        "spectramr.infrastructure.training.utils.domain_inference.infer_output_domain",
        lambda cfg: "image",
    )
    seen: list[torch.Tensor] = []

    class _Twin:
        marker_mask = torch.ones(1, 1, _H, _H)

        def __call__(self, x, external_b0_field=None):
            seen.append(x)
            return x, torch.zeros_like(x[:1]), x

    s = vfs.ConcreteVirtualFiducialStrategy.__new__(vfs.ConcreteVirtualFiducialStrategy)
    s.simulator, s.device = _Twin(), torch.device("cpu")
    s.config = SimpleNamespace(model=SimpleNamespace(conditioning=SimpleNamespace(sources=[])))
    s._null_space_hook = None
    s._phase_stego_hook = PhaseStegoHook(_cfg(), torch.device("cpu"))
    s._build_conditioning_context = lambda *a, **k: None
    s._forward_model = lambda corrupted_norm, marker_residual, **k: corrupted_norm

    target = _anatomy()
    scores = s._score_at_current_twin(target, None, cache_visuals=False)

    assert (seen[0].angle() - target.angle()).abs().max() > 0.1
    assert "val_marker_residual_after_magnitude" in scores


def test_both_twin_call_sites_go_through_twin_input() -> None:
    for fn in (
        vfs.ConcreteVirtualFiducialStrategy._compute_losses_impl,
        vfs.ConcreteVirtualFiducialStrategy._score_at_current_twin,
    ):
        assert "self._twin_input(" in inspect.getsource(fn)


def test_displacement_recovery_matches_the_twins_elastic_field() -> None:
    """Against the REAL twin (elastic motion, no B0/B1/noise-floor stages) the phase
    grid recovers the applied displacement along its gradient to well under a pixel."""
    from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator

    torch.manual_seed(0)
    size = 64
    twin = DigitalTwinSimulator(
        im_size=(size, size),
        enable_motion=True,
        motion_type="elastic",
        motion_amplitude=1.5,
        enable_b0=False,
        enable_b1=False,
        snr_range=(1e4, 1e4),
    )
    hook = PhaseStegoHook(
        _cfg(amplitude=1.0, spatial_freq_x=2.0, spatial_freq_y=2.0), torch.device("cpu")
    )
    clean = torch.ones(2, 1, size, size, dtype=torch.complex64)
    marked = hook.inject(clean)
    corrupted, _, joint_clean = twin(marked)
    assert twin.last_elastic_displacement_px is not None
    disp = twin.last_elastic_displacement_px
    key = "val_stego_displacement_rmse_px"
    real = hook.displacement_rmse_px(corrupted, joint_clean, disp)[key]
    # Controls: the wrong sign and another sample's field must score clearly worse, else the
    # metric would pass on any displacement of the right size.
    flipped = hook.displacement_rmse_px(corrupted, joint_clean, -disp)[key]
    swapped = hook.displacement_rmse_px(corrupted, joint_clean, disp.flip(0))[key]
    assert real < 0.6 * flipped and real < 0.6 * swapped
