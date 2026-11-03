"""Regression test for the fMRI Spatiotemporal-SFC Beltrami zero-gradient fix.

``SpatiotemporalAdaptiveSFCReconStrategy`` derived its SFC-Beltrami
coefficients ``mu``/``nu`` from the no-grad ``input_batch[:, 0]``, so
``beltrami_reg``/``temporal_reg`` were constants w.r.t. model parameters
(zero gradient) and ``lambda_mu``/``lambda_T`` were inert — the loss silently
collapsed to plain L1. The MRF sibling had already been fixed to derive from
the model output ``recon``; this pins the same fix for the fMRI strategy.
"""
from __future__ import annotations

import inspect
import types

import torch
import torch.nn as nn

import pytest

from spectramr.infrastructure.training.strategies import fmri_kspace_strategies as mod
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.fmri_kspace_strategies import (
    SpatiotemporalAdaptiveSFCReconStrategy,
)


class _Gen(nn.Module):
    """[B, C, T, H, W] -> [B, C, T, H, W] per-frame conv."""

    def __init__(self, c: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = self.conv(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def _make_strategy(
    gen: nn.Module, lambda_l1: float = 1.0
) -> SpatiotemporalAdaptiveSFCReconStrategy:
    s = object.__new__(SpatiotemporalAdaptiveSFCReconStrategy)
    s.lambda_mu = 1.0
    s.lambda_T = 1.0
    s.lambda_l1 = lambda_l1
    s.env = types.SimpleNamespace(generator=gen)
    return s


class _Ident(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_temporal_reg_is_zero_when_frames_share_their_mean() -> None:
    """#2363: the first-difference form was 1 - (1 + nu_0) / sum(1 + nu_t), i.e.
    1 - 1/T for a temporally constant series; smoothness must give zero."""
    frame = torch.randn(1, 1, 1, 8, 8)
    series = frame.expand(1, 1, 6, 8, 8).contiguous()
    out = _make_strategy(_Ident())._compute_losses_impl(series, series, epoch=0)
    assert float(out["spatiotemporal_sfc_temporal_reg"]) < 1e-5  # old form: 1 - 1/6


def test_temporal_reg_no_longer_biases_the_first_frame() -> None:
    torch.manual_seed(0)
    recon = torch.randn(2, 1, 8, 16, 16, requires_grad=True)
    s = _make_strategy(_Ident())
    s.lambda_mu = 0.0
    out = s._compute_losses_impl(recon, recon.detach(), epoch=0)
    out["g_total_loss"].backward()
    signs = torch.sign(recon.grad[0, 0].mean(dim=(-2, -1)))
    # The old term pushed frame 0 one way and every other frame the other way.
    assert not (signs[0] < 0 and bool((signs[1:] > 0).all()))


def test_inline_l1_takes_the_declared_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    """#2331: the arms declare l1 at 10.0; the strategy computed it at 1.0."""

    def _init(self, env=None, **kwargs):
        self.config = types.SimpleNamespace(training=types.SimpleNamespace())

    monkeypatch.setattr(BaseTrainingStrategy, "__init__", _init)
    monkeypatch.setattr(mod, "declared_loss_weights", lambda config: {"l1": 10.0})
    s = SpatiotemporalAdaptiveSFCReconStrategy(env=None)
    assert s.lambda_l1 == 10.0
    s.env = types.SimpleNamespace(generator=_Ident())
    s.lambda_mu = s.lambda_T = 0.0
    x, y = torch.zeros(1, 1, 4, 8, 8), torch.ones(1, 1, 4, 8, 8)
    out = s._compute_losses_impl(x, y, epoch=0)
    assert float(out["g_total_loss"]) == pytest.approx(10.0)
    assert float(out["spatiotemporal_sfc_recon"]) == pytest.approx(1.0)

    monkeypatch.setattr(mod, "declared_loss_weights", lambda config: {})
    with pytest.raises(ValueError, match=r"losses\.image_losses"):
        SpatiotemporalAdaptiveSFCReconStrategy(env=None)


def test_beltrami_regulariser_carries_gradient_to_model() -> None:
    gen = _Gen()
    inp = torch.randn(1, 2, 3, 8, 8)
    # target == recon so the L1 fidelity term contributes ~0 gradient; any
    # gradient that survives must come from the SFC-Beltrami regulariser.
    with torch.no_grad():
        target = gen(inp).clone()

    s = _make_strategy(gen)
    out = s._compute_losses_impl(inp, target, epoch=0)
    assert out["g_total_loss"].requires_grad

    gen.zero_grad()
    out["g_total_loss"].backward()
    gnorm = sum(
        p.grad.abs().sum().item() for p in gen.parameters() if p.grad is not None
    )
    # With the old input-derived mu/nu this would be ~0 (dead regulariser).
    assert gnorm > 0.0


def test_beltrami_and_temporal_terms_are_reported_nonzero() -> None:
    gen = _Gen()
    inp = torch.randn(1, 2, 3, 8, 8)
    target = torch.randn(1, 2, 3, 8, 8)
    s = _make_strategy(gen)
    out = s._compute_losses_impl(inp, target, epoch=0)
    assert out["spatiotemporal_sfc_beltrami_reg"].item() > 0.0
    assert out["spatiotemporal_sfc_temporal_reg"].item() >= 0.0


def test_source_derives_from_recon_not_input() -> None:
    src = inspect.getsource(
        SpatiotemporalAdaptiveSFCReconStrategy._compute_losses_impl
    )
    assert "x = recon[:, 0]" in src
    assert "x = input_batch[:, 0]" not in src


# --- idea_2: what BeltramiEPIDistortionStrategy validates and owns (#1768) ----


def _epi_strategy(field_hz: float):
    from spectramr.infrastructure.training.strategies.fmri_kspace_strategies import (
        BeltramiEPIDistortionStrategy,
    )

    class _Field(nn.Module):
        def forward(self, x):
            return torch.full_like(x, field_hz)

    s = object.__new__(BeltramiEPIDistortionStrategy)
    s.env = types.SimpleNamespace(generator=_Field())
    s.device = torch.device("cpu")
    s.t_esp = 0.5e-3
    s.config = types.SimpleNamespace(
        validation=types.SimpleNamespace(
            scoring=types.SimpleNamespace(compute=["psnr"], primary="psnr", transform=None)
        ),
        metrics=None,
    )
    return s


def test_validation_grades_the_epi_the_field_predicts_not_the_field_itself() -> None:
    """The generator emits Hz; ``val_psnr`` used to compare that field with the
    undistorted image. It now warps the target by the predicted field and grades
    that against the observed EPI, so the correct field scores best."""
    from spectramr.infrastructure.physics.epi_forward import apply_epi_distortion

    torch.manual_seed(0)
    yy = torch.linspace(0, 1, 32).view(1, 1, 32, 1).expand(2, 1, 32, 32)
    target = 0.5 + 0.4 * torch.sin(6.0 * yy)
    observed = apply_epi_distortion(target, torch.full_like(target, 40.0), t_esp=0.5e-3)

    right = _epi_strategy(40.0).validation_step(observed, target)
    wrong = _epi_strategy(-40.0).validation_step(observed, target)
    assert float(right["val_beltrami_epi_residual"]) < 1e-5
    assert float(right["val_psnr"]) > float(wrong["val_psnr"])


def test_both_fmri_strategies_declare_an_inline_objective() -> None:
    from spectramr.infrastructure.training.strategies.fmri_kspace_strategies import (
        BeltramiEPIDistortionStrategy,
    )

    assert SpatiotemporalAdaptiveSFCReconStrategy.inline_losses == frozenset({"l1"})
    assert SpatiotemporalAdaptiveSFCReconStrategy.folds_image_losses is False
    assert BeltramiEPIDistortionStrategy.inline_losses == frozenset({"beltrami_epi_residual"})
    assert BeltramiEPIDistortionStrategy.folds_image_losses is False
