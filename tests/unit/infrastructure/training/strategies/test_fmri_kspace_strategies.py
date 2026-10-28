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

from spectramr.infrastructure.physics.manifolds.beltrami_4d import (
    SeparableBeltrami4DSolver,
)
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


def _make_strategy(gen: nn.Module) -> SpatiotemporalAdaptiveSFCReconStrategy:
    s = object.__new__(SpatiotemporalAdaptiveSFCReconStrategy)
    s.lambda_mu = 1.0
    s.lambda_T = 1.0
    s.solver = SeparableBeltrami4DSolver()
    s.env = types.SimpleNamespace(generator=gen)
    return s


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
