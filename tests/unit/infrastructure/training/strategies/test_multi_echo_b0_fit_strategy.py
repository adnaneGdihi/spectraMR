"""Tests for ``MultiEchoB0FitStrategy`` (audit 2026-07 I1).

The real B0 estimator: the differentiable analytic fit (map_b0 SSOT) recovers
ΔB0 in Hz from the generator's refined echoes and is graded with B0FieldRMSE.
Unlike ``B0MappingStrategy`` (deformable registration), the physics fit is a
genuine layer in the loop (gradient flows) and the metric measures Hz.

Deterministic synthetic ground truth — no cluster data.
"""
from __future__ import annotations

import math
import types

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.multi_echo_b0_fit_strategy import (
    MultiEchoB0FitStrategy,
)


def _synth_two_echo(b0_true: torch.Tensor, t_shift: float) -> torch.Tensor:
    """Build [B, 4, H, W] real/imag-interleaved two-echo data whose phase
    difference encodes ``b0_true`` (Hz) in the GRE convention:
    echo2 = echo1·exp(+i·2π·ΔTE·ΔB0), on a non-zero static phase."""
    B, _, H, W = b0_true.shape
    mag = torch.rand(B, 1, H, W) + 0.5
    e0 = mag * torch.exp(torch.tensor(0.7j))
    e1 = e0 * torch.exp(1j * 2 * math.pi * t_shift * b0_true)
    return torch.cat([e0.real, e0.imag, e1.real, e1.imag], dim=1)


def _make_strategy(gen, *, t_shift=1e-3, lambda_field=1.0, lambda_echo=0.0):
    from spectramr.core.metrics.b0_field_rmse import B0FieldRMSE

    s = object.__new__(MultiEchoB0FitStrategy)
    s._t_shift = t_shift
    s._gamma = 2e-10
    s._max_cg = 80
    s._lambda_field = lambda_field
    s._lambda_echo = lambda_echo
    s._b0_metric = B0FieldRMSE()
    s.env = types.SimpleNamespace(generator=gen)
    return s


def _ramp_b0(H=32, W=32, amp=40.0, offset=20.0):
    yy, _ = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    return (amp * yy + offset)[None, None]


def test_analytic_fit_recovers_known_b0_in_hz() -> None:
    b0_true = _ramp_b0()
    inp = _synth_two_echo(b0_true, t_shift=1e-3)
    s = _make_strategy(nn.Identity())  # identity refinement -> echoes == input
    echoes = s._as_complex_echoes(s.generator_model(inp))
    b0_hat = s._fit_b0(echoes)
    interior = slice(2, -2)
    rmse = ((b0_hat - b0_true)[..., interior, interior] ** 2).mean().sqrt()
    assert float(rmse) < 3.0  # Hz — recovers the ±40 Hz field


def test_field_loss_gradient_flows_to_generator() -> None:
    b0_true = _ramp_b0()
    inp = _synth_two_echo(b0_true, t_shift=1e-3)
    gen = nn.Conv2d(4, 4, kernel_size=1)
    s = _make_strategy(gen)
    out = s._compute_losses_impl(inp, epoch=0, batch={"b0_map": b0_true})
    assert out["g_total_loss"].requires_grad
    gen.zero_grad()
    out["g_total_loss"].backward()
    gnorm = sum(
        p.grad.abs().sum().item() for p in gen.parameters() if p.grad is not None
    )
    assert gnorm > 0.0  # the physics fit is genuinely in the loop


def test_missing_reference_fails_loud() -> None:
    b0_true = _ramp_b0()
    inp = _synth_two_echo(b0_true, t_shift=1e-3)
    s = _make_strategy(nn.Identity())
    with pytest.raises(ValueError, match="reference B0 field"):
        s._compute_losses_impl(inp, epoch=0, batch={})


def test_single_echo_input_fails_loud() -> None:
    s = _make_strategy(nn.Identity())
    with pytest.raises(ValueError, match=">= 2 echoes"):
        s._as_complex_echoes(torch.randn(1, 2, 8, 8))  # only 1 complex echo


def test_validation_reports_hz_rmse() -> None:
    b0_true = _ramp_b0()
    inp = _synth_two_echo(b0_true, t_shift=1e-3)
    s = _make_strategy(nn.Identity())
    metrics = s.validation_step(inp, batch={"b0_map": b0_true})
    assert "val_b0_field_rmse" in metrics
    assert metrics["val_b0_field_rmse"] < 3.0


def test_registered_in_strategy_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["multi_echo_b0_fit"]
    assert path.endswith("MultiEchoB0FitStrategy")


def test_fit_matches_reference_extractor_convention() -> None:
    """Sign and global offset agree with ``b0_hz_from_dual_echo`` (the b0_map writer)."""
    from scripts.data.extract_oracle_bssfp import b0_hz_from_dual_echo

    dte = 4.56e-3
    b0_true = _ramp_b0()
    inp = _synth_two_echo(b0_true, t_shift=dte)
    s = _make_strategy(nn.Identity(), t_shift=dte)
    s._gamma = 1e-14  # gamma -> 0: the fit reduces to the voxelwise formula
    echoes = s._as_complex_echoes(inp)
    b0_hat = s._fit_b0(echoes)[0, 0]
    ref = b0_hz_from_dual_echo(
        echoes[0, 0].numpy(), echoes[0, 1].numpy(), delta_te_s=dte
    )
    assert torch.allclose(b0_hat, torch.from_numpy(ref), atol=0.05)
    assert torch.allclose(b0_hat, b0_true[0, 0], atol=0.05)
