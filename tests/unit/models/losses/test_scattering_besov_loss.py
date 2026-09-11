"""Tests for the scattering-Besov detail prior (B-1.3)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.losses.scattering_besov_loss import ScatteringBesovLoss


def _textured_target(n: int = 64) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij")
    return (0.5 + 0.3 * torch.sin(40 * xx) * torch.cos(35 * yy)).clamp(0, 1)[None, None]


def test_is_parameter_free() -> None:
    # Fixed filterbank -> no learnable params -> the lambda_scatter ablation is exactly param-matched.
    assert sum(p.numel() for p in ScatteringBesovLoss().parameters()) == 0


def test_penalizes_blur_more_than_sharp() -> None:
    # THE load-bearing oracle: the detail prior must penalise an over-smooth (blurred) prediction
    # MORE than a slightly-noisy sharp one, even when L1 prefers the blur (the failure mode L1 has).
    torch.manual_seed(0)
    target = _textured_target()
    blur = F.avg_pool2d(target, 5, stride=1, padding=2)
    sharp = (target + 0.02 * torch.randn_like(target)).clamp(0, 1)
    loss = ScatteringBesovLoss()
    ob = loss.compute(blur, target)
    os = loss.compute(sharp, target)
    assert float(ob["loss_total"]) > float(os["loss_total"])
    assert float(ob["loss_scatter"]) > float(os["loss_scatter"])
    assert float(ob["loss_besov"]) > float(os["loss_besov"])


def test_detail_energy_drops_for_blur() -> None:
    target = _textured_target()
    blur = F.avg_pool2d(target, 5, stride=1, padding=2)
    loss = ScatteringBesovLoss()
    assert float(loss.detail_energy(blur)) < float(loss.detail_energy(target))


def test_differentiable_into_prediction() -> None:
    torch.manual_seed(1)
    target = torch.rand(1, 1, 48, 48)
    pred = (target + 0.1 * torch.randn_like(target)).clamp(0, 1).requires_grad_(True)
    ScatteringBesovLoss().compute(pred, target)["loss_total"].backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0


def test_besov_s_upweights_finer_scales() -> None:
    # A larger Besov exponent s puts more weight on finer scales -> larger besov loss on a blur.
    torch.manual_seed(0)
    target = _textured_target()
    blur = F.avg_pool2d(target, 5, stride=1, padding=2)
    low = ScatteringBesovLoss(besov_s=0.5).compute(blur, target)["loss_besov"]
    high = ScatteringBesovLoss(besov_s=2.0).compute(blur, target)["loss_besov"]
    assert float(high) > float(low)


def test_besov_weight_direction_upweights_finest_band() -> None:
    # Regression (B-1.3 sign bug): the filterbank uses freq = 0.25 / 2**j, so the loop scale
    # index j INCREASES as the band gets COARSER (j=0 is the FINEST / highest-frequency band).
    # The Besov 2^(level*s) weighting must therefore land its LARGEST weight on the FINEST band
    # to "drive high-frequency detail" as documented. The pre-fix code used w = 2**(j*s), which
    # up-weighted the COARSEST band (the exact opposite of the SR detail prior's purpose).
    loss = ScatteringBesovLoss(n_scales=3, besov_s=1.0)
    w = loss.besov_scale_weights  # [n_scales], indexed by loop scale index j
    assert w.shape[0] == 3
    assert float(w[0]) > float(w[-1]), (
        "finest band (j=0, freq=0.25) must carry MORE Besov weight than the coarsest band"
    )
    # strictly increasing toward finer/higher-frequency bands (decreasing in j)
    assert bool(torch.all(w[:-1] > w[1:]))


def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        ScatteringBesovLoss(n_scales=0)
    with pytest.raises(ValueError):
        ScatteringBesovLoss(n_orientations=0)


def test_registered_and_creatable() -> None:
    import spectramr.models.losses  # noqa: F401 — fire @register_loss decorators
    from spectramr.models.losses.registry import LossRegistry

    assert "scattering_besov" in LossRegistry._custom_losses
    inst = LossRegistry.create("scattering_besov")
    assert isinstance(inst, ScatteringBesovLoss)
