"""Tests for the cartoon-texture (BV/G) safe-restoration loss (B-2.5)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.losses.cartoon_texture_loss import (
    CartoonTextureSafeLoss,
    _div,
    _grad_fwd,
    rof_cartoon,
)


def test_grad_div_are_true_adjoints() -> None:
    # ROF correctness rests on <grad u, p> = <u, -div p> for ARBITRARY p (the forward gradient and
    # the divergence must be a genuine adjoint pair). _div zeroes the gradient-field boundary so
    # this holds even when p has a non-zero boundary (not only for actual gradient-fields).
    torch.manual_seed(0)
    u = torch.randn(1, 1, 16, 16)
    gx, gy = _grad_fwd(u)
    px, py = torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16)
    lhs = float((gx * px + gy * py).sum())
    rhs = float((u * (-_div(px, py))).sum())
    assert abs(lhs - rhs) < 1e-4


def _scene() -> tuple:
    """Target = cartoon (hard edge) + texture; plus structure-hallucination / texture-swap / blur."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
    cartoon = (xx > 0.5).float() * 0.6 + 0.2
    tex1 = 0.1 * torch.sin(50 * xx) * torch.cos(45 * yy)
    tex2 = 0.1 * torch.sin(60 * xx) * torch.cos(30 * yy)
    target = (cartoon + tex1).clamp(0, 1)[None, None]
    hallu = ((xx > 0.4).float() * 0.6 + 0.2 + tex1).clamp(0, 1)[None, None]  # moved edge = false structure
    texswap = (cartoon + tex2).clamp(0, 1)[None, None]  # same structure, different texture
    blur = F.avg_pool2d(target, 5, stride=1, padding=2)  # texture absence
    return target, hallu, texswap, blur


def test_is_parameter_free() -> None:
    assert sum(p.numel() for p in CartoonTextureSafeLoss().parameters()) == 0


def test_rof_cartoon_removes_high_freq_texture() -> None:
    # The ROF cartoon removes oscillatory texture: its high-frequency energy (image minus a local
    # average) is lower than the input's. (Edge-agnostic — a mean-gradient measure would be
    # dominated by the preserved edge.)
    target, *_ = _scene()
    u = rof_cartoon(target)

    def _hf_energy(x):
        return float((x - F.avg_pool2d(x, 5, stride=1, padding=2)).abs().mean())

    assert _hf_energy(u) < _hf_energy(target)  # texture oscillation removed
    # decomposition is non-degenerate: it actually splits off a texture component.
    assert float((target - u).abs().mean()) > 1e-4


def test_confinement_oracle() -> None:
    # THE load-bearing oracle: at ~equal pixel-L1, the loss penalises a STRUCTURE hallucination
    # MORE than a texture swap (confinement), whereas plain L1 treats them equally.
    target, hallu, texswap, _ = _scene()
    loss = CartoonTextureSafeLoss()
    l1_hallu, l1_swap = float(F.l1_loss(hallu, target)), float(F.l1_loss(texswap, target))
    assert abs(l1_hallu - l1_swap) < 0.01  # L1 cannot distinguish them
    t_hallu = float(loss.compute(hallu, target)["loss_total"])
    t_swap = float(loss.compute(texswap, target)["loss_total"])
    assert t_hallu > 1.3 * t_swap  # structure hallucination costs clearly more (confinement)
    # and the cartoon term carries the structure signal
    assert float(loss.compute(hallu, target)["loss_cartoon"]) > 1.3 * float(
        loss.compute(texswap, target)["loss_cartoon"]
    )


def test_texture_absence_penalised() -> None:
    # An over-smooth (blurred) prediction loses texture energy -> texture term penalises it more
    # than a same-energy texture swap.
    target, _, texswap, blur = _scene()
    loss = CartoonTextureSafeLoss()
    assert float(loss.compute(blur, target)["loss_texture"]) > float(
        loss.compute(texswap, target)["loss_texture"]
    )


def test_cartoon_fidelity_lower_for_hallucinated_structure() -> None:
    target, hallu, texswap, _ = _scene()
    loss = CartoonTextureSafeLoss()
    assert float(loss.cartoon_fidelity(hallu, target)) < float(loss.cartoon_fidelity(texswap, target))


def test_differentiable_into_prediction() -> None:
    target, *_ = _scene()
    pred = (target + 0.05 * torch.randn_like(target)).clamp(0, 1).requires_grad_(True)
    CartoonTextureSafeLoss().compute(pred, target)["loss_total"].backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0


def test_rof_cartoon_gradient_bounded_on_high_contrast() -> None:
    """The differentiable ROF must not explode gradients on a high-contrast image.

    In flat regions ‖∇u‖→0, so the diffusivity ``1/‖∇u‖`` is enormous; the explicit
    TV-flow scheme then violates its CFL limit and the backward pass through 40
    iterations blows the gradient up to ``inf`` (the b25_cartoon_texture NaN loss,
    total_norm≈1.4e5). The stabilised scheme floors ‖∇u‖ so the diffusivity — and
    the backward gradient — stay bounded.
    """
    import math

    base = torch.zeros(1, 1, 32, 32)
    base[..., 8:24, 8:24] = 1.0  # bright square on black: sharp edges + large flats
    pred = base.clone().requires_grad_(True)
    u = rof_cartoon(pred)
    u.pow(2).mean().backward()
    gnorm = float(pred.grad.norm())
    assert math.isfinite(gnorm), f"ROF backward exploded to {gnorm}"
    assert gnorm < 50.0, f"ROF gradient not bounded: {gnorm}"


def test_rof_cartoon_bounded_on_noisy_input() -> None:
    """Same stability on a noisy high-contrast input (pre-fix gnorm ≈ 1e6)."""
    import math

    torch.manual_seed(0)
    base = torch.zeros(1, 1, 32, 32)
    base[..., 8:24, 8:24] = 1.0
    pred = (base + 0.3 * torch.randn_like(base)).clamp(0, 1).requires_grad_(True)
    u = rof_cartoon(pred)
    u.pow(2).mean().backward()
    gnorm = float(pred.grad.norm())
    assert math.isfinite(gnorm) and gnorm < 100.0, f"ROF gradient not bounded: {gnorm}"


def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        CartoonTextureSafeLoss(n_iter=0)
    with pytest.raises(ValueError):
        CartoonTextureSafeLoss(texture_pool=0)


def test_registered_and_creatable() -> None:
    import spectramr.models.losses  # noqa: F401 — fire @register_loss decorators
    from spectramr.models.losses.registry import LossRegistry

    assert "cartoon_texture_safe" in LossRegistry._custom_losses
    assert isinstance(LossRegistry.create("cartoon_texture_safe"), CartoonTextureSafeLoss)
