"""Cartoon-texture (BV/G) safe-restoration loss (MICCAI MRIxFields2026, B-2.5).

Meyer's cartoon-texture model splits an image into a piecewise-smooth CARTOON ``u`` (bounded
variation) and an oscillatory TEXTURE ``v`` (the G-norm dual of BV), ``f = u + v``. This loss
restores them ASYMMETRICALLY to CONFINE hallucination (the safety property restoration needs):

* the CARTOON (structure) is matched HARD with a pixel-exact L1 — the model cannot hallucinate
  false anatomy/edges, because any structural deviation moves ``u``;
* the TEXTURE (oscillation) is matched only LOOSELY in pooled energy — the model may synthesise
  plausible fine texture but is never forced to invent exact high-frequency pixels.

The cartoon is extracted by a few differentiable ROF (Rudin-Osher-Fatemi) total-variation
gradient-descent steps (parameter-free); the texture is the residual. Distinct from the plain
``tv`` penalty (a regulariser, no decomposition) and from B-1.3 scattering-Besov (which DRIVES
detail; this CONFINES hallucination — the opposite intent).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.losses.registry import register_loss
from spectramr.models.losses.validation import validate_loss_inputs


def _grad_fwd(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = F.pad(u[:, :, :, 1:] - u[:, :, :, :-1], (0, 1))
    gy = F.pad(u[:, :, 1:, :] - u[:, :, :-1, :], (0, 0, 0, 1))
    return gx, gy


def _div(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Backward-difference divergence — the TRUE adjoint of :func:`_grad_fwd` (Neumann).

    The forward gradient lives in the subspace ``{p : p[..., -1]=0}`` (its last column/row are
    the padding zeros). Zeroing that boundary here makes ``<grad u, p> = <u, -div p>`` hold for
    ARBITRARY ``p`` (verified to ~1e-6), not only gradient-fields. Inside the ROF loop ``p`` is
    already a gradient-field, so this is a no-op on the decomposition — it only guarantees the
    operator is a genuine adjoint pair.
    """
    px = F.pad(px[:, :, :, :-1], (0, 1))  # force px[..., -1] = 0 (the gradient-field boundary)
    py = F.pad(py[:, :, :-1, :], (0, 0, 0, 1))  # force py[..., -1, :] = 0
    dx = px - F.pad(px[:, :, :, :-1], (1, 0))
    dy = py - F.pad(py[:, :, :-1, :], (0, 0, 1, 0))
    return dx + dy


def rof_cartoon(
    f: torch.Tensor,
    tv_weight: float = 0.15,
    n_iter: int = 40,
    step: float = 0.1,
    grad_eps: float = 1e-3,
) -> torch.Tensor:
    """Differentiable ROF total-variation denoise -> the cartoon (BV) component of ``f``.

    Stabilised (2026-07-05). The TV-flow diffusivity ``1/‖∇u‖`` is unbounded as
    ``‖∇u‖→0`` (flat regions), where the old ``sqrt(gx²+gy²+1e-8)`` gave
    ``‖∇u‖≈1e-4`` and hence a diffusivity ``~1e4``. That violates the explicit
    scheme's CFL limit, so the 40-step *backward* pass blew the gradient to
    ``inf`` / ``~1e6`` and drove ``g_total_loss`` to NaN (the b25_cartoon_texture
    divergence). We floor ``‖∇u‖`` at the CFL-safe value ``8·step·tv_weight`` so
    the diffusivity — and the backward gradient through it — stay bounded, while
    gradients ABOVE the floor (real edges) keep the exact anisotropic TV
    direction. The floor scales with ``step·tv_weight`` so the CFL margin is
    invariant to those knobs.
    """
    norm_floor = max(grad_eps, 8.0 * step * tv_weight)
    u = f
    for _ in range(n_iter):
        gx, gy = _grad_fwd(u)
        norm = torch.sqrt(gx**2 + gy**2 + grad_eps**2).clamp_min(norm_floor)
        div = _div(gx / norm, gy / norm)
        u = u - step * ((u - f) - tv_weight * div)
    return u


@register_loss(name="cartoon_texture_safe", domain="image")
class CartoonTextureSafeLoss(nn.Module):
    """BV/G cartoon-texture decomposition with hallucination confinement (B-2.5)."""

    def __init__(
        self,
        tv_weight: float = 0.15,
        n_iter: int = 40,
        cartoon_weight: float = 1.0,
        texture_weight: float = 0.3,
        texture_pool: int = 4,
    ) -> None:
        super().__init__()
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1; got {n_iter}.")
        if texture_pool < 1:
            raise ValueError(f"texture_pool must be >= 1; got {texture_pool}.")
        self.tv_weight = float(tv_weight)
        self.n_iter = int(n_iter)
        self.cartoon_weight = float(cartoon_weight)
        self.texture_weight = float(texture_weight)
        self.texture_pool = int(texture_pool)

    def decompose(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cartoon u, texture v=img-u)."""
        if img.shape[1] != 1:
            img = img.mean(dim=1, keepdim=True)
        u = rof_cartoon(img, self.tv_weight, self.n_iter)
        return u, img - u

    @validate_loss_inputs
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(pred, target)["loss_total"]

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        u_p, v_p = self.decompose(pred)
        u_t, v_t = self.decompose(target)
        # Cartoon (structure): HARD pixel-exact match -> no hallucinated structure.
        loss_cartoon = F.l1_loss(u_p, u_t)
        # Texture (oscillation): LOOSE pooled-energy match -> plausible texture, not pixel-forced.
        e_p = F.adaptive_avg_pool2d(v_p.abs(), self.texture_pool)
        e_t = F.adaptive_avg_pool2d(v_t.abs(), self.texture_pool)
        loss_texture = F.l1_loss(e_p, e_t)
        total = self.cartoon_weight * loss_cartoon + self.texture_weight * loss_texture
        return {"loss_total": total, "loss_cartoon": loss_cartoon, "loss_texture": loss_texture}

    def cartoon_fidelity(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Structure-safety witness: relative cartoon agreement in [~0,1] (1 = faithful structure)."""
        u_p, _ = self.decompose(pred)
        u_t, _ = self.decompose(target)
        err = (u_p - u_t).abs().mean()
        scale = u_t.abs().mean().clamp_min(1e-8)
        return (1.0 - (err / scale)).clamp(0.0, 1.0).detach()
