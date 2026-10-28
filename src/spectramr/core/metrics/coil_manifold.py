r"""Per-pixel coil-manifold geometry: the projector, the null-space energy, their means.

A physical multi-coil image is rank one per pixel, :math:`x(r)=s(r)m(r)`. With the
unit map :math:`\hat s=s/\lVert s\rVert` and :math:`P=\hat s\hat s^{H}`,
:math:`P^{\perp}=I-P`, and any reconstruction :math:`f`,

.. math::

    \lVert Pf-x\rVert^2=\lVert f-x\rVert^2-\lVert P^{\perp}f\rVert^2
    \quad\text{per pixel, exactly,}

because :math:`P^{\perp}x=0` and :math:`P, P^{\perp}` are orthogonal. So projecting
never increases error, and :math:`\lVert P^{\perp}f\rVert^2` is a reference-free
lower bound on it. The identity needs :math:`P` to be a true projector at every
pixel, so where the map vanishes (outside ESPIRiT's support) :math:`\hat s` is
**zero** -- :math:`P=0`, which is exact for the zero anatomy there -- rather than
the clamped non-unit vector a division by ``eps`` would leave.

Everything here stays on the device: the reductions return 0-dim tensors and a
degenerate weight total yields ``nan`` through ``torch.where`` instead of a host
check (non-negotiable 9). The coil axis is dim 1 throughout, real/imag
interleaved as ``[R0, I0, R1, I1, ...]`` -- the layout M4Raw serves.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

__all__ = [
    "as_complex_coils",
    "as_interleaved",
    "coil_images",
    "null_space_energy",
    "project_kspace_onto_manifold",
    "project_onto_manifold",
    "require_complex_maps",
    "require_kspace_domain",
    "unit_maps",
    "weighted_mean",
]


def as_complex_coils(x: Tensor, *, name: str = "tensor") -> Tensor:
    """``[B, 2C, H, W]`` interleaved or ``[B, C, H, W]`` complex -> complex coils."""
    if torch.is_complex(x):
        return x
    if x.ndim != 4 or x.shape[1] % 2 != 0:
        raise ValueError(
            f"{name} must be complex [B, C, H, W] or real/imag-interleaved [B, 2C, H, W], "
            f"got {tuple(x.shape)}. An odd channel count means the coils were combined, "
            "and a combined image has no coil vector left to measure."
        )
    return torch.complex(x[:, 0::2], x[:, 1::2])


def as_interleaved(z: Tensor) -> Tensor:
    """Complex ``[B, C, H, W]`` -> real ``[B, 2C, H, W]`` in ``[R0, I0, ...]`` order."""
    out = torch.empty(z.shape[0], 2 * z.shape[1], *z.shape[2:], dtype=z.real.dtype, device=z.device)
    out[:, 0::2] = z.real
    out[:, 1::2] = z.imag
    return out


def coil_images(kspace: Tensor, *, name: str = "prediction") -> Tensor:
    """Complex coil images from k-space, through the physics SSOT."""
    return ifft2c(as_complex_coils(kspace, name=name))


def require_kspace_domain(domain: str | None, *, metric: str) -> None:
    """Raise unless the caller's tensors are k-space (``None`` = the documented default)."""
    if domain not in (None, "kspace"):
        raise ValueError(
            f"{metric} reads multi-coil k-space and takes coil images through ifft2c; "
            f"the caller says its tensors are {domain!r}. Measuring an image as k-space "
            "would return a finite number describing nothing."
        )


def require_complex_maps(maps: Tensor | None, like: Tensor, *, metric: str) -> Tensor:
    """The maps, or a ``ValueError`` naming what is wrong with them."""
    if maps is None:
        raise ValueError(
            f"{metric} needs coil sensitivity maps (MetricContext.coil_maps) and none "
            "reached it. Declare the `espirit_sensitivity` transform, which emits the "
            "complex maps under `sensitivity_complex`."
        )
    if not torch.is_complex(maps):
        raise ValueError(
            f"{metric} needs COMPLEX coil maps, got {maps.dtype}. A magnitude map makes "
            "I - s s^H a real projector, which drops the phase half of the null space."
        )
    if maps.shape != like.shape:
        raise ValueError(
            f"{metric}: coil maps {tuple(maps.shape)} and coil images {tuple(like.shape)} "
            "disagree; the projector is per pixel and per coil."
        )
    return maps


def unit_maps(maps: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """``(s_hat, ||s||^2)``: unit maps, exactly zero where the map vanishes."""
    norm2 = (maps.real.square() + maps.imag.square()).sum(1, keepdim=True)
    # Clamp before the root so the discarded branch of `where` stays finite.
    unit = maps / norm2.clamp(min=eps * eps).sqrt()
    unit = torch.where(norm2 > eps * eps, unit, torch.zeros_like(unit))
    return unit, norm2.squeeze(1)


def null_space_energy(images: Tensor, maps: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Per-pixel ``(||P_perp f||^2, ||s||^2)``, both ``[B, H, W]``.

    Uses ``||P_perp f||^2 = ||f||^2 - |s_hat^H f|^2`` so the ``C x C`` projector is
    never formed.
    """
    unit, norm2 = unit_maps(maps, eps)
    along = (unit.conj() * images).sum(1)
    total = (images.real.square() + images.imag.square()).sum(1)
    residual = (total - along.real.square() - along.imag.square()).clamp(min=0.0)
    return residual, norm2


def project_onto_manifold(images: Tensor, maps: Tensor, eps: float = 1e-6) -> Tensor:
    """``P f = s_hat (s_hat^H f)`` per pixel, on complex coil images."""
    unit, _ = unit_maps(maps, eps)
    return unit * (unit.conj() * images).sum(1, keepdim=True)


def project_kspace_onto_manifold(kspace: Tensor, maps: Tensor, eps: float = 1e-6) -> Tensor:
    """``F P F^H`` on multi-coil k-space, returned in the layout it arrived in."""
    images = coil_images(kspace)
    require_complex_maps(maps, images, metric="coil_manifold_projection")
    projected = fft2c(project_onto_manifold(images, maps, eps))
    return projected if torch.is_complex(kspace) else as_interleaved(projected)


def weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    """``sum(w v) / sum(w)`` as a 0-dim tensor; ``nan`` when every weight is zero.

    ``nan`` rather than a raise, because deciding to raise would need the total on
    the host. A zero total means no pixel carries a rank-one constraint, and a
    ``nan`` column says so without reading as a perfect score.
    """
    total = weights.sum()
    mean = (weights * values).sum() / total.clamp(min=torch.finfo(total.dtype).tiny)
    return torch.where(total > 0, mean, torch.full_like(mean, float("nan")))
