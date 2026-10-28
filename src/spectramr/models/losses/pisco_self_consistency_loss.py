r"""PISCO parallel-imaging self-consistency loss (SOTA plan P3.3).

PISCO (Spieker et al., "PISCO: Self-Supervised k-Space Regularization for Neural
Implicit MRI Reconstruction", 2024) regularises multi-coil reconstruction
**without a calibration region**. A genuine multi-coil image (image × smooth
coil sensitivities) satisfies a shift-invariant *linear self-consistency* in
k-space — the SPIRiT relation, where each k-space point is predictable from its
cross-coil neighbourhood. If that relation is real, the linear kernel ``G`` fit
on **disjoint subsets** of the acquired k-space must agree, so

.. math::
    \mathcal{L}_{\text{PISCO}} = \lVert G_A - G_B \rVert_F^2

is a calibration-free self-consistency penalty: ≈0 for parallel-imaging-consistent
data, large for noise / inconsistency. Reuses ``fft_ops`` only via the caller;
operates directly on multi-coil k-space.

Used by the IMJENSE-style scan-specific INR arm (``siren`` reconstruction) as a
k-space regulariser. Needs ≥ 2 coils (the cross-coil relation is the signal).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

__all__ = ["PiscoSelfConsistencyLoss", "pisco_self_consistency"]


def _to_real_interleaved(kspace: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] complex → [B, 2C, H, W] real (real coils then imag coils)."""
    if torch.is_complex(kspace):
        return torch.cat([kspace.real, kspace.imag], dim=1)
    return kspace  # already real-interleaved [B, 2C, H, W]


def _n_coils(kspace: torch.Tensor) -> int:
    return kspace.shape[1] if torch.is_complex(kspace) else kspace.shape[1] // 2


def _spirit_system(ri: torch.Tensor, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build SPIRiT (source-neighbourhood, target-centre) pairs via unfold.

    ``ri``: ``[B, F, H, W]`` real (``F = 2C``). Returns ``source [B, L, F·(k²−1)]``
    (the neighbourhood excluding the centre) and ``target [B, L, F]`` (the centre).
    """
    b, f, _, _ = ri.shape
    k = kernel_size
    patches = F.unfold(ri, kernel_size=k)  # [B, F·k², L]
    patches = patches.transpose(1, 2).reshape(b, -1, f, k * k)  # [B, L, F, k²]
    center = (k * k) // 2
    target = patches[..., center]  # [B, L, F]
    src_idx = [i for i in range(k * k) if i != center]
    source = patches[..., src_idx].reshape(b, -1, f * (k * k - 1))  # [B, L, D]
    return source, target


def _ridge_solve(s: torch.Tensor, t: torch.Tensor, ridge: float) -> torch.Tensor:
    """Per-batch ridge least squares: ``G = (SᵀS + λI)⁻¹ Sᵀ T``, ``[B, D, F]``."""
    sts = s.transpose(1, 2) @ s  # [B, D, D]
    eye = ridge * torch.eye(sts.shape[-1], device=s.device, dtype=s.dtype)
    g: torch.Tensor = torch.linalg.solve(sts + eye, s.transpose(1, 2) @ t)
    return g


def pisco_self_consistency(
    kspace: torch.Tensor,
    kernel_size: int = 3,
    ridge: float = 1e-2,
) -> torch.Tensor:
    r"""Calibrationless PISCO self-consistency ``‖G_A − G_B‖²`` (two disjoint subsets).

    Args:
        kspace: multi-coil k-space, complex ``[B, C, H, W]`` or real-interleaved
            ``[B, 2C, H, W]`` (``C ≥ 2``).
        kernel_size: SPIRiT neighbourhood size (odd).
        ridge: Tikhonov regularisation for the per-subset least-squares fit.

    Returns:
        Scalar self-consistency loss (≥ 0).
    """
    if _n_coils(kspace) < 2:
        raise ValueError(
            f"PISCO needs >= 2 coils (cross-coil self-consistency); got {_n_coils(kspace)}."
        )
    ri = _to_real_interleaved(kspace)
    source, target = _spirit_system(ri, kernel_size)
    n = source.shape[1]
    if n < 4:
        raise ValueError("k-space too small for PISCO (need more sliding windows).")
    # Interleave (even/odd windows) into two disjoint subsets so neither is
    # spatially biased (a contiguous top/bottom split could legitimately differ).
    idx = torch.arange(n, device=source.device)
    a, b = idx[0::2], idx[1::2]
    s_a, t_a = source[:, a], target[:, a]
    s_b, t_b = source[:, b], target[:, b]
    g_a = _ridge_solve(s_a, t_a, ridge)
    g_b = _ridge_solve(s_b, t_b, ridge)
    return ((g_a - g_b) ** 2).mean()


@register_loss(
    name="pisco_self_consistency",
    aliases=["pisco", "PISCO"],
    domain="kspace",
)
class PiscoSelfConsistencyLoss(nn.Module):
    """Registered PISCO calibrationless self-consistency regulariser.

    Forward uses ``prediction`` (the model's multi-coil k-space); ``target`` is
    ignored (PISCO is self-supervised). ``context['kspace']`` overrides the
    tensor PISCO operates on when present.
    """

    def __init__(self, kernel_size: int = 3, ridge: float = 1e-2) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.ridge = float(ridge)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        kspace = prediction
        if context is not None and context.get("kspace") is not None:
            kspace = context["kspace"]
        return pisco_self_consistency(kspace, kernel_size=self.kernel_size, ridge=self.ridge)
