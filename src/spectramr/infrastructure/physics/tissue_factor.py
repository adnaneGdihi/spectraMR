"""Tissue-dependent contrast factor for the factored relaxometric head (VFU-05).

The digital twin divides the 3T anatomy by a three-class gain map
(:meth:`MultiAcquisitionSimulator.tissue_classes`), so the contrast transfer is
tissue-dependent. One scalar ``kappa`` cannot represent it, and a scalar
factored head reduces to an output scale.

Here the head predicts a per-pixel class posterior and ``kappa(x)`` is the
posterior-weighted mean of the DECLARED per-class gains. The gain values stay
analytic (handed over from the relaxometry); what is learned is where each
tissue is, from the low-field input. The class map is not read from the target,
which would be an oracle absent at inference.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.infrastructure.physics.multi_acquisition import TISSUE_ORDER

N_TISSUES = len(TISSUE_ORDER)


def gain_vector(gains: dict[str, float], *, device=None, dtype=None) -> Tensor:
    """Declared gains in :data:`TISSUE_ORDER`, ``[N_TISSUES]``; raises on a missing class."""
    missing = [t for t in TISSUE_ORDER if t not in gains]
    if missing:
        raise ValueError(f"tissue gains lack classes {missing}; expected {list(TISSUE_ORDER)}")
    return torch.tensor([float(gains[t]) for t in TISSUE_ORDER], device=device, dtype=dtype)


def tissue_kappa_map(logits: Tensor, gains: Tensor) -> Tensor:
    """``kappa(x) = sum_c softmax(logits)_c * gain_c``, ``[B, 1, H, W]``.

    Bounded by construction to ``[min(gains), max(gains)]``, so the head cannot
    absorb an intensity map outside what the declared relaxometry allows.
    """
    if logits.ndim != 4 or logits.shape[1] != N_TISSUES:
        raise ValueError(f"expected [B, {N_TISSUES}, H, W] logits, got {tuple(logits.shape)}")
    weights = torch.softmax(logits, dim=1)
    return (weights * gains.to(logits.device, logits.dtype).view(1, -1, 1, 1)).sum(
        dim=1, keepdim=True
    )


def class_agreement(logits: Tensor, classes: Tensor) -> Tensor:
    """Fraction of voxels whose predicted class matches the simulator's, scalar tensor."""
    return (logits.argmax(dim=1, keepdim=True) == classes).float().mean()


__all__ = ["N_TISSUES", "class_agreement", "gain_vector", "tissue_kappa_map"]
