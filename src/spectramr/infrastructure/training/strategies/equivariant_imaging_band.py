r"""The acquired phase-encode band, for the EI and SSDU objectives.

M4Raw stores about 195 of 256 phase-encode columns; the rest are exact zeros.
``complementary_rep_masking`` publishes that support as ``support_mask``. Its
complement is known to be zero without any reference, so an objective may score
it, and a group action that moves content onto those columns (a 90-degree
rotation swaps the band-limited axis with the full-band readout axis) should not
be matched there:

.. math::

    P\,x = F^{-1}\bigl(\mathbb 1_{\mathrm{supp}} \odot F x\bigr).
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.data.transforms.complementary_rep_masking import SUPPORT_MASK_KEY
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.multicoil_ei_operator import interleaved_to_complex
from spectramr.infrastructure.training.strategies.mixins.reconstruction import _as_aux_mapping

__all__ = ["band_project", "batch_support", "scored_anchor"]


def batch_support(batch: Any, like: torch.Tensor) -> torch.Tensor:
    """The acquired support as a bool mask in ``like``'s ``[B, 1, H, W]`` layout.

    Args:
        batch: The raw training batch (a dict or a ``TrainingBatch``).
        like: The batch context's sampling mask, whose layout the support takes.

    Raises:
        ValueError: when the batch carries no support or its plane differs from ``like``'s.
    """
    raw = _as_aux_mapping(batch) if batch is not None else {}
    support = raw.get(SUPPORT_MASK_KEY) if isinstance(raw, dict) else None
    if isinstance(support, dict):
        support = support.get("data")
    if not isinstance(support, torch.Tensor):
        raise ValueError(
            f"this objective scores the acquired band and the batch carries no "
            f"`{SUPPORT_MASK_KEY}`; declare the `complementary_rep_masking` transform."
        )
    support = support.to(like.device)
    if support.dim() == like.dim() + 1 and support.shape[-1] == 1:
        support = support[..., 0]
    if support.shape[-2:] != like.shape[-2:]:
        raise ValueError(
            f"`{SUPPORT_MASK_KEY}` {tuple(support.shape)} does not share the mask's plane "
            f"{tuple(like.shape)}"
        )
    return support > 0


def band_project(image: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """:math:`P x` for a real/imag-interleaved image ``[B, 2C, H, W]``."""
    kspace = fft2c(interleaved_to_complex(image)) * support.to(torch.float32)
    return as_interleaved(ifft2c(kspace)).to(image.dtype)


def scored_anchor(
    predicted: torch.Tensor, measured: torch.Tensor, select: torch.Tensor, per_selected: bool
) -> torch.Tensor:
    r""":math:`\lVert S(A\hat x - y)\rVert^2`, averaged over all entries or the scored ones.

    Args:
        predicted: Complex :math:`A\hat x` with the scored mask already applied.
        measured: Complex :math:`y`, zero outside the input mask.
        select: The scored locations, broadcastable to ``predicted``.
        per_selected: Divide by the number of scored entries instead of all of them.
    """
    resid = predicted - measured
    sq = (resid.conj() * resid).real * select.to(resid.real.dtype)
    if not per_selected:
        return sq.mean()
    count = select.to(resid.real.dtype).expand_as(sq).sum()
    return sq.sum() / count.clamp_min(1.0)
