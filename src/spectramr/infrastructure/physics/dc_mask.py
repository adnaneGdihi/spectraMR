"""One owner for "what channel layout does a DC layer's sampling mask need".

The sampling mask is **coil-independent by physics**: the same Cartesian /
radial / spiral pattern is acquired on every coil, and a coefficient is observed
or not observed as a whole -- never half-observed because its imaginary part
happens to be zero. What varies is only the *layout* a caller happens to hold it
in, and the DC layers reduce their own inputs to complex channels before the
blend, so a mask that arrived interleaved no longer matches.

Five layers each answered that question for themselves. ``MaskedReplacementDataConsistency``
collapsed with ``amax``; ``AdaptiveDataConsistency``, ``NoiseAdaptiveDataConsistency``
and ``KANAdaptiveDataConsistency`` sliced ``mask[:, :C]`` (right only because every
channel carries the same pattern -- it pairs mask channel ``[Re0, Im0, Re1, Im1]``
with coils ``0..3``); ``TargetAwareFSDC`` answered nothing at all. The 2026-09-16
cluster run is what that costs: five arms reached validation and raised
``The size of tensor a (4) must match the size of tensor b (8)`` from the two
un-guarded sites, having trained for their whole budget first (non-negotiable 17).
"""

from __future__ import annotations

import torch

__all__ = ["align_dc_mask"]


def align_dc_mask(mask: torch.Tensor, target_channels: int) -> torch.Tensor:
    """Reduce *mask* to a channel count that broadcasts against the DC blend.

    Args:
        mask: Sampling mask ``[B, C_mask, H, W]`` (or any shape whose ``-3``
            axis is the channel axis). Complex masks are read through their
            real part; the dtype is otherwise untouched, so a bool mask stays
            bool for a ``torch.where`` and a float mask stays float for a
            multiply.
        target_channels: Channel count of the tensors the mask is about to be
            combined with -- the *complex* count, after a caller has folded an
            interleaved ``[Re0, Im0, ...]`` input down to coils.

    Returns:
        The mask unchanged when it already broadcasts (``1`` or
        ``target_channels`` channels); the pairwise ``maximum`` of interleaved
        Re/Im channels when it carries ``2 * target_channels`` of them; a
        1-channel ``amax`` collapse otherwise.

    The interleaved branch reduces with ``maximum`` rather than taking either
    half, for the reason ``paired_magnitude`` states in
    ``models/diffusion/kspace_process.py``: observation is a property of the
    COEFFICIENT, so a pair is observed when either of its halves is. The
    fallback is the coil-wise logical OR -- a frequency any coil sampled is
    sampled -- which is the rule ``MaskedReplacementDataConsistency`` has enforced since
    the 2026-05-10 diff_varnet crash.
    """
    if torch.is_complex(mask):
        mask = mask.real
    if mask.ndim < 3:
        return mask
    channels = mask.shape[-3]
    if channels in (1, target_channels):
        return mask
    if channels == 2 * target_channels:
        return torch.maximum(mask[..., 0::2, :, :], mask[..., 1::2, :, :])
    return mask.amax(dim=-3, keepdim=True)
