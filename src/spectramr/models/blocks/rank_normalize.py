r"""Rank-normalize block: exact monotone-contrast invariance (MCGI core).

The per-image empirical-CDF rank transform

.. math::

   R(\mathbf x)(v) = F_{\mathbf x}\bigl(\mathbf x(v)\bigr)

is *exactly* invariant to the group :math:`G_+` of strictly increasing intensity
remaps, because the empirical CDF transforms covariantly,
:math:`F_{g\circ\mathbf x}(g(t)) = F_{\mathbf x}(t)`, so evaluating at the image
gives identical ranks: :math:`R(g\circ\mathbf x) = R(\mathbf x)`. An encoder
fronted by :math:`R` therefore factors through the :math:`G_+`-orbit and is
contrast-agnostic by construction (for anatomy-defined tasks; invariant features
discard absolute-contrast information).

Tied voxels (e.g. a zero background) take the *midrank*
:math:`(\#\{x_j < x_i\} + \tfrac12(\#\{x_j = x_i\} - 1)) / (N-1)`, which is
deterministic, makes :math:`R(-\mathbf x) = 1 - R(\mathbf x)` hold exactly, and
costs :math:`O(N \log N)` time and :math:`O(N)` memory. The same transform runs in
training and inference: the block has no parameters and nothing learned sits
upstream of it, so a differentiable surrogate would only route gradient to the
loaded image (#2409).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-(batch, channel) empirical-CDF midrank transform to ``[0, 1]``.

    Args:
        x: Input ``[B, C, *spatial]``. Ranks are computed within each
            ``(batch, channel)`` map over the flattened spatial dims.

    Returns:
        Tensor of the same shape and dtype as ``x``, valued in ``[0, 1]``; a
        constant map is ``0.5`` everywhere.
    """
    b, c = x.shape[0], x.shape[1]
    flat = x.reshape(b, c, -1).contiguous()
    n = flat.shape[-1]
    sorted_flat = flat.sort(dim=-1).values
    below = torch.searchsorted(sorted_flat, flat, right=False)
    not_above = torch.searchsorted(sorted_flat, flat, right=True)
    # Integer numerator keeps the transform bit-identical under monotone remaps.
    out = (below + not_above - 1).to(x.dtype) / (2 * max(n - 1, 1))
    if n == 1:
        out = torch.full_like(out, 0.5)
    return out.reshape_as(x)


class RankNormalize2d(nn.Module):
    """``nn.Module`` wrapper over :func:`rank_normalize` (identical in train and eval)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rank_normalize(x)


__all__ = ["RankNormalize2d", "rank_normalize"]
