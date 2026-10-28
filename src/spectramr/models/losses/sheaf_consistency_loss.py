r"""Cellular-sheaf consistency loss for k-space / multi-contrast reconstruction.

This loss gives the ``sheaf_kspace`` / ``sheaf_multi_contrast`` strategy keys a
*genuine* sheaf-theoretic objective rather than a vanilla reconstruction
surrogate. It consumes the primitives in
:mod:`spectramr.infrastructure.physics.cellular_sheaf` directly.

A reconstructed volume is interpreted as a section of a *cellular sheaf*
:math:`\mathcal{F}` on a cover graph :math:`G=(V,E)`: each channel of the
``[B, C, H, W]`` tensor (a contrast, a coil, or an overlapping k-space patch)
is a **cover cell** / vertex, with stalk
:math:`\mathcal{F}(v)=\mathbb{R}^{HW}` carrying the flattened spatial signal.
Adjacent cells are joined by edges whose restriction maps are the identity
(the simplest "agreement" sheaf): two cells are *consistent* when they carry
the same section on their shared support.

The objective is the **sheaf-Laplacian quadratic form**

.. math::

    \langle x, L_{\mathcal{F}}\,x\rangle
        = \sum_{e=(u,v)} \| \mathcal{F}_{v\trianglelefteq e}x_v
                          - \mathcal{F}_{u\trianglelefteq e}x_u \|^2 ,

which is :math:`0` iff :math:`x` is a **global section**
(:math:`x\in H^0(G;\mathcal{F})`) — i.e. all cover cells agree — and strictly
positive when local sections disagree [Hansen-Ghrist 2019]. We additionally
penalise the **Mayer-Vietoris obstruction** :math:`\|[w]\|_{H^1}` of glueing
the *prediction* and *target* on a two-patch cover, which vanishes iff the two
agree on their overlap.

See :mod:`spectramr.infrastructure.physics.cellular_sheaf` for the algebraic
primitives.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.cellular_sheaf import (
    mayer_vietoris_obstruction_norm,
    sheaf_laplacian_quadratic_form,
)
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="sheaf_consistency",
    aliases=["cellular_sheaf", "sheaf_kspace_consistency"],
    domain="image",
)
class SheafConsistencyLoss(nn.Module):
    r"""Cellular-sheaf consistency penalty over a cover of cover cells.

    The prediction's channels are treated as the cells of a cellular cover
    (contrasts / coils / overlapping k-space patches). The loss penalises the
    sheaf-Laplacian energy of the prediction (disagreement between cells under
    identity restriction maps) plus the Mayer-Vietoris obstruction between the
    prediction and the target.

    The ``prediction`` and ``target`` are ``[B, C, H, W]`` (or
    ``[B, C, ...]``) real tensors. ``C`` is the number of cover cells; the
    remaining axes are flattened into the per-cell stalk.

    Args:
        laplacian_weight: Coefficient on the sheaf-Laplacian quadratic form
            of the prediction. Default 1.0.
        obstruction_weight: Coefficient on the Mayer-Vietoris obstruction
            norm between prediction and target. Default 1.0.
        reduction: ``"mean"`` (average over the batch) or ``"sum"``.
    """

    def __init__(
        self,
        laplacian_weight: float = 1.0,
        obstruction_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if laplacian_weight < 0:
            raise ValueError("laplacian_weight must be >= 0")
        if obstruction_weight < 0:
            raise ValueError("obstruction_weight must be >= 0")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}")
        self.laplacian_weight = laplacian_weight
        self.obstruction_weight = obstruction_weight
        self.reduction = reduction

    @staticmethod
    def _path_edges(num_cells: int, device: torch.device) -> torch.Tensor:
        """Edges of a path graph over the cover cells (each cell agrees with
        its neighbour). For a single cell there are no edges.
        """
        if num_cells < 2:
            return torch.empty(0, 2, dtype=torch.long, device=device)
        u = torch.arange(num_cells - 1, device=device)
        v = u + 1
        return torch.stack([u, v], dim=1)

    @staticmethod
    def _pixelwise_edges(
        batch: int, num_cells: int, stalk: int, device: torch.device
    ) -> torch.Tensor:
        """Path edges ``(b, c, p) -- (b, c + 1, p)`` over flattened ``(B*C*n)`` vertices."""
        idx = torch.arange(batch * num_cells * stalk, device=device).view(batch, num_cells, stalk)
        u = idx[:, :-1, :].reshape(-1)
        return torch.stack([u, u + stalk], dim=1)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} and target shape "
                f"{tuple(target.shape)} must match."
            )
        if prediction.ndim < 3:
            raise ValueError(f"expected [B, C, ...] with >=3 dims; got {tuple(prediction.shape)}.")

        batch = prediction.shape[0]
        num_cells = prediction.shape[1]
        # Per sample: (C cells, n=stalk dim) where n = product of spatial dims.
        pred_flat = prediction.reshape(batch, num_cells, -1)
        tgt_flat = target.reshape(batch, num_cells, -1)
        stalk = pred_flat.shape[-1]
        num_edges = self._path_edges(num_cells, prediction.device).shape[0]

        # Every per-sample term enters the reduction linearly, so the batch sum of
        # each term is what is needed.
        total = prediction.new_zeros(())
        if self.laplacian_weight > 0 and num_edges > 0:
            # The identity-restriction sheaf on stalk R^n is the direct sum of n
            # stalk-1 sheaves on the same path graph, so one stalk-1 call carries
            # the quadratic form without the (E, n, n) identity maps (17 GB at 256^2).
            edges = self._pixelwise_edges(batch, num_cells, stalk, prediction.device)
            unit = prediction.new_ones(1, 1, 1).expand(edges.shape[0], 1, 1)
            lap = sheaf_laplacian_quadratic_form(pred_flat.reshape(-1, 1), edges, unit, unit)
            total = total + self.laplacian_weight * (lap / (num_edges * stalk))

        # Mayer-Vietoris obstruction between prediction and target on a two-patch
        # cover (whole-section overlap). Zero iff they agree.
        if self.obstruction_weight > 0:
            section_len = num_cells * stalk
            for b in range(batch):
                pred_section = pred_flat[b].reshape(-1)
                tgt_section = tgt_flat[b].reshape(-1)
                obstruction = mayer_vietoris_obstruction_norm(
                    pred_section, tgt_section, pred_section, tgt_section
                )
                total = total + self.obstruction_weight * (obstruction / section_len)

        if self.reduction == "mean":
            return total / batch
        return total
