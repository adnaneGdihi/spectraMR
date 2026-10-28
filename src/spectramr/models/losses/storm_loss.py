"""SToRM smoothness-on-manifolds regularisation loss.

Canonical-home migration from ``spectramr/models/geometric/storm.py`` to
``spectramr/models/losses/`` (CLAUDE.md pitfall #12). The
``@register_loss`` decorator must live under ``models/losses/``; the
underlying manifold-Laplacian computation stays in
``models/geometric/storm.py`` because it is a reusable geometric
primitive (also consumed by ``GraphImagePrior``), not a loss.

Reference
---------
Poddar et al., "Dynamic MRI Using SmooThness Regularization on
Manifolds (SToRM)," IEEE TMI 35(4), 2016.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.models.losses.registry import register_loss

if TYPE_CHECKING:
    from spectramr.models.geometric.storm import ManifoldLaplacian

# Imported lazily inside ``__init__`` so the losses package can be
# imported before the geometric package has finished initialising
# (avoids ``losses → geometric → losses`` cycle at import time).


def _uniform_temporal_laplacian(
    batch: int, frames: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    r"""``[B, T, T]`` path-graph Laplacian linking each frame to its neighbours.

    Diagonal = node degree (1 at the two endpoints, 2 in the interior), −1 on
    both off-diagonals. This makes ``Tr(X L X^T)`` equal to
    ``sum_i ||x_i - x_{i+1}||^2`` — the sum of squared consecutive frame
    differences, which is what "temporal smoothness" means and is >= 0 by
    construction.

    **This replaces a loop that did not build a Laplacian at all.** The previous
    code wrote four entries per iteration with ``=`` rather than ``+=``, so each
    step overwrote the last one's diagonal and the interior diagonal ended at 1
    instead of the degree 2. The result had non-zero row sums and a NEGATIVE
    minimum eigenvalue (−0.73 at T=5), so the quadratic form was **unbounded
    below**: a constant frame sequence scored −3, and scaling it up drove the
    "smoothness penalty" toward −∞. The regulariser did not merely fail to
    smooth — it rewarded blowing the reconstruction up, while showing a loss
    curve going satisfyingly down. ``test_uniform_laplacian_is_psd_and_penalises
    _frame_differences`` pins the properties that make it a Laplacian.

    Building it with ``diag_embed`` also drops the O(T) kernel launches the loop
    issued *every training step* to fill a matrix that only depends on the batch
    shape (CLAUDE.md #9).
    """
    if frames < 2:
        raise ValueError(
            f"a temporal Laplacian needs >= 2 frames, got {frames} — a single "
            "frame has no temporal neighbour to be smooth against."
        )
    main = torch.full((frames,), 2.0, device=device, dtype=dtype)
    # Endpoints have one neighbour, interior nodes two. Degree IS the diagonal.
    main[0] = 1.0
    main[-1] = 1.0
    off = torch.full((frames - 1,), -1.0, device=device, dtype=dtype)
    lap = (
        torch.diag_embed(main) + torch.diag_embed(off, offset=1) + torch.diag_embed(off, offset=-1)
    )
    return lap.unsqueeze(0).expand(batch, frames, frames)


@register_loss(
    "storm",
    aliases=["manifold_smoothness", "storm_regularization"],
    domain="image",
    workflows=frozenset({Regime.DYNAMIC}),
    tasks=frozenset({Task.RECONSTRUCTION}),
)
class SToRMRegularizer(nn.Module):
    """SToRM: Smoothness Regularization on Manifolds.

    Regularises dynamic MRI reconstruction by enforcing smoothness on the
    learned temporal manifold via the trace ``Tr(X L X^T)``, penalising
    differences between frames connected on the manifold graph.

    Tagged ``mri_dynamic``: the manifold is over the TEMPORAL frame axis of a
    ``[B, T, C, H, W]`` series, which is what makes an acquisition dynamic.

    Reference:
        Poddar & Jacob, "Dynamic MRI using smoothness regularization on
        manifolds (SToRM)", IEEE Trans Med Imaging 35(4):1106-1115, 2016.

    Args:
        lambda_manifold: regularisation strength (scalar).
        learn_laplacian: when True, learn ``L`` from navigators using a
            :class:`ManifoldLaplacian` block; when False, use the uniform
            temporal-smoothness Laplacian — a deliberate, declared choice.
        num_frames: number of temporal frames ``T``.
        nav_dim: navigator-signal dimensionality ``D``.
    """

    def __init__(
        self,
        lambda_manifold: float = 1.0,
        learn_laplacian: bool = True,
        num_frames: int = 20,
        nav_dim: int = 64,
    ) -> None:
        super().__init__()
        self.lambda_manifold = lambda_manifold
        self.learn_laplacian = learn_laplacian
        self.manifold: ManifoldLaplacian | None
        if learn_laplacian:
            # Lazy import to avoid losses ↔ geometric circular dependency
            # at package-load time.
            from spectramr.models.geometric.storm import ManifoldLaplacian

            self.manifold = ManifoldLaplacian(num_frames, nav_dim)
        else:
            self.manifold = None

    def forward(
        self,
        X: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        navigators: torch.Tensor | None = None,
        L: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the manifold-smoothness penalty.

        Args:
            X: ``[B, T, C, H, W]`` dynamic image sequence.
            target: Unused. The registry calls ``loss(prediction, target)``;
                taking it here keeps the target from binding to ``navigators``.
            navigators: ``[B, T, D]`` navigator signals when
                ``learn_laplacian`` is True.
            L: ``[B, T, T]`` pre-computed Laplacian; mutually exclusive
                with ``navigators``.

        Returns:
            Scalar manifold-smoothness loss.
        """
        B, T = X.shape[0], X.shape[1]

        if L is None:
            if self.learn_laplacian:
                if navigators is None:
                    raise ValueError(
                        "SToRMRegularizer(learn_laplacian=True) needs `navigators` "
                        "[B, T, D] (or a precomputed `L`) to build the manifold "
                        "graph. It previously fell through to the UNIFORM "
                        "temporal Laplacian here — silently turning learned "
                        "manifold smoothness into plain temporal TV, which is a "
                        "different regulariser reported under the name SToRM "
                        "(pitfall #9). Pass navigators, precompute L, or declare "
                        "learn_laplacian=False to choose the uniform Laplacian on "
                        "purpose."
                    )
                assert self.manifold is not None
                L, _ = self.manifold(navigators)
            else:
                L = _uniform_temporal_laplacian(B, T, device=X.device, dtype=X.dtype)

        x_flat = X.reshape(B, T, -1)  # [B, T, P], P = C*H*W

        # Tr(X^T L X) = sum_{i,p} X[i,p] (L X)[i,p]: contracting L with X first stays
        # at [B, T, P], where the [B, P, P] product needs 68.7 GB at 256^2.
        lx = torch.bmm(L, x_flat)  # [B, T, P]
        loss = (x_flat * lx).sum(dim=(1, 2)).mean()
        return self.lambda_manifold * loss


__all__ = ["SToRMRegularizer"]
