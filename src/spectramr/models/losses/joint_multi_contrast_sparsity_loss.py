r"""Joint multi-contrast gradient-sparsity loss.

Encodes the prior that co-registered multi-contrast images of the same
subject share *edge support*. Stacking the gradient fields across contrasts
into a matrix :math:`G \\in \\mathbb{R}^{N \\times C}` (one row per voxel,
one column per contrast) and penalising its mixed :math:`\\ell_{2,1}` norm,

.. math::

   \\| \\nabla \\mathbf{X} \\|_{2,1}
   \\;=\\;
   \\sum_i \\sqrt{\\sum_c \\| (\\nabla \\mathbf{x}_c)_i \\|_2^2 + \\varepsilon^2},

drives whole *rows* — i.e. the multi-contrast vector at each voxel — to zero
together. Voxels in flat tissue regions go silent across every contrast;
voxels at anatomical boundaries fire across all contrasts that resolve them.
This is the joint-CS prior of Bilgic et al. [7], adapted for use as an
auxiliary loss term inside a deep multi-contrast reconstruction network.

The gain over independent recovery is largest at high acceleration (R≥6),
where any single contrast lacks the in-plane information to localise edges
on its own. By coupling the sparsity across contrasts, edges that are
strong in one contrast (e.g. T2 → CSF) help anchor the same voxel's
recovery in a contrast where the edge is weaker (e.g. T1 → CSF).

The implementation supports either:

- A single tensor of shape ``(B, C, H, W)`` where ``C`` is the contrast
  axis (one prediction stack per sample), or
- A list of tensors all of shape ``(B, 1, H, W)`` — useful when each
  contrast comes from a separate branch of a multi-head model.

References
----------
[7] B. Bilgic, V. K. Goyal, E. Adalsteinsson, "Multi-contrast reconstruction
    with Bayesian compressed sensing," *Magn. Reson. Med.*, vol. 66, no. 6,
    pp. 1601–1615, 2011.

This loss is part of the *Pattern A/C joint reconstruction* described in
``docs/multi_contrast.md``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

__all__ = ["JointMultiContrastSparsityLoss"]


def _spatial_gradients(x: torch.Tensor) -> torch.Tensor:
    """Forward-difference 2D gradients with edge padding (B, C, H, W, 2)."""
    # Pad on the trailing edge so dx, dy share spatial shape with x.
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.stack([dx, dy], dim=-1)


@register_loss(name="joint_multi_contrast_sparsity", domain="image")
class JointMultiContrastSparsityLoss(nn.Module):
    r"""Group-sparse :math:`\\ell_{2,1}` penalty on the stacked gradient field.

    Args:
        eps: Numerical stability epsilon under the square root.
        normalise: If True, divide the final loss by the number of voxels so the
            scale is comparable across image sizes. Default True.
        per_contrast_weights: Optional sequence of per-contrast weights with
            length matching the contrast axis; useful when one contrast is
            noisier and should contribute less to the coupling.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        normalise: bool = True,
        per_contrast_weights: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.eps = eps
        self.normalise = normalise
        if per_contrast_weights is not None:
            w = torch.tensor(per_contrast_weights, dtype=torch.float32)
            if (w < 0).any():
                raise ValueError("per_contrast_weights must be non-negative")
            self.register_buffer("_weights", w)
        else:
            self._weights = None

    def forward(
        self,
        prediction: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the joint multi-contrast sparsity penalty on ``prediction``.

        ``target`` is accepted for API compatibility with the loss registry
        but is not used — the prior acts on the prediction's own gradients.
        """
        x = self._coerce(prediction)
        # x: (B, C, H, W). Gradients: (B, C, H, W, 2).
        grads = _spatial_gradients(x)
        # Per-voxel L2 across (contrast, dx/dy) → (B, H, W).
        # Permute to put contrast last for clarity: (B, H, W, C, 2).
        grads = grads.permute(0, 2, 3, 1, 4)
        if self._weights is not None:
            if self._weights.shape[0] != grads.shape[-2]:
                raise ValueError(
                    f"per_contrast_weights length {self._weights.shape[0]} != "
                    f"contrast axis size {grads.shape[-2]}"
                )
            grads = grads * self._weights.view(1, 1, 1, -1, 1)
        per_voxel = torch.sqrt(grads.pow(2).sum(dim=(-1, -2)) + self.eps**2)
        loss = per_voxel.sum()
        if self.normalise:
            loss = loss / per_voxel.numel()
        return loss

    @staticmethod
    def _coerce(
        prediction: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(prediction, list):
            if not prediction:
                raise ValueError("prediction list is empty")
            # Expected per-contrast: (B, 1, H, W). Concatenate along channel.
            return torch.cat(prediction, dim=1)
        if prediction.dim() != 4:
            raise ValueError(
                f"prediction must be (B, C, H, W) or list of (B, 1, H, W); "
                f"got shape {tuple(prediction.shape)}"
            )
        return prediction
