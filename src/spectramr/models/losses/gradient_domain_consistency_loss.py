r"""Gradient-domain consistency loss for cross-contrast translation.

Co-registered multi-contrast images of the same anatomy share *edge
support* even though their intensity statistics differ. A translation
generator that hallucinates lesions or smooths boundaries violates this
invariant. The gradient-domain consistency loss — first popularised in
the cycle-consistent translation literature — measures the residual
between the source and translated images *after* a high-pass operator
:math:`\nabla`:

.. math::

   \mathcal{L}_{\text{grad}}
   \;=\;
   \big\| \nabla \mathbf{x}_{\text{src}}
        - \nabla G(\mathbf{x}_{\text{src}}) \big\|_p,

with optional channel-wise normalisation by the per-image gradient
energy (``normalise_per_image=True``) so absolute intensity scaling
between contrasts doesn't dominate the gradient comparison.

Three gradient operators are supported:

- ``finite_diff`` (default) — first-order forward differences with
  edge padding, fastest.
- ``sobel`` — Sobel kernel applied via depthwise conv; smoother edges.
- ``laplacian`` — discrete Laplacian, sensitive to small structures.

The loss is an *auxiliary* term on top of a primary translation loss
(e.g. L1 + perceptual). Recommended weight: ``0.05–0.5`` depending on
how aggressively you want to forbid hallucination.

References
----------
[6]  J.-Y. Zhu, T. Park, P. Isola, A. A. Efros, "Unpaired image-to-image
     translation using cycle-consistent adversarial networks (CycleGAN),"
     ICCV 2017.
[14] T. Park, M.-Y. Liu, T.-C. Wang, J.-Y. Zhu, "Semantic image synthesis
     with spatially-adaptive normalization (SPADE)," CVPR 2019.

§2 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

__all__ = ["GradientDomainConsistencyLoss"]


def _finite_diff(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.cat([dx, dy], dim=1)


def _make_sobel_kernel(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    ky = kx.t()
    kxy = torch.stack([kx, ky], dim=0).unsqueeze(1)  # (2, 1, 3, 3)
    return kxy.repeat(channels, 1, 1, 1)  # (2*C, 1, 3, 3)


def _sobel(x: torch.Tensor) -> torch.Tensor:
    C = x.shape[1]
    weight = _make_sobel_kernel(C, x.device, x.dtype)
    grouped = F.conv2d(x, weight, padding=1, groups=C)
    return grouped


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    C = x.shape[1]
    k = (
        torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=x.device,
            dtype=x.dtype,
        )
        .view(1, 1, 3, 3)
        .repeat(C, 1, 1, 1)
    )
    return F.conv2d(x, k, padding=1, groups=C)


_OPERATORS = {
    "finite_diff": _finite_diff,
    "sobel": _sobel,
    "laplacian": _laplacian,
}


@register_loss(name="gradient_domain_consistency", domain="image")
class GradientDomainConsistencyLoss(nn.Module):
    r"""Penalise gradient-field divergence between source and translation.

    Args:
        operator: One of ``"finite_diff"``, ``"sobel"``, ``"laplacian"``.
        norm: Distance norm — ``"l1"``, ``"l2"``, or ``"smooth_l1"``.
        normalise_per_image: If True, divide each image's gradient field
            by its own RMS energy before comparing — makes the loss
            scale-invariant under contrast-dependent intensity rescaling.
        anchor: ``"target"`` (default) — the *source* image is the gradient
            anchor: its gradient field is detached, so only ``prediction``
            receives gradient; ``"prediction"`` — symmetric, both contribute.
            ``"target"`` is appropriate for translation tasks where the
            target image is the unmodified source.
    """

    _NORM_FNS: ClassVar[dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = {
        "l1": lambda a, b: (a - b).abs().mean(),
        "l2": lambda a, b: (a - b).pow(2).mean(),
        "smooth_l1": lambda a, b: F.smooth_l1_loss(a, b),
    }

    def __init__(
        self,
        operator: str = "finite_diff",
        norm: str = "l1",
        normalise_per_image: bool = False,
        anchor: str = "target",
    ) -> None:
        super().__init__()
        if operator not in _OPERATORS:
            raise ValueError(f"operator must be one of {list(_OPERATORS)}, got {operator!r}")
        if norm not in self._NORM_FNS:
            raise ValueError(f"norm must be one of {list(self._NORM_FNS)}, got {norm!r}")
        if anchor not in ("target", "prediction"):
            raise ValueError(f"anchor must be 'target' or 'prediction', got {anchor!r}")
        self.operator = operator
        self.norm = norm
        self.normalise_per_image = normalise_per_image
        self.anchor = anchor

    @staticmethod
    def _normalise(g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Per-image RMS over (C, H, W); the floor goes under the root because
        # sqrt'(0) is infinite and a flat image (zero gradient field) would NaN.
        rms = g.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps * eps).sqrt()
        return g / rms

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} != target shape {tuple(target.shape)}."
            )
        op = _OPERATORS[self.operator]
        g_pred = op(prediction)
        g_tgt = op(target)
        if self.anchor == "target":
            g_tgt = g_tgt.detach()
        if self.normalise_per_image:
            g_pred = self._normalise(g_pred)
            g_tgt = self._normalise(g_tgt)
        return self._NORM_FNS[self.norm](g_pred, g_tgt)
