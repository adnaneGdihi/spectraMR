r"""Gram-matrix style loss — feature-distribution matching for harmonisation.

For VGG features :math:`\phi_l(\cdot)` extracted at a layer :math:`l`, the
Gram matrix is :math:`G_l(\mathbf{x}) = \phi_l\phi_l^\top / (C\,H\,W)`
(channel inner product, normalised by spatial extent). The style loss is

.. math::
    \mathcal{L}_{\text{style}} = \sum_l \|G_l(\hat{\mathbf{x}})-G_l(\mathbf{x})\|_F^2

This is the classical Gatys-style transfer term: it matches *texture
statistics* across feature maps regardless of spatial layout. In MRI it
is useful for **harmonisation** (matching scanner-specific texture
without distorting anatomy) and as an auxiliary to perceptual loss in
i2i / SR pipelines where a pure VGG feature distance is too rigid.

Reference:
    L. A. Gatys, A. S. Ecker, M. Bethge, "Image style transfer using
    convolutional neural networks," *Proc. IEEE CVPR*, 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.perceptual_loss import VGGFeatureExtractor
from spectramr.models.losses.registry import register_loss


def _gram(features: torch.Tensor) -> torch.Tensor:
    """Compute normalised Gram matrix from a feature map.

    Args:
        features: Tensor of shape ``(B, C, H, W)``.

    Returns:
        Gram matrix ``(B, C, C)``, normalised by ``C * H * W``, in float32.
    """
    b, c, h, w = features.shape
    # Each entry sums H*W products, which overflows float16 (max 65504) on an
    # early VGG layer at MRI resolution, so the product runs in float32 with
    # autocast disabled.
    flat = features.reshape(b, c, h * w).float()
    with torch.autocast(device_type=flat.device.type, enabled=False):
        gram = torch.bmm(flat, flat.transpose(1, 2))
    return gram / (c * h * w)


@register_loss(name="gram_style", aliases=["GramStyleLoss", "style_loss"], domain="image")
class GramStyleLoss(nn.Module):
    r"""Texture-statistics matching via VGG Gram matrices.

    Args:
        layers: VGG19 relu names to extract features from. Defaults to
            ``["relu1_1", "relu2_1", "relu3_1", "relu4_1"]`` which span
            low-to-mid scales (style is dominated by these).
        use_input_norm: Apply ImageNet mean/std normalisation before VGG.
        weights: Optional per-layer weights. Defaults to uniform.
    """

    def __init__(
        self,
        layers: list[str] | None = None,
        use_input_norm: bool = True,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
        self.layers = layers
        if weights is None:
            weights = [1.0] * len(layers)
        if len(weights) != len(layers):
            raise ValueError(f"weights len {len(weights)} != layers len {len(layers)}")
        self.layer_weights: torch.Tensor
        self.register_buffer("layer_weights", torch.tensor(weights))
        self.extractor = VGGFeatureExtractor(layers=layers, use_input_norm=use_input_norm)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        feats_pred = self.extractor(prediction)
        with torch.no_grad():
            feats_target = self.extractor(target)
        loss = prediction.new_zeros(())
        for w, fp, ft in zip(self.layer_weights, feats_pred, feats_target, strict=True):
            gp, gt = _gram(fp), _gram(ft)
            loss = loss + w * (gp - gt).pow(2).mean()
        return loss
