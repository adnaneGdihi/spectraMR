r"""MCGI -- Monotone-Contrast-Group-Invariant encoder.

An encoder :math:`\Phi = \Psi \circ R` whose invariance to monotone intensity
remaps is guaranteed *by construction* rather than learned in expectation
(contrast the randomization-trained approximate invariance of SynthSeg /
SynthMorph). :math:`R` is the per-image empirical-CDF rank transform
(:mod:`spectramr.models.blocks.rank_normalize`), which is exactly invariant to the
group :math:`G_+` of strictly increasing intensity maps; :math:`\Psi` is any
backbone.

For order-reversing contrast pairs (T1w vs T2w) adjoin the involution
:math:`\sigma:\mathbf x\mapsto-\mathbf x` and symmetrize,

.. math::

   \Phi_{\mathrm{sym}}(\mathbf x)
     = \tfrac12\bigl(\Psi(R(\mathbf x)) + \Psi(R(-\mathbf x))\bigr),

a :math:`\mathbb Z_2`-invariant section of the quotient, so :math:`\Phi_{\mathrm
{sym}}` is exactly invariant on the full group :math:`G_+ \rtimes \mathbb Z_2`.

Scope (honest): invariant features discard absolute-contrast information, so MCGI
is for anatomy-defined tasks (segmentation, registration) and not for contrast
quantification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.config.schemas.training.mcgi import MCGIConfig
from spectramr.models.blocks.rank_normalize import RankNormalize2d
from spectramr.models.registry import register_model


def _conv_backbone(
    in_channels: int, hidden_channels: int, out_channels: int, depth: int
) -> nn.Sequential:
    """A small same-resolution conv stack standing in for :math:`\\Psi`.

    The invariance guarantee is independent of the backbone (it sits *behind*
    the rank transform), so a plain conv stack is sufficient and honest.
    """
    layers: list[nn.Module] = []
    c_in = in_channels
    for _ in range(max(depth - 1, 0)):
        layers += [nn.Conv2d(c_in, hidden_channels, 3, padding=1), nn.GELU()]
        c_in = hidden_channels
    layers += [nn.Conv2d(c_in, out_channels, 3, padding=1)]
    return nn.Sequential(*layers)


_DEFAULTS = {name: field.default for name, field in MCGIConfig.model_fields.items()}


@register_model(
    name="mcgi_encoder",
    training_mode="supervised",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    supports_contrast_conditioning=False,
    strict_kwargs=True,
)
class MCGIEncoder(nn.Module):
    r"""Monotone-contrast-group-invariant encoder :math:`\Phi = \Psi \circ R`.

    The MCGI knobs are validated by
    :class:`~spectramr.config.schemas.training.mcgi.MCGIConfig`, which owns their
    defaults and legal values; the resolved model is kept as ``self.config``.

    Args:
        in_channels: Input image channels.
        hidden_channels: Backbone hidden width.
        out_channels: Output feature channels.
        depth: Number of conv layers in the backbone.
        rank_transform: Apply :math:`R` before the backbone. ``False`` gives the
            non-invariant one-knob control (same backbone on raw intensities).
        symmetrize_order_reversal: Pool over :math:`\sigma:\mathbf x\mapsto-\mathbf
            x` to extend invariance to :math:`G_+\rtimes\mathbb Z_2`.
        hard_rank_eval: Must be ``True`` (#2409).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = _DEFAULTS["hidden_channels"],
        out_channels: int = 1,
        depth: int = _DEFAULTS["depth"],
        *,
        rank_transform: bool = _DEFAULTS["rank_transform"],
        symmetrize_order_reversal: bool = _DEFAULTS["symmetrize_order_reversal"],
        hard_rank_eval: bool = _DEFAULTS["hard_rank_eval"],
    ) -> None:
        super().__init__()
        self.config = MCGIConfig(
            hidden_channels=hidden_channels,
            depth=depth,
            rank_transform=rank_transform,
            symmetrize_order_reversal=symmetrize_order_reversal,
            hard_rank_eval=hard_rank_eval,
        )
        self.symmetrize_order_reversal = self.config.symmetrize_order_reversal
        self.rank: nn.Module = RankNormalize2d() if self.config.rank_transform else nn.Identity()
        self.backbone = _conv_backbone(
            in_channels, self.config.hidden_channels, out_channels, self.config.depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(self.rank(x))
        if self.symmetrize_order_reversal:
            feat_rev = self.backbone(self.rank(-x))
            feat = 0.5 * (feat + feat_rev)
        return feat


__all__ = ["MCGIEncoder"]
