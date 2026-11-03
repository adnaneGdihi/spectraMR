"""One-shot multi-parameter mapping: a U-Net trunk and per-parameter heads.

The co-registered T1/T2/FLAIR triplet goes through a configurable U-Net whose output
is a ``base_channels`` feature map; :class:`MultiParameterHead` turns it into
positive T1 and T2 (ms) and PD maps, plus optional segmentation logits.
``predict_parameters`` exposes (rho, T1, T2) under the names the TSE/IR render
(``tse_signal``) and ``CalibratedQMapStrategy`` read, so the maps are trained
self-supervised: each series is re-rendered with its header protocol and compared
with the acquired one. M4Raw carries no reference maps.
"""

from __future__ import annotations

import torch
from torch import nn

from spectramr.models.multi_param.multi_parameter_mapping import MultiParameterHead
from spectramr.models.reconstruction.unet import UNet, UNetConfig
from spectramr.models.registry import register_model

__all__ = ["MultiParamMapping"]


@register_model(
    name="multi_param_mapping",
    training_mode="multi_param_mapping",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
)
class MultiParamMapping(nn.Module):
    """U-Net trunk + :class:`MultiParameterHead`; ``forward`` returns the parameter maps."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        spatial_dims: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        t1_scale_ms: float = 1000.0,
        t2_scale_ms: float = 100.0,
        num_seg_classes: int = 0,
    ) -> None:
        super().__init__()
        if spatial_dims != 2:
            raise ValueError(f"multi_param_mapping is 2-D; got spatial_dims={spatial_dims}")
        if depth < 1 or base_channels < 1:
            raise ValueError(f"depth={depth} and base_channels={base_channels} must be >= 1")
        self.trunk = UNet(
            UNetConfig(
                in_channels=in_channels,
                out_channels=base_channels,
                features=tuple(base_channels * 2**i for i in range(depth + 1)),
                depth=depth,
            )
        )
        self.head = MultiParameterHead(
            base_channels,
            num_seg_classes=num_seg_classes,
            t1_scale_ms=t1_scale_ms,
            t2_scale_ms=t2_scale_ms,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(torch.relu(self.trunk(x)))

    def predict_parameters(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """(rho, T1 ms, T2 ms) in the render's names; PD is the apparent proton density."""
        maps = self(x)
        return {"rho": maps["pd"], "t1": maps["t1_ms"], "t2": maps["t2_ms"]}
