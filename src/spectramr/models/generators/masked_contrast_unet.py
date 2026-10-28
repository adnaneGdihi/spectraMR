"""C2 control of the tissue-map cohort: any subset of contrasts in, every contrast out, no physics.

A three-level skip-connected U-Net maps the masked [T1, T2, FLAIR] triplet plus its
availability planes straight to the three contrasts. It is trained by the same strategy, with
the same contrast dropout, gain projection and leave-one-contrast-out scoring as the
tissue-map models, so a difference against them measures the physics bottleneck rather than
the training scheme. Like them it never reads a hidden contrast, and a global input scale
passes straight through (normalised in, multiplied back out).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model

__all__ = ["MaskedContrastUNet"]

_N = 3


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.LeakyReLU(0.2),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.LeakyReLU(0.2),
    )


@register_model(name="masked_contrast_unet", training_mode="reconstruction", input_domain="image")
class MaskedContrastUNet(nn.Module):
    """Skip-connected U-Net from the masked triplet to all three contrasts.

    Args:
        in_channels: Must be 3, the [T1, T2, FLAIR] triplet in noise units.
        out_channels: Accepted for the registry contract; three contrasts are predicted.
        base_channels: Width of the first level (doubled per level).
    """

    #: The strategy passes the protocol (unused here) and an availability mask.
    takes_protocol: ClassVar[bool] = True

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        if in_channels != _N:
            raise ValueError(f"in_channels={in_channels}; the input is the 3-contrast triplet")
        c = base_channels
        self.enc1, self.enc2, self.enc3 = _block(2 * _N, c), _block(c, 2 * c), _block(2 * c, 4 * c)
        self.up2, self.dec2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), _block(4 * c, 2 * c)
        self.up1, self.dec1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), _block(2 * c, c)
        self.head = nn.Conv2d(c, _N, 1)

    def render_triplet(
        self,
        x: torch.Tensor,
        protocol: Any = None,
        available: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """No maps, and one predicted image per contrast; ``protocol`` is not used."""
        if x.ndim != 4 or x.shape[1] != _N:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(x.shape)}")
        b = x.shape[0]
        avail = (
            torch.ones(b, _N, device=x.device)
            if available is None
            else available.to(x.device, x.dtype)
        )
        if (avail.sum(1) < 1).any():
            raise ValueError("every sample needs at least one available contrast")
        a4 = avail.view(b, _N, 1, 1)
        x = x * a4  # a dropped contrast is never read below this line
        with torch.no_grad():
            scale = (
                torch.quantile(x.flatten(1).float(), 0.99, dim=1).clamp_min(1.0).view(b, 1, 1, 1)
            )
        e1 = self.enc1(torch.cat([x / scale, a4.expand(-1, -1, *x.shape[2:])], dim=1))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = F.softplus(self.head(d1)) * scale
        return {}, [out[:, k : k + 1] for k in range(_N)]

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Registry contract: the strategy reaches the model through render_triplet."""
        raise RuntimeError(
            "masked_contrast_unet is driven by CalibratedQMapStrategy through "
            "render_triplet(x, protocol, available)."
        )
