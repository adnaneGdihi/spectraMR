"""GAN Architecture Variants
=============================

Distinct adversarial generator architectures for MRI reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

# ---------------------------------------------------------------------------
# 1. Energy-Based GAN — generator trained against energy-based discriminator
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


@register_model(name="energy_based_gan", training_mode="gan")
class EnergyBasedGANGenerator(nn.Module, IGenerator):
    """Generator for energy-based GAN: residual blocks with Langevin-style refinement."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 64,
        n_res_blocks: int = 6,
        n_refine_steps: int = 3,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.n_refine = n_refine_steps
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, bf, 7, padding=3), nn.InstanceNorm2d(bf), nn.GELU()
        )
        self.down = nn.Sequential(
            nn.Conv2d(bf, bf * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(bf * 2), nn.GELU()
        )
        self.res_blocks = nn.Sequential(*[_ResBlock(bf * 2) for _ in range(n_res_blocks)])
        self.up = nn.Sequential(
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2), nn.InstanceNorm2d(bf), nn.GELU()
        )
        self.head = nn.Conv2d(bf, out_channels, 7, padding=3)
        # Learnable refinement step size
        self.step_size = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.stem(x)
        h = self.up(self.res_blocks(self.down(h)))
        pred = torch.tanh(self.head(h))
        # Iterative refinement (Langevin-like)
        for _ in range(self.n_refine):
            pred = pred + self.step_size * (x[:, : pred.shape[1]] - pred)
        return pred


# ---------------------------------------------------------------------------
# 2. SAGAN — Self-Attention GAN for MRI
# ---------------------------------------------------------------------------


class _SelfAttention2d(nn.Module):
    """Self-attention module (Zhang et al., SAGAN)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.key = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, 1, HW, E) single head; scale=1.0 keeps SAGAN's unscaled logits.
        q = self.query(x).flatten(2).transpose(1, 2).unsqueeze(1)
        k = self.key(x).flatten(2).transpose(1, 2).unsqueeze(1)
        v = self.value(x).flatten(2).transpose(1, 2).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        out = out.squeeze(1).transpose(1, 2).reshape(B, C, H, W)
        return x + self.gamma * out


@register_model(name="sagan", training_mode="gan")
class SAGANGenerator(nn.Module, IGenerator):
    """Self-Attention GAN: residual generator with SA layers at mid-resolution."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 64,
        n_res: int = 6,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, bf, 7, padding=3), nn.InstanceNorm2d(bf), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(bf, bf * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(bf * 2), nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(bf * 2, bf * 4, 3, stride=2, padding=1), nn.InstanceNorm2d(bf * 4), nn.ReLU()
        )
        self.res = nn.Sequential(*[_ResBlock(bf * 4) for _ in range(n_res)])
        self.sa = _SelfAttention2d(bf * 4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2), nn.InstanceNorm2d(bf * 2), nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2), nn.InstanceNorm2d(bf), nn.ReLU()
        )
        self.head = nn.Sequential(nn.Conv2d(bf, out_channels, 7, padding=3), nn.Tanh())

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.stem(x)
        h = self.down2(self.down1(h))
        h = self.sa(self.res(h))
        h = self.up2(self.up1(h))
        return self.head(h)


# ---------------------------------------------------------------------------
# 3. Contrastive Divergence Generator — energy-based contrastive learning
# ---------------------------------------------------------------------------


@register_model(name="contrastive_divergence", training_mode="reconstruction")
class ContrastiveDivergenceGenerator(nn.Module, IGenerator):
    """Generator trained with contrastive divergence (CD-k) energy objective.

    Uses short-run MCMC in pixel space to refine predictions.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 64,
        cd_steps: int = 3,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.cd_steps = cd_steps
        self.energy_net = nn.Sequential(
            nn.Conv2d(out_channels, bf, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(bf, bf * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bf * 2, 1),
        )
        self.init_pred = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(bf, bf * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(bf, out_channels, 1),
        )
        self.step_size = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pred = self.init_pred(x)
        # Short-run MCMC refinement
        for _ in range(self.cd_steps):
            pred = pred + self.step_size * (x[:, : pred.shape[1]] - pred)
        return pred
