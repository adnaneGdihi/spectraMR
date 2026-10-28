r"""Mixture-of-experts VAE (``moe_vae``).

A Gaussian VAE encoder feeds a top-1 router
(:class:`spectramr.models.blocks.moe.top1_router.Top1Router`); the selected
expert decoder reconstructs the image. The load-balance auxiliary loss
prevents the gate from collapsing onto one expert. Following Shazeer
*et al.* [1] the routing gradient is propagated through the top-1 gate
probability via a straight-through scaling.

The router reads the latent and ``forward`` takes only the image, so any
per-contrast specialisation is emergent -- the model is never told which
contrast it is seeing. It therefore does not declare
``supports_contrast_conditioning`` (#1938).

Reference:
    [1] N. Shazeer *et al.*, "Outrageously large neural networks: the
        sparsely-gated mixture-of-experts layer," ICLR 2017.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.blocks.moe.top1_router import Top1Router
from spectramr.models.registry import register_model


class _ExpertDecoder(nn.Module):
    """One expert: latent -> image, mirroring the shared trunk."""

    def __init__(self, latent_dim: int, hidden: int, pooled_size: int, out_channels: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.pooled_size = pooled_size
        self.fc = nn.Linear(latent_dim, hidden * pooled_size * pooled_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden // 2, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, out_channels, 3, 1, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(-1, self.hidden, self.pooled_size, self.pooled_size)
        return self.net(h)


@register_model(
    name="moe_vae",
    training_mode="vae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class MoEVAE(nn.Module):
    r"""Mixture-of-experts VAE with top-1 routing.

    Args:
        in_channels: Input image channels.
        out_channels: Reconstruction channels.
        num_experts: Number of expert decoders ``K`` (``>= 2``).
        latent_dim: Gaussian latent width.
        hidden: Trunk / expert channel width.
        pooled_size: Spatial size pooled to before the latent heads.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_experts: int = 4,
        latent_dim: int = 64,
        hidden: int = 128,
        pooled_size: int = 4,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError(f"moe_vae requires num_experts >= 2, got {num_experts}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.pooled_size = pooled_size
        feature_dim = hidden * pooled_size * pooled_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, 4, 2, 1),
            nn.GroupNorm(8, hidden // 2),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, hidden, 4, 2, 1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)

        self.router = Top1Router(latent_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                _ExpertDecoder(latent_dim, hidden, pooled_size, out_channels)
                for _ in range(num_experts)
            ]
        )

    @property
    def name(self) -> str:
        return "moe_vae"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.pool(self.encoder(x))
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    @staticmethod
    def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
        """KL( N(mu, var) || N(0, I) ), batch-mean scalar."""
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Returns a dict consumed by ``VAETrainingStrategy``:
            ``reconstruction`` — ``x_hat``;
            ``kl_override`` — the standard-normal KL (batch-mean scalar);
            ``aux_loss`` — the load-balance term (>= 1.0 at uniform load).

        The Gaussian latent here *does* have a (mu, logvar), but we route KL
        via ``kl_override`` so the load-balance ``aux_loss`` travels with it
        through the same additive hook.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        k_star, gate_val, gate_probs = self.router(z)

        out_hw = x.shape[-2:]
        x_hat = x.new_zeros(x.shape[0], self.out_channels, *out_hw)
        for k in range(self.num_experts):
            mask = k_star == k
            if bool(mask.any()):
                dec = self.experts[k](z[mask])
                if dec.shape[-2:] != out_hw:
                    dec = F.interpolate(dec, size=out_hw, mode="bilinear", align_corners=False)
                x_hat[mask] = dec

        # Straight-through scaling so the routing gradient reaches the gate.
        gv = gate_val.view(-1, 1, 1, 1)
        x_hat = x_hat * gv + (x_hat * (1.0 - gv)).detach()

        kl = self.kl_standard_normal(mu, logvar)
        aux_loss = self.router.load_balance_loss(k_star, gate_probs)
        # NB: deliberately omit mu/logvar so the strategy's Gaussian-KL path
        # contributes ~0 (zeros) and KL is counted exactly once via kl_override.
        return {
            "reconstruction": x_hat,
            "kl_override": kl,
            "aux_loss": aux_loss,
        }


__all__ = ["MoEVAE"]
