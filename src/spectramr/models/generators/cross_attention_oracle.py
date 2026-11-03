"""Method A: Cross-Attention Oracle UNet.

Extracts the Point Spread Function (PSF) from the marker residual ΔM and
uses it as Keys/Values in a cross-attention bottleneck. The corrupted anatomy
features are the Queries — the network explicitly asks the marker
"how exactly were you distorted?" and receives the exact inverse filter.

Architecture:
    Corrupted Joint Image → UNet Encoder → [Bottleneck Cross-Attention] → Decoder
                                                 ↑
    Marker Residual ΔM → Oracle Encoder → Degradation Token (K, V)

Reference:
    Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class MarkerCrossAttention(nn.Module):
    """Cross-attention layer that extracts physics from marker residual.

    The marker residual ΔM = (corrupted_marker - ideal_marker) is a pure
    representation of the PSF and noise floor. This module encodes ΔM into
    a degradation token and uses it to condition the anatomy reconstruction.

    Args:
        embed_dim: Embedding dimension for attention.
        num_heads: Number of attention heads.
        oracle_channels: Input channels for the oracle encoder.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        oracle_channels: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Oracle encoder: ΔM → degradation token
        self.oracle_encoder = nn.Sequential(
            nn.Conv2d(oracle_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(16, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(32, 128),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, embed_dim),
        )

        # Cross-attention: anatomy queries × degradation keys/values
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

        # Projection for anatomy features → query tokens
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Layer norm for pre-attention normalization
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(
        self,
        anatomy_features: torch.Tensor,
        marker_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention conditioning from marker residual.

        Args:
            anatomy_features: Bottleneck features [B, D] or [B, N, D].
            marker_residual: ΔM = corrupted_marker - ideal_marker [B, 2, H, W].

        Returns:
            Conditioned features [B, N, D].

        forward method for MarkerCrossAttention.

        Executes PyTorch tensor operations.

        Args:
            anatomy_features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_residual (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode degradation from marker residual
        degradation_token = self.oracle_encoder(marker_residual)  # [B, D]
        degradation_token = degradation_token.unsqueeze(1)  # [B, 1, D]

        # Ensure anatomy features have sequence dimension
        if anatomy_features.dim() == 2:
            anatomy_features = anatomy_features.unsqueeze(1)  # [B, 1, D]

        # Pre-norm cross-attention
        Q = self.norm_q(self.query_proj(anatomy_features))
        KV = self.norm_kv(degradation_token)

        # Cross-attend: anatomy asks marker "how were you distorted?"
        attn_out, _ = self.cross_attn(query=Q, key=KV, value=KV)

        # Residual connection
        output = anatomy_features + self.output_proj(attn_out)

        return output


@register_model(
    name="cross_attention_oracle_unet",
    training_mode="reconstruction",
    # Recon backbone fed an IMAGE tensor (the strategy IFFTs k-space first; arms
    # rss-combine to magnitude image) — image-domain. See the VF DC-blob fix
    # (KNOWN_IMAGE) in docs; misclassifying it k-space caused the IFFT'd blob.
    spatial_dims=(2,),
    # 1-ch magnitude image or 2-ch real/imag (complex_image) — both are the
    # post-IFFT image-space representation the backbone consumes.
    input_domain=("image", "complex_image"),
    output_domain=("image", "complex_image"),
)
class CrossAttentionOracleUNet(nn.Module):
    """UNet with cross-attention bottleneck for marker-guided reconstruction.

    The encoder-decoder processes the corrupted anatomy while the
    MarkerCrossAttention at the bottleneck injects degradation knowledge
    extracted from the Virtual Fiducial marker.

    Args:
        in_channels: Input channels (2 for real/imag).
        out_channels: Output channels (2 for real/imag).
        features: Feature dimensions per encoder level.
        embed_dim: Cross-attention embedding dimension.
        num_heads: Number of attention heads.
        oracle_channels: Channels for the marker residual (2 = real/imag).
        use_marker: ``False`` builds no cross-attention branch, so the network is the
            same encoder-decoder run blind: the marker-off control (pitfall #17).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] | None = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        oracle_channels: int = 2,
        use_marker: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if not isinstance(use_marker, bool):
            raise TypeError(f"use_marker must be a bool, got {use_marker!r}.")
        self.use_marker = use_marker

        if features is None:
            features = [32, 64, 128, 256]

        self.features = features
        self.embed_dim = embed_dim

        # ── Encoder ──
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.encoders.append(self._conv_block(prev_ch, feat))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = feat

        # ── Bottleneck ──
        bottleneck_ch = features[-1] * 2
        self.bottleneck = self._conv_block(features[-1], bottleneck_ch)

        # Marker branch. Off, it is not built: the VF strategy then sees no
        # ``cross_attention`` attribute and passes no marker, and no parameter idles.
        if use_marker:
            self.to_tokens = nn.AdaptiveAvgPool2d((4, 4))
            self.token_proj = nn.Linear(bottleneck_ch * 16, embed_dim)
            self.cross_attention = MarkerCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                oracle_channels=oracle_channels,
            )
            self.from_tokens = nn.Linear(embed_dim, bottleneck_ch * 16)

        # ── Decoder ──
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 1, -1, -1):
            self.upconvs.append(
                nn.ConvTranspose2d(bottleneck_ch, features[i], kernel_size=2, stride=2)
            )
            self.decoders.append(self._conv_block(features[i] * 2, features[i]))
            bottleneck_ch = features[i]

        # ── Output ──
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # ── Latent head (for distillation) ──
        self.latent_dim = embed_dim
        self.latent_head = nn.Identity()  # Z_teacher = cross-attn output

        logger.info(
            "[CrossAttentionOracleUNet] in=%d, out=%d, features=%s, embed_dim=%d, params=%d",
            in_channels,
            out_channels,
            features,
            embed_dim,
            sum(p.numel() for p in self.parameters()),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        marker_residual: torch.Tensor | None = None,
        return_latent: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional marker conditioning.

        Args:
            x: Corrupted input [B, C, H, W].
            marker_residual: ΔM = corrupted_marker - ideal_marker [B, 2, H, W].
                If None, cross-attention is skipped (blind mode).
            return_latent: If True, also return the latent Z for distillation.
            **kwargs: Additional kwargs (ignored).

        Returns:
            Reconstruction [B, out_channels, H, W].
            If return_latent: (reconstruction, Z_latent).

        forward method for CrossAttentionOracleUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_residual (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            return_latent (bool): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # ── Encoder ──
        skip_connections = []
        h = x
        for encoder, pool in zip(self.encoders, self.pools, strict=False):
            h = encoder(h)
            skip_connections.append(h)
            h = pool(h)

        # ── Bottleneck ──
        h = self.bottleneck(h)
        B, C_b, H_b, W_b = h.shape

        # ── Cross-Attention (if marker available) ──
        if marker_residual is not None and self.use_marker:
            # Spatial → tokens
            tokens = self.to_tokens(h)  # [B, C_b, 4, 4]
            tokens = tokens.flatten(1)  # [B, C_b * 16]
            tokens = self.token_proj(tokens)  # [B, embed_dim]

            # Cross-attend with degradation
            conditioned = self.cross_attention(tokens, marker_residual)

            # Extract latent for distillation
            z_latent = conditioned.squeeze(1)  # [B, embed_dim]

            # Tokens → spatial
            h_recon = self.from_tokens(z_latent)  # [B, C_b * 16]
            h_recon = h_recon.view(B, C_b, 4, 4)

            # Upsample back to bottleneck spatial size
            h = h + F.interpolate(h_recon, size=(H_b, W_b), mode="bilinear", align_corners=False)
        else:
            z_latent = None

        # ── Decoder ──
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders, strict=False)):
            h = upconv(h)
            skip = skip_connections[-(i + 1)]

            # Handle size mismatch
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.output_conv(h)

        if return_latent and z_latent is not None:
            return output, z_latent
        return output


__all__ = ["CrossAttentionOracleUNet", "MarkerCrossAttention"]
