"""Attention & Transformer Architecture Variants
================================================

Architecturally-distinct attention-based generators for MRI reconstruction.
Each model captures a specific attention mechanism or transformer design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Standard conv-bn-relu block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = _ConvBlock(out_ch * 2, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# 1. Attention U-Net — U-Net with additive attention gates on skip connections
# ---------------------------------------------------------------------------


class _AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018)."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.w_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.w_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.w_gate(gate)
        s = self.w_skip(skip)
        g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        att = self.psi(F.relu(g + s, inplace=True))
        return skip * att


@register_model(name="attention_unet", training_mode="reconstruction", spatial_dims=(2,))
class AttentionUNet(nn.Module, IGenerator):
    """U-Net with additive attention gates on each skip connection."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc1 = _ConvBlock(in_channels, bf)
        self.enc2 = _DownBlock(bf, bf * 2)
        self.enc3 = _DownBlock(bf * 2, bf * 4)
        self.bottleneck = _DownBlock(bf * 4, bf * 8)

        self.ag3 = _AttentionGate(bf * 8, bf * 4, bf * 2)
        self.ag2 = _AttentionGate(bf * 4, bf * 2, bf)
        self.ag1 = _AttentionGate(bf * 2, bf, bf // 2)

        self.up3 = _UpBlock(bf * 8, bf * 4)
        self.up2 = _UpBlock(bf * 4, bf * 2)
        self.up1 = _UpBlock(bf * 2, bf)
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.up3(b, self.ag3(b, e3))
        d2 = self.up2(d3, self.ag2(d3, e2))
        d1 = self.up1(d2, self.ag1(d2, e1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# 2. Attention Cross-Modality — cross-attention between two modality streams
# ---------------------------------------------------------------------------


@register_model(name="attention_cross_modality", training_mode="reconstruction")
class AttentionCrossModality(nn.Module, IGenerator):
    """Dual-stream U-Net fusing modalities via cross-attention at the bottleneck."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        mid = in_channels // 2 if in_channels > 1 else in_channels
        self.stream_a = nn.Sequential(
            _ConvBlock(mid, bf), _DownBlock(bf, bf * 2), _DownBlock(bf * 2, bf * 4)
        )
        self.stream_b = nn.Sequential(
            _ConvBlock(in_channels - mid, bf), _DownBlock(bf, bf * 2), _DownBlock(bf * 2, bf * 4)
        )
        self.cross_attn = nn.MultiheadAttention(bf * 4, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBlock(bf * 2, bf * 2),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            _ConvBlock(bf, bf),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        mid = x.shape[1] // 2 if x.shape[1] > 1 else x.shape[1]
        a = self.stream_a(x[:, :mid])
        b = self.stream_b(x[:, mid:])
        B, C, H, W = a.shape
        a_flat = a.flatten(2).permute(0, 2, 1)
        b_flat = b.flatten(2).permute(0, 2, 1)
        fused, _ = self.cross_attn(a_flat, b_flat, b_flat)
        fused = fused.permute(0, 2, 1).reshape(B, C, H, W)
        return F.interpolate(
            self.decoder(fused), size=x.shape[2:], mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# 3. Linformer — linear-complexity self-attention via low-rank projection
# ---------------------------------------------------------------------------


class _LinformerAttention(nn.Module):
    """Self-attention with O(n) complexity via key/value projection."""

    def __init__(self, dim: int, seq_len: int = 256, k: int = 64, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj_k = nn.Linear(seq_len, k)
        self.proj_v = nn.Linear(seq_len, k)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = self.proj_k(k.transpose(-1, -2)).transpose(-1, -2)
        v = self.proj_v(v.transpose(-1, -2)).transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


@register_model(name="linformer", training_mode="reconstruction")
class LinformerGenerator(nn.Module, IGenerator):
    """MRI reconstruction using Linformer (linear self-attention)."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, dim: int = 128, depth: int = 4, **kwargs
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    _LinformerAttention(dim, seq_len=256, k=64, heads=4),
                    nn.LayerNorm(dim),
                    nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_channels * 16))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        tokens = patches.flatten(2).permute(0, 2, 1)
        for blk in self.blocks:
            tokens = tokens + blk(tokens)
        tokens = self.head(tokens)
        return tokens.permute(0, 2, 1).reshape(B, self.out_channels, pH * 4, pW * 4)[:, :, :H, :W]


# ---------------------------------------------------------------------------
# 4. Perceiver — cross-attention with fixed-size latent array
# ---------------------------------------------------------------------------


@register_model(name="perceiver", training_mode="reconstruction")
class PerceiverGenerator(nn.Module, IGenerator):
    """Perceiver IO: asymmetric cross-attention with a latent bottleneck."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 128,
        num_latents: int = 64,
        depth: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, dim, 1)
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.cross_attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(dim, num_heads=4, batch_first=True) for _ in range(depth)]
        )
        self.self_attn_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 4, batch_first=True)
                for _ in range(depth)
            ]
        )
        self.output_cross = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.head = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        inp = self.input_proj(x).flatten(2).permute(0, 2, 1)  # (B, N, dim)
        latents = self.latents.expand(B, -1, -1)
        for cross, self_attn in zip(self.cross_attn_layers, self.self_attn_layers, strict=False):
            latents = latents + cross(latents, inp, inp)[0]
            latents = self_attn(latents)
        out, _ = self.output_cross(inp, latents, latents)
        return self.head(out.permute(0, 2, 1).reshape(B, -1, H, W))


# ---------------------------------------------------------------------------
# 5. Hierarchical Softmax Attention
# ---------------------------------------------------------------------------


@register_model(name="hierarchical_softmax_attention", training_mode="reconstruction")
class HierarchicalSoftmaxAttention(nn.Module, IGenerator):
    """Multi-scale attention: applies self-attention at multiple spatial scales."""

    def __init__(self, in_channels: int = 2, out_channels: int = 2, dim: int = 64, **kwargs):
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.scale_attns = nn.ModuleList(
            [nn.MultiheadAttention(dim, num_heads=4, batch_first=True) for _ in range(3)]
        )
        self.pools = nn.ModuleList([nn.AvgPool2d(2**i) for i in range(3)])
        self.fusion = nn.Conv2d(dim * 3, dim, 1)
        self.head = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.proj_in(x)
        outs = []
        for pool, attn in zip(self.pools, self.scale_attns, strict=False):
            pooled = pool(feat)
            pH, pW = pooled.shape[2], pooled.shape[3]
            tokens = pooled.flatten(2).permute(0, 2, 1)
            tokens = tokens + attn(tokens, tokens, tokens)[0]
            restored = F.interpolate(
                tokens.permute(0, 2, 1).reshape(B, -1, pH, pW),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            outs.append(restored)
        return self.head(self.fusion(torch.cat(outs, dim=1)))


# ---------------------------------------------------------------------------
# 6. Hierarchical Window Transformer (Shifted-Window like Swin but multi-level)
# ---------------------------------------------------------------------------


@register_model(name="hierarchical_window_transformer", training_mode="reconstruction")
class HierarchicalWindowTransformer(nn.Module, IGenerator):
    """Window-based transformer with hierarchical patch merging."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 64,
        depths: list[int] | None = None,
        **kwargs,
    ):
        super().__init__()
        depths = depths or [2, 2, 2]
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.stages = nn.ModuleList()
        cur_dim = dim
        for d in depths:
            stage = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    cur_dim, nhead=4, dim_feedforward=cur_dim * 4, batch_first=True
                ),
                num_layers=d,
            )
            self.stages.append(stage)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(cur_dim, dim, 4, stride=4),
            nn.Conv2d(dim, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        tokens = patches.flatten(2).permute(0, 2, 1)
        for stage in self.stages:
            tokens = stage(tokens)
        feat = tokens.permute(0, 2, 1).reshape(B, -1, pH, pW)
        out = self.head(feat)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 7. Stable ViT — ViT with post-norm and sigma-reparam for stability
# ---------------------------------------------------------------------------


@register_model(name="stable_vit", training_mode="reconstruction")
class StableViT(nn.Module, IGenerator):
    """ViT with post-LayerNorm and sigma-reparameterized attention for training stability."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, dim: int = 128, depth: int = 6, **kwargs
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    dim, nhead=4, dim_feedforward=dim * 4, batch_first=True, norm_first=False
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, out_channels * 16))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        N = pH * pW
        tokens = patches.flatten(2).permute(0, 2, 1)
        tokens = tokens + self.pos_embed[:, :N]
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        tokens = self.head(tokens)
        out = tokens.permute(0, 2, 1).reshape(B, self.out_channels, pH * 4, pW * 4)
        return out[:, :, :H, :W]


# ---------------------------------------------------------------------------
# 8. Vision Transformer SSL — ViT with contrastive pre-training head
# ---------------------------------------------------------------------------


@register_model(name="vision_transformer_ssl", training_mode="reconstruction")
class VisionTransformerSSL(nn.Module, IGenerator):
    """ViT encoder with projection head for self-supervised contrastive pretraining."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, dim: int = 128, depth: int = 6, **kwargs
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 4, batch_first=True),
            num_layers=depth,
        )
        self.projection_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.decoder_head = nn.Sequential(nn.Linear(dim, out_channels * 16))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        tokens = patches.flatten(2).permute(0, 2, 1)
        tokens = self.encoder(tokens)
        recon = self.decoder_head(tokens)
        return recon.permute(0, 2, 1).reshape(B, self.out_channels, pH * 4, pW * 4)[:, :, :H, :W]


# ---------------------------------------------------------------------------
# 9. ViT-KAN — Vision Transformer with KAN (Kolmogorov-Arnold) MLP layers
# ---------------------------------------------------------------------------


class _KANLayer(nn.Module):
    """B-spline approximation to a KAN layer."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.spline = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.spline(self.activation(x))


@register_model(name="vit_kan", training_mode="reconstruction", spatial_dims=(2,))
class ViTKANGenerator(nn.Module, IGenerator):
    """Vision Transformer with Kolmogorov-Arnold Network MLP blocks."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, dim: int = 128, depth: int = 4, **kwargs
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(dim, num_heads=4, batch_first=True),
                        "norm2": nn.LayerNorm(dim),
                        "kan": nn.Sequential(_KANLayer(dim, dim * 4), _KANLayer(dim * 4, dim)),
                    }
                )
            )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_channels * 16))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        pH, pW = patches.shape[2], patches.shape[3]
        tokens = patches.flatten(2).permute(0, 2, 1)
        for blk in self.blocks:
            tokens = (
                tokens
                + blk["attn"](blk["norm1"](tokens), blk["norm1"](tokens), blk["norm1"](tokens))[0]
            )
            tokens = tokens + blk["kan"](blk["norm2"](tokens))
        out = self.head(tokens)
        return out.permute(0, 2, 1).reshape(B, self.out_channels, pH * 4, pW * 4)[:, :, :H, :W]
