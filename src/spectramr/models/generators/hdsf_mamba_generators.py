"""
Hierarchical Dilated Space-Filling Architecture (HDSF-Mamba) Generative Models.

Implements a purely linear $O(N)$ architecture explicitly bypassing the "Local Meander Trap"
of continuous space-filling curves within volumetric State Space Models. Decouples state space
into:
1. Micro-scale local sweep (Windowed Hilbert 3D)
2. Macro-scale global sweep (Dilated Morton 3D)
3. Cross-seam stream (Axis-Permuted Hilbert 3D)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.grad_checkpointing import NestedGradCheckpointingMixin
from spectramr.models.blocks.topology_linearizer import (
    ImageTopologyLinearizer,
    resolution_agnostic_mode,
)
from spectramr.models.constants import DEFAULT_DIFFUSION_TIMESTEPS
from spectramr.models.generators.hilbert_mamba_generators import (
    _MambaEncoder,
)
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class HDSFMambaEncoder(nn.Module):
    """Hierarchical Dilated Space-Filling Mamba Encoder.

    Implements a 3-stream parallel SSM forward pass targeting local windowed,
    cross-seam permuted, and globally dilated configurations simultaneously.
    """

    def __init__(
        self, d_model: int, num_layers: int, d_state: int, block_size: int = 16, dilation: int = 4
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.dilation = dilation

        self.local_mamba = _MambaEncoder(d_model, num_layers, d_state)
        self.global_mamba = _MambaEncoder(d_model, num_layers, d_state)
        self.seam_mamba = _MambaEncoder(d_model, num_layers, d_state)

        self.fusion = nn.Conv3d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        # Dictionary to store dynamically instantiated continuous-topology linearizers
        # (they cache to buffers on their first pass avoiding any runtime cost)
        self._linearizers = nn.ModuleDict()

    def _get_lin(
        self, shape: tuple[int, ...], mode: str, device: torch.device
    ) -> ImageTopologyLinearizer:
        # Route strict hilbert modes through the resolution-agnostic *_rect
        # sibling so a non-cubic dynamic sub-grid (e.g. the dilated global
        # stream) cannot AssertionError; byte-identical on cubic pow2 blocks.
        mode = resolution_agnostic_mode(mode)
        key = f"{mode}_{shape[0]}_{shape[1]}_{shape[2]}"
        if key not in self._linearizers:
            lin = ImageTopologyLinearizer(shape, mode=mode).to(device)
            self._linearizers[key] = lin
        return self._linearizers[key]

    def _power_of_2_pad(self, x: int) -> int:
        """Finds next power of 2 for dimension to ensure Hilbert 3D constraints."""
        import math

        return 2 ** math.ceil(math.log2(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process 3D volumetric tensor through HDSF.

        Args:
            x: [B, C, D, H, W]

        forward method for HDSFMambaEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, D, H, W = x.shape
        # Enforce cubic block requirements for Hilbert3D (D,H,W must be equal power of 2 locally)
        # To handle any input, pad to multiple of block_size
        pad_d = (self.block_size - D % self.block_size) % self.block_size
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        else:
            x_padded = x

        B_p, C_p, D_p, H_p, W_p = x_padded.shape
        b = self.block_size

        num_d = D_p // b
        num_h = H_p // b
        num_w = W_p // b

        # Stream A: Micro-Scale Local Sweep (Windowed Hilbert)
        # Reshape into [N, C, b, b, b] blocks
        blocks = x_padded.view(B_p, C_p, num_d, b, num_h, b, num_w, b)
        blocks = blocks.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, C_p, b, b, b)

        lin_local = self._get_lin((b, b, b), "hilbert_3d", x.device)
        seq_local = lin_local(blocks).permute(0, 2, 1)  # [N, L, C]
        out_local = self.local_mamba(seq_local).permute(0, 2, 1)  # [N, C, L]
        out_blocks_local = lin_local.reverse(out_local)  # [N, C, b, b, b]

        out_blocks_local = out_blocks_local.view(B_p, num_d, num_h, num_w, C_p, b, b, b)
        out_local_spatial = out_blocks_local.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            B_p, C_p, D_p, H_p, W_p
        )

        # Stream C: Cross-Seam Sweep (Axis-Permuted Hilbert)
        # Permute coordinates locally (X,Y,Z -> Y,Z,X => W,H,D for torch shape D,H,W)
        blocks_permuted = blocks.permute(0, 1, 3, 4, 2)
        seq_seam = lin_local(blocks_permuted).permute(0, 2, 1)
        out_seam = self.seam_mamba(seq_seam).permute(0, 2, 1)
        out_blocks_seam_permuted = lin_local.reverse(out_seam)
        out_blocks_seam = out_blocks_seam_permuted.permute(0, 1, 4, 2, 3)  # Unpermute

        out_blocks_seam = out_blocks_seam.view(B_p, num_d, num_h, num_w, C_p, b, b, b)
        out_seam_spatial = out_blocks_seam.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            B_p, C_p, D_p, H_p, W_p
        )

        # Stream B: Macro-Scale Global Sweep (Dilated Morton)
        s = min(self.dilation, D_p, H_p, W_p)
        x_global = x_padded[:, :, ::s, ::s, ::s]

        # Ensure the spatial subgrid satisfies powers for strict indexing linearity bounds if needed
        lin_global = self._get_lin(x_global.shape[2:], "morton_3d", x.device)

        seq_global = lin_global(x_global).permute(0, 2, 1)
        out_global = self.global_mamba(seq_global).permute(0, 2, 1)
        out_global_spatial = lin_global.reverse(out_global)

        # Inverse mapping natively maps back coordinates iteratively through the inverse_idx lookup.
        # We trilinear upsample this global branch to the native padded resolution
        out_global_spatial = F.interpolate(
            out_global_spatial, size=(D_p, H_p, W_p), mode="trilinear", align_corners=False
        )

        # Topological Fusion
        fused = out_local_spatial + out_seam_spatial + out_global_spatial
        out_spatial = self.fusion(fused)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            out_spatial = out_spatial[:, :, :D, :H, :W]

        return out_spatial


@register_model(
    name="hdsf_mamba_unet",
    training_mode="reconstruction",
    # Hierarchical Dilated Space-Filling Mamba U-Net — explicitly a
    # "deep volumetric 3D U-Net" per the docstring.
    spatial_dims=(3,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class HDSFMambaUNet(nn.Module):
    """Hierarchical Dilated Space-Filling Mamba U-Net.

    A deep volumetric 3D U-Net explicitly optimized for volumetric
    sequences using partitioned Space Filling State Space blocks.
    Ensures O(N) linear compute scale mapped onto 1D streams locally and globally.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        block_size: int = 16,
        dilation: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        # Use 3D variants since we operate on volumes
        self.enc_convs = nn.ModuleList()
        self.enc_mambas = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for feat in features:
            self.enc_convs.append(
                nn.Sequential(
                    nn.Conv3d(prev_ch, feat, 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, feat), feat),
                    nn.Conv3d(feat, feat, 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, feat), feat),
                )
            )
            # Use HDSFMambaEncoder instead of _MambaEncoder
            self.enc_mambas.append(
                HDSFMambaEncoder(feat, num_mamba_layers, d_state, block_size, dilation)
            )
            self.pools.append(nn.MaxPool3d(2))
            prev_ch = feat

        # Bottleneck
        self.bottleneck_mamba = HDSFMambaEncoder(
            features[-1], num_mamba_layers * 2, d_state, block_size, dilation
        )

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_mambas = nn.ModuleList()

        for i in range(len(features) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose3d(features[i], features[i - 1], 2, stride=2))
            self.dec_convs.append(
                nn.Sequential(
                    nn.Conv3d(features[i - 1] * 2, features[i - 1], 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, features[i - 1]), features[i - 1]),
                    nn.Conv3d(features[i - 1], features[i - 1], 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, features[i - 1]), features[i - 1]),
                )
            )
            self.dec_mambas.append(
                HDSFMambaEncoder(features[i - 1], num_mamba_layers, d_state, block_size, dilation)
            )

        self.final_up = nn.ConvTranspose3d(features[0], features[0], 2, stride=2)

        self.head = nn.Sequential(
            nn.Conv3d(features[0], features[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(features[0], out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward 3D pass.
        Note: If input is 4D (no D), we add an artificial slice dim.

        forward method for HDSFMambaUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        is_4d = x.ndim == 4
        if is_4d:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]

        skips = []
        h = x
        for conv, mamba, pool in zip(self.enc_convs, self.enc_mambas, self.pools, strict=False):
            h = conv(h)
            h = h + mamba(h)
            skips.append(h)
            if h.shape[2] == 1:
                h = F.max_pool3d(h, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                h = pool(h)

        h = h + self.bottleneck_mamba(h)

        for up, conv, mamba, skip in zip(
            self.ups,
            self.dec_convs,
            self.dec_mambas,
            reversed(skips[:-1]),
            strict=False,
        ):
            h = up(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = conv(h)
            h = h + mamba(h)

        h = self.final_up(h)
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode="trilinear", align_corners=False)

        out = self.head(h)

        # Residual branch anchor directly into base intensity fields
        out = out + x[:, : out.shape[1]]

        if is_4d:
            out = out.squeeze(2)

        return out


@register_model(
    name="hdsf_hld_mamba",
    training_mode="diffusion",
    # Hierarchical Dilated SF Latent Diffusion Mamba — operates on
    # volumetric latents (3D structure prior).
    spatial_dims=(3,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class HDSFHLDMamba(NestedGradCheckpointingMixin, nn.Module):
    """Hierarchical Dilated Space-Filling Latent Diffusion Mamba.

    Operates as the global structural denoiser natively mapping out complex latent
    fields and noise parameter outputs over completely unrolled state sequences.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        d_model: int = 128,
        num_layers: int = 6,
        d_state: int = 16,
        time_embed_dim: int = 256,
        block_size: int = 16,
        dilation: int = 4,
        num_timesteps: int = DEFAULT_DIFFUSION_TIMESTEPS,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # The builder injects training.diffusion.timesteps, the range t is drawn from (#2442).
        self.num_timesteps = int(num_timesteps)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(min(8, d_model), d_model),
        )

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, d_model),
        )

        self.encoder = HDSFMambaEncoder(d_model, num_layers, d_state, block_size, dilation)

        self.head = nn.Sequential(
            nn.Conv3d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(d_model, out_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """forward method for HDSFHLDMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        is_4d = x.ndim == 4
        if is_4d:
            x = x.unsqueeze(2)

        feat = self.stem(x)  # [B, d_model, D, H, W]

        # Inject timestep embedding (spatial broadcast)
        if timesteps is not None:
            t_norm = timesteps.float() / self.num_timesteps
            t_emb = self.time_mlp(t_norm.unsqueeze(-1))  # [B, d_model]
            t_emb = t_emb.view(*t_emb.shape, 1, 1, 1)
            feat = feat + t_emb

        encoded = feat + self.encoder(feat)
        out = self.head(encoded)

        if is_4d:
            out = out.squeeze(2)

        return out
