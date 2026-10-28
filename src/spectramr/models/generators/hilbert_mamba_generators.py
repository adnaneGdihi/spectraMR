"""Hilbert-Mamba Generators for ULF-to-HF MRI Reconstruction.

Ten novel generator architectures that combine Hilbert space-filling curve
linearization with Mamba (Structured State Space Models) for paired
Ultra-Low Field to High Field MRI super-resolution.

Key idea: A Hilbert curve maps 2D/3D grids into locality-preserving 1D
sequences.  Mamba processes these sequences in O(L) time vs O(L²) for
Transformers, enabling tractable volumetric processing.

Methods:
    1. CT-Mamba    — Continuous-time discretization resampling
    2. FE-Mamba    — Fractal expansion Kronecker upsampling
    3. CRM-Mamba   — Cross-resolution modulated SSM
    4. HW-Mamba    — Hilbert-wavelet spectral detail prediction
    5. MM-Mamba    — Macro-micro nested hierarchical processing
    6. MDI-Mamba   — Multi-directional isotropic fusion
    7. 2.5D-Mamba  — Orthogonal slice-to-volume cross-plane
    8. HLD-Mamba   — Hilbert latent diffusion noise predictor
    9. INR-Mamba   — Implicit neural representation continuous SR
   10. Hilbert-Mamba U-Net — Combined baseline (FE + MDI)

Reference:
    Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces." arXiv:2312.00752.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.mamba_block import MambaBlock
from spectramr.models.blocks.topology_linearizer import (
    ImageTopologyLinearizer,
    resolution_agnostic_mode,
)
from spectramr.models.constants import DEFAULT_DIFFUSION_TIMESTEPS
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


def _squeeze_volume_to_2d(
    x: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    """Fold 5D ``[B, C, H, W, D]`` volumes into 4D ``[B*D, C, H, W]``.

    TorchIO validation returns full volumes (depth last) instead of 2D
    patches, and the Conv2d stems take 4D input. Every slice is scored by
    the 2D generator, so depth folds into the batch. A trailing ``1`` (a
    TorchIO 2D patch) is squeezed as before.

    Returns:
        Tuple of (4D tensor, original shape or ``None`` if no-op).
    """
    if x.ndim == 5 and x.shape[-1] == 1:
        return x.squeeze(-1), x.shape
    if x.ndim == 5:
        b, c, h, w, d = x.shape
        return x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w), x.shape
    return x, None


def _unsqueeze_to_volume(out: torch.Tensor, original_shape: tuple[int, ...] | None) -> torch.Tensor:
    """Unfold ``[B*D, C, H, W]`` back to ``[B, C, H, W, D]``; inverse of the fold."""
    if original_shape is None:
        return out
    if original_shape[-1] == 1:
        return out.unsqueeze(-1)
    b, _, _, _, d = original_shape
    _, c_out, h, w = out.shape
    return out.reshape(b, d, c_out, h, w).permute(0, 2, 3, 4, 1)


# ═══════════════════════════════════════════════════════════════════════
# Shared building blocks
# ═══════════════════════════════════════════════════════════════════════


class _HilbertLinearize(nn.Module):
    """Lazy Hilbert linearizer that builds its permutation on first call.

    Supports both 2D and 3D modes and caches per spatial resolution.
    The ``mode`` parameter controls which linearization scheme is used,
    allowing the user to select among all modes registered in
    ``ImageTopologyLinearizer`` (hilbert_2d, hilbert_3d, morton_2d, etc.).
    """

    def __init__(self, mode: str = "hilbert_2d") -> None:
        super().__init__()
        self.mode = mode
        self._linearizer: ImageTopologyLinearizer | None = None
        self._cached_shape: tuple[int, ...] | None = None

    def _get_linearizer(
        self, spatial_shape: tuple[int, ...], device: torch.device
    ) -> ImageTopologyLinearizer:
        if self._cached_shape != spatial_shape:
            self._linearizer = ImageTopologyLinearizer(
                spatial_shape, mode=resolution_agnostic_mode(self.mode)
            )
            self._linearizer = self._linearizer.to(device)
            self._cached_shape = spatial_shape
        return self._linearizer  # type: ignore[return-value]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten spatial dims into Hilbert-ordered 1D sequence.

        Args:
            x: ``[B, C, H, W]`` or ``[B, C, D, H, W]``.

        Returns:
            Tuple of (sequence ``[B, C, L]``, original spatial shape).

        forward method for _HilbertLinearize.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, tuple[int, ...]]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        spatial = x.shape[2:]
        lin = self._get_linearizer(spatial, x.device)
        return lin(x), spatial

    def reverse(self, seq: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
        """Restore spatial dims from Hilbert-ordered 1D sequence."""
        lin = self._get_linearizer(spatial_shape, seq.device)
        return lin.reverse(seq)


class _MambaEncoder(nn.Module):
    """Stack of Mamba blocks operating on [B, L, D] sequences."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process [B, L, D] sequence through Mamba stack with residuals.

        forward method for _MambaEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)


class _ConvStem(nn.Module):
    """Patch-embed or channel projection for Hilbert-Mamba generators."""

    def __init__(self, in_ch: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(min(8, d_model), d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ConvStem.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.proj(x)


class _ConvHead(nn.Module):
    """Output head: project back to target channels."""

    def __init__(self, d_model: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _ConvHead.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════
# Method 5: MM-Mamba — Macro-Micro Nested
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="mm_mamba",
    training_mode="reconstruction",
    # Macro-Micro Nested Mamba: explicitly designed for tractable 3D
    # volumetric processing. Doc: divides Hilbert curve into local
    # blocks. Accepts both 2D (degenerate single-block) and 3D.
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class MMMamba(nn.Module):
    """Macro-Micro Nested Mamba for tractable 3D volumetric processing.

    Divides the Hilbert curve into local blocks (micro-curves), processes
    each independently, then runs a macro-level Mamba over block summaries.
    Global context is fused back into local features for HF generation.

    Math:
        h_micro^(i) = MaxPool(Mamba_micro(P_i))
        Z_global = Mamba_macro([h_micro^(1), ..., h_micro^(K)])
        output_i = Mamba_dec(P_i + context_i)

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        micro_layers: Mamba blocks per micro encoder.
        macro_layers: Mamba blocks for macro encoder.
        block_size: Tokens per micro block.
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        micro_layers: int = 2,
        macro_layers: int = 2,
        block_size: int = 1024,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)

        self.micro_encoder = _MambaEncoder(d_model, micro_layers, d_state)
        self.macro_encoder = _MambaEncoder(d_model, macro_layers, d_state)

        # Global context injection
        self.context_proj = nn.Linear(d_model, d_model)
        self.decoder = _MambaEncoder(d_model, micro_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF → HF with macro-micro hierarchy.

        forward method for MMMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]
        B, L, D = seq.shape

        # Pad to multiple of block_size
        pad_len = (self.block_size - L % self.block_size) % self.block_size
        if pad_len > 0:
            seq = F.pad(seq, (0, 0, 0, pad_len))

        num_blocks = seq.shape[1] // self.block_size

        # Micro: process each block independently
        blocks = seq.reshape(B * num_blocks, self.block_size, D)
        micro_out = self.micro_encoder(blocks)  # [B*K, block, D]

        # Macro: pool each block → summary, run global Mamba
        block_summaries = micro_out.mean(dim=1)  # [B*K, D]
        block_summaries = block_summaries.reshape(B, num_blocks, D)
        global_ctx = self.macro_encoder(block_summaries)  # [B, K, D]

        # Fuse global context back into local features
        ctx_expanded = self.context_proj(global_ctx)  # [B, K, D]
        ctx_expanded = ctx_expanded.unsqueeze(2).expand(-1, -1, self.block_size, -1)
        ctx_expanded = ctx_expanded.reshape(B, -1, D)  # [B, K*block, D]

        fused = micro_out.reshape(B, -1, D) + ctx_expanded
        decoded = self.decoder(fused)

        # Remove padding and restore spatial
        decoded = decoded[:, :L, :].permute(0, 2, 1)
        out = self.linearize.reverse(decoded, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 6: MDI-Mamba — Multi-Directional Isotropic
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="mdi_mamba",
    training_mode="reconstruction",
    # Multi-Directional Isotropic Mamba: K parallel scan directions
    # (Hilbert / Morton / Snake / Zigzag — all 2D variants per the
    # DEFAULT_MODES). Acts as a multi-head Mamba via scan-direction
    # diversity. The 2D restriction is from the default mode list.
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class MDIMamba(nn.Module):
    """Multi-Directional Isotropic Mamba.

    Processes the input through K different Hilbert curve orientations
    (or other linearization modes), each with its own Mamba branch,
    then averages in spatial domain to compensate for boundary tears.

    Math:
        V̂_HF = (1/K) Σ_k H_k⁻¹(Mamba_k(H_k(V^ULF)))

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Mamba blocks per branch.
        num_directions: Number of scan directions (K).
        linearization_modes: List of linearization modes to use. If not
            provided, defaults to ``["hilbert_2d", "morton_2d",
            "snake_2d", "zigzag_2d"]`` (first K entries).
    """

    # Default modes ordered by locality quality
    DEFAULT_MODES = [
        "hilbert_2d",
        "morton_2d",
        "snake_2d",
        "zigzag_2d",
        "radial_inside_out",
        "radial_outside_in",
    ]

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        num_directions: int = 4,
        d_state: int = 16,
        linearization_modes: list[str] | None = None,
    ) -> None:
        super().__init__()
        # Resolve modes: user-specified or first K defaults
        if linearization_modes is not None:
            self.modes = linearization_modes[:num_directions]
        else:
            self.modes = self.DEFAULT_MODES[:num_directions]

        self.num_directions = len(self.modes)
        self.stem = _ConvStem(in_channels, d_model)

        # Per-direction Mamba branches + linearizers
        self.linearizers = nn.ModuleList([_HilbertLinearize(mode=m) for m in self.modes])
        self.branches = nn.ModuleList(
            [_MambaEncoder(d_model, num_layers, d_state) for _ in self.modes]
        )

        self.head = _ConvHead(d_model, out_channels)

        logger.info("[MDIMamba] modes=%s, d_model=%d", self.modes, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: multi-directional average in spatial domain.

        forward method for MDIMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        accum = torch.zeros_like(feat)

        for lin, branch in zip(self.linearizers, self.branches, strict=False):
            seq, shape = lin(feat)  # [B, D, L]
            seq = seq.permute(0, 2, 1)  # [B, L, D]
            processed = branch(seq)  # [B, L, D]
            processed = processed.permute(0, 2, 1)
            restored = lin.reverse(processed, shape)
            accum = accum + restored

        averaged = accum / self.num_directions
        out = self.head(averaged) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Method 7: 2.5D-Mamba — Orthogonal Slice-to-Volume
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="two_half_d_mamba",
    training_mode="reconstruction",
    # 2.5D: input is 3D volume; processes 2D in-plane slices then a
    # cross-plane Mamba pass. Accepts 3D input only (the 2.5D pipeline
    # requires a depth axis to fuse).
    spatial_dims=(3,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class TwoHalfDMamba(nn.Module):
    """2.5D Orthogonal Slice-to-Volume Mamba.

    Processes 2D slices in-plane with Hilbert-linearized Mamba, then
    runs a secondary Mamba cross-plane along the Z-axis to restore
    volumetric coherence.

    For 2D inputs (single slices), acts as a standard Hilbert-Mamba.
    For 3D inputs, the cross-plane pass enables inter-slice context.

    Math:
        f_z = Mamba_in_plane(H_2D(slice_z))  for each slice z
        y_{z,i} = Mamba_z_axis(F[:, i, :])   cross-plane at position i

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        in_plane_layers: Mamba blocks for in-plane processing.
        cross_plane_layers: Mamba blocks for Z-axis processing.
        linearization_mode: Linearization mode for 2D in-plane Hilbert.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        in_plane_layers: int = 4,
        cross_plane_layers: int = 2,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
    ) -> None:
        super().__init__()
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.in_plane_mamba = _MambaEncoder(d_model, in_plane_layers, d_state)
        self.cross_plane_mamba = _MambaEncoder(d_model, cross_plane_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: 2D or 3D input → enhanced output.

        For 2D: [B, C, H, W] → in-plane only.
        For 3D: [B, C, H, W, D] (TorchIO convention) → in-plane + cross-plane.

        Note: Unlike other Hilbert generators, TwoHalfDMamba natively
        supports 5D input so no squeeze is needed for the 3D path.
        The 5D path handles full volumes with cross-plane Mamba.

        forward method for TwoHalfDMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, H, W, D)):
                TorchIO convention with depth as the last axis. Earlier
                revisions of this file unpacked ``[B, C, D, H, W]`` and
                produced ``Hilbert 2D requires square power-of-2 dims,
                got (H, D)`` on rectangular volumes like the 2026-05-10
                ``exp_hm_07_25d_mamba`` smoke crash (``patch_size:
                [256, 256, 16]``).

        Returns:
            torch.Tensor: Output tensor in the same shape convention as the input.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        is_3d = x.dim() == 5

        if is_3d:
            # TorchIO yields [B, C, H, W, D] for volumetric patches.
            B, C, H, W, D = x.shape
            # Process each slice in-plane: fold depth into batch.
            x_2d = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
            feat = self.stem(x_2d)  # [B*D, d, H, W]
            seq, shape = self.linearize(feat)  # [B*D, d, L]
            seq = seq.permute(0, 2, 1)  # [B*D, L, d]
            in_plane = self.in_plane_mamba(seq)  # [B*D, L, d]

            # Reshape for cross-plane: [B, D, L, d]
            d_model = in_plane.shape[-1]
            L = in_plane.shape[1]
            cross_input = in_plane.reshape(B, D, L, d_model)

            # Cross-plane Mamba at each spatial position
            cross_input = cross_input.permute(0, 2, 1, 3)  # [B, L, D, d]
            cross_input = cross_input.reshape(B * L, D, d_model)
            cross_out = self.cross_plane_mamba(cross_input)  # [B*L, D, d]
            cross_out = cross_out.reshape(B, L, D, d_model)
            cross_out = cross_out.permute(0, 2, 1, 3)  # [B, D, L, d]
            cross_out = cross_out.reshape(B * D, L, d_model)

            # Restore spatial
            cross_out = cross_out.permute(0, 2, 1)  # [B*D, d, L]
            restored = self.linearize.reverse(cross_out, shape)  # [B*D, d, H, W]
            out = self.head(restored)  # [B*D, C_out, H, W]
            # Unfold back to TorchIO 5-D layout ``[B, C_out, H, W, D]``.
            return out.reshape(B, D, -1, H, W).permute(0, 2, 3, 4, 1)
        else:
            # Standard 2D: in-plane only
            feat = self.stem(x)
            seq, shape = self.linearize(feat)
            seq = seq.permute(0, 2, 1)
            processed = self.in_plane_mamba(seq)
            processed = processed.permute(0, 2, 1)
            out = self.linearize.reverse(processed, shape)
            return self.head(out) + x[:, : self.head.proj[-1].out_channels]


# ═══════════════════════════════════════════════════════════════════════
# Method 8 (Loss): Soft-DTW — see soft_dtw_loss.py
# The generator for Method 8 is HilbertMambaUNet with the DTW loss.
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Method 9: HLD-Mamba — Hilbert Latent Diffusion
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="hld_mamba",
    training_mode="diffusion",
    # Hilbert Latent Diffusion Mamba: 1D noise predictor in a latent
    # token sequence. The latent comes from a separately-trained VAE
    # and is itself spatial (2D), so spatial_dims is image-rank.
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class HLDMamba(nn.Module):
    """Hilbert Latent Diffusion Mamba — 1D noise predictor for diffusion.

    Replaces the heavy 2D/3D U-Net noise predictor in diffusion models
    with a 1D Mamba block operating on Hilbert-linearized sequences.
    The ULF input is concatenated with the noisy HF target for conditioning.

    Math:
        y_t = √ᾱ_t · y + √(1-ᾱ_t) · ε       # forward diffusion
        ε_θ = Mamba(y_t ∥ x↑ ∥ Emb(t))        # noise prediction

    Args:
        in_channels: Input channels (noisy HF + ULF conditioning).
        out_channels: Output channels (predicted noise, same as HF).
        d_model: Mamba model dimension.
        num_layers: Number of Mamba blocks.
        time_embed_dim: Timestep embedding dimension.
        linearization_mode: Linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        d_model: int = 128,
        num_layers: int = 6,
        d_state: int = 16,
        time_embed_dim: int = 256,
        linearization_mode: str = "hilbert_2d",
        num_timesteps: int = DEFAULT_DIFFUSION_TIMESTEPS,
    ) -> None:
        super().__init__()
        # The builder injects training.diffusion.timesteps, the range t is drawn from (#2442).
        self.num_timesteps = int(num_timesteps)
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, d_model),
        )

        self.encoder = _MambaEncoder(d_model, num_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict noise ε_θ from noisy input + timestep.

        Args:
            x: Noisy input [B, C, H, W] (concatenated noisy_hf + ulf).
            timesteps: Diffusion timesteps [B] (normalized 0→1).

        Returns:
            Predicted noise [B, C_out, H, W].

        forward method for HLDMamba.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]

        # Inject timestep embedding (normalize to [0,1] to prevent large activations)
        if timesteps is not None:
            t_norm = timesteps.float() / self.num_timesteps
            t_emb = self.time_mlp(t_norm.unsqueeze(-1))  # [B, D]
            seq = seq + t_emb.unsqueeze(1)  # broadcast across L

        decoded = self.encoder(seq)
        decoded = decoded.permute(0, 2, 1)
        out = self.linearize.reverse(decoded, shape)
        out = self.head(out)
        return _unsqueeze_to_volume(out, vol_shape)


# ═══════════════════════════════════════════════════════════════════════
# Combined Baseline: Hilbert-Mamba UNet
# ═══════════════════════════════════════════════════════════════════════


@register_model(
    name="hilbert_mamba_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class HilbertMambaUNet(nn.Module):
    """Hilbert-Mamba U-Net — encoder-decoder with skip connections.

    A full U-Net architecture where each encoder/decoder level uses
    Hilbert-linearized Mamba blocks.  Combines the core ideas of fractal
    linearization (Method 2) with multi-directional averaging (Method 6)
    in a single practical architecture.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        features: Feature dimensions per encoder level.
        num_mamba_layers: Mamba blocks per level.
        linearization_mode: Primary linearization mode for Hilbert curve.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.linearize = _HilbertLinearize(mode=linearization_mode)

        # Encoder
        self.enc_convs = nn.ModuleList()
        self.enc_mambas = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for feat in features:
            self.enc_convs.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, feat, 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, feat), feat),
                    nn.Conv2d(feat, feat, 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, feat), feat),
                )
            )
            self.enc_mambas.append(_MambaEncoder(feat, num_mamba_layers, d_state))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = feat

        # Bottleneck
        self.bottleneck_mamba = _MambaEncoder(features[-1], num_mamba_layers * 2, d_state)

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_mambas = nn.ModuleList()

        for i in range(len(features) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(features[i], features[i - 1], 2, stride=2))
            self.dec_convs.append(
                nn.Sequential(
                    nn.Conv2d(features[i - 1] * 2, features[i - 1], 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, features[i - 1]), features[i - 1]),
                    nn.Conv2d(features[i - 1], features[i - 1], 3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(min(8, features[i - 1]), features[i - 1]),
                )
            )
            self.dec_mambas.append(_MambaEncoder(features[i - 1], num_mamba_layers, d_state))

        self.final_up = nn.ConvTranspose2d(features[0], features[0], 2, stride=2)
        self.head = _ConvHead(features[0], out_channels)

    def _mamba_pass(self, feat: torch.Tensor, mamba: _MambaEncoder) -> torch.Tensor:
        """Hilbert-linearize → Mamba → restore spatial."""
        seq, shape = self.linearize(feat)
        seq = seq.permute(0, 2, 1)
        processed = mamba(seq)
        processed = processed.permute(0, 2, 1)
        return self.linearize.reverse(processed, shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: U-Net with Hilbert-Mamba at every level.

        forward method for HilbertMambaUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        skips = []

        # Encoder
        h = x
        for conv, mamba, pool in zip(self.enc_convs, self.enc_mambas, self.pools, strict=False):
            h = conv(h)
            h = h + self._mamba_pass(h, mamba)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        h = h + self._mamba_pass(h, self.bottleneck_mamba)

        # Decoder
        for up, conv, mamba, skip in zip(
            self.ups,
            self.dec_convs,
            self.dec_mambas,
            reversed(skips[:-1]),
            strict=False,
        ):
            h = up(h)
            # Handle potential size mismatch from pooling
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = conv(h)
            h = h + self._mamba_pass(h, mamba)

        h = self.final_up(h)
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)

        out = self.head(h)

        # Residual: anchor untrained output near the input
        out_ch = out.shape[1]
        out = out + x[:, :out_ch]
        return _unsqueeze_to_volume(out, vol_shape)


__all__ = [
    "HLDMamba",
    "HilbertMambaUNet",
    "MDIMamba",
    "MMMamba",
    "TwoHalfDMamba",
]
