"""Manifold-Constrained Consistency Models (MC-CM).

Innovation III from SOTA Roadmap: Single-step generation with spectral fidelity.

Key Insight:
    Consistency Models (CMs) map any diffusion trajectory point x_t directly to x_0,
    enabling single-step generation. However, standard distillation can cause blur.
    MC-CM adds spectral loss and hard data consistency to preserve high frequencies.

Mathematical Formulation:
    Consistency function: f_θ(x_t) = Net_θ(x_t) - A†(A·Net_θ(x_t) - y)

    Training loss:
        L = ||f_θ(x_t) - f_teacher(x_t)||² + λ||W ⊙ F(f_θ - f_teacher)||²

    where W weights high-frequency k-space more heavily.

Benefits:
    - Single-step inference (~20ms)
    - Hard data consistency by construction
    - Spectral loss preserves fine details

References:
    - [4] arxiv:2509.22736 - Consistency Models as Priors

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.forward_operator import (
    MultiCoilDataConsistency,
    MultiCoilForwardOperator,
)
from spectramr.models.registry import register_model


class ConsistencyUNet(nn.Module):
    """U-Net for consistency model with time conditioning.

    Predicts the denoised image x_0 from any noisy state x_t.

    Args:
        in_channels: Input channels
        base_channels: Base channel count
        channel_mults: Channel multipliers per resolution
        time_embed_dim: Time embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        time_embed_dim: int = 128,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            base_channels (int): Description.
            channel_mults (tuple[int, ...]): Description.
            time_embed_dim (int): Description.
        """
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial conv
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        for mult in channel_mults[:-1]:
            out_ch = base_channels * mult
            self.encoders.append(ResBlockWithTime(ch, out_ch, time_embed_dim))
            self.downsamples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch

        # Bottleneck
        self.bottleneck = ResBlockWithTime(ch, base_channels * channel_mults[-1], time_embed_dim)
        ch = base_channels * channel_mults[-1]

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for mult in reversed(channel_mults[:-1]):
            out_ch = base_channels * mult
            self.upsamples.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))
            self.decoders.append(
                ResBlockWithTime(out_ch * 2, out_ch, time_embed_dim)  # Skip connection
            )
            ch = out_ch

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy input (B, C, H, W)
            t: Timestep (B,) in [0, 1]

        Returns:
            Predicted clean image (B, C, H, W)
        """
        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)

        # Initial
        h = self.in_conv(x)

        # Encoder
        skips = []
        for enc, down in zip(self.encoders, self.downsamples, strict=False):
            h = enc(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips), strict=False):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h, t_emb)

        return self.out_conv(h)


class ResBlockWithTime(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_embed_dim (int): Description.
        """
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ResBlockWithTime.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


@register_model(name="mc_cm", training_mode="diffusion")
class ManifoldConstrainedCM(nn.Module):
    """Manifold-Constrained Consistency Model.

    Features:
    1. Hard data consistency built into architecture
    2. Spectral consistency loss for high-frequency preservation
    3. Single-step generation capability

    Args:
        student_net: U-Net student network
        forward_operator: MRI forward operator for DC
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
    """

    def __init__(
        self,
        student_net: ConsistencyUNet | None = None,
        forward_operator: MultiCoilForwardOperator | None = None,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        # Config args
        in_channels: int = 2,
        hidden_channels: int = 64,
    ) -> None:
        """__init__.

        Args:
            student_net (ConsistencyUNet | None): Description.
            forward_operator (MultiCoilForwardOperator | None): Description.
            sigma_min (float): Description.
            sigma_max (float): Description.
            in_channels (int): Description.
            hidden_channels (int): Description.
        """
        super().__init__()

        if student_net is None:
            student_net = ConsistencyUNet(
                in_channels=in_channels,
                base_channels=hidden_channels,
            )

        self.student = student_net
        self.forward_op = forward_operator
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if forward_operator is not None:
            self.dc_layer = MultiCoilDataConsistency(forward_operator, mode="hard")
        else:
            self.dc_layer = None

        # EMA target network (for consistency training)
        self.ema_student = None

    def consistency_function(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply consistency function with hard DC.

        f_θ(x) = Net_θ(x) - A†(A·Net_θ(x) - y)

        Args:
            x_t: Noisy input (B, 2, H, W)
            t: Timestep (B,)
            y: k-space measurements (optional)

        Returns:
            Consistent prediction (B, 2, H, W)
        """
        # Network prediction
        x_pred = self.student(x_t, t)

        # Apply hard data consistency if available
        if self.dc_layer is not None and y is not None:
            # Convert to complex for DC
            x_complex = torch.complex(x_pred[:, 0], x_pred[:, 1]).unsqueeze(1)
            x_dc = self.dc_layer(x_complex, y)
            x_pred = torch.cat([x_dc.real, x_dc.imag], dim=1).squeeze(2)

        return x_pred

    def compute_loss(
        self,
        x_0: torch.Tensor,
        y: torch.Tensor | None = None,
        teacher_pred: torch.Tensor | None = None,
        spectral_weight: float = 0.1,
    ) -> torch.Tensor:
        """Compute MC-CM training loss.

        L = L_consistency + λ·L_spectral

        Args:
            x_0: Clean target (B, 2, H, W)
            y: k-space measurements (optional)
            teacher_pred: Teacher predictions (for distillation)
            spectral_weight: Weight for spectral loss

        Returns:
            Total loss
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.rand(B, device=device)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t

        # Add noise
        noise = torch.randn_like(x_0)
        x_t = x_0 + sigma[:, None, None, None] * noise

        # Student prediction
        student_pred = self.consistency_function(x_t, t, y)

        # Target (teacher or ground truth)
        if teacher_pred is not None:
            target = teacher_pred
        else:
            target = x_0

        # Consistency loss (pixel space)
        loss_consistency = F.mse_loss(student_pred, target)

        # Spectral loss (frequency space)
        pred_fft = fft2c(student_pred)  # Use physics module for proper centering
        target_fft = fft2c(target)  # Use physics module for proper centering

        # High-frequency weighting matrix
        H, W = x_0.shape[-2:]
        weight = self._get_spectral_weights(H, W, device)

        spectral_error = (pred_fft - target_fft).abs() * weight
        loss_spectral = spectral_error.mean()

        return loss_consistency + spectral_weight * loss_spectral

    def _get_spectral_weights(
        self,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get high-frequency weighting matrix.

        Penalizes high-frequency errors more heavily.
        """
        # fftshift is required: the spectra this weight multiplies come from
        # fft2c (DC at H/2, W/2), while fftfreq orders DC at index 0. Without
        # the shift the ramp is spatially inverted — DC gets the Nyquist
        # weight and the high frequencies get ~1, the opposite of the
        # documented intent.
        fy = torch.fft.fftshift(torch.fft.fftfreq(H, device=device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(W, device=device))

        fy, fx = torch.meshgrid(fy, fx, indexing="ij")
        freq_mag = torch.sqrt(fy**2 + fx**2)

        # Linear ramp from 1 (DC) to 2 (Nyquist)
        weight = 1.0 + freq_mag / freq_mag.max()

        return weight.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Single-step (or few-step) sampling.

        Args:
            y: k-space measurements (B, coils, H, W)
            num_steps: Number of refinement steps (1 for pure CM)

        Returns:
            Reconstructed image (B, 2, H, W)
        """
        B = y.shape[0]
        H, W = y.shape[-2:]
        device = y.device

        # Initialize from zero-filled recon or noise
        if self.forward_op is not None:
            x_zf = self.forward_op.adjoint(y)
            x = torch.cat([x_zf.real, x_zf.imag], dim=1).squeeze(2)
            # Add small noise
            x = x + 0.1 * torch.randn_like(x)
        else:
            x = torch.randn(B, 2, H, W, device=device)

        # Consistency mapping (can refine with multiple steps)
        for i in range(num_steps):
            t = torch.ones(B, device=device) * (1.0 - i / num_steps)
            x = self.consistency_function(x, t, y)

        return x


def create_manifold_cm(
    forward_operator: MultiCoilForwardOperator | None = None,
    base_channels: int = 64,
) -> ManifoldConstrainedCM:
    """Factory function for MC-CM.

    Args:
        forward_operator: MRI forward operator
        base_channels: U-Net base channels

    Returns:
        Configured MC-CM model
    """
    student = ConsistencyUNet(
        in_channels=2,
        base_channels=base_channels,
        channel_mults=(1, 2, 4, 8),
        time_embed_dim=128,
    )

    return ManifoldConstrainedCM(
        student_net=student,
        forward_operator=forward_operator,
    )


__all__ = [
    "ConsistencyUNet",
    "ManifoldConstrainedCM",
    "create_manifold_cm",
]
