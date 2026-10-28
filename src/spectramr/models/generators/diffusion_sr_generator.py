#!/usr/bin/env python3
"""Diffusion Super-Resolution Model

This module implements a state-of-the-art diffusion-based super-resolution
model for spectraMR. The diffusion model uses a U-Net architecture with
time embeddings and conditional generation for high-quality image
super-resolution.
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.blocks.normalization import get_adaptive_groups
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models."""

    def __init__(self, dim: int):
        """__init__.

        Args:
            dim (int): Description.
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal time embeddings.

        Args:
            t: Time steps tensor of shape (batch_size,)

        Returns:
            Time embeddings of shape (batch_size, dim)

        forward method for TimeEmbedding.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class DiffusionAttentionBlock(nn.Module):
    """Self-attention block for diffusion models."""

    def __init__(self, channels: int, num_heads: int = 8):
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(get_adaptive_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DiffusionAttentionBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        h = self.num_heads

        # Normalization
        x_norm = self.norm(x)

        # QKV projection
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.reshape(B, h, C // h, H * W).transpose(2, 3)  # (B, h, H*W, C//h)
        k = k.reshape(B, h, C // h, H * W).transpose(2, 3)
        v = v.reshape(B, h, C // h, H * W).transpose(2, 3)

        # Attention
        out = F.scaled_dot_product_attention(q, k, v)  # default scale (C // h) ** -0.5
        out = out.transpose(2, 3).reshape(B, C, H, W)

        # Output projection
        out = self.proj(out)
        return x + out


class DiffusionResBlock(nn.Module):
    """Residual block with time embedding for diffusion models."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_emb_dim (int): Description.
            dropout (float): Description.
        """
        super().__init__()

        self.time_emb = nn.Linear(time_emb_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(
            get_adaptive_groups(out_channels),
            out_channels,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(
            get_adaptive_groups(out_channels),
            out_channels,
        )

        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Time embedding
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DiffusionResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        t_emb = self.time_emb(t_emb)
        t_emb = t_emb[:, :, None, None]  # Add spatial dimensions

        # First convolution
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h + t_emb)

        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)

        # Skip connection
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        has_attention: bool = False,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_emb_dim (int): Description.
            has_attention (bool): Description.
        """
        super().__init__()

        self.res_block = DiffusionResBlock(in_channels, out_channels, time_emb_dim)
        self.attention = DiffusionAttentionBlock(out_channels) if has_attention else None
        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DownBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.res_block(x, t_emb)
        if self.attention:
            x = self.attention(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        has_attention: bool = False,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_emb_dim (int): Description.
            has_attention (bool): Description.
        """
        super().__init__()

        self.res_block = DiffusionResBlock(in_channels, out_channels, time_emb_dim)
        self.attention = DiffusionAttentionBlock(out_channels) if has_attention else None
        self.upsample = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for UpBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.res_block(x, t_emb)
        if self.attention:
            x = self.attention(x)
        x = self.upsample(x)
        return x


class DiffusionUNet(nn.Module):
    """U-Net architecture for diffusion models."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        num_res_blocks: int = 2,
        attention_levels: list[int] | None = None,
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
            time_emb_dim (int): Description.
            num_res_blocks (int): Description.
            attention_levels (Optional[list[int]]): Description.
            dropout (float): Description.
        """
        super().__init__()

        if attention_levels is None:
            attention_levels = [2, 3]  # Apply attention at these levels

        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        ch = base_channels
        for level in range(4):  # 4 downsampling levels
            has_attention = level in attention_levels
            self.encoder.append(
                nn.ModuleList(
                    [
                        DiffusionResBlock(ch, ch, time_emb_dim, dropout)
                        for _ in range(num_res_blocks)
                    ]
                    + [DownBlock(ch, ch * 2, time_emb_dim, has_attention)],
                ),
            )
            ch *= 2

        # Middle block
        self.middle = nn.ModuleList(
            [
                DiffusionResBlock(ch, ch, time_emb_dim, dropout),
                DiffusionAttentionBlock(ch),
                DiffusionResBlock(ch, ch, time_emb_dim, dropout),
            ],
        )

        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder_channels = []  # Store expected channels for each level
        for level in range(4):  # 4 upsampling levels
            has_attention = (3 - level) in attention_levels
            expected_ch = ch // 2
            self.decoder_channels.append(expected_ch)
            self.decoder.append(
                nn.ModuleList(
                    [
                        UpBlock(ch, expected_ch, time_emb_dim, has_attention),
                    ]
                    + [
                        DiffusionResBlock(
                            expected_ch,
                            expected_ch,
                            time_emb_dim,
                            dropout,
                        )
                        for _ in range(num_res_blocks)
                    ],
                ),
            )
            ch //= 2

        # Final layers
        self.final_norm = nn.GroupNorm(
            get_adaptive_groups(base_channels),
            base_channels,
        )
        self.final_conv = nn.Conv2d(
            base_channels,
            self.out_channels,
            3,
            padding=1,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the diffusion U-Net.

        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Time steps of shape (B,)

        Returns:
            Output tensor of shape (B, C, H, W)

        """
        # Time embedding
        t_emb = self.time_embedding(t)

        # Initial convolution
        x = self.init_conv(x)

        # Encoder with skip connections
        skips = []
        for level_modules in self.encoder:
            blocks = list(level_modules.children())
            for block in blocks[:-1]:  # Residual blocks
                x = block(x, t_emb)
            skips.append(x)
            x = blocks[-1](x, t_emb)  # Downsampling block

        # Middle block
        for block in self.middle:
            if isinstance(block, DiffusionResBlock):
                x = block(x, t_emb)
            else:  # DiffusionAttentionBlock
                x = block(x)

        # Decoder with skip connections
        for i, level_modules in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            blocks = list(level_modules.children())
            x = blocks[0](x, t_emb)  # Upsampling block

            # Concatenate skip connection with proper channel handling
            if x.shape[1] != skip.shape[1]:
                # Resize upsampled features to match skip connection channels
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                # Use 1x1 conv to match channels if needed
                if x.shape[1] != skip.shape[1]:
                    adapter = nn.Conv2d(x.shape[1], skip.shape[1], 1)
                    adapter = adapter.to(x.device)
                    x = adapter(x)

            x = torch.cat([x, skip], dim=1)

            # Project back to expected channels for residual blocks
            expected_channels = self.decoder_channels[i]
            if x.shape[1] != expected_channels:
                proj = nn.Conv2d(x.shape[1], expected_channels, 1).to(x.device)
                x = proj(x)

            # Residual blocks
            for block in blocks[1:]:
                x = block(x, t_emb)

        # Final layers
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion process for super-resolution."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        scale_factor: int = 4,
    ):
        """__init__.

        Args:
            model (nn.Module): Description.
            timesteps (int): Description.
            beta_start (float): Description.
            beta_end (float): Description.
            scale_factor (int): Description.
        """
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.scale_factor = scale_factor

        # Type hints for buffers
        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.alphas_cumprod: torch.Tensor
        self.alphas_cumprod_prev: torch.Tensor
        self.sqrt_alphas_cumprod: torch.Tensor
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor
        self.sqrt_recip_alphas_cumprod: torch.Tensor
        self.sqrt_recipm1_alphas_cumprod: torch.Tensor
        self.posterior_variance: torch.Tensor
        self.posterior_log_variance_clipped: torch.Tensor
        self.posterior_mean_coef1: torch.Tensor
        self.posterior_mean_coef2: torch.Tensor

        # Beta schedule
        self.register_buffer(
            "betas",
            torch.linspace(beta_start, beta_end, timesteps),
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer(
            "alphas_cumprod",
            torch.cumprod(self.alphas, dim=0),
        )
        self.register_buffer(
            "alphas_cumprod_prev",
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0),
        )

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(self.alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / self.alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / self.alphas_cumprod - 1),
        )

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        variance = self.betas * (1.0 - self.alphas_cumprod_prev)
        variance = variance / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_variance", variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(self.posterior_variance, min=1e-20)),
        )
        coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", coef1)
        coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_mean_coef2", coef2)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0).

        Args:
            x_start: Original clean images
            t: Time steps
            noise: Optional noise tensor

        Returns:
            Noisy images at time t

        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod,
            t,
            x_start.shape,
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diffusion losses.

        Args:
            x_start: Clean target images
            t: Time steps
            noise: Optional noise tensor

        Returns:
            Loss value

        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion step: p(x_{t-1} | x_t).

        Args:
            x: Current noisy images
            t: Current time steps

        Returns:
            Denoised images

        """
        # Predict noise
        predicted_noise = self.model(x, t)

        # Get coefficients
        posterior_mean_coef1 = self._extract(
            self.posterior_mean_coef1,
            t,
            x.shape,
        )
        posterior_mean_coef2 = self._extract(
            self.posterior_mean_coef2,
            t,
            x.shape,
        )
        posterior_variance = self._extract(self.posterior_variance, t, x.shape)

        # Compute posterior mean
        posterior_mean = posterior_mean_coef1 * x + posterior_mean_coef2 * predicted_noise

        # Add noise for t > 0
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        return posterior_mean + torch.sqrt(posterior_variance) * noise

    def p_sample_loop(
        self,
        shape: tuple[int, ...],
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Full reverse diffusion process.

        Args:
            shape: Shape of output tensor
            num_inference_steps: Number of denoising steps

        Returns:
            Generated images

        """
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)

        # Use fewer steps for inference
        timesteps = torch.linspace(
            self.timesteps - 1,
            0,
            num_inference_steps,
            dtype=torch.long,
            device=device,
        )

        for i in range(len(timesteps)):
            t = timesteps[i].repeat(x.shape[0])
            x = self.p_sample(x, t)

        return x

    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Extract values from tensor a at indices t."""
        out = a[t.to(torch.long)]
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)


@register_model(
    name="diffusion_sr",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class DiffusionSRGenerator(nn.Module):
    """Diffusion-based Super-Resolution Generator.

    This generator uses diffusion models to perform high-quality
    super-resolution by learning to denoise from random noise
    conditioned on low-resolution inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        scale_factor: int = 1,
        timesteps: int = 1000,
        base_channels: int = 64,
        time_emb_dim: int = 128,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            scale_factor (int): Description.
            timesteps (int): Description.
            base_channels (int): Description.
            time_emb_dim (int): Description.
        """
        super().__init__()

        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.default_image_size = 64
        self.num_inference_steps = 50

        # U-Net for diffusion
        unet = DiffusionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
        )

        # Diffusion process
        self.diffusion = GaussianDiffusion(
            model=unet,
            timesteps=timesteps,
            scale_factor=scale_factor,
        )

        # Low-resolution encoder for conditioning
        self.lr_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, base_channels, 3, padding=1),
            nn.ReLU(),
        )

        # Upsampling network
        self.upsampler = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 2x upsampling
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 2x upsampling
            nn.ReLU(),
            nn.Conv2d(base_channels, self.out_channels, 3, padding=1),
        )

    def forward(
        self,
        lr_image: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate super-resolved image from low-resolution input.

        Args:
            lr_image: Low-resolution input image
            timesteps: Optional diffusion timesteps tensor

        Returns:
            Super-resolved high-resolution image

        forward method for DiffusionSRGenerator.

        Executes PyTorch tensor operations.

        Args:
            lr_image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if lr_image.numel() == 0 or lr_image.shape[0] == 0:
            raise RuntimeError("Input tensor must not be empty")

        batch_size = lr_image.shape[0]
        device = lr_image.device

        if timesteps is None:
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif timesteps.dim() == 0:
            timesteps = timesteps.expand(batch_size).to(device)
        else:
            timesteps = timesteps.to(device)

        # Encode low-resolution features
        lr_features = self.lr_encoder(lr_image)

        # Generate high-resolution image using diffusion
        hr_shape = (
            batch_size,
            self.out_channels,
            lr_image.shape[2] * self.scale_factor,
            lr_image.shape[3] * self.scale_factor,
        )

        hr_guess = self.upsampler(lr_features)
        hr_guess = F.interpolate(
            hr_guess,
            size=hr_shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        prediction = self.diffusion.model(hr_guess, timesteps)
        prediction = F.interpolate(
            prediction,
            size=(lr_image.shape[2], lr_image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return torch.tanh(prediction)

    def generate(
        self,
        lr_image: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate super-resolved image with control over inference steps.

        Args:
            lr_image: Low-resolution input
            num_inference_steps: Number of denoising steps

        Returns:
            Super-resolved image

        """
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        if lr_image is not None:
            hr_shape = (
                lr_image.shape[0],
                self.out_channels,
                lr_image.shape[2] * self.scale_factor,
                lr_image.shape[3] * self.scale_factor,
            )
            device = lr_image.device
        else:
            hr_shape = (
                1,
                self.out_channels,
                self.default_image_size,
                self.default_image_size,
            )
            device = next(self.parameters()).device

        sample = self.diffusion.p_sample_loop(hr_shape, steps).to(device)
        return sample

    def get_output_shape(
        self,
        input_shape: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        if input_shape is None:
            input_shape = (
                1,
                self.in_channels,
                self.default_image_size,
                self.default_image_size,
            )
        batch_size, channels, height, width = input_shape
        return (
            batch_size,
            self.out_channels,
            height * self.scale_factor,
            width * self.scale_factor,
        )

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def p_losses(
        self,
        x_start: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """p_losses.

        Args:
            x_start (Optional[torch.Tensor]): Description.
            t (Optional[torch.Tensor]): Description.
            noise (Optional[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if x_start is None:
            shape = (
                1,
                self.out_channels,
                self.default_image_size,
                self.default_image_size,
            )
            device = next(self.parameters()).device
            x_start = torch.randn(shape, device=device)
        if t is None:
            device = x_start.device
            t = torch.randint(0, self.diffusion.timesteps, (x_start.shape[0],), device=device)
        return self.diffusion.p_losses(x_start, t, noise)

    def p_sample(
        self,
        x: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """p_sample.

        Args:
            x (Optional[torch.Tensor]): Description.
            t (Optional[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if x is None:
            shape = (
                1,
                self.out_channels,
                self.default_image_size,
                self.default_image_size,
            )
            device = next(self.parameters()).device
            x = torch.randn(shape, device=device)
        if t is None:
            device = x.device
            t = torch.full(
                (x.shape[0],),
                self.diffusion.timesteps - 1,
                dtype=torch.long,
                device=device,
            )
        return self.diffusion.p_sample(x, t)

    def p_sample_loop(
        self,
        shape: tuple[int, ...] | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """p_sample_loop.

        Args:
            shape (Optional[tuple[int, ...]]): Description.
            num_inference_steps (Optional[int]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if shape is None:
            shape = (
                1,
                self.out_channels,
                self.default_image_size,
                self.default_image_size,
            )
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        return self.diffusion.p_sample_loop(shape, steps)

    def q_sample(
        self,
        x_start: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """q_sample.

        Args:
            x_start (Optional[torch.Tensor]): Description.
            t (Optional[torch.Tensor]): Description.
            noise (Optional[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.
        """
        if x_start is None:
            shape = (
                1,
                self.out_channels,
                self.default_image_size,
                self.default_image_size,
            )
            device = next(self.parameters()).device
            x_start = torch.randn(shape, device=device)
        if t is None:
            device = x_start.device
            t = torch.randint(0, self.diffusion.timesteps, (x_start.shape[0],), device=device)
        return self.diffusion.q_sample(x_start, t, noise)


def create_diffusion_sr_generator(
    in_channels: int = 3,
    out_channels: int = 3,
    scale_factor: int = 1,
) -> DiffusionSRGenerator:
    """Create a diffusion-based super-resolution generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        scale_factor: Super-resolution scale factor

    Returns:
        DiffusionSRGenerator instance

    """
    return DiffusionSRGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        scale_factor=scale_factor,
    )
