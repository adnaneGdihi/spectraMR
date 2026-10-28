"""Physics-Guided Tissue Diffusion Generator.

Implements DDPM-based diffusion operating directly in tissue parameter space
(ρ, T1, T2, T2*, ADC, χ) — no pre-trained VAE required. The tissue parameter
space is itself a physics-defined latent space. The Bloch equations serve as
the "decoder" (tissue → image), replacing the VAE decoder entirely.

At each reverse sampling step, a Bloch Data Consistency (DC) correction
projects the denoised tissue maps back onto the manifold of physically
plausible parameters:
    x̂_{t-1} = denoise(x_t, t) - η ∇_{x_t} ||y_target - Bloch(x_t, θ_seq)||²

Reference:
    Chung et al., "Score-MRI: Score-Based Generative Models for MRI
    Reconstruction," IEEE TMI, 2022.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""

    def __init__(self, dim: int):
        """Initialize embedding.

        Args:
            dim: Embedding dimension.
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding.

        Args:
            t: Timestep tensor [B] in [0, 1].

        Returns:
            Embedding [B, dim].

        forward method for SinusoidalPositionEmbedding.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TissueResBlock(nn.Module):
    """Residual block with timestep conditioning via FiLM."""

    def __init__(self, channels: int, time_dim: int):
        """Initialize residual block.

        Args:
            channels: Number of feature channels.
            time_dim: Dimension of timestep embedding.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU()

        # FiLM from timestep
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with timestep conditioning.

        Args:
            x: Input features [B, C, H, W].
            t_emb: Timestep embedding [B, time_dim].

        Returns:
            Output features [B, C, H, W].

        forward method for TissueResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # FiLM conditioning from timestep
        gamma_beta = self.time_mlp(t_emb)
        gamma, beta = gamma_beta.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = (1 + gamma) * self.norm2(h) + beta

        h = self.act(h)
        h = self.conv2(h)
        return h + x


class TissueDenoisingUNet(nn.Module):
    """Denoising U-Net operating on tissue parameter maps.

    Predicts noise ε from noisy tissue maps x_t at timestep t.
    Architecture: simple 3-level U-Net with timestep conditioning.
    """

    def __init__(
        self,
        tissue_channels: int = 6,
        base_dim: int = 64,
        time_dim: int = 128,
        n_res_blocks: int = 2,
        *,
        num_timesteps: int,
    ):
        """Initialize denoising U-Net.

        Args:
            tissue_channels: Number of tissue parameter channels (ρ, T1, T2, T2*, ADC, χ).
            base_dim: Base feature dimension.
            time_dim: Timestep embedding dimension.
            n_res_blocks: Residual blocks per level.
            num_timesteps: The generator's diffusion horizon; ``t`` is divided by it.
        """
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        d0 = base_dim  # Level 0
        d1 = base_dim * 2  # Level 1
        d2 = base_dim * 4  # Level 2

        # Input projection
        self.in_conv = nn.Conv2d(tissue_channels, d0, 3, padding=1, padding_mode="reflect")

        # Encoder Level 0 → Level 1
        self.enc0 = nn.ModuleList([TissueResBlock(d0, time_dim) for _ in range(n_res_blocks)])
        self.down0 = nn.Sequential(nn.Conv2d(d0, d1, 3, stride=2, padding=1))

        # Encoder Level 1 → Level 2
        self.enc1 = nn.ModuleList([TissueResBlock(d1, time_dim) for _ in range(n_res_blocks)])
        self.down1 = nn.Sequential(nn.Conv2d(d1, d2, 3, stride=2, padding=1))

        # Bottleneck
        self.mid = nn.ModuleList(
            [
                TissueResBlock(d2, time_dim),
                TissueResBlock(d2, time_dim),
            ]
        )

        # Decoder Level 2 → Level 1
        self.up1 = nn.ConvTranspose2d(d2, d1, 4, stride=2, padding=1)
        # After concatenation with skip: d1 + d1 = 2*d1
        self.dec1_proj = nn.Conv2d(d1 * 2, d1, 1)
        self.dec1 = nn.ModuleList([TissueResBlock(d1, time_dim) for _ in range(n_res_blocks)])

        # Decoder Level 1 → Level 0
        self.up0 = nn.ConvTranspose2d(d1, d0, 4, stride=2, padding=1)
        # After concatenation with skip: d0 + d0 = 2*d0
        self.dec0_proj = nn.Conv2d(d0 * 2, d0, 1)
        self.dec0 = nn.ModuleList([TissueResBlock(d0, time_dim) for _ in range(n_res_blocks)])

        # Output projection
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, d0), d0),
            nn.SiLU(),
            nn.Conv2d(d0, tissue_channels, 3, padding=1, padding_mode="reflect"),
        )
        # Zero-init output for stable start
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise from noisy tissue maps.

        Args:
            x_noisy: Noisy tissue parameters [B, tissue_channels, H, W].
            t: Timestep [B] in [0, T].

        Returns:
            Predicted noise [B, tissue_channels, H, W].

        forward method for TissueDenoisingUNet.

        Executes PyTorch tensor operations.

        Args:
            x_noisy (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        t_norm = t.float() / self.num_timesteps
        t_emb = self.time_embed(t_norm)

        # Encoder
        h = self.in_conv(x_noisy)
        for block in self.enc0:
            h = block(h, t_emb)
        skip0 = h  # [B, d0, H, W]

        h = self.down0(h)
        for block in self.enc1:
            h = block(h, t_emb)
        skip1 = h  # [B, d1, H/2, W/2]

        h = self.down1(h)  # [B, d2, H/4, W/4]

        # Bottleneck
        for block in self.mid:
            h = block(h, t_emb)

        # Decoder
        h = self.up1(h)  # [B, d1, H/2, W/2]
        if h.shape[2:] != skip1.shape[2:]:
            h = nn.functional.interpolate(
                h, size=skip1.shape[2:], mode="bilinear", align_corners=False
            )
        h = torch.cat([h, skip1], dim=1)  # [B, 2*d1, H/2, W/2]
        h = self.dec1_proj(h)  # [B, d1, H/2, W/2]
        for block in self.dec1:
            h = block(h, t_emb)

        h = self.up0(h)  # [B, d0, H, W]
        if h.shape[2:] != skip0.shape[2:]:
            h = nn.functional.interpolate(
                h, size=skip0.shape[2:], mode="bilinear", align_corners=False
            )
        h = torch.cat([h, skip0], dim=1)  # [B, 2*d0, H, W]
        h = self.dec0_proj(h)  # [B, d0, H, W]
        for block in self.dec0:
            h = block(h, t_emb)

        return self.out_conv(h)


@register_model(name="tissue_diffusion_generator", training_mode="diffusion")
class TissueDiffusionGenerator(nn.Module, IGenerator):
    """Physics-Guided Tissue Diffusion Generator.

    Operates DDPM-based diffusion directly in tissue parameter space.
    No pre-trained VAE required — the tissue parameter space IS the
    physics-defined latent space, with Bloch equations as the decoder.

    During inference, reverse sampling includes Bloch Data Consistency
    correction at each step to enforce physical plausibility.
    """

    # Physiological bounds for normalization (maps to [0, 1])
    TISSUE_BOUNDS = {
        "rho": (0.0, 1.0),
        "t1": (100.0, 5000.0),
        "t2": (10.0, 2500.0),
        "t2star": (5.0, 2000.0),
        "adc": (0.1e-3, 3.5e-3),
        "chi": (-1.0, 1.0),
    }
    TISSUE_KEYS = ["rho", "t1", "t2", "t2star", "adc", "chi"]

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        tissue_channels: int = 6,
        base_dim: int = 64,
        n_diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        dc_step_size: float = 0.1,
        **kwargs,
    ):
        """Initialize tissue diffusion generator.

        Args:
            in_channels: Input image channels (for IGenerator compatibility).
            out_channels: Output image channels.
            tissue_channels: Number of tissue parameter channels.
            base_dim: Base dimension of denoising U-Net.
            n_diffusion_steps: Number of diffusion timesteps.
            beta_start: Starting noise schedule beta.
            beta_end: Ending noise schedule beta.
            dc_step_size: Step size η for Bloch DC correction.
        """
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.tissue_channels = tissue_channels
        self.n_steps = n_diffusion_steps
        self.dc_step_size = dc_step_size

        # Denoising U-Net
        self.denoiser = TissueDenoisingUNet(
            tissue_channels=tissue_channels,
            base_dim=base_dim,
            num_timesteps=n_diffusion_steps,
        )

        # Noise schedule (linear)
        betas = torch.linspace(beta_start, beta_end, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @property
    def name(self) -> str:
        """Model name."""
        return "TissueDiffusionMRI"

    @property
    def in_channels(self) -> int:
        """Input channels."""
        return self._in_channels

    def get_parameter_count(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Output shape matches input shape."""
        return input_shape

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space (IGenerator interface).

        For diffusion models, this runs the full reverse sampling process.

        Args:
            x: Conditioning input or dummy tensor for shape inference.
            **kwargs: Additional arguments (bloch_layer, target_image, seq_metadata).

        Returns:
            Generated tissue parameter maps stacked as channels [B, tissue_channels, H, W].
        """
        B, _, H, W = x.shape
        shape = (B, self.tissue_channels, H, W)
        tissue_dict = self.ddpm_sample(
            shape=shape,
            device=x.device,
            bloch_layer=kwargs.get("bloch_layer"),
            target_image=kwargs.get("target_image"),
            seq_metadata=kwargs.get("seq_metadata"),
        )
        # Stack tissue maps as channels for generic output
        return self.normalize_tissue(tissue_dict)

    def normalize_tissue(self, tissue_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize tissue parameters to [0, 1] and stack as channels.

        Args:
            tissue_dict: Dict of tissue parameter tensors.

        Returns:
            Normalized stacked tensor [B, tissue_channels, H, W].
        """
        channels = []
        for key in self.TISSUE_KEYS:
            if key in tissue_dict:
                lo, hi = self.TISSUE_BOUNDS[key]
                val = (tissue_dict[key] - lo) / (hi - lo + 1e-8)
                channels.append(val.clamp(0, 1))
            else:
                # Default to 0.5 (midpoint) if not available
                channels.append(torch.full_like(next(iter(tissue_dict.values())), 0.5))
        return torch.cat(channels, dim=1)

    def denormalize_tissue(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert [0,1]-normalized stacked tensor back to physical units.

        Args:
            x: Normalized tensor [B, tissue_channels, H, W].

        Returns:
            Dict of tissue parameter tensors in physical units.
        """
        result = {}
        for i, key in enumerate(self.TISSUE_KEYS):
            lo, hi = self.TISSUE_BOUNDS[key]
            result[key] = x[:, i : i + 1].clamp(0, 1) * (hi - lo) + lo
        return result

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise at timestep t.

        Args:
            x_start: Clean tissue maps [B, tissue_channels, H, W].
            t: Timestep indices [B].
            noise: Optional pre-sampled noise.

        Returns:
            Tuple of (noisy_tissue, noise).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        tissue_maps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward: predict noise from noisy tissue maps.

        Args:
            x: Input image (unused in diffusion training, kept for IGenerator compat).
            t: Diffusion timestep [B]. If None, sampled randomly.
            tissue_maps: Normalized tissue maps [B, tissue_channels, H, W].
                If None, uses x as input (fallback).

        Returns:
            Tuple of (predicted_noise, target_noise).

        forward method for TissueDiffusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            tissue_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = x.shape[0]
        device = x.device

        if tissue_maps is None:
            # Fallback: treat input as already-normalized tissue maps
            tissue_maps = x

        if t is None:
            t = torch.randint(0, self.n_steps, (B,), device=device)

        x_noisy, noise = self.q_sample(tissue_maps, t)
        noise_pred = self.denoiser(x_noisy, t)

        return noise_pred, noise

    @torch.no_grad()
    def ddpm_sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        bloch_layer=None,
        target_image: torch.Tensor | None = None,
        seq_metadata: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """DDPM reverse sampling with optional Bloch DC correction.

        Args:
            shape: Shape of tissue maps [B, tissue_channels, H, W].
            device: Device for computation.
            bloch_layer: Optional MultiPhysicsBlochLayer for DC correction.
            target_image: Target MRI image for DC loss gradient.
            seq_metadata: Sequence parameters dict for Bloch synthesis.

        Returns:
            Dict of denoised tissue parameters in physical units.
        """
        x = torch.randn(shape, device=device)

        for t_idx in reversed(range(self.n_steps)):
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.denoiser(x, t)

            # DDPM reverse step
            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            x_0_pred = (x - (1 - alpha_cumprod).sqrt() * noise_pred) / alpha_cumprod.sqrt()
            x_0_pred = x_0_pred.clamp(0, 1)  # Tissue maps in [0, 1]

            if t_idx > 0:
                noise = torch.randn_like(x)
                sigma = beta.sqrt()
                x = alpha.sqrt() * x_0_pred + sigma * noise
            else:
                x = x_0_pred

            # Bloch DC correction (if available)
            if bloch_layer is not None and target_image is not None and seq_metadata is not None:
                with torch.enable_grad():
                    x_dc = x.detach().requires_grad_(True)
                    tissue_dict = self.denormalize_tissue(x_dc)
                    synth = bloch_layer(tissue_dict, seq_metadata)
                    dc_loss = nn.functional.mse_loss(synth, target_image)
                    dc_grad = torch.autograd.grad(dc_loss, x_dc)[0]
                    x = x - self.dc_step_size * dc_grad

        return self.denormalize_tissue(x.clamp(0, 1))
