"""Latent Diffusion Model (LDM) Generator
=====================================

This module implements a complete Latent Diffusion Model for
MRI super-resolution. LDMs compress images to a latent space
using an autoencoder, then perform diffusion in the compressed
latent space for more efficient and higher-quality generation.

Architecture:
- Encoder: Compresses input images to latent space
- Decoder: Reconstructs images from latent space
- Diffusion Model: Performs denoising in latent space
- Conditioning: Supports time and cross-attention conditioning

Reference:
Rombach et al. "High-Resolution Image Synthesis with Latent
Diffusion Models" CVPR 2022
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.diffusion.base_diffusion import Diffusion
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

from ..blocks.acquisition_film import AcquisitionTableFiLM
from ..latent_diffusion.latent_unet import LatentUNet
from .latent_cfg import axis_keep_masks, guided_noise, resolve_latent_cfg
from .latent_scale_calibration import (
    AUTO_LATENT_SCALE,
    global_latent_std,
    parse_latent_scaling_factor,
)


@dataclass
class LatentDiffusionGeneratorConfig:
    """Configuration for Latent Diffusion Generator parameters."""

    in_channels: int = 1
    out_channels: int = 1
    latent_channels: int = 4
    base_channels: int = 64
    timesteps: int = 1000
    beta_schedule: str = "linear"
    device: str | None = None
    spatial_dims: int = 2  # 2 for 2D, 3 for 3D
    # [Experiment 32b] Contrast-Aware Conditioning
    use_contrast_guidance: bool = False
    num_contrasts: int = 4
    contrast_embed_dim: int = 128
    # [PHYSICS FIX] Conditional Translation (SR3-style concatenation)
    # When True, doubles UNet input channels for condition + noisy target
    conditional_translation: bool = False

    # [DIFFBOOST] Text Conditioning
    text_conditioning_enabled: bool = False
    text_model_id: str = "openai/clip-vit-large-patch14"


class ResnetBlock(nn.Module):
    """ResBlock with GroupNorm and SiLU (LDM Standard)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dropout: float = 0.0,
        num_groups: int = 32,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            dropout (float): Description.
            num_groups (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # GroupNorm handling: ensure channels divisible by groups
        groups = num_groups if in_channels % num_groups == 0 else max(1, in_channels // 2)

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        groups_out = num_groups if out_channels % num_groups == 0 else max(1, out_channels // 2)
        self.norm2 = nn.GroupNorm(groups_out, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for ResnetBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    """Self-Attention Block for LDM."""

    def __init__(self, in_channels: int, num_groups: int = 32):
        """__init__.

        Args:
            in_channels (int): Description.
            num_groups (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        groups = num_groups if in_channels % num_groups == 0 else max(1, in_channels // 2)
        self.norm = nn.GroupNorm(groups, in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for AttnBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Single-head SDPA over the h*w tokens; default scale is c**-0.5.
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        k = k.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        v = v.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = h_.squeeze(1).transpose(1, 2).reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_


class Downsample(nn.Module):
    """Downsampling with convolution."""

    def __init__(self, in_channels, with_conv=True):
        """__init__.

        Args:
            in_channels (Any): Description.
            with_conv (Any): Description.
        """
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            # no stride, use avg pool? Original LDM uses stride 2 conv
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Downsample.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """Upsampling with convolution."""

    def __init__(self, in_channels, with_conv=True):
        """__init__.

        Args:
            in_channels (Any): Description.
            with_conv (Any): Description.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Upsample.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class AutoencoderKL(nn.Module):
    """Variational Autoencoder with KL regularization for latent diffusion.

    Compresses images to a latent space with learned variance for
    stable training and reconstruction.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 64,
        num_layers: int = 4,
        downsample_factor: int = 8,
        spatial_dims: int = 2,  # 2 for 2D, 3 for 3D
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            latent_channels (int): Description.
            base_channels (int): Description.
            num_layers (int): Description.
            downsample_factor (int): Description.
            spatial_dims (int): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.spatial_dims = spatial_dims
        self.downsample_factor = downsample_factor

        # Encoder
        self.encoder = self._build_encoder(
            in_channels,
            base_channels,
            latent_channels,
            num_layers,
        )

        # Decoder
        self.decoder = self._build_decoder(
            latent_channels,
            base_channels,
            out_channels,
            num_layers,
        )

        # Quantization convolution (for stable latents)
        if spatial_dims == 2:
            self.quant_conv = nn.Conv2d(latent_channels * 2, latent_channels * 2, 1)
            self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        else:
            self.quant_conv = nn.Conv3d(latent_channels * 2, latent_channels * 2, 1)
            self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)

    def _build_encoder(
        self,
        in_ch: int,
        base_ch: int,
        latent_ch: int,
        num_layers: int,
    ) -> nn.Sequential:
        """Build encoder network with ResBlocks and GroupNorm."""

        # 1. Initial Conv
        layers = [nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1)]

        # 2. Downsampling Stack
        ch = base_ch
        # For f=8 (2^3), we need 3 levels. If num_layers=4, it implies 4 levels (f=16)?
        # Let's trust downsample_factor logic
        num_resolutions = 0
        current_res = 1
        while current_res < self.downsample_factor:
            current_res *= 2
            num_resolutions += 1

        # Channel multipliers for each level: 1, 2, 4, 8...
        ch_mult = [1, 2, 4, 8]  # Standard LDM

        for i_level in range(num_resolutions):
            mult = ch_mult[i_level]
            prev_mult = ch_mult[i_level - 1] if i_level > 0 else 1

            block_in = base_ch * prev_mult
            block_out = base_ch * mult

            # Add ResBlocks (usually 2 per level)
            for _ in range(2):
                layers.append(ResnetBlock(block_in, block_out))
                block_in = block_out  # Next block input is current output

            # Downsample if not at final resolution
            if i_level < num_resolutions:
                layers.append(Downsample(block_out))

        # 3. Middle Block (Res - Attn - Res)
        current_ch = base_ch * ch_mult[num_resolutions - 1]
        layers.append(ResnetBlock(current_ch, current_ch))
        layers.append(AttnBlock(current_ch))
        layers.append(ResnetBlock(current_ch, current_ch))

        # 4. Final Norm + Conv to Latent
        groups = 32 if current_ch % 32 == 0 else max(1, current_ch // 2)
        layers.append(nn.GroupNorm(groups, current_ch, eps=1e-6, affine=True))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(current_ch, latent_ch * 2, kernel_size=3, stride=1, padding=1))

        return nn.Sequential(*layers)

    def _build_decoder(
        self,
        latent_ch: int,
        base_ch: int,
        out_ch: int,
        num_layers: int,
    ) -> nn.Sequential:
        """Build decoder network with ResBlocks and GroupNorm."""

        # Determine depth from downsample factor
        num_resolutions = 0
        current_res = 1
        while current_res < self.downsample_factor:
            current_res *= 2
            num_resolutions += 1

        ch_mult = [1, 2, 4, 8]
        current_ch = base_ch * ch_mult[num_resolutions - 1]

        layers = [
            nn.Conv2d(latent_ch, current_ch, kernel_size=3, stride=1, padding=1),
            # Middle Block
            ResnetBlock(current_ch, current_ch),
            AttnBlock(current_ch),
            ResnetBlock(current_ch, current_ch),
        ]

        # Upsampling Stack
        for i_level in reversed(range(num_resolutions)):
            mult = ch_mult[i_level]
            block_out = base_ch * mult

            for _ in range(3):  # Usually 3 blocks in decoder
                layers.append(ResnetBlock(current_ch, block_out))
                current_ch = block_out

            layers.append(Upsample(current_ch))

        # Final Output
        groups = 32 if current_ch % 32 == 0 else max(1, current_ch // 2)
        layers.append(nn.GroupNorm(groups, current_ch, eps=1e-6, affine=True))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(current_ch, out_ch, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (B, C, H, W) for 2D
               or (B, C, D, H, W) for 3D

        Returns:
            Tuple of (mean, logvar) for latent distribution

        """
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to reconstruction.

        Args:
            z: Latent tensor of shape (B, latent_channels, H', W') for 2D
               or (B, latent_channels, D', H', W') for 3D

        Returns:
            Reconstructed tensor of shape (B, out_channels, H, W) for 2D
            or (B, out_channels, D, H, W) for 3D

        """
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, mean, logvar)

        forward method for AutoencoderKL.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        z = self.quant_conv(torch.cat([z, torch.zeros_like(z)], dim=1))
        z = z[:, : self.latent_channels]  # Take only mean part
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar


# ``spatial_dims=(2,)``, NOT ``(2, 3)``. This model has never supported 3-D and
# declaring that it did was an advertisement the code could not honour
# (CLAUDE.md #8): ``AutoencoderKL`` branches correctly for quant_conv/
# post_quant_conv but its encoder/decoder submodules are hardcoded ``nn.Conv2d``,
# so a spatial_dims=3 instance CONSTRUCTS and then dies at the first encode with
# "Expected 3D (unbatched) or 4D (batched) input to conv2d". The declaration is
# what the audit's spec card and check_workflow_spatial_rank read, so it is how a
# 3-D arm got written against this model in the first place. Narrowed so the
# refusal happens at pre-flight. ``stubs.py`` aliases ``latent_diffusion`` to this
# same registry entry, so both names are covered by this one declaration.
# Widening it back to (2, 3) requires making the AutoencoderKL submodules
# dimension-aware FIRST — see the constructor guard below and issue #1063.
@register_model(
    name="latent_gan_generator",
    training_mode="gan",
    supports_contrast_conditioning=True,
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class LatentDiffusionGenerator(nn.Module, IGenerator):
    """Complete Latent Diffusion Model for MRI super-resolution.

    Combines an autoencoder for latent compression with a diffusion model
    operating in the compressed latent space.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        latent_channels: Number of latent channels
        base_channels: Base number of channels for U-Net
        timesteps: Number of diffusion timesteps
        beta_schedule: Schedule for beta values

    """

    def __init__(
        self,
        config: LatentDiffusionGeneratorConfig | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        latent_channels: int | None = None,
        base_channels: int | None = None,
        timesteps: int | None = None,
        beta_schedule: str | None = None,
        device: str | None = None,
        spatial_dims: int | None = None,
        **kwargs,
    ):
        """Initialize Latent Diffusion Generator.

        Args:
            config: Configuration object with all parameters. If provided,
                   individual parameters are ignored.
            in_channels: Number of input channels (ignored if config provided)
            out_channels: Number of output channels (ignored if config provided)
            latent_channels: Number of latent channels (ignored if config provided)
            base_channels: Base number of channels for U-Net (ignored if config provided)
            timesteps: Number of diffusion timesteps (ignored if config provided)
            beta_schedule: Schedule for beta values (ignored if config provided)
            device: Device to run on (ignored if config provided)
            spatial_dims: Spatial dimensions (ignored if config provided)

        """
        super().__init__()

        # Use config if provided, otherwise create from individual parameters
        if config is not None:
            self.config = config
        else:
            self.config = LatentDiffusionGeneratorConfig(
                in_channels=in_channels if in_channels is not None else 1,
                out_channels=out_channels if out_channels is not None else 1,
                latent_channels=latent_channels if latent_channels is not None else 4,
                base_channels=base_channels if base_channels is not None else 64,
                timesteps=timesteps if timesteps is not None else 1000,
                beta_schedule=beta_schedule if beta_schedule is not None else "linear",
                device=device,
                spatial_dims=spatial_dims if spatial_dims is not None else 2,
                use_contrast_guidance=kwargs.get("use_contrast_guidance", False),
                num_contrasts=kwargs.get("num_contrasts", 4),
                # [SR3] Thread the conditioning knobs (previously swallowed by
                # **kwargs → conditional_translation defaulted False, so the
                # ULF-conditioning concat was never built — an inert facade,
                # pitfall #16). Read + validated below.
                conditional_translation=bool(kwargs.get("conditional_translation", False)),
                contrast_embed_dim=int(kwargs.get("contrast_embed_dim", 128)),
                # [DIFFBOOST]
                text_conditioning_enabled=kwargs.get("text_conditioning_enabled", False),
                text_model_id=kwargs.get("text_model_id", "openai/clip-vit-large-patch14"),
            )

        # Validate the conditioning contract (#9/#15): a requested but
        # unimplemented ``conditioning_key`` must RAISE (only ``concat`` — SR3
        # spatial concatenation — is wired; it had ZERO consumers before this).
        # An UNCONDITIONAL LDM (conditional_translation=False, no key) is valid.
        _key_raw = kwargs.get("conditioning_key")
        if _key_raw is not None and str(_key_raw).lower() != "concat":
            raise ValueError(
                f"conditioning_key={_key_raw!r} is not implemented; only 'concat' "
                "(SR3-style spatial concatenation) is wired for the "
                "latent-diffusion ULF condition."
            )
        self._conditioning_key = "concat" if self.config.conditional_translation else None
        # The contrast embedding is ADDED to the UNet time embedding (dim 128),
        # so a mismatched contrast_embed_dim is a runtime broadcast crash.
        if self.config.use_contrast_guidance and self.config.contrast_embed_dim != 128:
            raise ValueError(
                "contrast_embed_dim must equal the UNet time_emb_dim (128) — the "
                f"contrast embedding is added to it; got {self.config.contrast_embed_dim}."
            )

        # Set instance attributes from config
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.latent_channels = self.config.latent_channels
        self.spatial_dims = self.config.spatial_dims

        # Autoencoder for latent space. ``vae_backbone`` selects between the
        # from-scratch custom AutoencoderKL (stage-1 recipe) and a frozen
        # pretrained Stable-Diffusion VAE (transfer learning — the far stronger
        # latent prior; see pretrained_vae_adapter.py and
        # docs/ldm_two_stage_ulf_to_hf.md). Unknown values RAISE (CLAUDE.md #9).
        self._vae_backbone = str(kwargs.get("vae_backbone", "custom") or "custom").lower()
        if self._vae_backbone in ("custom", "kl", "autoencoder_kl"):
            # 2-D ONLY, and it must RAISE rather than build something unrunnable
            # (CLAUDE.md #9). The `sd` backbone below has always had this guard;
            # the custom one did not, so `spatial_dims=3` constructed happily and
            # then died at the first encode with a bare
            # "Expected 3D (unbatched) or 4D (batched) input to conv2d" — a shape
            # error deep in a forward pass, with nothing pointing at the config
            # key that caused it. AutoencoderKL branches Conv2d/Conv3d for
            # quant_conv/post_quant_conv but its encoder/decoder submodules are
            # hardcoded nn.Conv2d, so the 3-D build was never viable.
            # Lifting this means making those submodules dimension-aware first,
            # and only THEN fixing the separate 5-D conditioning gap in
            # _process_slices (issue #1063) — in that order, since the encode
            # fails before _process_slices is ever reached.
            if self.spatial_dims != 2:
                raise ValueError(
                    "vae_backbone='custom' (AutoencoderKL) is 2D-only; got "
                    f"spatial_dims={self.spatial_dims}. Its encoder/decoder "
                    "submodules are hardcoded nn.Conv2d, so a 3-D instance "
                    "builds but cannot encode. Run slice-mode 2D "
                    "(model.spatial_dims: 2 with data.sampling.enable_slice_2d: "
                    "true), or use a genuinely volumetric model."
                )
            self.autoencoder = AutoencoderKL(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                latent_channels=self.latent_channels,
                base_channels=self.config.base_channels,
                spatial_dims=self.spatial_dims,
            )
        elif self._vae_backbone in ("sd", "sd_vae", "diffusers_sd", "stable_diffusion"):
            if self.spatial_dims != 2:
                raise ValueError(
                    "vae_backbone='sd' (Stable-Diffusion VAE) is 2D-only; got "
                    f"spatial_dims={self.spatial_dims}. Run slice-mode 2D "
                    "(data.slice_2d: true)."
                )
            from spectramr.models.generators.pretrained_vae_adapter import (
                DEFAULT_SD_VAE_ID,
                PretrainedSDVAEAdapter,
            )

            self.autoencoder = PretrainedSDVAEAdapter(
                pretrained_id=str(kwargs.get("vae_pretrained_id", DEFAULT_SD_VAE_ID)),
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                latent_channels=self.latent_channels,
                freeze=bool(kwargs.get("freeze_vae", True)),
            )
        else:
            raise ValueError(
                f"Unknown vae_backbone={self._vae_backbone!r}. Expected 'custom' "
                "(train AutoencoderKL from scratch) or 'sd' (frozen pretrained "
                "Stable-Diffusion VAE). CLAUDE.md #9 forbids a silent fallback."
            )

        # Diffusion model operating in latent space.
        #
        # [SCHEDULE SSOT] The reverse process used by ``sample()`` MUST match the
        # forward process the strategy q_samples with, which is driven by
        # ``training.diffusion.{num_timesteps,noise_schedule}``. This dataclass
        # default ("linear") was never wired from YAML, so an arm declaring
        # ``noise_schedule: cosine`` trained on a cosine forward trajectory and
        # then inverted it with LINEAR posterior coefficients — the reverse
        # trajectory diverges and the decoded image collapses to ~black
        # (stage-2 LDM: train_psnr≈32 in latent space, val_psnr≈6 dB).
        # ``DiffusionTrainingStrategy`` now calls ``set_diffusion_schedule`` at
        # setup to bind this to the training SSOT; ``beta_schedule`` is also
        # accepted as a model_kwarg so a standalone build can pin it (#15).
        # self.config already folds in the explicit beta_schedule/timesteps args
        # (the model builder passes model_kwargs, not a config object).
        self.beta_schedule = str(self.config.beta_schedule)
        self.timesteps = int(self.config.timesteps)
        self.diffusion = Diffusion(
            self.timesteps,
            self.beta_schedule,
            # `self.config` is a LatentDiffusionGeneratorConfig, NOT
            # TrainingSettings: it carries `device` and has no `run:` block.
            # Phase 4b's `device -> run.device` rename was applied by name
            # rather than by receiver, turning a working read into a hard
            # AttributeError on generator construction.
            self.config.device or "cpu",
        )

        # [DIFFBOOST] CLIP Text Model
        self.tokenizer = None
        self.text_encoder = None
        self.context_dim = None

        if self.config.text_conditioning_enabled:
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for text conditioning. Install with `pip install transformers`."
                ) from exc

            self.tokenizer = CLIPTokenizer.from_pretrained(self.config.text_model_id)
            self.text_encoder = CLIPTextModel.from_pretrained(self.config.text_model_id)
            self.text_encoder.requires_grad_(False)  # Freeze CLIP
            self.context_dim = self.text_encoder.config.hidden_size

        # U-Net for denoising in latent space
        # [PHYSICS FIX] Double input channels for conditional translation (SR3-style)
        unet_in_channels = (
            self.latent_channels * 2
            if self.config.conditional_translation
            else self.latent_channels
        )

        self._channel_mult = _parse_channel_multipliers(kwargs.get("channel_multipliers"))
        self.denoise_net = LatentUNet(
            in_channels=unet_in_channels,
            out_channels=self.latent_channels,  # Output is just noise prediction
            time_emb_dim=128,
            cond_channels=0,  # Spatial conditioning via concatenation, not cross-attention
            base_channels=self.config.base_channels,
            channel_mult=self._channel_mult,
            num_res_blocks=2,
            attention_levels=(),
            dropout=0.1,
            # [DIFFBOOST]
            context_dim=self.context_dim,
        )

        # Scale factor for latent space (learned or fixed)
        self.scale_factor = nn.Parameter(torch.ones(1))

        # Per-axis CFG (LM-19): training dropout per axis and sampling guidance weights.
        self._cfg = resolve_latent_cfg(
            kwargs.get("cfg_drop_probs"),
            kwargs.get("cfg_weights"),
            has_condition=bool(self.config.conditional_translation),
            has_contrast=bool(self.config.use_contrast_guidance),
        )
        import logging

        logging.getLogger(__name__).info(
            "[LDM] CFG: %s; channel_multipliers=%s", self._cfg.to_dict(), self._channel_mult
        )

        # Pattern-C acquisition FiLM on the contrast embedding (LM-20); the table is
        # bound by the training strategy from data.multi_contrast.acquisition_params.
        acq_film = kwargs.get("acquisition_film", False)
        if not isinstance(acq_film, bool):
            raise ValueError(f"acquisition_film must be true/false; got {acq_film!r}.")
        if acq_film and not self.config.use_contrast_guidance:
            raise ValueError(
                "acquisition_film modulates the contrast embedding, so it needs "
                "use_contrast_guidance: true."
            )
        self.acquisition_film = (
            AcquisitionTableFiLM(self.config.num_contrasts, self.config.contrast_embed_dim)
            if acq_film
            else None
        )

        # [Experiment 32b] Contrast Embedding
        if self.config.use_contrast_guidance:
            self.contrast_embedder = nn.Embedding(
                self.config.num_contrasts,
                (
                    self.config.contrast_embed_dim
                    if hasattr(self.config, "contrast_embed_dim")
                    else 128
                ),
            )

        # [SR3] The ULF condition is encoded through the SAME frozen autoencoder
        # as the target (see ``encode_condition``), so its latent is guaranteed
        # to match ``z`` in spatial size and distribution. The old trainable
        # ConditioningEncoder used a DIFFERENT downsampling arithmetic
        # (floor((N-1)/2)+1 vs the VAE's floor((N-2)/2)+1) → desynced on odd
        # sizes and hit a no-op ``pass`` "safety check" → torch.cat crash. No
        # separate module is built for the wired ``concat`` mode.

        # [AUDIT FIX] Latent Standardization Buffers
        self.register_buffer("latent_mean", torch.zeros(1, self.latent_channels, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, self.latent_channels, 1, 1))

        # [LATENT SCALING] Rombach et al. (CVPR 2022) scale the raw VAE latent to
        # ~unit variance before diffusion (z_scaled = s·z). ``set_latent_statistics``
        # had NO caller → latent_std stayed 1.0 → the "standardize to N(0,1)" in
        # encode_to_latent was a dead identity (stage-1 logged latent_std≈0.408).
        # The custom backbone reads ``latent_scaling_factor`` (a single positive
        # scalar s): wire it via latent_std = 1/s so encode does (z-0)/(1/s)=s·z and
        # decode inverts it. Optional (default None → std=1.0, byte-identical to the
        # historical behavior, so the ~16 existing custom-backbone LDM configs are
        # unaffected). The 'sd' backbone self-scales internally, so setting the knob
        # there RAISES (double-scaling corrupts) (#9/#15).
        lsf = parse_latent_scaling_factor(kwargs.get("latent_scaling_factor"))
        if lsf is not None:
            if self._vae_backbone in (
                "sd",
                "sd_vae",
                "diffusers_sd",
                "stable_diffusion",
            ):
                raise ValueError(
                    "latent_scaling_factor is for the custom backbone only; the "
                    "'sd' backbone applies its own scaling_factor internally, so "
                    "setting it here would double-scale (#9). Remove the knob for "
                    "vae_backbone='sd'."
                )
        if lsf == AUTO_LATENT_SCALE:
            # Calibrated on the first training encode; the flag is a buffer so a
            # resumed or predicted checkpoint keeps the scale it trained with.
            self.register_buffer("latent_scale_calibrated", torch.zeros((), dtype=torch.bool))
            self._latent_scale_ready = False
            self.register_load_state_dict_post_hook(_sync_latent_scale_ready)
        elif lsf is not None:
            self.set_latent_statistics(
                torch.zeros(self.latent_channels),
                torch.full((self.latent_channels,), 1.0 / lsf),
            )
        self._latent_scaling_factor = lsf

        # The stage-1 VAE is loaded by the training builder (load_autoencoder_state),
        # never here: models/ does no checkpoint IO, and inference already holds the
        # autoencoder inside the stage-2 checkpoint. A path beside the hub backbone is
        # still a config error, because the 'sd' VAE comes from its hub id.
        if kwargs.get("vae_checkpoint_path") and self._vae_backbone not in (
            "custom",
            "kl",
            "autoencoder_kl",
        ):
            raise ValueError(
                f"vae_checkpoint_path is set but vae_backbone={self._vae_backbone!r} "
                "does not load a stage-1 checkpoint (the pretrained VAE is loaded "
                "from its hub id). Remove vae_checkpoint_path, or use "
                "vae_backbone='custom'."
            )

    def load_autoencoder_state(
        self, state: Mapping[str, torch.Tensor], *, freeze: bool, source: str
    ) -> None:
        """Load a stage-1 VAE's state dict into ``self.autoencoder``.

        The training builder reads the stage-1 checkpoint's generator part and calls
        this; the keys may sit under ``autoencoder.`` / ``vae.`` / ``module.``. Any
        incomplete restore raises, because stage 2 cannot train in a random latent
        space (#9).

        Args:
            state: The stage-1 generator's state dict.
            freeze: If True, set ``requires_grad=False`` on every autoencoder
                parameter so the LDM only trains its diffusion U-Net + conditioning
                encoder.
            source: Where ``state`` came from, for the error messages.
        """
        # Strip a wrapping prefix when the stage-1 model embedded the
        # autoencoder under e.g. ``autoencoder.``, ``vae.``, or ``module.``.
        autoencoder_state = {}
        # Parameters AND buffers: a buffer absent from ``ae_param_names`` would
        # otherwise land in ``missing`` below and trip the completeness check.
        ae_param_names = {n for n, _ in self.autoencoder.named_parameters()} | {
            n for n, _ in self.autoencoder.named_buffers()
        }
        for prefix in ("autoencoder.", "vae.", "module.autoencoder.", "module.", ""):
            stripped = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            hits = {k for k in stripped if k in ae_param_names}
            if hits:
                autoencoder_state = {k: v for k, v in stripped.items() if k in ae_param_names}
                break

        if not autoencoder_state:
            # #9 — raise rather than degrade to a random-initialised VAE.
            raise RuntimeError(
                f"vae_checkpoint_path={source} loaded but contained no keys "
                "matching the LDM autoencoder. Names diverge between the stage-1 "
                "and stage-2 VAE architectures — check that stage-1 used "
                "model_type=autoencoder_kl_pretrain with the SAME latent_channels/"
                "base_channels as this arm."
            )

        missing, unexpected = self.autoencoder.load_state_dict(autoencoder_state, strict=False)
        if missing:
            # #9 — ``strict=False`` tolerates MISSING keys, leaving them at their
            # random init. The emptiness guard above only catches a *total* miss, so a
            # stage-1 VAE whose key set merely diverges (different depth: matching
            # ``encoder.down.0.*``, absent ``encoder.down.3.*``) loaded a PARTIALLY
            # random autoencoder and still reported PASS. Shape drift already raises
            # inside load_state_dict; name drift is what slips through. Demand a
            # complete restore.
            raise RuntimeError(
                f"vae_checkpoint_path={source} restored only "
                f"{len(autoencoder_state)} of {len(ae_param_names)} autoencoder "
                f"tensors; {len(missing)} would stay randomly initialised "
                f"(first: {sorted(missing)[:5]}). Stage 2 cannot train in a "
                "partially random latent space — stage-1 and stage-2 VAE "
                "architectures must match."
            )
        if freeze:
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)
            self.autoencoder.eval()

        import logging

        logging.getLogger(__name__).info(
            "[LDM] Loaded pretrained AutoencoderKL from %s (matched=%d, "
            "missing=%d, unexpected=%d, frozen=%s)",
            source,
            len(autoencoder_state),
            len(missing),
            len(unexpected),
            freeze,
        )

    def set_diffusion_schedule(
        self,
        timesteps: int,
        beta_schedule: str,
        device: str | None = None,
        betas: torch.Tensor | None = None,
    ) -> None:
        """Rebind the reverse-diffusion schedule to the training SSOT.

        ``sample()`` inverts the forward process the strategy q_sampled with, so
        the two MUST share a schedule. The strategy owns that SSOT
        (``training.diffusion.{num_timesteps,noise_schedule}``) and calls this at
        setup. Raises on an unknown schedule (#9 — no silent fallback).

        Args:
            timesteps: Number of diffusion steps (strategy's ``num_timesteps``).
            beta_schedule: ``"linear"`` or ``"cosine"``.
            device: Device for the schedule buffers.
            betas: The forward process's ACTUAL betas. Binding by name alone was
                not enough: the strategy's DiffusionScheduler builds ``cosine``
                as ``cos(pi/2 * t/T)^2`` (s=0, clamp [0, 0.999]) while
                ``base_diffusion.cosine_beta_schedule`` builds the
                Nichol-Dhariwal ``s=0.008`` variant (clip [1e-4, 0.9999]). Same
                name, different beta sequence, so sampling still inverted a
                trajectory training never ran. Passing the betas makes the
                forward schedule the single source of truth.
        """
        self.timesteps = int(timesteps)
        self.beta_schedule = str(beta_schedule)
        self.config.timesteps = self.timesteps
        self.config.beta_schedule = self.beta_schedule
        # Diffusion raises ValueError("unknown beta schedule: ...") on a bad name
        # when no explicit betas are supplied.
        self.diffusion = Diffusion(
            self.timesteps,
            self.beta_schedule,
            device or self.config.device or "cpu",
            betas=betas,
        )

    def set_latent_statistics(self, mean: torch.Tensor, std: torch.Tensor):
        """Update latent statistics for standardization."""
        if mean.numel() == self.latent_channels:
            mean = mean.reshape(1, self.latent_channels, 1, 1)
        if std.numel() == self.latent_channels:
            std = std.reshape(1, self.latent_channels, 1, 1)

        self.latent_mean.data.copy_(mean.to(self.latent_mean.device))
        self.latent_std.data.copy_(std.to(self.latent_std.device))

    def encode_condition(self, condition_image: torch.Tensor) -> torch.Tensor:
        """Encode the ULF condition through the SAME frozen autoencoder path as
        the target (``encode_to_latent``), guaranteeing the condition latent
        matches ``z`` in spatial size and distribution for the SR3 concat."""
        return self.encode_to_latent(condition_image)

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space (deterministic encoding)."""
        with torch.no_grad():
            mean, logvar = self.autoencoder.encode(x)
            # Use mean for deterministic encoding (no sampling)
            z = mean
            # NB: apply NEITHER quant_conv NOR post_quant_conv here.
            #
            # post_quant_conv: ``decode`` applies it, so applying it on encode
            # too double-transforms the round trip.
            #
            # quant_conv: it is UNTRAINED on the from-scratch stage-1 path.
            # AutoencoderKL.encode() returns chunk(encoder(x)) and never touches
            # quant_conv, and stage 1 trains through exactly that encode/decode
            # pair (AutoencoderPretrainGenerator.forward), so no gradient ever
            # reaches quant_conv — it stays at random init in the stage-1
            # checkpoint. Applying it here interposed a random 1x1 channel mix
            # (plus bias, of the same order as the latent itself) between the
            # trained encoder and the trained decoder that nothing inverts, so
            # even a perfect denoiser decoded to noise. The cat-with-zeros /
            # re-slice dance was itself the tell: quant_conv is shaped for the
            # encoder's raw 2C moments, not for the already-chunked mean.
            #
            # This is a no-op for the pretrained SD backbone, whose adapter sets
            # quant_conv = nn.Identity() because diffusers applies its own
            # inside encode() (pretrained_vae_adapter.py:169).

            if self._latent_scaling_factor == AUTO_LATENT_SCALE and not self._latent_scale_ready:
                self._calibrate_latent_scale(z)
            # [AUDIT FIX] Standardize to N(0,1) for Diffusion
            z = (z - self.latent_mean) / (self.latent_std + 1e-8)
        return z

    def _calibrate_latent_scale(self, mu: torch.Tensor) -> None:
        """Set ``latent_std`` to the pooled std of the first training batch's ``mu``."""
        if not self.training:
            raise RuntimeError(
                "latent_scaling_factor='auto' is not calibrated: it is measured on the "
                "first TRAINING encode, and this model is in eval mode with no calibrated "
                "checkpoint loaded. Train at least one step, or load a stage-2 checkpoint."
            )
        std = global_latent_std(mu).clamp_min(1e-6)
        self.latent_std.data.fill_(0.0).add_(std)
        self.latent_scale_calibrated.fill_(True)
        self._latent_scale_ready = True
        import logging

        # One host read, once per run: the calibrated value is the provenance.
        value = float(std)
        logging.getLogger(__name__).info(
            "[LDM] latent_scaling_factor=auto calibrated: std(mu)=%.6g -> scale=%.6g "
            "over %d latent elements (stored in the latent_std buffer).",
            value,
            1.0 / value,
            mu.numel(),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: dict | None = None,
        condition_image: torch.Tensor | None = None,  # ULF Input
    ) -> torch.Tensor:
        """Forward pass for training (noise prediction).

        Args:
            x: Noisy latent input (B, C, H, W). If x has `in_channels`
              (image-space) rather than `latent_channels`, it is encoded
              through the autoencoder first — this lets the standard
              diffusion training strategy hand the raw target/noisy image
              to the model without knowing it operates in latent space.
            timesteps: Diffusion timesteps (B,)
            context: Optional context dictionary
            condition_image: Source image for conditioning (B, C_in, H_in, W_in)
        """
        # Auto-encode raw image-space inputs to latents. The strategy adds
        # noise in image space (matching model.in_channels), so the first
        # iteration after q_sample will hand us a (B, in_channels, H, W)
        # tensor that needs encoding before the latent U-Net can consume
        # it. Skip the encode when x is already in latent shape.
        if (
            x.dim() >= 4
            and x.shape[1] == self.in_channels
            and self.in_channels != self.latent_channels
        ):
            mean, _logvar = self.autoencoder.encode(x)
            x = mean[:, : self.latent_channels]

        # Mechanism guard (#16/#20): a conditional-translation model MUST be
        # handed its condition. Silently running unconditionally when the
        # strategy forgot to pass condition_image is the exact facade this fix
        # removes — raise instead of degrading to an unconditional denoiser.
        if self.config.conditional_translation and condition_image is None:
            raise ValueError(
                "conditional_translation=True but condition_image is None: the "
                "ULF condition was not threaded to the generator. Ensure the "
                "diffusion strategy populates gen_kwargs['condition_image']."
            )

        # Same guard for the OTHER conditioning channel. `_process_slices` gates
        # the contrast path on `use_contrast_guidance and contrast_idx is not
        # None`, so a model that declares guidance but never receives the index
        # trains contrast-AGNOSTICALLY while advertising per-contrast
        # conditioning — carrying an nn.Embedding whose gradient is always zero
        # (pitfall #16). Nothing caught it: check_multi_contrast_model_support
        # guards data->model, and check_contrast_conditioning_strategy_threaded
        # keys off a DIFFERENT kwarg (`use_contrast_conditioning`), so this
        # direction was uncovered.
        #
        # Deliberately in forward() and NOT in sample(): unconditional sampling
        # with contrast_idx=None is legitimate under classifier-free guidance, so
        # raising there would break a real use case. Training silently losing its
        # conditioning is the failure worth being loud about.
        # NB read through `context`, which is how forward() receives it — there is
        # no bare contrast_idx parameter on this signature.
        if self.config.use_contrast_guidance and (context or {}).get("contrast_idx") is None:
            raise ValueError(
                "use_contrast_guidance=True but contrast_idx is None: the batch "
                "did not carry it, so this forward would run contrast-agnostic "
                "while advertising per-contrast conditioning. Set "
                "data.multi_contrast.enabled: true with a contrast_map so the "
                "dataset stamps contrast_idx, or set use_contrast_guidance: "
                "false."
            )

        # Encode conditioning if present
        cond_latent = None
        if self.config.conditional_translation and condition_image is not None:
            cond_latent = self.encode_condition(condition_image)

        # [CFG TRAINING] Per-axis dropout; absent knobs keep the 10% condition drop.
        keep = axis_keep_masks(self._cfg, x.shape[0], x.device) if self.training else {}
        if cond_latent is not None and "condition" in keep:
            cond_latent = cond_latent * keep["condition"].view(-1, 1, 1, 1)

        # Handle 3D slicing if needed, but usually we train on 2D/slabs
        return self._process_slices(
            x,
            timesteps,
            operation="denoise",
            cond_latent=cond_latent,
            contrast_idx=context.get("contrast_idx") if context else None,
            text_context=context.get("text_context") if context else None,
            contrast_keep=keep.get("contrast"),
        )

    def _process_slices(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        operation: str = "denoise",
        cond_latent: torch.Tensor | None = None,
        contrast_idx: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
        contrast_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process latent tensor using 2D U-Net.

        For 2D inputs: process directly [B, C, H, W]
        For 3D inputs: process slice by slice [B, C, D, H, W]

        Args:
            z: Latent tensor [B, C, H, W] or [B, C, D, H, W]
            t: Time tensor [B]
            operation: 'denoise' or 'sample'
            text_context: Text embeddings [B, Seq, Dim]

        Returns:
            Processed tensor with same shape as input

        """
        if z.dim() == 4:  # 2D case [B, C, H, W]
            # Prepare Time and Contrast Context
            time_context = None
            contrast_context = None
            if self.config.use_contrast_guidance and contrast_idx is not None:
                # Logic: We compute t_emb using the UNet's time_mlp, and extract contrast_emb
                time_context = self.denoise_net.time_mlp(t)
                contrast_context = self._contrast_context(contrast_idx, contrast_keep)

            # [CONDITIONING] Concatenate cond_latent for conditional translation
            unet_input = z
            if cond_latent is not None:
                # Ensure cond_latent matches z shape
                if cond_latent.shape != z.shape:
                    # Should be handled by encoder, but safety check
                    pass  # Assume correct from encoder
                unet_input = torch.cat([z, cond_latent], dim=1)

            if operation == "denoise":
                return self.denoise_net(
                    unet_input,
                    t,
                    time_context=time_context,
                    contrast_context=contrast_context,
                    text_context=text_context,
                )
            # sample
            return self.denoise_net(
                unet_input,
                t,
                time_context=time_context,
                contrast_context=contrast_context,
                text_context=text_context,
            )

        if z.dim() == 5:  # 3D case [B, C, D, H, W]
            B, C, D, H, W = z.shape
            results = []

            for d in range(D):
                # Extract slice [B, C, H, W]
                z_slice = z[:, :, d, :, :]

                # Recalculate context per slice if memory constrained, or reuse?
                time_context = None
                contrast_context = None
                if self.config.use_contrast_guidance and contrast_idx is not None:
                    time_context = self.denoise_net.time_mlp(t)
                    contrast_context = self._contrast_context(contrast_idx, contrast_keep)

                if operation == "denoise":
                    # Denoise slice
                    result_slice = self.denoise_net(
                        z_slice,
                        t,
                        time_context=time_context,
                        contrast_context=contrast_context,
                        text_context=text_context,
                    )
                else:  # sample
                    # For sampling, we need to handle the reverse process
                    result_slice = self.denoise_net(
                        z_slice,
                        t,
                        time_context=time_context,
                        contrast_context=contrast_context,
                        text_context=text_context,
                    )

                results.append(result_slice)

            # Stack back to 5D [B, C, D, H, W]
            return torch.stack(results, dim=2)

        raise ValueError(
            f"Unsupported tensor dimensions: {z.dim()}. Expected 4D (2D) or 5D (3D)",
        )

    def _contrast_context(
        self, contrast_idx: torch.Tensor, contrast_keep: torch.Tensor | None
    ) -> torch.Tensor:
        """Contrast embedding, FiLM-ed by acquisition when on, then CFG-dropped."""
        context = self.contrast_embedder(contrast_idx)
        if self.acquisition_film is not None:
            context = self.acquisition_film(context, contrast_idx)
        if contrast_keep is not None:
            context = context * contrast_keep.view(-1, 1)
        return context

    @property
    def uses_acquisition_film(self) -> bool:
        return self.acquisition_film is not None

    def set_acquisition_params(
        self, contrast_map: Mapping[str, int], rows: Sequence[Mapping[str, object]]
    ) -> None:
        """Bind the per-contrast acquisition table the FiLM reads (strategy setup)."""
        if self.acquisition_film is None:
            raise RuntimeError("set_acquisition_params called but acquisition_film is off.")
        self.acquisition_film.bind(contrast_map, rows)

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to image."""
        # [AUDIT FIX] Un-standardize from N(0,1) to VAE Latent Space
        z = z * self.latent_std + self.latent_mean
        return self.autoencoder.decode(z)

    def sample(
        self,
        shape: tuple[int, ...],
        device: str = "cpu",
        contrast_idx: torch.Tensor | None = None,
        condition_image: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Sample new images from the latent diffusion model.

        Args:
            shape: Shape of output tensor (B, C, D, H, W)
            device: Device to place samples on
            contrast_idx: Physics class label [B]
            condition_image: ULF image for conditional translation [B, C, H, W]
            num_inference_steps: Number of reverse steps to run. ``None`` (the
                default) runs the full DDPM chain, one step per training
                timestep. A smaller value runs an evenly-strided DDIM chain with
                exactly that many denoiser evaluations. This is the parameter
                ``DiffusionTrainingStrategy._generate_validation_prediction``
                probes for when forwarding ``validation.sampler_steps``; without
                it the configured step count is silently dropped and validation
                pays the full-chain cost.

        Returns:
            Generated images

        """
        # Start from random noise in latent space
        batch_size = shape[0]
        ds = self.autoencoder.downsample_factor
        if self._cfg.guided and len(shape) != 4:
            raise ValueError(
                "cfg_weights guidance is implemented for 2-D latents only; got a "
                f"{len(shape)}-D sample shape. Run slice-mode 2D or drop cfg_weights."
            )

        if len(shape) == 4:
            latent_shape = (
                batch_size,
                self.latent_channels,
                shape[2] // ds,
                shape[3] // ds,
            )
        else:
            latent_shape = (
                batch_size,
                self.latent_channels,
                shape[2] // ds,
                shape[3] // ds,
                shape[4] // ds,
            )

        z_t = torch.randn(latent_shape, device=device)

        if contrast_idx is not None:
            contrast_idx = contrast_idx.to(device)

        # [PHYSICS FIX] Prepare Condition Latent
        z_cond = None
        if self.config.conditional_translation and condition_image is not None:
            # Encode the ULF through the frozen autoencoder (same path as the
            # target latent) so z_cond matches z_t in spatial size for the concat.
            z_cond = self.encode_condition(condition_image.to(device))

            # If 3D latent, expand condition check
            if len(latent_shape) == 5:  # [B, C, D, H, W]
                # Replicate conditioning across depth if condition_image is 2D
                D = latent_shape[2]
                z_cond = z_cond.unsqueeze(2).expand(-1, -1, D, -1, -1)

        # Build the reverse schedule. The default (None) keeps the historical
        # full DDPM chain -- one denoiser call per training timestep, stepped by
        # p_sample_step, which is only valid for t -> t-1. A configured step
        # count strides the chain instead and must use the skip-capable DDIM
        # update, otherwise each jump would apply a single step's worth of
        # variance across many timesteps.
        n_total = int(self.diffusion.timesteps)
        requested = n_total if num_inference_steps is None else int(num_inference_steps)
        strided = requested < n_total
        if strided:
            n_steps = max(1, requested)
            # Span from the HIGH end: torch.linspace(a, b, 1) returns [a], so
            # building low->high would put a 1-step schedule at t=0 -- asking the
            # denoiser to clean pure noise it is told is already clean. Starting
            # at n_total-1 makes num_inference_steps=1 the correct single
            # max-noise -> x_0 hop. For n_steps >= 2 both spans end at t=0.
            schedule = sorted(
                {round(v) for v in torch.linspace(n_total - 1, 0, n_steps).tolist()},
                reverse=True,
            )
        else:
            schedule = list(range(n_total - 1, -1, -1))

        # Denoise step by step
        for t_index, t in enumerate(schedule):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # SR3 Concatenation
            unet_input = z_t
            if z_cond is not None:
                unet_input = torch.cat([z_t, z_cond], dim=1)

            if self._cfg.guided:
                predicted_noise = guided_noise(
                    self._stacked_denoiser(contrast_idx, batch_size),
                    z_t,
                    t_tensor,
                    z_cond,
                    self.config.use_contrast_guidance and contrast_idx is not None,
                    self._cfg,
                )
            else:
                predicted_noise = self._process_slices(
                    unet_input, t_tensor, "denoise", contrast_idx=contrast_idx
                )

            if strided:
                # -1 on the last hop means "decode the x_0 estimate" (alpha_bar_prev = 1).
                t_prev = schedule[t_index + 1] if t_index + 1 < len(schedule) else -1
                z_t = self.diffusion.ddim_step(
                    z_t,
                    t_tensor,
                    torch.full((batch_size,), t_prev, device=device, dtype=torch.long),
                    predicted_noise,
                )
            else:
                # The timestep, not the loop counter: noise is dropped only at t == 0 (#482).
                z_t = self.diffusion.p_sample_step(z_t, t_tensor, predicted_noise, t)

        # Decode to image space, then crop/pad to the requested output size.
        # The latent used ``shape[i] // ds`` (a floor), so decode can land a few
        # pixels short of ``shape`` when the in-plane size is not a multiple of
        # ds; align to the caller's shape so metrics/DC see the exact geometry.
        decoded = self.decode_from_latent(z_t)
        return self._match_spatial(decoded, shape[2:])

    def _stacked_denoiser(self, contrast_idx: torch.Tensor | None, batch_size: int):
        """Denoise a stack of ``n * batch_size`` inputs, repeating ``contrast_idx`` to match."""

        def denoise(unet_input, t, keep):
            reps = unet_input.shape[0] // batch_size
            idx = None if contrast_idx is None else contrast_idx.repeat(reps)
            return self._process_slices(
                unet_input, t, "denoise", contrast_idx=idx, contrast_keep=keep
            )

        return denoise

    @staticmethod
    def _match_spatial(x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
        """Center-crop / symmetric zero-pad the trailing spatial dims to ``size``."""
        size = tuple(int(s) for s in size)
        n = len(size)
        if tuple(x.shape[-n:]) == size:
            return x
        # Center-crop any oversized trailing dim first.
        for axis, target in enumerate(size):
            dim = x.dim() - n + axis
            cur = x.shape[dim]
            if cur > target:
                x = x.narrow(dim, (cur - target) // 2, target)
        # Then symmetric-pad any undersized trailing dim. F.pad's list runs from
        # the LAST dim backwards: [d(-1)_lo, d(-1)_hi, d(-2)_lo, d(-2)_hi, ...].
        pad: list[int] = [0, 0] * n
        need_pad = False
        for axis, target in enumerate(size):
            cur = x.shape[x.dim() - n + axis]
            if cur < target:
                need_pad = True
                total = target - cur
                slot = (n - 1 - axis) * 2  # last trailing dim → slot 0
                pad[slot] = total // 2
                pad[slot + 1] = total - total // 2
        if need_pad:
            x = F.pad(x, pad)
        return x

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "latent_diffusion"

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space."""
        # For latent diffusion, z is the shape specification
        if isinstance(z, torch.Tensor) and z.dim() == 1:
            # z contains shape information
            shape = tuple(z.int().tolist())
            return self.sample(shape, device=str(z.device))
        # z is actual latent tensor to decode
        return self.decode_from_latent(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        return input_shape  # Same as input for reconstruction

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_latent_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get latent space shape for given input shape."""
        ds = self.autoencoder.downsample_factor
        return (
            input_shape[0],
            self.latent_channels,
            input_shape[2] // ds,
            input_shape[3] // ds,
            input_shape[4] // ds,
        )

    def to(self, *args, **kwargs):
        """Move model to device, including diffusion process."""
        super().to(*args, **kwargs)
        # Also move the diffusion process
        device = args[0] if args else kwargs.get("device")
        if device is not None:
            self.diffusion.to(device)
        return self


def _parse_channel_multipliers(raw: object) -> tuple[int, ...]:
    """``model_kwargs.channel_multipliers`` -> the latent U-Net's ``channel_mult``.

    Absent keeps the historical ``(1, 2, 4)``. Anything but a non-empty list of
    positive ints raises.
    """
    if raw is None:
        return (1, 2, 4)
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        raise ValueError(f"channel_multipliers must be a list of positive ints; got {raw!r}.")
    mult = tuple(raw)  # type: ignore[arg-type]
    if not mult or not all(isinstance(m, int) and not isinstance(m, bool) and m > 0 for m in mult):
        raise ValueError(f"channel_multipliers must be a non-empty list of positive ints; got {raw!r}.")
    return mult


def _sync_latent_scale_ready(module: nn.Module, incompatible_keys: object) -> None:
    """After ``load_state_dict``, mirror the loaded calibration flag (one sync, at load)."""
    flag = getattr(module, "latent_scale_calibrated", None)
    if flag is not None:
        module._latent_scale_ready = bool(flag)


def create_latent_diffusion_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    **kwargs,
) -> LatentDiffusionGenerator:
    """Factory function for creating latent diffusion generator.

    Args:
        in_channels: Number of input/output channels
        out_channels: Number of output classes (unused for diffusion)
        **kwargs: Additional arguments for LatentDiffusionGenerator

    Returns:
        Configured LatentDiffusionGenerator instance

    """
    return LatentDiffusionGenerator(
        in_channels=in_channels,
        out_channels=in_channels,
        **kwargs,
    )
