"""Conditional Diffusion Generator Implementation
=============================================

Conditional diffusion model for physics-informed reconstruction
in multi-stage training pipelines.
"""

import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from spectramr.config.schemas.enums import SchedulerTypes
from spectramr.models.blocks.embeddings import (
    SinusoidalPositionEmbedding as SinusoidalPositionEmbeddings,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

from .mixins import DiffusionMixin

logger = logging.getLogger(__name__)

# SinusoidalPositionEmbeddings imported from blocks.embeddings.


class ConditionalDiffusionBlock(nn.Module):
    """Conditional diffusion block with physics-informed conditioning"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        conditioning_dim: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_emb_dim (int): Description.
            conditioning_dim (Optional[int]): Description.
        """
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conditioning_mlp = (
            nn.Linear(conditioning_dim, out_channels) if conditioning_dim is not None else None
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.activation = nn.SiLU()

        # Skip connection
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with conditional information

        forward method for ConditionalDiffusionBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            conditioning (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        # Add conditioning if provided
        if conditioning is not None and self.conditioning_mlp is not None:
            cond_emb = self.conditioning_mlp(conditioning)
            h = h + cond_emb[:, :, None, None]

        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)

        return h + self.skip(x)


class ConditionalDiffusionUNet(nn.Module):
    """Conditional UNet for diffusion model"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 128,
        conditioning_dim: int | None = None,
        num_res_blocks: int = 2,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
            channel_mults (tuple[int, ...]): Description.
            time_emb_dim (int): Description.
            conditioning_dim (Optional[int]): Description.
            num_res_blocks (int): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.conditioning_dim = conditioning_dim

        # Time embeddings
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling blocks
        self.downs = nn.ModuleList()
        channels = base_channels
        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                self.downs.append(
                    ConditionalDiffusionBlock(
                        channels,
                        base_channels * mult,
                        time_emb_dim,
                        conditioning_dim,
                    ),
                )
                channels = base_channels * mult

            if i < len(channel_mults) - 1:
                self.downs.append(nn.Conv2d(channels, channels, 3, stride=2, padding=1))
                self.downs.append(nn.GroupNorm(8, channels))
                self.downs.append(nn.SiLU())

        # Middle block
        self.middle = ConditionalDiffusionBlock(
            channels,
            channels,
            time_emb_dim,
            conditioning_dim,
        )

        # Upsampling blocks
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            for _ in range(num_res_blocks):
                self.ups.append(
                    ConditionalDiffusionBlock(
                        channels + base_channels * mult,
                        base_channels * mult,
                        time_emb_dim,
                        conditioning_dim,
                    ),
                )
                channels = base_channels * mult

            if i > 0:
                self.ups.append(
                    nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
                )
                self.ups.append(nn.GroupNorm(8, channels))
                self.ups.append(nn.SiLU())

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through conditional diffusion UNet

        forward method for ConditionalDiffusionUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            conditioning (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Time embeddings
        time_emb = self.time_emb(time)
        time_emb = self.time_mlp(time_emb)

        # Initial convolution
        x = self.init_conv(x)

        # Downsampling with skip connections
        skips = []
        for layer in self.downs:
            if isinstance(layer, ConditionalDiffusionBlock):
                x = layer(x, time_emb, conditioning)
                skips.append(x)
            else:
                x = layer(x)

        # Middle block
        x = self.middle(x, time_emb, conditioning)

        # Upsampling with skip connections
        for layer in self.ups:
            if isinstance(layer, ConditionalDiffusionBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, time_emb, conditioning)
            else:
                x = layer(x)

        return self.final_conv(x)


@register_model(name="conditional_diffusion", training_mode="diffusion")
class ConditionalDiffusionGenerator(DiffusionMixin, nn.Module, IGenerator):
    """Conditional diffusion model for physics-informed reconstruction.

    Features:
    - Time-step conditioning for diffusion process
    - Physics-informed conditioning (e.g., k-space masks)
    - UNet architecture with residual blocks
    - Sinusoidal position embeddings
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 128,
        conditioning_dim: int | None = None,
        num_res_blocks: int = 2,
        noise_schedule: str = "cosine",
        model_id: str | None = None,
        device: str | None = None,
        load_pipeline: bool = False,
        pipeline_class: type[Any] | None = None,
        conditioning_type: str = "cross_attn",
        base_model: str | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
            channel_mults (tuple[int, ...]): Description.
            time_emb_dim (int): Description.
            conditioning_dim (Optional[int]): Description.
            num_res_blocks (int): Description.
            noise_schedule (str): Description.
            model_id (Optional[str]): Description.
            device (Optional[str]): Description.
            load_pipeline (bool): Description.
            pipeline_class (Optional[type[Any]]): Description.
            conditioning_type (str): Description.
            base_model (Optional[str]): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditioning_dim = conditioning_dim
        self.noise_schedule = noise_schedule
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.conditioning_type = conditioning_type
        self.base_model = base_model
        self._pipeline_class = pipeline_class
        self._pipeline = None

        # Adjust in_channels for concat conditioning
        unet_in_channels = in_channels
        # If concat, we assume the input to the UNet will be x + conditioning
        # However, if in_channels in config is already set to sum (e.g. 2), we trust it.
        # But we need to handle the concatenation in forward.

        # Diffusion model
        self.model = ConditionalDiffusionUNet(
            in_channels=unet_in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            time_emb_dim=time_emb_dim,
            conditioning_dim=conditioning_dim,
            num_res_blocks=num_res_blocks,
        )

        # Initialize diffusion buffers via DiffusionMixin
        # Using 'cosine' for cosine schedule, 'linear' for linear
        schedule_type = "cosine" if self.noise_schedule == SchedulerTypes.COSINE.value else "linear"
        self._initialize_diffusion_buffers(timesteps=1000, beta_schedule=schedule_type)

        if self.model_id is not None and load_pipeline:
            self._pipeline = self._load_external_pipeline()

    def _resolve_device(self, requested: str | None) -> torch.device:
        """Resolve the computation device, preferring requested value."""
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _select_pipeline_class(self) -> type[Any] | None:
        """Return the preferred external diffusion pipeline class."""
        if self._pipeline_class is not None:
            return self._pipeline_class
        # Imported here, one try per name: diffusers costs ~0.8 s and is only
        # needed once an external pipeline is actually requested.
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError:
            logger.warning("diffusers.StableDiffusionPipeline unavailable (pip install diffusers)")
        else:
            return StableDiffusionPipeline
        try:
            from diffusers import DiffusionPipeline
        except ImportError:
            logger.warning("diffusers.DiffusionPipeline unavailable (pip install diffusers)")
        else:
            return DiffusionPipeline
        return None

    def _load_external_pipeline(self) -> Any:
        """Instantiate the external diffusion pipeline when requested."""
        if self.model_id is None:
            return None

        pipeline_cls = self._select_pipeline_class()
        if pipeline_cls is None:
            raise RuntimeError(
                "diffusers is not installed; cannot load an external pipeline.",
            )

        pipeline = pipeline_cls.from_pretrained(self.model_id)
        if hasattr(pipeline, "to"):
            pipeline = pipeline.to(self.device)
        return pipeline

    def _ensure_external_pipeline(self) -> Any:
        """Ensure the external pipeline is instantiated before use."""
        if self.model_id is None:
            return None

        if self._pipeline is None:
            self._pipeline = self._load_external_pipeline()
        return self._pipeline

    def _image_to_tensor(self, image: Any) -> torch.Tensor:
        """Convert pipeline image outputs into torch tensors."""
        if hasattr(image, "images"):
            return self._image_to_tensor(image.images)

        if isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("Pipeline returned no images to process.")
            return self._image_to_tensor(image[0])

        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)

        if hasattr(image, "numpy"):
            array = image.numpy()
        else:
            array = np.asarray(image, dtype=np.float32)

        if array.ndim == 2:
            array = array[:, :, None]

        channels = array.shape[2]
        if channels < self.out_channels:
            pad = np.zeros(
                (
                    array.shape[0],
                    array.shape[1],
                    self.out_channels - channels,
                ),
                dtype=array.dtype,
            )
            array = np.concatenate((array, pad), axis=2)
        elif channels > self.out_channels:
            array = array[:, :, : self.out_channels]

        tensor = torch.from_numpy(array.astype(np.float32))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "conditional_diffusion"

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion process (q sampling)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(
            -1,
            1,
            1,
            1,
        )
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise from noisy sample"""
        return self.model(x_t, t, conditioning)

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reverse diffusion process (single step)"""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(
            -1,
            1,
            1,
            1,
        )

        # Predict noise
        predicted_noise = self.predict_noise(x_t, t, conditioning)

        # Compute mean
        mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return mean
        # Add noise for stochasticity
        noise = torch.randn_like(x_t)
        variance = betas_t
        return mean + torch.sqrt(variance) * noise

    def sample(
        self,
        batch_size: int = 1,
        conditioning: torch.Tensor | None = None,
        timesteps: int = 1000,
        height: int = 256,
        width: int = 256,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples using reverse diffusion"""
        device = next(self.parameters()).device

        # Start from pure noise
        if noise is None:
            x_t = torch.randn(
                batch_size,
                self.in_channels,
                height,
                width,
                device=device,
            )
        else:
            x_t = noise.to(device)
            if x_t.size(0) != batch_size:
                raise ValueError(
                    "Provided noise tensor batch size does not match requested batch size.",
                )

        conditioning_tensor = conditioning.to(device) if conditioning is not None else None

        # Reverse diffusion process
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_tensor, conditioning_tensor)

        return x_t

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        return input_shape

    def generate(
        self,
        noise: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples from conditional diffusion model"""
        if self.model_id is not None:
            pipeline = self._ensure_external_pipeline()
            pipeline_kwargs = dict(kwargs)
            prompt = pipeline_kwargs.pop("prompt", "")
            height = pipeline_kwargs.pop("height", 256)
            width = pipeline_kwargs.pop("width", 256)

            result = pipeline(
                prompt,
                height=height,
                width=width,
                **pipeline_kwargs,
            )
            return self._image_to_tensor(result)

        batch_size = kwargs.get("batch_size", 1)
        if noise is not None:
            batch_size = noise.size(0)

        height = kwargs.get("height", 256)
        width = kwargs.get("width", 256)
        timesteps = kwargs.get("timesteps", 1000)

        sample_method = getattr(self.sample, "__func__", None)

        if (
            sample_method is ConditionalDiffusionGenerator.sample
            and noise is None
            and (height != 256 or width != 256)
        ):
            device = next(self.parameters()).device
            noise = torch.randn(
                batch_size,
                self.in_channels,
                height,
                width,
                device=device,
            )

        sample_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "conditioning": conditioning,
            "timesteps": timesteps,
        }

        if sample_method is ConditionalDiffusionGenerator.sample:
            sample_kwargs["height"] = height
            sample_kwargs["width"] = width
            if noise is not None:
                sample_kwargs["noise"] = noise
        elif noise is not None:
            sample_kwargs["noise"] = noise

        return self.sample(**sample_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through diffusion model

        forward method for ConditionalDiffusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            conditioning (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.conditioning_type == "concat" and conditioning is not None:
            # Spatial conditioning via concatenation
            # Ensure conditioning is on same device and size
            if conditioning.device != x.device:
                conditioning = conditioning.to(x.device)

            # Concatenate along channel dimension
            # x: [B, C, H, W], conditioning: [B, C_cond, H, W]
            x_in = torch.cat([x, conditioning], dim=1)
            # Pass None for conditioning to UNet since we already concatenated
            return self.model(x_in, t, conditioning=None)

        return self.model(x, t, conditioning)


def create_conditional_diffusion_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 64,
    channel_mults: tuple[int, ...] = (1, 2, 4, 8),
    time_emb_dim: int = 128,
    conditioning_dim: int | None = None,
    num_res_blocks: int = 2,
    noise_schedule: str = "cosine",
    model_id: str | None = None,
    device: str | None = None,
    load_pipeline: bool = False,
    pipeline_class: type[Any] | None = None,
) -> ConditionalDiffusionGenerator:
    """Create a conditional diffusion generator instance.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        channel_mults: Channel multipliers for each level
        time_emb_dim: Time embedding dimension
        conditioning_dim: Conditioning input dimension
        num_res_blocks: Number of residual blocks per level
        noise_schedule: Type of noise schedule ("linear" or "cosine")
        model_id: Optional diffusers model identifier to leverage external
            pipelines for image generation
        device: Preferred device for both internal model and external pipeline
        load_pipeline: Whether to load the external pipeline eagerly
        pipeline_class: Explicit pipeline class override, if desired

    Returns:
        Configured ConditionalDiffusionGenerator instance

    """
    return ConditionalDiffusionGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        time_emb_dim=time_emb_dim,
        conditioning_dim=conditioning_dim,
        num_res_blocks=num_res_blocks,
        noise_schedule=noise_schedule,
        model_id=model_id,
        device=device,
        load_pipeline=load_pipeline,
        pipeline_class=pipeline_class,
        conditioning_type="cross_attn",  # Default
        base_model=None,
    )
