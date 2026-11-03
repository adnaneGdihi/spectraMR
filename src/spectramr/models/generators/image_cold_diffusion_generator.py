from typing import Any

import torch
from torch import Tensor, nn

from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator
from spectramr.models.diffusion.architectures.image_cold_diffusion_unet import (
    ImageColdDiffusionUNet,
)
from spectramr.models.generators.mixins.diffusion import DiffusionMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


_DEGRADATION_TYPES = frozenset({"noise", "physical"})


@register_model(
    name="image_cold_diffusion",
    training_mode="diffusion",
    # Image-space cold diffusion. The backbone is configurable: a real backbone
    # consumes a 1-ch magnitude image (domain ``image``), while a complex backbone
    # (``backbone_type: complex_unet`` + ``use_complex_conv``) consumes a 2-ch (or
    # 2C-ch coil-stacked) real/imag image (domain ``complex_image``). Declare both
    # so a coil-compressed complex arm (svd → ifft → complex_image) audits clean.
    spatial_dims=(2,),
    input_domain=("image", "complex_image"),
    output_domain=("image", "complex_image"),
)
class ImageColdDiffusionGenerator(nn.Module, DiffusionMixin, IGenerator):
    """
    Image-Space Cold Diffusion Generator.

    Fully supports:
    - Progressive physical corruptions via DigitalTwinSimulator
    - Timestep-conditioned denoising backbones
    - Cross-contrast conditioning via FiLM embeddings
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        timesteps: int = 100,
        beta_schedule: str = "linear",
        degradation_type: str = "physical",
        digital_twin_kwargs: dict | None = None,
        cond_dim: int | None = None,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()

        if degradation_type not in _DEGRADATION_TYPES:
            raise ValueError(
                f"Unknown degradation_type {degradation_type!r}; "
                f"valid: {sorted(_DEGRADATION_TYPES)}."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.degradation_type = degradation_type

        # Initialize Diffusion Buffers
        self._initialize_diffusion_buffers(timesteps=timesteps, beta_schedule=beta_schedule)

        # Initialize DigitalTwinSimulator for 'physical' degradation
        self.simulator = None
        if self.degradation_type == "physical":
            dt_kwargs = digital_twin_kwargs or {}
            # Filter config-level keys that are not DigitalTwinSimulator constructor params
            # Note: degradation_ranges IS a valid DigitalTwinSimulator param — keep it.
            _config_only_keys = {"enabled"}
            dt_kwargs = {k: v for k, v in dt_kwargs.items() if k not in _config_only_keys}
            self.simulator = DigitalTwinSimulator(
                **dt_kwargs,
            )
            # Cold schedule ranges from 1.0 -> 0.0 (Clean -> Corrupted interpolation weighting in reverse)
            # wait, schedule is normalized timestep t/T.
            schedule_linspace = torch.linspace(1.0, 0.0, timesteps)
            self.register_buffer("cold_schedule", schedule_linspace)

        # Initialize Denoising Backbone
        self.denoising_model = ImageColdDiffusionUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_dim=256,
            cond_dim=cond_dim,
            dropout=dropout,
        )

    @property
    def device(self) -> torch.device:
        return self.betas.device

    @property
    def name(self) -> str:
        return "image_cold_diffusion"

    def _degrade(self, x_0_pred: Tensor, t: Tensor) -> Tensor:
        """Degrade predicted x_0 down to x_t."""
        if self.degradation_type == "noise":
            return self.q_sample(x_start=x_0_pred, t=t)
        elif self.degradation_type == "physical" and self.simulator is not None:
            # schedule_val = 1.0 (start t=0), 0.0 (end t=T)
            schedule_val = self.cold_schedule[t.long()]
            degradation_factor = 1.0 - schedule_val  # 0.0 -> 1.0 (Clean -> Corrupted)

            B, C, H, W = x_0_pred.shape
            out = torch.zeros_like(x_0_pred)
            for i in range(B):
                cf = degradation_factor[i].item()
                # DigitalTwinSimulator expects complex image-space input.
                # Cast magnitude input to complex64 so the simulator can perform
                # internal FFT / physics ops, then convert back to real magnitude.
                clean_slice = x_0_pred[i : i + 1].to(torch.complex64)
                corrupted, _, _ = self.simulator(clean_anatomy=clean_slice, corruption_factor=cf)
                # Simulator returns complex — take magnitude to stay in real float domain.
                out[i : i + 1] = corrupted.abs().to(x_0_pred.dtype)
            return out
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")

    def _predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        # ``DiffusionMixin`` declares this abstract; cold diffusion predicts x_0
        # directly and has no noise to invert. A ``pass`` body returned None into
        # any caller that reached it (2026-09-03): say so instead.
        raise NotImplementedError(
            "ImageColdDiffusionGenerator predicts x_0 directly; there is no noise "
            "to invert (cold diffusion has no _predict_start_from_noise)."
        )

    def _model_forward(self, x: Tensor, t: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError(
            "ImageColdDiffusionGenerator routes the model call through forward(); "
            "_model_forward is the mixin's abstract slot, not a cold-diffusion path."
        )

    @torch.no_grad()
    def p_sample(
        self,
        x: Tensor,
        t: Tensor,
        t_index: int,
        **kwargs: Any,
    ) -> Tensor:
        """Reverse cold diffusion sampling step."""
        cond = kwargs.get("cond")
        guidance_scale = kwargs.get("guidance_scale", 1.0)

        if guidance_scale > 1.0 and cond is not None:
            uncond_pred = self.denoising_model(x, t, cond=None)
            cond_pred = self.denoising_model(x, t, cond=cond)
            pred_x_start = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        else:
            pred_x_start = self.denoising_model(x, t, cond=cond)

        if t_index == 0:
            return pred_x_start

        prev_t = torch.full_like(t, max(0, t_index - 1))
        return self._degrade(pred_x_start, prev_t)

    def forward(
        self, x: Tensor, cond: Tensor | None = None, t: Tensor | None = None, **kwargs
    ) -> Tensor:
        """
        Forward pass for training and validation.
        The input `x` is already corrupted (x_t) by DiffusionTrainingStrategy.q_sample().

        forward method for ImageColdDiffusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if t is None:
            # Default to t=0 if no timestep provided
            t = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        # Predict x_0 from x_t (x)
        return self.denoising_model(x, t, cond=cond)

    @torch.no_grad()
    def generate(self, z: Tensor, **kwargs: Any) -> Tensor:
        """Restore ``z``, the degraded input, by the reverse loop from t = T-1 to 0.

        Cold diffusion starts from the measurement, not from noise (#2383).
        """
        x = z
        for i in reversed(range(self.timesteps)):
            t = torch.full((z.shape[0],), i, device=z.device, dtype=torch.long)
            x = self.p_sample(x, t, i, **kwargs)
        return x

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
