"""One-shot latent regression with the stage-2 LDM's own modules (LM-13 control).

Objective 1 of the ULF->HF LDM cohort compared the conditional LDM against a
raster-scan Mamba-UNet trained with a different loss, so the delta mixed
architecture, objective and sampler. This control holds the LDM's parts fixed
and removes only the diffusion:

* the same frozen autoencoder (custom AutoencoderKL from stage 1, or the sd-VAE),
  encode path and latent standardisation;
* the same latent U-Net topology, SR3 concat of the ULF latent and contrast
  embedding;
* one deterministic pass: the noisy-latent slot is zero and the timestep is
  fixed at ``timesteps - 1``, i.e. the LDM's input at pure noise, and the U-Net
  output is read as the HF latent rather than as noise.

The model composes a :class:`LatentDiffusionGenerator` rather than copying its
modules, so an architecture change to the LDM reaches the control too.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from spectramr.models.registry import register_model

from .latent_diffusion_generator import LatentDiffusionGenerator

__all__ = ["LatentRegressionGenerator"]

_PREFIX = "ldm."


@register_model(
    name="latent_regression",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    supports_contrast_conditioning=True,
    requires_paired_data=True,
)
class LatentRegressionGenerator(nn.Module):
    """Predict the HF latent from the ULF latent in one U-Net pass, then decode."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.ldm = LatentDiffusionGenerator(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
        if not self.ldm.config.conditional_translation:
            raise ValueError(
                "latent_regression maps the ULF latent to the HF latent, so it needs "
                "conditional_translation: true (the SR3 concat it reuses)."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, contrast_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.ldm.config.use_contrast_guidance and contrast_id is None:
            raise ValueError(
                "use_contrast_guidance=True but contrast_id is None: the batch did not "
                "carry it, so this control would run contrast-agnostic while the LDM "
                "it is compared against is contrast-conditioned."
            )
        cond = self.ldm.encode_condition(x)
        t = torch.full((x.shape[0],), self.ldm.timesteps - 1, device=x.device, dtype=torch.long)
        unet_input = torch.cat([torch.zeros_like(cond), cond], dim=1)
        z_hat = self.ldm._process_slices(unet_input, t, "denoise", contrast_idx=contrast_id)
        return self.ldm._match_spatial(self.ldm.decode_from_latent(z_hat), tuple(x.shape[2:]))

    def load_autoencoder_state(
        self, state: Mapping[str, torch.Tensor], *, freeze: bool, source: str
    ) -> None:
        """Delegate the stage-1 load, stripping this wrapper's ``ldm.`` prefix.

        The builder re-freezes before a checkpoint load by passing this model's own
        ``state_dict()``, whose keys start ``ldm.autoencoder.``.
        """
        stripped = {
            (k[len(_PREFIX) :] if k.startswith(_PREFIX) else k): v for k, v in state.items()
        }
        self.ldm.load_autoencoder_state(stripped, freeze=freeze, source=source)

    @property
    def name(self) -> str:
        return "latent_regression"
