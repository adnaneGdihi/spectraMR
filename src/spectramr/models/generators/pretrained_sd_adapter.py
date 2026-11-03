"""Pretrained Stable Diffusion UNet with a measurement adapter (ControlNet or concat).

The backbone is a ``diffusers`` ``UNet2DConditionModel`` loaded with ``from_pretrained``
and frozen; only the pixel adapters (``conv_in``/``conv_out``, re-sized to the MRI
channel count), a learned null prompt and the conditioning adapter train.

The measurement reaches the model through ``training.diffusion.condition_on_input``:
the strategy concatenates the undersampled image after the noised target, so the
first ``out_channels`` input channels are ``x_t`` and the rest are the control image.

* ``adapter_type: controlnet`` -- a trainable copy of the UNet encoder
  (``ControlNetModel.from_unet``) reads ``x_t`` plus the control image and adds its
  zero-initialised residuals to the frozen UNet's skip connections (Zhang et al.,
  ICCV 2023). The conditioning embedding keeps full resolution because the UNet runs
  on pixels, not on 8x-downsampled latents.
* ``adapter_type: concat`` -- ``conv_in`` reads every input channel (SR3-style).

``diffusers`` is optional (``.[diffusion]``); construction without it raises.
"""

from __future__ import annotations

import copy
import importlib
import logging
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn

from spectramr.models.generators.option_parsing import parse_enum_option
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

DEFAULT_SD_BASE_MODEL = "stabilityai/stable-diffusion-2-1-base"

__all__ = [
    "DEFAULT_SD_BASE_MODEL",
    "PretrainedSDAdapterGenerator",
    "SDAdapterType",
    "load_sd_unet",
]


class SDAdapterType(StrEnum):
    """How the control image enters the pretrained UNet."""

    CONTROLNET = "controlnet"
    CONCAT = "concat"


def _import_diffusers() -> Any:
    try:
        return importlib.import_module("diffusers")
    except Exception as exc:  # ImportError incl. sys.modules['diffusers'] = None
        raise ImportError(
            "model_type 'pretrained_sd_adapter' requires the optional 'diffusers' package. "
            "Install it with `pip install -e '.[diffusion]'`."
        ) from exc


def load_sd_unet(base_model_path: str) -> nn.Module:
    """Load the ``unet`` subfolder of a Stable Diffusion repo id or local directory."""
    diffusers = _import_diffusers()
    try:
        return diffusers.UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the pretrained Stable Diffusion UNet '{base_model_path}'. On an "
            "offline compute node the weights must already be in the local HuggingFace cache: "
            f"pre-fetch them on a login node (`huggingface-cli download {base_model_path}`, "
            "sharing HF_HOME with the job), or point model.model_kwargs.base_model_path at a "
            f"local directory that holds a `unet/` subfolder. Underlying error: {exc}"
        ) from exc


def _resized_conv_in(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """``conv`` re-sized to ``in_channels`` inputs, each new channel taking the summed filter."""
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, padding=conv.padding)
    with torch.no_grad():
        summed = conv.weight.sum(dim=1, keepdim=True) / in_channels
        new.weight.copy_(summed.expand(-1, in_channels, -1, -1))
        new.bias.copy_(conv.bias)
    return new


def _resized_conv_out(conv: nn.Conv2d, out_channels: int) -> nn.Conv2d:
    """``conv`` re-sized to ``out_channels`` outputs, each taking the mean pretrained filter."""
    new = nn.Conv2d(conv.in_channels, out_channels, conv.kernel_size, padding=conv.padding)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(dim=0, keepdim=True).expand(out_channels, -1, -1, -1))
        new.bias.fill_(float(conv.bias.mean()))
    return new


@register_model(
    name="pretrained_sd_adapter",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    supports_contrast_conditioning=False,
    strict_kwargs=True,
)
class PretrainedSDAdapterGenerator(nn.Module):
    """Frozen pretrained SD UNet denoiser conditioned on the measurement through an adapter.

    Args:
        in_channels: ``out_channels`` noised-target channels plus the control channels.
        out_channels: Channels of the predicted noise / image.
        base_model_path: HuggingFace repo id or local directory of a Stable Diffusion
            checkpoint with a ``unet/`` subfolder.
        adapter_type: ``controlnet`` or ``concat`` (see the module docstring).
        freeze_backbone: Freeze the pretrained UNet except its re-sized pixel adapters.
        unet: Pre-built ``UNet2DConditionModel`` (test seam); skips ``from_pretrained``.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_model_path: str = DEFAULT_SD_BASE_MODEL,
        adapter_type: str = SDAdapterType.CONTROLNET.value,
        freeze_backbone: bool = True,
        unet: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.adapter_type = parse_enum_option(SDAdapterType, adapter_type, "adapter_type")
        if not isinstance(freeze_backbone, bool):
            raise ValueError(f"freeze_backbone must be a bool, got {freeze_backbone!r}.")
        control_channels = int(in_channels) - int(out_channels)
        if control_channels < 0:
            raise ValueError(f"in_channels={in_channels} is below out_channels={out_channels}.")
        if self.adapter_type is SDAdapterType.CONTROLNET and control_channels == 0:
            raise ValueError(
                "adapter_type='controlnet' needs a control image: set "
                "training.diffusion.condition_on_input: true and in_channels = 2 * out_channels."
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_model_path = base_model_path
        self.freeze_backbone = freeze_backbone

        unet = load_sd_unet(base_model_path) if unet is None else unet
        cfg = unet.config
        if getattr(cfg, "addition_embed_type", None) or getattr(cfg, "class_embed_type", None):
            raise ValueError(
                f"'{base_model_path}' needs added/class embeddings (e.g. SDXL); only "
                "text-conditioned SD 1.x/2.x UNets are supported."
            )

        self.controlnet: nn.Module | None = None
        if self.adapter_type is SDAdapterType.CONTROLNET:
            # Built before conv_in is re-sized: from_unet copies weights by the UNet config.
            self.controlnet = _import_diffusers().ControlNetModel.from_unet(
                unet,
                conditioning_channels=control_channels,
                conditioning_embedding_out_channels=(16,),
            )
        backbone_in = self.out_channels if self.controlnet is not None else self.in_channels
        unet.conv_in = _resized_conv_in(unet.conv_in, backbone_in)
        unet.conv_out = _resized_conv_out(unet.conv_out, self.out_channels)
        if freeze_backbone:
            unet.requires_grad_(False)
            unet.conv_in.requires_grad_(True)
            unet.conv_out.requires_grad_(True)
        if self.controlnet is not None:
            self.controlnet.conv_in = copy.deepcopy(unet.conv_in).requires_grad_(True)
        self.unet = unet
        self.null_context = nn.Parameter(torch.zeros(1, 1, int(cfg.cross_attention_dim)))

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "pretrained_sd_adapter: base=%s adapter=%s freeze_backbone=%s trainable=%d/%d params",
            base_model_path,
            self.adapter_type.value,
            freeze_backbone,
            trainable,
            total,
        )

    def set_grad_checkpointing(self, enable: bool) -> None:
        """Toggle ``diffusers`` gradient checkpointing on the UNet and the ControlNet."""
        for module in (self.unet, self.controlnet):
            if module is None:
                continue
            if enable:
                module.enable_gradient_checkpointing()
            else:
                module.disable_gradient_checkpointing()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict from ``x = [x_t, control]`` stacked on the channel axis, ``[B, C, H, W]``."""
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"pretrained_sd_adapter expects [B, {self.in_channels}, H, W], got {tuple(x.shape)}."
            )
        context = self.null_context.expand(x.shape[0], -1, -1).to(x.dtype)
        if self.controlnet is None:
            return self.unet(x, timesteps, encoder_hidden_states=context, return_dict=False)[0]
        noisy, control = x[:, : self.out_channels], x[:, self.out_channels :]
        down, mid = self.controlnet(
            noisy,
            timesteps,
            encoder_hidden_states=context,
            controlnet_cond=control,
            return_dict=False,
        )
        return self.unet(
            noisy,
            timesteps,
            encoder_hidden_states=context,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
            return_dict=False,
        )[0]
