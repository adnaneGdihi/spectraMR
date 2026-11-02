"""Two-stage spectral cascade of conditional diffusion denoisers.

Stage 1 denoises the low-pass band of the image inside a circle of radius
``stage_1.cutoff_frequency`` k-space samples. Stage 2 denoises the high-pass
residual and is conditioned on the stage-1 output. Each stage is a
time-conditioned U-Net that predicts its band's clean image (x0) from
``[y, (stage-1 output,) x_t]``, where ``y`` is the measurement. The band split
and the noising schedule belong to ``CascadedDiffusionStrategy``. Setting
``stage_2: null`` with ``stage_1.kspace_filter: all_pass`` gives the matched
single-stage control.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import UNet, UNetConfig
from spectramr.models.registry import register_model

#: Closed vocabularies of the stage blocks; anything else raises (non-negotiable 3).
STAGE_1_FILTERS: tuple[str, ...] = ("low_pass_circle", "all_pass")
STAGE_2_CONDITIONING: tuple[str, ...] = ("stage_1_output",)
STAGE_2_TARGETS: tuple[str, ...] = ("high_pass_residual",)

_STAGE_1_KEYS = frozenset({"kspace_filter", "cutoff_frequency"})
_STAGE_2_KEYS = frozenset({"input_conditioning", "prediction_target"})


def _check_keys(block: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise ValueError(
            f"model_kwargs.{name} has unknown keys {unknown}; allowed {sorted(allowed)}."
        )


def _parse_stage_1(block: Mapping[str, Any] | None) -> tuple[str, int | None]:
    if block is None:
        raise ValueError("model_kwargs.stage_1 is required (kspace_filter, cutoff_frequency).")
    _check_keys(block, _STAGE_1_KEYS, "stage_1")
    kspace_filter = block.get("kspace_filter")
    if kspace_filter not in STAGE_1_FILTERS:
        raise ValueError(
            f"model_kwargs.stage_1.kspace_filter={kspace_filter!r} not in {STAGE_1_FILTERS}."
        )
    cutoff = block.get("cutoff_frequency")
    if kspace_filter == "all_pass":
        if cutoff is not None:
            raise ValueError(
                "stage_1.cutoff_frequency is read only by kspace_filter=low_pass_circle."
            )
        return kspace_filter, None
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 1:
        raise ValueError(
            f"model_kwargs.stage_1.cutoff_frequency must be a positive int radius in k-space "
            f"samples, got {cutoff!r}."
        )
    return kspace_filter, cutoff


def _parse_stage_2(block: Mapping[str, Any] | None, kspace_filter: str) -> bool:
    if kspace_filter == "all_pass":
        if block is not None:
            raise ValueError(
                "stage_2 needs a stage_1 low_pass_circle split; all_pass leaves no residual."
            )
        return False
    if block is None:
        raise ValueError(
            "stage_1.kspace_filter=low_pass_circle needs a stage_2 block for the residual."
        )
    _check_keys(block, _STAGE_2_KEYS, "stage_2")
    if block.get("input_conditioning") not in STAGE_2_CONDITIONING:
        raise ValueError(
            f"model_kwargs.stage_2.input_conditioning={block.get('input_conditioning')!r} "
            f"not in {STAGE_2_CONDITIONING}."
        )
    if block.get("prediction_target") not in STAGE_2_TARGETS:
        raise ValueError(
            f"model_kwargs.stage_2.prediction_target={block.get('prediction_target')!r} "
            f"not in {STAGE_2_TARGETS}."
        )
    return True


@register_model(name="cascaded_diffusion", training_mode="cascaded_diffusion", strict_kwargs=True)
class CascadedDiffusionGenerator(nn.Module, IGenerator):
    """Stage-1 low-band and stage-2 high-band residual x0-denoisers."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] | list[int] = (32, 64, 128, 256),
        time_dim: int = 256,
        stage_1: Mapping[str, Any] | None = None,
        stage_2: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kspace_filter, self.cutoff_frequency = _parse_stage_1(stage_1)
        self.has_stage_2 = _parse_stage_2(stage_2, self.kspace_filter)
        widths = tuple(int(f) for f in features)
        if len(widths) < 2:
            raise ValueError(f"features needs at least two widths, got {widths}.")

        def _unet(cin: int) -> UNet:
            # Explicit UNetConfig: the kwargs path validates ``features`` but builds its defaults.
            return UNet(
                config=UNetConfig(
                    in_channels=cin,
                    out_channels=self.out_channels,
                    features=widths,
                    depth=len(widths) - 1,
                    time_dim=int(time_dim),
                )
            )

        # Stage 1 sees [y, x_t]; stage 2 sees [y, stage-1 output, r_t].
        self.stage_1_net = _unet(self.in_channels + self.out_channels)
        self.stage_2_net = (
            _unet(self.in_channels + 2 * self.out_channels) if self.has_stage_2 else None
        )

    def denoise_stage_1(
        self, y: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """x0 estimate of the stage-1 band from ``[y, x_t]`` at ``timesteps``."""
        return self.stage_1_net(torch.cat([y, x_t], dim=1), timesteps)

    def denoise_stage_2(
        self,
        y: torch.Tensor,
        stage_1_output: torch.Tensor,
        r_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """x0 estimate of the high-pass residual, conditioned on the stage-1 output."""
        if self.stage_2_net is None:
            raise RuntimeError("This cascade was built without stage_2 (kspace_filter=all_pass).")
        return self.stage_2_net(torch.cat([y, stage_1_output, r_t], dim=1), timesteps)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        stage_1_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 1 on ``x = [y, x_t]``, or stage 2 on ``x = [y, r_t]`` when given ``stage_1_output``."""
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        y, z = x[:, : self.in_channels], x[:, self.in_channels :]
        if stage_1_output is None:
            return self.denoise_stage_1(y, z, timesteps)
        return self.denoise_stage_2(y, stage_1_output, z, timesteps)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        return "cascaded_diffusion"
