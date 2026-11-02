"""Inference for ``training_mode: cascaded_diffusion``: the validation sampler, at predict.

The input is the measurement ``y`` the cascade was conditioned on. Sampling runs
the stage-1 DDIM chain, then the stage-2 chain conditioned on its sample, and
returns their sum, through the same :class:`CascadeSampler` validation uses.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.infrastructure.training.utils.cascade_sampler import (
    CascadeSampler,
    resolve_cascade_schedule,
)

from .base_inference_strategy import BaseInferenceStrategy


def _section(config: dict[str, Any], *path: str) -> Any:
    node: Any = config
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
    return node


class CascadedDiffusionInferenceStrategy(BaseInferenceStrategy):
    """Two-chain DDIM sampling of the spectral cascade (stage 1, then stage 2 on its sample)."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(model, device, config)
        conditioning = _section(self.config, "model", "conditioning") or {}
        if conditioning.get("enabled") and conditioning.get("sources"):
            raise ValueError(
                "cascaded_diffusion infer cannot apply model.conditioning: the FiLM "
                "conditioner is built by the training strategy and is not part of the generator."
            )
        schedule = resolve_cascade_schedule(
            _section(self.config, "training", "diffusion"),
            _section(self.config, "validation", "sampling", "steps"),
        )
        self.sampler = CascadeSampler(model, schedule.build_scheduler(device), schedule)
        self.out_channels = int(_section(self.config, "model", "out_channels"))
        # The validation seed, so infer reproduces the sample validation scored.
        self.seed = int(_section(self.config, "run", "seed")) + 1

    @property
    def training_mode(self) -> TrainingModeTypes:
        return TrainingModeTypes.DIFFUSION

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return input_tensor.to(self.device).detach()

    def run_inference(self, input_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.sampler.sample(input_tensor, self.out_channels, self.seed)

    def postprocess_output(self, output_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return output_tensor.detach().cpu()

    def get_strategy_info(self) -> dict[str, Any]:
        info = super().get_strategy_info()
        schedule = self.sampler.schedule
        info.update(
            {
                "paradigm": "cascaded_diffusion",
                "has_stage_2": bool(self.sampler.cascade.has_stage_2),
                "num_timesteps": schedule.num_timesteps,
                "sampling_steps": schedule.sampling_steps,
                "seed": self.seed,
            }
        )
        return info
