"""The spectral-cascade denoiser calls and DDIM sampler, shared by validation and ``infer``.

``CascadedDiffusionStrategy`` (training, validation) and
``CascadedDiffusionInferenceStrategy`` (``spectramr infer``) both sample through
:class:`CascadeSampler`, so the two paths run the same stage-1 chain, the same
stage-2 chain conditioned on its sample, and the same band projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.physics.spectral_cascade_bands import (
    low_pass_circle_mask,
    split_low_high,
)
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import DiffusionScheduler
from spectramr.infrastructure.training.utils.diffusion_mixin import build_training_scheduler
from spectramr.models.diffusion.reverse_path import resolve_reverse_steps

__all__ = ["CascadeSampler", "CascadeSchedule", "resolve_cascade_schedule"]

_TYPES = (None, "ddim")
_SCHEDULES = ("linear", "cosine")
_SAMPLERS = ("ddim",)
_PREDICTION_TYPES = ("sample",)


def _field(block: Any, name: str) -> Any:
    value = block.get(name) if isinstance(block, dict) else getattr(block, name, None)
    return getattr(value, "value", value)


@dataclass(frozen=True)
class CascadeSchedule:
    """The forward process and reverse step count read from ``training.diffusion``."""

    num_timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float
    sampling_steps: int

    def build_scheduler(self, device: torch.device) -> DiffusionScheduler:
        return build_training_scheduler(
            num_timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule,
            device=device,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )


def resolve_cascade_schedule(diffusion: Any, validation_steps: int | None) -> CascadeSchedule:
    """Read and validate the schedule from a ``training.diffusion`` block or its dump."""
    if diffusion is None:
        raise ValueError("The spectral cascade needs a training.diffusion block.")
    checks = (
        ("type", _TYPES),
        ("noise_schedule", _SCHEDULES),
        ("sampler", _SAMPLERS),
        ("prediction_type", _PREDICTION_TYPES),
    )
    for name, allowed in checks:
        if _field(diffusion, name) not in allowed:
            raise ValueError(
                f"training.diffusion.{name}={_field(diffusion, name)!r} is not implemented by "
                f"the spectral cascade; allowed {allowed}."
            )
    num_timesteps = int(_field(diffusion, "timesteps"))
    steps = resolve_reverse_steps(
        validation_steps, _field(diffusion, "sampling_steps"), num_timesteps
    )
    if steps is None or int(steps) < 1:
        raise ValueError("The spectral cascade needs a positive sampling step count.")
    return CascadeSchedule(
        num_timesteps=num_timesteps,
        beta_schedule=str(_field(diffusion, "noise_schedule")),
        beta_start=float(_field(diffusion, "beta_start")),
        beta_end=float(_field(diffusion, "beta_end")),
        sampling_steps=int(steps),
    )


class CascadeSampler:
    """Band-projected stage calls and the two-chain DDIM sampler of the cascade."""

    def __init__(
        self, generator: nn.Module, scheduler: DiffusionScheduler, schedule: CascadeSchedule
    ) -> None:
        self.generator = generator
        self.scheduler = scheduler
        self.schedule = schedule
        self._band_masks: dict[tuple[int, int, str], torch.Tensor] = {}
        self.cascade  # noqa: B018 -- refuse a non-cascade model at construction

    @property
    def cascade(self) -> Any:
        cascade = unwrap_model(self.generator)
        for attr in ("kspace_filter", "cutoff_frequency", "has_stage_2"):
            if not hasattr(cascade, attr):
                raise TypeError(
                    "The spectral cascade needs model_type: cascaded_diffusion, got "
                    f"{type(cascade).__name__}."
                )
        return cascade

    def split_bands(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(P_L x, P_H x)``; the all-pass control puts everything in the low band."""
        cascade = self.cascade
        if cascade.kspace_filter == "all_pass":
            return x, torch.zeros_like(x)
        h, w = int(x.shape[-2]), int(x.shape[-1])
        key = (h, w, str(x.device))
        mask = self._band_masks.get(key)
        if mask is None:
            mask = low_pass_circle_mask(h, w, cascade.cutoff_frequency, device=x.device)
            self._band_masks[key] = mask
        return split_low_high(x, mask)

    def stage(
        self,
        y: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        stage_1_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One denoiser call, projected onto the band of the stage it ran."""
        out = self.generator(
            torch.cat([y, z_t.to(y.dtype)], dim=1), t, stage_1_output=stage_1_output
        )
        out = out[0] if isinstance(out, tuple) else out
        low, high = self.split_bands(out)
        return low if stage_1_output is None else high

    def ddim_chain(
        self,
        y: torch.Tensor,
        shape: tuple[int, ...],
        gen: torch.Generator,
        stage_1_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM over a strided schedule with the x0-parameterised denoiser."""
        horizon = self.schedule.num_timesteps
        seq = sorted(
            set(
                torch.linspace(horizon - 1, 0, self.schedule.sampling_steps).round().long().tolist()
            ),
            reverse=True,
        )
        z = torch.randn(shape, generator=gen).to(y.device, non_blocking=True)
        x0_hat = z
        abar = self.scheduler.alphas_cumprod
        for i, t in enumerate(seq):
            t_vec = torch.full((shape[0],), t, dtype=torch.long, device=y.device)
            x0_hat = self.stage(y, z, t_vec, stage_1_output=stage_1_output)
            t_prev = seq[i + 1] if i + 1 < len(seq) else -1
            eps = (z - abar[t].sqrt() * x0_hat) / (1.0 - abar[t]).sqrt()
            z = self.scheduler.ddim_step(z, t, t_prev, eps)
        return x0_hat

    def sample(self, y: torch.Tensor, out_channels: int, seed: int) -> torch.Tensor:
        """Stage-1 chain, then the stage-2 chain conditioned on its sample; their sum."""
        shape = (y.shape[0], out_channels, *y.shape[2:])
        gen = torch.Generator(device="cpu").manual_seed(seed)
        s1 = self.ddim_chain(y, shape, gen)
        if not self.cascade.has_stage_2:
            return s1
        return s1 + self.ddim_chain(y, shape, gen, stage_1_output=s1)
