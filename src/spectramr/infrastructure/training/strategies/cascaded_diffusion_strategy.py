r"""Two-stage spectral cascade of conditional x0-denoising diffusion models.

The target splits as :math:`x_0 = P_L x_0 + P_H x_0`, with :math:`P_L` the
low-pass circle of ``model_kwargs.stage_1`` (:mod:`~spectramr.infrastructure.physics.spectral_cascade_bands`).
Stage 1 is a Gaussian DDPM on :math:`P_L x_0` conditioned on the measurement
:math:`y`. Stage 2 is a DDPM on the high-pass residual :math:`P_H x_0`,
conditioned on :math:`y` and on the stage-1 output. Each stage's output is
projected onto its own band, so the composed estimate
:math:`\hat x_0 = P_L \hat x_0^{(1)} + P_H \hat x_0^{(2)}` gives the declared
image losses an objective that, for MSE, is exactly the sum of the two
per-band errors (Parseval).

Training draws independent timesteps for the two stages. Stage 2 is conditioned
on the stage-1 x0 estimate at the stage-1 timestep, detached, so it sees
conditioning of every quality the stage-1 sampler passes through. Validation
runs the stage-1 deterministic DDIM chain, then the stage-2 chain conditioned on
that sample, and returns their sum. Both stages use the scheduler built by
:func:`build_training_scheduler` from ``training.diffusion``.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

from spectramr.infrastructure.training.utils.cascade_sampler import (
    CascadeSampler,
    resolve_cascade_schedule,
)

from .reconstruction import ReconstructionTrainingStrategy

#: ``training.diffusion`` fields this strategy reads; any other explicitly set field raises.
_READ_DIFFUSION_FIELDS = frozenset(
    {
        "type",
        "timesteps",
        "noise_schedule",
        "beta_start",
        "beta_end",
        "sampler",
        "sampling_steps",
        "prediction_type",
    }
)


class CascadedDiffusionStrategy(ReconstructionTrainingStrategy):
    """Stage 1 on the low band, stage 2 on the high-pass residual given stage 1's output."""

    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = True

    def _refuse_inert_diffusion_knobs(self) -> None:
        """Validate ``training.diffusion`` against what this strategy reads."""
        diffusion = self.config.training.diffusion
        if diffusion is None:
            raise ValueError("CascadedDiffusionStrategy needs a training.diffusion block.")
        unread = sorted(
            (set(diffusion.model_fields_set) | set(diffusion.model_extra or {}))
            - _READ_DIFFUSION_FIELDS
        )
        if unread:
            raise ValueError(
                f"training.diffusion.{unread} is set but CascadedDiffusionStrategy never reads "
                f"it; it reads {sorted(_READ_DIFFUSION_FIELDS)}."
            )

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("cascaded_diffusion",))
        self._refuse_inert_diffusion_knobs()
        validation = getattr(self.config, "validation", None)
        schedule = resolve_cascade_schedule(
            self.config.training.diffusion,
            validation.sampling.steps if validation is not None else None,
        )
        self.scheduler = schedule.build_scheduler(self.device)
        self._seed = int(self.config.run.seed)
        self._sampler = CascadeSampler(
            cast(nn.Module, self.env.generator), self.scheduler, schedule
        )
        cascade = self._cascade_model()
        if self.logging_service:
            self.logging_service.log_info(
                f"[CascadedDiffusion] stage_1={cascade.kspace_filter} "
                f"cutoff={cascade.cutoff_frequency} stage_2={cascade.has_stage_2} "
                f"(conditioned on detached stage-1 x0), T={self.num_timesteps}, "
                f"DDIM steps={self.sampling_steps}"
            )
            self._log_config_features(self.logging_service)

    @property
    def num_timesteps(self) -> int:
        return self._sampler.schedule.num_timesteps

    @property
    def sampling_steps(self) -> int:
        return self._sampler.schedule.sampling_steps

    def _cascade_model(self) -> Any:
        return self._cascade_sampler().cascade

    def _cascade_sampler(self) -> CascadeSampler:
        """The shared sampler, bound to the current (possibly re-wrapped) generator."""
        self._sampler.generator = cast(nn.Module, self.env.generator)
        return self._sampler

    def split_bands(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(P_L x, P_H x)``; the all-pass control puts everything in the low band."""
        return self._cascade_sampler().split_bands(x)

    def _stage(
        self,
        y: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        stage_1_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._cascade_sampler().stage(y, z_t, t, stage_1_output=stage_1_output)

    def _generate_predictions(
        self,
        lr_image: torch.Tensor,
        forward_kwargs: dict[str, Any],
        batch_context: dict[str, Any],
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        y = self._apply_input_conditioning(lr_image, batch_context)
        x0 = batch_context["hr"]
        if x0.shape[0] != y.shape[0] or x0.shape[2:] != y.shape[2:]:
            raise ValueError(
                f"target {tuple(x0.shape)} and measurement {tuple(y.shape)} must share batch "
                "and grid: the cascade concatenates them on the channel axis."
            )
        low, high = self.split_bands(x0)
        batch = x0.shape[0]
        t1 = self.scheduler.sample_timesteps(batch)
        s1 = self._stage(y, self.scheduler.q_sample(low, t1), t1)
        if not self._cascade_model().has_stage_2:
            return s1, None
        t2 = self.scheduler.sample_timesteps(batch)
        s2 = self._stage(y, self.scheduler.q_sample(high, t2), t2, stage_1_output=s1.detach())
        return s1 + s2, None

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        lr_image, _ = self._prepare_generator_inputs(batch_context, input_batch, validation=True)
        y = self._apply_input_conditioning(lr_image, batch_context)
        out_channels = int(self.config.model.out_channels)
        return self._cascade_sampler().sample(y, out_channels, self._seed + 1)
