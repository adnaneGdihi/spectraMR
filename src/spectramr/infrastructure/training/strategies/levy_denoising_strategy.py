r"""Conditional alpha-stable denoising: train ``D(y, x_t, sigma) -> x_0``, sample with DLIM.

Forward corruption (variance-exploding, one shot)
:math:`x_t = x_0 + \sigma_t\,\varepsilon`, :math:`\varepsilon\sim S_\alpha(0,1)` i.i.d.
symmetric stable [Shariatian et al., DLPM, arXiv:2407.18609]. The network input
is the channel concatenation ``[y, x_t, log sigma_t]`` of the conditioning image
``y`` (the ULF input), the noisy target and a constant noise-level map, and its
output is :math:`\hat x_0`.

Objective. The declared ``losses.image_losses`` act on :math:`\hat x_0` against
:math:`x_0`. With an MSE term the minimiser is :math:`\mathbb{E}[x_0\mid y,x_t,\sigma_t]`
under any noise law: the data are bounded, so the objective stays finite for
:math:`\alpha<2` even though :math:`\varepsilon` has infinite variance (the
noise-prediction target :math:`\varepsilon` would not). Tweedie's identity, and so a
score, holds only at :math:`\alpha=2`; nothing here relies on it.

Validation runs the deterministic DLIM step of DLPM (Algorithm 4) in its
variance-exploding form, :math:`x_{i+1}=\hat x_0+(\sigma_{i+1}/\sigma_i)(x_i-\hat x_0)`,
on a geometric noise ladder from :math:`\sigma_{\max}` to :math:`\sigma_{\min}`, and
returns the last :math:`\hat x_0`; ``resetting_sde`` (:mod:`.resetting_sde_sampler`) is
the stochastic alternative at :math:`\alpha=2`. Noise draws come from seeded CPU streams, so
validation is reproducible and does not consume the training stream.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn

from spectramr.config.schemas.training.levy_denoising import LEVY_DENOISING_MODES
from spectramr.models.diffusion.alpha_stable_levy import sample_alpha_stable

from .reconstruction import ReconstructionTrainingStrategy
from .resetting_sde_sampler import noise_ladder, resetting_sde_sampler
from .rng_streams import rank_seeded_generator

#: (strategy, conditioning, generator) -> x0 estimate. One entry per declared sampler.
Sampler = Callable[["LevyDenoisingStrategy", torch.Tensor, torch.Generator], torch.Tensor]


def dlim_sampler(
    strategy: LevyDenoisingStrategy, cond: torch.Tensor, gen: torch.Generator
) -> torch.Tensor:
    """Deterministic DLIM: keep the predicted noise direction, shrink its scale."""
    cfg = strategy.levy_cfg
    sigmas = noise_ladder(cfg.sigma_max, cfg.sigma_min, cfg.sampling_steps)
    x = sigmas[0] * strategy.draw_noise(strategy.target_shape(cond), cond.device, gen)
    x0 = x
    for i, sigma in enumerate(sigmas):
        x0 = strategy.denoise(cond, x, torch.full((cond.shape[0],), sigma, device=cond.device))
        if i + 1 < len(sigmas):
            x = x0 + (sigmas[i + 1] / sigma) * (x - x0)
    return x0


SAMPLERS: dict[str, Sampler] = {"dlim": dlim_sampler, "resetting_sde": resetting_sde_sampler}


class LevyDenoisingStrategy(ReconstructionTrainingStrategy):
    """Train a conditional x0-denoiser under alpha-stable corruption; validate by sampling."""

    #: The builder's declared image losses are the denoising objective (on x0_hat).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = True
    checkpointed_streams: ClassVar[tuple[str, ...]] = ("_levy_rng",)

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=LEVY_DENOISING_MODES)
        self._refuse_inert_diffusion_knobs()
        cfg = self.config.training.levy_denoising
        if cfg is None:
            raise ValueError(
                "LevyDenoisingStrategy needs a training.levy_denoising block "
                "(alpha, sigma_min, sigma_max, sampler, sampling_steps)."
            )
        if cfg.sampler not in SAMPLERS:
            raise ValueError(
                f"training.levy_denoising.sampler {cfg.sampler!r} not in {sorted(SAMPLERS)}."
            )
        self.levy_cfg = cfg
        self._seed = int(self.config.run.seed)
        self._levy_rng = rank_seeded_generator(self._seed)
        self._out_channels = int(self.config.model.out_channels)
        if self.logging_service:
            self._log_config_features(self.logging_service)

    def target_shape(self, cond: torch.Tensor) -> tuple[int, ...]:
        return (cond.shape[0], self._out_channels, *cond.shape[2:])

    def draw_noise(
        self, shape: tuple[int, ...], device: torch.device, gen: torch.Generator
    ) -> torch.Tensor:
        """Unit-scale symmetric stable draw on the seeded CPU stream, optionally clipped."""
        eps = sample_alpha_stable(shape, self.levy_cfg.alpha, generator=gen)
        if self.levy_cfg.noise_clip is not None:
            eps = eps.clamp_(-self.levy_cfg.noise_clip, self.levy_cfg.noise_clip)
        return eps.to(device, non_blocking=True)

    def draw_sigma(self, batch: int, device: torch.device) -> torch.Tensor:
        """Log-uniform noise scale per sample on ``[sigma_min, sigma_max]``."""
        lo, hi = math.log(self.levy_cfg.sigma_min), math.log(self.levy_cfg.sigma_max)
        u = torch.rand(batch, generator=self._levy_rng)
        return torch.exp(lo + (hi - lo) * u).to(device, non_blocking=True)

    def denoise(self, cond: torch.Tensor, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """``D([y, x_t, log sigma]) -> x0_hat``; the model's in_channels must match."""
        level = (
            sigma.log()
            .reshape(-1, 1, *([1] * (cond.ndim - 2)))
            .expand(cond.shape[0], 1, *cond.shape[2:])
        )
        net_in = torch.cat([cond, x_t.to(cond.dtype), level.to(cond.dtype)], dim=1)
        expected = int(self.config.model.in_channels)
        if self.levy_cfg.denoiser_in_channels(cond.shape[1], x_t.shape[1]) != expected:
            raise ValueError(
                f"LevyDenoisingStrategy feeds [y ({cond.shape[1]}), x_t ({x_t.shape[1]}), "
                f"log sigma (1)] = {net_in.shape[1]} channels, but model.in_channels={expected}."
            )
        out = cast(nn.Module, self.env.generator)(net_in)
        return out[0] if isinstance(out, tuple) else out

    def _generate_predictions(
        self,
        lr_image: torch.Tensor,
        forward_kwargs: dict[str, Any],
        batch_context: dict[str, Any],
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        cond = self._apply_input_conditioning(lr_image, batch_context)
        x0 = batch_context["hr"]
        if x0.shape != self.target_shape(cond):
            raise ValueError(
                f"target {tuple(x0.shape)} does not match the denoiser output "
                f"{self.target_shape(cond)}: the conditioning and target grids must agree."
            )
        sigma = self.draw_sigma(x0.shape[0], x0.device)
        eps = self.draw_noise(tuple(x0.shape), x0.device, self._levy_rng)
        x_t = x0 + sigma.reshape(-1, *([1] * (x0.ndim - 1))) * eps
        return self.denoise(cond, x_t, sigma), None

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        lr_image, _ = self._prepare_generator_inputs(batch_context, input_batch, validation=True)
        cond = self._apply_input_conditioning(lr_image, batch_context)
        gen = torch.Generator(device="cpu").manual_seed(self._seed + 1)
        return SAMPLERS[self.levy_cfg.sampler](self, cond, gen)
