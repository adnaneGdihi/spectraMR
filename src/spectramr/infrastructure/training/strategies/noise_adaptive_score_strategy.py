"""Noise-Adaptive Score-Based Reconstruction (ULF-PR-6 / II.3).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-6.

A sigma-conditioned VE-SDE score model for low-SNR ULF reconstruction, conditioned on
the undersampled measurement ``y`` by channel concatenation. The denoiser uses the
Karras (EDM) preconditioning of :mod:`spectramr.models.diffusion.edm_schedule`::

    D(x; sigma, y) = c_skip(sigma) x + c_out(sigma) F([c_in(sigma) x, y], c_noise(sigma)),
    s(x; sigma, y) = (D(x; sigma, y) - x) / sigma².

- :meth:`_compute_losses_impl` draws ``sigma`` log-uniformly on ``[sigma_min, sigma_max]``,
  forms ``x_sigma = x0 + sigma ε`` and minimises the Karras-weighted denoising score
  matching loss ``λ(sigma) ‖D(x_sigma; sigma, y) - x0‖²``, optionally re-weighted by a per-pixel
  ``batch["snr_map"]``. Declared ``losses.image_losses`` fold onto ``D`` against x0.
- :meth:`reverse_sde_sample` is the Euler-Maruyama VE-SDE solver. Validation runs
  it conditioned on ``y`` for ``validation.sampling.steps`` steps.

References: Karras et al., NeurIPS 2022 (EDM); Song et al., ICLR 2021 (score SDEs).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch
from torch import nn

from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field
from spectramr.models.diffusion.edm_schedule import (
    edm_loss_weight,
    edm_preconditioning_coefficients,
)

from .reconstruction import ReconstructionTrainingStrategy

#: Reverse-SDE steps at validation when ``validation.sampling.steps`` is unset
#: (the sampler's own default).
_DEFAULT_SAMPLER_STEPS = 50


class NoiseAdaptiveScoreStrategy(ReconstructionTrainingStrategy):
    """sigma-conditioned, measurement-conditioned VE-SDE score-matching strategy."""

    #: Loss ownership: the DSM term is inline; declared image losses fold onto D(x_sigma; sigma, y).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = True
    #: The step scores D at a random sigma, so it reports no train_* metrics.
    reports_train_metrics: ClassVar[bool] = False

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5

    @classmethod
    def standalone_infer_blocker(cls, config: object) -> str | None:
        """``infer`` calls the bare generator; reconstruction needs the conditioned reverse SDE."""
        del config
        return (
            "NoiseAdaptiveScoreStrategy reconstructs by reverse-SDE sampling conditioned on "
            "the measurement; the bare generator forward is not that estimator."
        )

    def _sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Log-uniform sigma on ``[sigma_min, sigma_max]``, drawn on ``device`` (no host sync)."""
        log_sig = torch.empty(batch_size, device=device).uniform_(
            math.log(self.sigma_min), math.log(self.sigma_max)
        )
        return log_sig.exp()

    def _time_conditioned_generator(self) -> nn.Module:
        """The generator, refusing one whose denoiser ignores the timestep (sigma would be inert)."""
        gen = cast(nn.Module, self.env.generator)
        inner = unwrap_model(gen)
        if getattr(inner, "_denoiser_takes_time", None) is not True:
            raise ValueError(
                f"{type(self).__name__} conditions the denoiser on c_noise(sigma) through "
                "forward(x, t); the generator "
                f"{type(inner).__name__} has no time-conditioned denoiser. Use "
                "model_type: cold_diffusion with a denoising_model that declares time_dim."
            )
        return gen

    def _denoise(
        self, gen: nn.Module, x_sigma: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EDM-preconditioned ``D(x; sigma, y)`` and the network input it was fed."""
        c_skip, c_out, c_in, c_noise = edm_preconditioning_coefficients(sigma, self.sigma_data)
        shape = (-1,) + (1,) * (x_sigma.dim() - 1)
        model_input = torch.cat([c_in.view(shape) * x_sigma, cond], dim=1)
        # A non-None t makes ColdDiffusionGenerator call its denoiser directly, unnoised.
        f = gen(model_input, c_noise)
        return c_skip.view(shape) * x_sigma + c_out.view(shape) * f, model_input

    def score(
        self, gen: nn.Module, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Score ``s(x; sigma, y) = (D(x; sigma, y) - x) / sigma²`` for the reverse SDE."""
        denoised, _ = self._denoise(gen, x, sigma, cond)
        return (denoised - x) / sigma.view((-1,) + (1,) * (x.dim() - 1)).pow(2)

    def _conditioning_input(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor | None, **kwargs: Any
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """The batch context and the measurement ``y`` the denoiser is conditioned on."""
        batch_context = self._prepare_batch_context(input_batch, target_batch, **kwargs)
        cond, forward_kwargs = self._prepare_generator_inputs(batch_context, input_batch)
        if forward_kwargs:
            raise ValueError(
                f"{type(self).__name__} forwards only (x, c_noise) to the generator; the "
                f"batch asked for {sorted(forward_kwargs)}, which it would drop."
            )
        return batch_context, cond

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        gen = self._time_conditioned_generator()
        batch_context, cond = self._conditioning_input(input_batch, target_batch, **kwargs)
        x0 = batch_context["hr"]
        if x0.shape[0] != cond.shape[0] or x0.shape[2:] != cond.shape[2:]:
            raise ValueError(
                f"target {tuple(x0.shape)} and measurement {tuple(cond.shape)} differ in "
                "batch or spatial extent; the score model concatenates them."
            )
        if x0.shape[1] != int(self.config.model.out_channels):
            # D = c_skip x + c_out F would broadcast a narrower target silently.
            raise ValueError(
                f"target has {x0.shape[1]} channels but model.out_channels is "
                f"{self.config.model.out_channels}; the score is defined on the target's space."
            )

        sigma = self._sample_sigma(x0.shape[0], x0.device)
        x_sigma = x0 + sigma.view((-1,) + (1,) * (x0.dim() - 1)) * torch.randn_like(x0)
        denoised, model_input = self._denoise(gen, x_sigma, sigma, cond)

        self._declare_model_input(
            {"model_input": model_input, "noised": x_sigma, "target": x0},
            in_kspace_keys=set(),
            extra={
                "model_input_key": "model_input",
                "sigma_range": [self.sigma_min, self.sigma_max],
                "sigma_sampling": "log_uniform",
                "sigma_data": self.sigma_data,
                "c_noise": "0.25*ln(sigma)",
                "conditioning": "channel concat [c_in * x_sigma, y]",
                "note": (
                    "The network sees [c_in * (x0 + sigma*eps), y], formed inside the "
                    "step; 'first_steps/input_prepared' is y alone."
                ),
            },
        )

        weight = edm_loss_weight(sigma, self.sigma_data).view((-1,) + (1,) * (x0.dim() - 1))
        per_pixel = weight * (denoised - x0).pow(2)
        snr_map = read_batch_field(self._resolve_legacy_batch(input_batch, kwargs), "snr_map")
        if snr_map is not None:
            per_pixel = per_pixel * (snr_map.to(per_pixel.device) ** 2).clamp_min(1e-6)
        dsm = per_pixel.mean()

        components: dict[str, torch.Tensor] = {"loss_score_dsm": dsm.detach()}
        total = dsm
        aux = self._apply_builder_image_losses(denoised, x0, components)
        if aux is not None:
            total = total + aux
        self._last_prediction = denoised
        self._last_target = x0
        return {"g_total_loss": total, "loss": total, **components}

    def _validation_sampler_steps(self) -> int:
        """``validation.sampling.steps`` when declared, else the sampler default."""
        validation = self.config.validation
        steps = validation.sampling.steps if validation is not None else None
        return int(steps) if steps is not None else _DEFAULT_SAMPLER_STEPS

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reconstruct by reverse-SDE sampling conditioned on the measurement."""
        gen = self._time_conditioned_generator()
        cond, forward_kwargs = self._prepare_generator_inputs(
            batch_context, input_batch, validation=True
        )
        if forward_kwargs:
            raise ValueError(
                f"{type(self).__name__} forwards only (x, c_noise) to the generator; the "
                f"batch asked for {sorted(forward_kwargs)}, which it would drop."
            )
        out_channels = int(self.config.model.out_channels)
        shape = (cond.shape[0], out_channels, *cond.shape[2:])
        noise = torch.Generator(device=cond.device).manual_seed(int(self.config.run.seed))
        return self.reverse_sde_sample(
            lambda x, s: self.score(gen, x, s, cond),
            shape,
            n_steps=self._validation_sampler_steps(),
            device=cond.device,
            generator=noise,
        )

    @torch.no_grad()
    def reverse_sde_sample(
        self,
        score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        n_steps: int = _DEFAULT_SAMPLER_STEPS,
        device: torch.device | str = "cpu",
        snr_map: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler-Maruyama reverse-time VE-SDE sampling on a log-spaced sigma grid."""
        device = torch.device(device)
        sigmas = torch.exp(
            torch.linspace(
                math.log(self.sigma_max), math.log(self.sigma_min), n_steps + 1, device=device
            )
        )
        x = torch.randn(*shape, device=device, generator=generator) * sigmas[0]
        for k in range(n_steps):
            sigma_now, sigma_next = sigmas[k], sigmas[k + 1]
            score = score_fn(x, sigma_now.expand(shape[0]))
            d_sigma2 = sigma_now**2 - sigma_next**2
            drift = d_sigma2 * score
            noise = torch.randn(*shape, device=device, generator=generator)
            noise = noise * torch.sqrt(d_sigma2.clamp_min(0.0))
            if snr_map is not None:
                noise = noise * snr_map
            x = x + drift + noise
        return x
