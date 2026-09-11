r"""Ambient Diffusion reconstruction strategy — Phase B step 5.

Ambient Diffusion (Daras et al., NeurIPS 2023) learns a clean image prior from
*corrupted* data alone. For accelerated MRI (Aali et al., Ambient Diffusion
Posterior Sampling) the SSDU Λ/Θ split is lifted onto a diffusion model: the
network is trained, at every diffusion noise level, from the Λ-undersampled
measurement, and its clean-image estimate :math:`\hat{x}_0` is supervised on the
**held-out** Θ k-space via :class:`~spectramr.models.losses.ambient_consistency_loss.AmbientConsistencyLoss`.

The per-step objective is

.. math::
    \mathcal{L} = \underbrace{\lVert \boldsymbol{\epsilon}_\theta(x_t, t) -
    \boldsymbol{\epsilon} \rVert^2}_{\text{diffusion denoising}}
    + \lambda \underbrace{\lVert M_\Theta (F\,\hat{x}_0 - \mathbf{y}) \rVert^2}_{\text{ambient consistency}} ,

with :math:`x_t = q_{\text{sample}}(A^{*}_\Lambda \mathbf{y}, t,
\boldsymbol{\epsilon})` and :math:`\hat{x}_0` recovered from the eps prediction.
At Θ-fraction 0 / noise → 0 the objective reduces to the SSDU held-out
residual, so Ambient provably generalizes SSDU onto a diffusion prior.

Inference is **A-DPS**: this strategy subclasses ``DiffusionTrainingStrategy``,
so validation runs the diffusion sampling path; with ``inference_sampler: dds``
on the score generator the conditioned posterior reverse loop
(``DDSReconSampler``, reusing the T1 ``cg_data_consistency`` primitive) is
invoked — i.e. A-DPS reuses the DDS machinery rather than a redundant sampler.

Scope: assumes a single (coil-combined) complex forward operator and a score /
eps-prediction generator (exposing ``_model_forward`` + ``_predict_start_from_noise``).
**The R=8 superiority reported on fastMRI is untested at 0.3T — do not advertise it.**
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)
from spectramr.infrastructure.training.strategies.ssdu_strategy import split_acquired_mask
from spectramr.models.losses.ambient_consistency_loss import ambient_consistency_residual

logger = logging.getLogger(__name__)


def ambient_diffusion_step(
    model_eps: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    predict_x0: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    q_sample: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x_start: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Run one diffusion step and recover the clean-image estimate.

    Args:
        model_eps: ``(x_t, t) -> eps_pred`` (the score/eps network).
        predict_x0: ``(x_t, t, eps) -> x0`` (the schedule's eps→x₀ map).
        q_sample: ``(x_start, t, noise) -> x_t`` (forward diffusion).
        x_start: the Λ zero-filled surrogate clean image.
        t: diffusion timesteps.
        noise: the Gaussian noise injected by ``q_sample``.

    Returns:
        ``(eps_pred, x0_pred, x_t)``. ``x_t`` is returned rather than discarded
        because it is the tensor the network is fed, and the caller owes it to
        the debug-snapshot contract (non-negotiable 14): the strategy's
        ``first_steps/input_prepared`` is captured before any of this runs.
        Recomputing ``q_sample`` at the call site to recover it would duplicate
        work in the training loop for no reason (non-negotiable 9).
    """
    x_t = q_sample(x_start, t, noise)
    eps_pred = model_eps(x_t, t)
    x0_pred = predict_x0(x_t, t, eps_pred)
    return eps_pred, x0_pred, x_t


class AmbientDiffusionStrategy(DiffusionTrainingStrategy):
    """Reference-free diffusion reconstruction via the Ambient (SSDU-lifted) objective."""

    #: Loss ownership (issue #1918). Ambient-noise objective: mse_loss(eps_pred, noise) predicts the NOISE, not
    #: the target, so no declared image loss is computed here -- and the hook
    #: overrides DiffusionTrainingStrategy's without calling it or the computer.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        cfg = getattr(self.config.training, "ambient", None) if self.config.training else None
        if cfg is not None:
            self.theta_fraction = float(cfg.theta_fraction)
            self.ambient_weight = float(cfg.ambient_weight)
            self.denoise_weight = float(cfg.denoise_weight)
            seed = int(cfg.split_seed)
        else:
            self.theta_fraction = 0.4
            self.ambient_weight = 1.0
            self.denoise_weight = 1.0
            seed = 0
        self._ambient_rng = torch.Generator(device="cpu").manual_seed(seed)
        logger.info(
            "AmbientDiffusionStrategy: theta_fraction=%.3g, ambient_weight=%.3g, "
            "denoise_weight=%.3g (source=%s)",
            self.theta_fraction,
            self.ambient_weight,
            self.denoise_weight,
            "config" if cfg is not None else "defaults",
        )

    def _resolve_measurement_and_mask(
        self, input_batch: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the complex measured k-space ``y`` and acquired mask.

        Ambient is self-supervised — it needs the measurement and acquisition
        mask, never a fully-sampled target (mirrors SSDU's contract). Raises if
        the dataloader does not expose them (no silent fallback, pitfall #9).
        """
        batch = kwargs.get("batch")
        y = None
        mask = None
        if isinstance(batch, dict):
            y = batch.get("measured_kspace", batch.get("kspace_measured"))
            mask = batch.get("mask")
        if y is None:
            raise ValueError(
                "AmbientDiffusionStrategy requires the measured k-space "
                "(batch['measured_kspace']); Ambient is self-supervised with no "
                "fully-sampled target. Ensure dataset_type=kspace exposes it."
            )
        if mask is None:
            # Derive the acquired mask from the measurement support (non-zero).
            mag = y.abs().sum(dim=1, keepdim=True)
            mask = (mag > 1e-8).to(y.real.dtype)
        return y, mask

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del target_batch, epoch  # Ambient is reference-free.
        gen = (
            self.generator_model.module
            if hasattr(self.generator_model, "module")
            else self.generator_model
        )
        for attr in ("_model_forward", "_predict_start_from_noise", "q_sample"):
            if not hasattr(gen, attr):
                raise ValueError(
                    f"AmbientDiffusionStrategy needs a score/eps generator exposing "
                    f"{attr!r} (e.g. score_based_diffusion); got {type(gen).__name__}."
                )

        y, acquired = self._resolve_measurement_and_mask(input_batch, **kwargs)

        # SSDU Λ/Θ split lifted onto the diffusion objective.
        lam, theta = split_acquired_mask(
            acquired, theta_fraction=self.theta_fraction, generator=self._ambient_rng
        )
        mask_dtype = y.real.dtype if y.is_complex() else y.dtype
        # Λ zero-filled image surrogate (the diffusion "clean" stand-in).
        x_start = ifft2c(y * lam.to(mask_dtype)).real

        timesteps = torch.randint(0, int(gen.timesteps), (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        eps_pred, x0_pred, x_t = ambient_diffusion_step(
            gen._model_forward,
            gen._predict_start_from_noise,
            gen.q_sample,
            x_start,
            timesteps,
            noise,
        )

        # Everything the model sees is built in this method, so
        # ``first_steps/input_prepared`` -- captured before the forward pass --
        # is the raw measurement, not the input. Declare the real one; the
        # wrapper emits it under ``diffusion_step`` (non-negotiable 14).
        #
        # Deliberately NOT keyed ``model_input``. That name is in the canonical
        # set ``save_debug_snapshot`` IFFTs unconditionally for a k-space arm --
        # and ambient's input is IMAGE-domain regardless, because ``x_start``
        # comes back through ``ifft2c`` above. Borrowing the canonical name here
        # would IFFT an image and reproduce findings-booklet VIS-1.
        self._declare_model_input(
            {
                "ambient_x_t": x_t,
                "lambda_zero_filled": x_start,
                "acquired_mask": acquired,
            },
            in_kspace_keys=set(),
            extra={
                "model_input_key": "ambient_x_t",
                "note": (
                    "Ambient feeds q_sample(x_start, t, noise), where x_start is "
                    "the LAMBDA-masked zero-filled image surrogate -- both formed "
                    "inside this step. Reference-free: there is no clean target, "
                    "so none is captured. 'acquired_mask' is the sampling "
                    "pattern before the Lambda/Theta split, shown as a mask "
                    "rather than transformed."
                ),
            },
        )

        denoise = torch.nn.functional.mse_loss(eps_pred, noise)
        consistency = ambient_consistency_residual(x0_pred, y, theta, weight=self.ambient_weight)
        total = self.denoise_weight * denoise + consistency
        return {
            "g_total_loss": total,
            "loss": total,
            "loss_diffusion_denoise": denoise.detach(),
            "loss_ambient_consistency": consistency.detach(),
        }


__all__ = ["AmbientDiffusionStrategy", "ambient_diffusion_step"]
