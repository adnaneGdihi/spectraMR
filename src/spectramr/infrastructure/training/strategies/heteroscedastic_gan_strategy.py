"""Adversarial ULF restoration with a heteroscedastic ``[mean, log-variance]`` head.

The generator emits two channels per one-channel target. The critic, the declared
image losses and the validation metrics see only the mean; the generator objective
adds the Gaussian NLL of both channels with the Hoult-anchored variance prior,
through :func:`heteroscedastic_ulf_loss`, the same owner
:class:`HeteroscedasticULFStrategy` uses, configured by the same
``training.heteroscedastic_ulf`` block.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.losses.heteroscedastic_losses import heteroscedastic_ulf_loss

from .gan import GANTrainingStrategy
from .heteroscedastic_ulf_strategy import _as_mean_logvar, _hoult_sigma_per_sample

__all__ = ["HeteroscedasticGANStrategy"]


class HeteroscedasticGANStrategy(GANTrainingStrategy):
    """GAN whose generator also predicts a per-pixel log-variance, trained by Gaussian NLL."""

    #: The generator emits ``[mean, logvar]`` for a one-channel target.
    predicts_distribution_params: ClassVar[bool] = True

    def _setup_strategy_specific_components(self) -> None:
        super()._setup_strategy_specific_components()
        cfg = self.config.training.heteroscedastic_ulf
        if cfg is None:
            raise ConfigurationError(
                "HeteroscedasticGANStrategy needs training.heteroscedastic_ulf "
                "(source_field, lambda_var_prior, logvar band); it is absent."
            )
        self._het_cfg = cfg
        # The paired ULF route stamps no per-sample source field, so the anchor is the declared one.
        self._het_sigma = _hoult_sigma_per_sample(
            None, cfg.sigma_floor, cfg.max_sigma, source_field=cfg.source_field
        )
        if getattr(self, "logging_service", None):
            self.logging_service.log_info(
                f"[HeteroscedasticGAN] NLL + variance prior on [mean, logvar]: "
                f"source_field={cfg.source_field} T, sigma_target={float(self._het_sigma[0]):.4g}, "
                f"lambda_var_prior={cfg.lambda_var_prior}, lambda_mean={cfg.lambda_mean}, "
                f"logvar=[{cfg.logvar_min}, {cfg.logvar_max}]"
            )

    def _split_generator_output(self, output: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return _as_mean_logvar(output)

    def _auxiliary_generator_loss(
        self, image: torch.Tensor, auxiliary: Any, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self._het_cfg
        sigma = self._het_sigma
        if sigma.device != image.device or sigma.dtype != image.dtype:
            # Move the constant once: a per-step pageable H2D copy syncs the stream.
            sigma = self._het_sigma = sigma.to(device=image.device, dtype=image.dtype)
        terms = heteroscedastic_ulf_loss(
            image,
            auxiliary,
            target,
            sigma,
            lambda_var_prior=cfg.lambda_var_prior,
            lambda_mean=cfg.lambda_mean,
            logvar_min=cfg.logvar_min,
            logvar_max=cfg.logvar_max,
        )
        total = terms.pop("loss_total")
        components = {f"het_{k.removeprefix('loss_')}": v for k, v in terms.items()}
        components["het_logvar_mean"] = auxiliary.detach().mean()
        return total, components
