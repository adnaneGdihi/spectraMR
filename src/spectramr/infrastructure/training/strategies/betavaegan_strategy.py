"""β-TC-VAE-GAN training strategy.

Drives the cross-paradigm composition the ``beta_vae_gan`` model exists to
validate (TODO/phase3_model_implementations_detailed.md §12): the standard
GAN two-player loop (adversarial + reconstruction, via
:class:`GANTrainingStrategy`) plus the β-TC-VAE disentanglement term added
to the generator loss.

This is the "GANTrainingStrategy with a ``beta_tc_vae_aux`` objective"
described in the plan. We reuse the full, proven GAN machinery (n_critic
discriminator updates, AMP, gradient penalty, metrics) unchanged and only
augment the generator-side loss with the model-provided β-TC scalar, so the
composition is exercised in training without duplicating the GAN loop.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy


class BetaVAEGANStrategy(GANTrainingStrategy):
    """GAN strategy that adds the β-TC-VAE disentanglement term to G's loss.

    The term enters through ``_auxiliary_generator_loss``, the hook both the
    generator step and the loss report call; the GAN ``train_step`` never calls
    ``_compute_losses_impl``, so a term added only there never trained.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        gen = self.env.generator if self.env else None
        if not callable(getattr(gen, "beta_tc_term", None)):
            raise TypeError(
                "BetaVAEGANStrategy requires a generator exposing 'beta_tc_term(x)', got "
                f"{type(gen).__name__}; without it the run is a plain GAN. Use the "
                "beta_vae_gan model_type."
            )
        self._beta_tc_input: torch.Tensor | None = None

    def _beta_tc_weight(self) -> float:
        """Auxiliary weight for the β-TC term (default 1.0).

        The total-correlation strength is already controlled by the model's
        ``beta`` (inside ``BetaTCVAELoss``); this is the outer mixing weight
        against the adversarial + reconstruction terms.
        """
        gan_cfg = (
            getattr(self.config.training, "gan", None)
            if self.config and self.config.training
            else None
        )
        w = getattr(gan_cfg, "beta_tc_weight", None) if gan_cfg is not None else None
        return float(w) if w is not None else 1.0

    def _train_generator_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        self._beta_tc_input = input_batch
        return super()._train_generator_step(input_batch, target_batch, *args, **kwargs)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        self._beta_tc_input = input_batch
        return super()._compute_losses_impl(input_batch, target_batch, epoch, **kwargs)

    def _auxiliary_generator_loss(
        self, image: torch.Tensor, auxiliary: Any, target: torch.Tensor
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """``weight * beta_tc_term(input)``, logged as ``loss_beta_tc``."""
        if self._beta_tc_input is None:
            raise RuntimeError(
                "BetaVAEGANStrategy: no generator input recorded for the beta-TC term."
            )
        beta_tc = self.generator_model.beta_tc_term(self._beta_tc_input)
        return self._beta_tc_weight() * beta_tc, {"loss_beta_tc": beta_tc}
