"""Unified VAE Loss Computer - SSOT Implementation.

Handles VAE-specific loss computation:
- Reconstruction loss (L1/L2)
- KL divergence (regularization)
- Optional adversarial loss
- Optional perceptual loss
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from spectramr.domain.exceptions import ConfigurationError
from spectramr.models.losses.computers.base import (
    DEFAULT_LOSS_DEVICE,
    BaseLossComputer,
    LossOutput,
)
from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    _call_safe_loss,
)

logger = logging.getLogger(__name__)


# The KL-annealing knobs carry two spellings: the one the experiment YAMLs were
# authored with (`enable_kl_annealing` / `kl_anneal_start` / `kl_anneal_end`)
# and the one this resolver and the schema use. `LatentTrainingConfigSchema`
# folds them with `AliasChoices` for validated blocks; this table covers the
# raw-dict path (a block reaching us through `extra='allow'`), so BOTH shapes
# resolve identically. SSOT for the pairing — keep in sync with
# `config/schemas/training/base.py::LatentTrainingConfigSchema`.
def _explicit_fields(src: Any) -> frozenset[str]:
    """Names the author actually WROTE on ``src`` — never schema defaults.

    A pydantic model reports this as ``model_fields_set`` (populated by the
    canonical field name even when the YAML used a validation alias). Extras
    admitted under ``extra='allow'`` live in ``model_extra``. A raw mapping is
    its own key set. Anything else (a ``SimpleNamespace`` in a test, a legacy
    object) has no way to distinguish set-from-default, so fall back to its
    attribute names.
    """
    if isinstance(src, Mapping):
        return frozenset(src)
    fields_set = getattr(src, "model_fields_set", None)
    if fields_set is not None:
        extras = getattr(src, "model_extra", None) or {}
        return frozenset(fields_set) | frozenset(extras)
    return frozenset(vars(src)) if hasattr(src, "__dict__") else frozenset()


_KL_ALIASES: dict[str, tuple[str, ...]] = {
    "anneal_kl_beta": ("anneal_kl_beta", "enable_kl_annealing"),
    "kl_beta_start": ("kl_beta_start", "kl_anneal_start"),
    "kl_beta_end": ("kl_beta_end", "kl_anneal_end"),
    "kl_anneal_steps": ("kl_anneal_steps",),
}


# Names of registered losses whose ``forward`` does NOT take ``(pred, target)``
# as the first two positional arguments — they are routed through dedicated
# code paths and must never be miscalled here. Keep in sync with the same
# constant in ``unified_gan.py``.
_NON_PRED_TARGET_LOSS_NAMES = frozenset(
    {
        "kl",
        "kl_divergence",
        "vq_kl",
        "beta_tc_vae",
        "BetaTCVAELoss",
        "vq",
        "VQLoss",
        "latent_vq",
        "vqgan",
        "VQGANLoss",
        "helmholtz_pde",
        "smoothness_loss",
        "flow_smoothness",
        "smooth_loss",
        "latent_regularization",
        "LatentRegularizationLoss",
        "modality_swap",
        "ModalitySwapLoss",
    }
)


class UnifiedVAELossComputer(BaseLossComputer):
    """Unified VAE loss computer implementing SSOT pattern."""

    def __init__(self, config: Any, device: torch.device = DEFAULT_LOSS_DEVICE):
        """Initialize VAE loss computer.

        Args:
            config: Configuration with VAE settings
            device: Compute device
        """
        self.reconstruction_loss_fn: nn.Module | None = None
        self.perceptual_loss_fn: nn.Module | None = None
        # Duck-typed: the GAN loss library object exposing ``compute_generator_loss``.
        self.adversarial_loss_fn: Any = None

        super().__init__(config, device)

    def _initialize_losses(self) -> None:
        """Initialize loss functions using LossBuilder.

        Uses the LossBuilder pattern for SSOT loss creation.
        Falls back to defaults if config is missing.
        """
        from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

        if self.config is not None:
            # Build errors with a real config MUST propagate (pitfall #9). The
            # old `try/except Exception: pass` silently collapsed the
            # perceptual/adversarial/latent stack to a bare L1 on any builder
            # failure (a typo'd loss key, a missing sub-block) — review
            # 2026-07-01. Fail loud instead.
            builder = LossBuilder(self.config, self.device)
            losses = builder.build_reconstruction_losses().build_latent_losses().build()
            self.reconstruction_loss_fn = losses.get("l1", nn.L1Loss())
            self.perceptual_loss_fn = losses.get("perceptual")
            self.adversarial_loss_fn = losses.get("adversarial")
            # KL loss is typically computed inline, not as a module.
            return

        # No config supplied (direct construction in a test/script) — the ONLY
        # sanctioned minimal fallback.
        self.reconstruction_loss_fn = nn.L1Loss()
        self.perceptual_loss_fn = None
        self.adversarial_loss_fn = None

    def _get_loss_weight(
        self, loss_name: str, epoch: int = 0, iteration: int = 0, **kwargs: Any
    ) -> float:
        """Get loss weight from strict v6.0 config (training.vae section)."""
        if loss_name == "kl":
            # v6.0: modern training.vae config path with safe fallback.
            # When ``anneal_kl_beta`` is set the KL weight linearly warms up from
            # ``kl_beta_start`` to ``kl_beta_end`` over ``kl_anneal_steps``
            # iterations (audit 2026-06: these start/steps knobs were previously
            # unbacked — only kl_beta_end was read and the live path ignored
            # ``iteration`` — so KL annealing was a silent no-op). Knobs resolve
            # from training.vae, then legacy training.latent, then training.
            try:
                vae_cfg = getattr(self.config.training, "vae", None)
                latent_cfg = getattr(self.config.training, "latent", None)

                def _resolve(name: str, default: Any) -> Any:
                    for src in (vae_cfg, latent_cfg, self.config.training):
                        if src is None:
                            continue
                        # Two shapes reach us. A schema-validated block is a
                        # model (attribute access); one admitted through
                        # extra='allow' is a plain dict, where getattr silently
                        # returns None — the trap that made `training.vae`
                        # inert (audit 2026-07-18). Read both, and accept
                        # either KL spelling.
                        #
                        # Only an EXPLICITLY SET field counts as a declaration.
                        # A declared schema supplies defaults for absent keys,
                        # so treating "has an attribute" as "was configured"
                        # would let this block's defaults shadow a real value
                        # in the next block down. The ldm_two_stage stage-1
                        # arms depend on exactly that fallthrough: they set the
                        # beta bounds under `vae` and the anneal flag/steps
                        # under `latent`.
                        provided = _explicit_fields(src)
                        for key in _KL_ALIASES.get(name, (name,)):
                            if key not in provided:
                                continue
                            val = (
                                src.get(key)
                                if isinstance(src, Mapping)
                                else getattr(src, key, None)
                            )
                            if val is not None:
                                return val
                    return default

                beta_end = float(_resolve("kl_beta_end", 1.0))
                if not bool(_resolve("anneal_kl_beta", False)):
                    return beta_end
                beta_start = float(_resolve("kl_beta_start", 0.0))
                warmup = int(_resolve("kl_anneal_steps", 10000))
                if iteration > 0 and warmup > 0:
                    progress = min(iteration / warmup, 1.0)
                    return beta_start + progress * (beta_end - beta_start)
                return beta_start
            except AttributeError:
                logger.warning("kl_beta config access failed, using default 1.0.")
                return 1.0

        # Fallback to base behavior (lambda_NAME)
        return super()._get_loss_weight(loss_name, epoch, iteration, **kwargs)

    def _configured_latent_loss(self) -> tuple[Any, str]:
        """Resolve the latent regulariser from ``losses.latent.latent_loss_type``.

        Returns ``(loss_module, name)``. Returns ``(None, "kl_divergence")`` for
        the default KL so the standard path stays byte-identical for existing
        arms. Built once and cached. Like every other loss, a non-default choice
        is instantiated through the loss registry (``create_loss``).
        """
        cached: tuple[Any, str] | None = getattr(self, "_latent_loss_cache", None)
        if cached is not None:
            return cached
        latent_cfg = self.config.losses.latent if self.config.losses is not None else None
        name = getattr(latent_cfg, "latent_loss_type", None) or "kl_divergence"
        resolved: tuple[Any, str]
        if str(name).lower() in ("kl", "kl_divergence"):
            resolved = (None, "kl_divergence")
        else:
            from spectramr.models.losses.registry import create_loss

            resolved = (create_loss(str(name)), str(name))
        self._latent_loss_cache = resolved
        return resolved

    @staticmethod
    def _apply_latent_loss(
        loss_fn: Any, name: str, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Call a registered latent loss with the inputs its signature expects.

        Latent losses have heterogeneous signatures (the framework tracks them in
        ``_NON_PRED_TARGET_LOSS_NAMES``): the KL family takes ``(mu, log_var)``;
        norm-/kernel-based losses operate on a reparameterised sample ``z`` (and
        a prior sample for MMD). Unknown names raise (NN#3) rather than risk a
        silent mis-call.
        """
        key = name.lower()
        out: torch.Tensor
        if key in ("kl", "kl_divergence", "vq_kl"):
            out = loss_fn(mu, logvar)
            return out
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        if key == "latent_regularization":
            out = loss_fn(z)
            return out
        if key in ("mmd", "maximum_mean_discrepancy"):
            out = loss_fn(z, torch.randn_like(z))
            return out
        raise ConfigurationError(
            f"latent_loss_type={name!r} is not a recognised latent loss. "
            "Supported: kl_divergence, latent_regularization, mmd."
        )

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        losses_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute VAE losses (reconstruction + KL).

        Args:
            pred: Reconstructed image from decoder
            target: Original image
            epoch: Current epoch
            iteration: Current iteration
            losses_dict: Optional dictionary of additional losses
            **kwargs: posterior (mean, logvar), prior, etc.

        Returns:
            LossOutput with total loss and components
        """
        components: dict[str, torch.Tensor] = {}

        # 1. RECONSTRUCTION LOSS
        lambda_rec = self._get_loss_weight("reconstruction", epoch, iteration)
        if lambda_rec > 0:
            # Use provided loss function or default to L1
            if self.reconstruction_loss_fn:
                rec_loss = self.reconstruction_loss_fn(pred, target)
            else:
                rec_loss = torch.nn.functional.l1_loss(pred, target)

            components["reconstruction"] = rec_loss

        # 2. LATENT REGULARIZATION (KL by default; configurable via
        #    ``losses.latent.latent_loss_type`` and dispatched through the loss
        #    registry like any other loss).
        lambda_kl = self._get_loss_weight("kl", epoch, iteration)
        if lambda_kl > 0:
            posterior = kwargs.get("posterior")
            if posterior is not None:
                mu, logvar = posterior
                # BetaTCVAE (if present in losses_dict) owns the KL term itself.
                beta_tc = bool(
                    losses_dict
                    and any(fn.__class__.__name__ == "BetaTCVAELoss" for fn in losses_dict.values())
                )
                latent_fn, latent_name = self._configured_latent_loss()
                if beta_tc:
                    pass  # KL handled by the BetaTCVAE loss elsewhere
                elif latent_fn is None:
                    # Default standard-KL path — byte-identical to the prior
                    # behaviour, so existing VAE arms are unaffected.
                    # KL divergence: D_KL(q(z|x) || p(z)) with p(z) = N(0, I).
                    components["kl"] = (
                        -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                    )
                else:
                    components["kl"] = self._apply_latent_loss(latent_fn, latent_name, mu, logvar)

        # 3. PERCEPTUAL LOSS (optional)
        lambda_percep = self._get_loss_weight("perceptual", epoch, iteration)
        if lambda_percep > 0 and self.perceptual_loss_fn:
            percep_loss = self.perceptual_loss_fn(pred, target)
            components["perceptual"] = percep_loss

        # 4. ADVERSARIAL LOSS (optional, for adversarial VAE)
        lambda_adv = self._get_loss_weight("adversarial", epoch, iteration)
        discriminator = kwargs.get("discriminator")

        if lambda_adv > 0 and discriminator and self.adversarial_loss_fn:
            fake_pred = discriminator(pred)
            if hasattr(self.adversarial_loss_fn, "compute_generator_loss"):
                adv_loss = self.adversarial_loss_fn.compute_generator_loss(fake_outputs_d=fake_pred)
                components["adversarial"] = adv_loss

        # 5. DYNAMIC COMPONENT LOSSES (from losses_dict)
        if losses_dict:
            for loss_name, loss_fn in losses_dict.items():
                if loss_name in components or loss_name in [
                    "reconstruction",
                    "kl",
                    "kl_divergence",
                    "perceptual",
                    "adversarial",
                    "l1",  # Handled by reconstruction
                ]:
                    continue
                try:
                    # [PHYSICS FIX] BetaTCVAELoss requires latent variables, not images!
                    if loss_fn.__class__.__name__ == "BetaTCVAELoss":
                        posterior = kwargs.get("posterior")
                        if posterior is not None:
                            mu, logvar = posterior
                            z = kwargs.get("z")
                            if z is None:
                                # Fallback: Resample z if the strategy didn't provide it
                                std = torch.exp(0.5 * logvar)
                                eps = torch.randn_like(std)
                                z = mu + eps * std
                            loss_val = loss_fn(z=z, mu=mu, log_var=logvar)
                        else:
                            logger.warning(
                                "[unified_vae] BetaTCVAELoss requires `posterior` "
                                "kwarg from the strategy; received None → SKIPPED."
                            )
                            continue
                    elif (
                        loss_name in _NON_PRED_TARGET_LOSS_NAMES
                        or loss_fn.__class__.__name__ in _NON_PRED_TARGET_LOSS_NAMES
                    ):
                        # Latent / VQ / KL-like losses must be routed via
                        # dedicated computer paths, not the (pred, target) loop.
                        logger.warning(
                            "[unified_vae] Skipping loss %r (%s) — non-(pred,target) "
                            "signature; route it through a dedicated computer.",
                            loss_name,
                            loss_fn.__class__.__name__,
                        )
                        continue
                    else:
                        # Forward all dispatch kwargs (smaps, mask, posterior, …)
                        # so signature-aware losses receive what they need.
                        loss_val = _call_safe_loss(loss_fn, pred, target, **kwargs)

                    if isinstance(loss_val, torch.Tensor):
                        # Store the RAW component — ``_stack_components`` applies
                        # ``_get_loss_weight`` exactly once (base.py:310-315). The
                        # pre-multiply here was a w^2 double-application, invisible
                        # only while every dynamic weight resolved to 1.0 or 0.0.
                        # The same fix already landed in the diffusion computer.
                        components[loss_name] = loss_val
                except Exception as _exc:
                    logger.warning(
                        "[unified_vae] Loss %r (%s) failed: %s — being SKIPPED.",
                        loss_name,
                        loss_fn.__class__.__name__,
                        _exc,
                    )
                    logger.debug("Suppressed exception traceback:", exc_info=True)

        # ``iteration`` reaches the stack as it reaches every gate above: resolved
        # at step 0, the KL anneal would sit at ``kl_beta_start`` and every
        # warm-up-gated term would weigh 0.0 for the whole run (#1648).
        total = self._stack_components(components, epoch=epoch, iteration=iteration)

        return LossOutput(total=total, components=components, metrics={})


class UnifiedVQVAELossComputer(BaseLossComputer):
    """Unified VQ-VAE loss computer (discrete latent codes).

    Handles vector quantization loss and commitment loss.
    """

    def __init__(self, config: Any, device: torch.device = DEFAULT_LOSS_DEVICE):
        """Initialize VQ-VAE loss computer."""
        self.reconstruction_loss_fn: nn.Module | None = None
        self.perceptual_loss_fn: nn.Module | None = None
        self.adversarial_loss_fn: nn.Module | None = None

        super().__init__(config, device)

    def _initialize_losses(self) -> None:
        """Initialize loss functions."""
        pass

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        losses_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute VQ-VAE losses.

        Args:
            pred: Reconstructed image
            target: Original image
            epoch: Current epoch
            iteration: Current iteration
            losses_dict: Optional dictionary of additional losses
            **kwargs: quantized, codebook_loss, etc.

        Returns:
            LossOutput with total loss and components
        """
        components: dict[str, torch.Tensor] = {}

        # 1. RECONSTRUCTION LOSS
        lambda_rec = self._get_loss_weight("reconstruction", epoch, iteration)
        if lambda_rec > 0:
            if self.reconstruction_loss_fn:
                rec_loss = self.reconstruction_loss_fn(pred, target)
            else:
                rec_loss = torch.nn.functional.l1_loss(pred, target)

            components["reconstruction"] = rec_loss

        # 2. CODEBOOK LOSS (VQ loss)
        lambda_codebook = self._get_loss_weight("codebook", epoch, iteration)
        if lambda_codebook > 0:
            codebook_loss = kwargs.get("codebook_loss")
            if codebook_loss is not None:
                components["codebook"] = codebook_loss

        # 3. COMMITMENT LOSS
        lambda_commit = self._get_loss_weight("commitment", epoch, iteration)
        if lambda_commit > 0:
            commitment_loss = kwargs.get("commitment_loss")
            if commitment_loss is not None:
                components["commitment"] = commitment_loss

        # 4. PERCEPTUAL LOSS
        lambda_percep = self._get_loss_weight("perceptual", epoch, iteration)
        if lambda_percep > 0 and self.perceptual_loss_fn:
            percep_loss = self.perceptual_loss_fn(pred, target)
            components["perceptual"] = percep_loss

        # 5. DYNAMIC COMPONENT LOSSES (from losses_dict)
        if losses_dict:
            for loss_name, loss_fn in losses_dict.items():
                if loss_name in components or loss_name in [
                    "reconstruction",
                    "codebook",
                    "commitment",
                    "perceptual",
                    "vq_loss",  # Usually alias for codebook/commitment sum
                    "perplexity",  # Metric, not loss
                    "l1",
                ]:
                    continue
                if (
                    loss_name in _NON_PRED_TARGET_LOSS_NAMES
                    or loss_fn.__class__.__name__ in _NON_PRED_TARGET_LOSS_NAMES
                ):
                    logger.warning(
                        "[unified_vqvae] Skipping loss %r (%s) — non-(pred,target) signature.",
                        loss_name,
                        loss_fn.__class__.__name__,
                    )
                    continue
                try:
                    loss_val = _call_safe_loss(loss_fn, pred, target, **kwargs)
                    if isinstance(loss_val, torch.Tensor):
                        # Raw component — see the w^2 note in UnifiedVAELossComputer.
                        components[loss_name] = loss_val
                except Exception as _exc:
                    logger.warning(
                        "[unified_vqvae] Loss %r (%s) failed: %s — SKIPPED.",
                        loss_name,
                        loss_fn.__class__.__name__,
                        _exc,
                    )
                    logger.debug("Suppressed exception traceback:", exc_info=True)

        # ``iteration`` threaded for the same reason as in ``UnifiedVAELossComputer``.
        total = self._stack_components(components, epoch=epoch, iteration=iteration)

        return LossOutput(total=total, components=components, metrics={})


__all__ = [
    "UnifiedVAELossComputer",
    "UnifiedVQVAELossComputer",
]
