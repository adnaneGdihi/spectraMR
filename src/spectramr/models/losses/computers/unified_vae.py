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


def _refuse_undeclared_reconstruction(config: Any) -> None:
    """Refuse a VAE/VQ-VAE arm that declares no live reconstruction term (decision 16).

    Without one the objective is the latent term alone and the decoder never learns to
    reconstruct; the computer no longer substitutes an undeclared L1 (#2280).
    """
    from spectramr.models.losses.weights import build_loss_weight_table

    if config is None:
        raise ConfigurationError(
            "A VAE/VQ-VAE loss computer needs a config: no reconstruction loss is declared "
            "without one, and it no longer substitutes an L1."
        )
    table = build_loss_weight_table(getattr(config, "losses", None))
    # Canonical names: a declared ``mse`` is ``l2`` here even where the builder keys it ``mse``.
    if any(n in table and table[n].enabled and table[n].weight > 0 for n in ("l1", "l2")):
        return
    spec = table.get("reconstruction")
    declared = (
        f"losses.reconstruction ({spec.source}) is declared at weight {spec.weight}, but "
        if spec is not None and spec.enabled and spec.weight > 0
        else ""
    )
    raise ConfigurationError(
        f"{declared}no reconstruction loss is declared to carry the VAE objective. Declare "
        "`- {name: mse, weight: <w>}` (or l1) under losses.image_losses; the computer no "
        "longer substitutes an L1."
    )


def _declared_reconstruction(losses: Mapping[str, Any]) -> tuple[str | None, Any]:
    """The reconstruction term the arm DECLARED (``l1`` or ``l2``), as the builder built it.

    A computer may not fall back to an L1 the arm never declared (#2280). ``None`` when
    the arm declares neither, which the caller turns into a refusal or an absent term.
    """
    for name in ("l1", "l2"):
        if name in losses:
            return name, losses[name]
    return None, None


class UnifiedVAELossComputer(BaseLossComputer):
    """Unified VAE loss computer implementing SSOT pattern."""

    def __init__(self, config: Any, device: torch.device = DEFAULT_LOSS_DEVICE):
        """Initialize VAE loss computer.

        Args:
            config: Configuration with VAE settings
            device: Compute device
        """
        self.reconstruction_loss_fn: nn.Module | None = None
        self.reconstruction_name: str | None = None
        self.perceptual_loss_fn: nn.Module | None = None
        # Duck-typed: the GAN loss library object exposing ``compute_generator_loss``.
        self.adversarial_loss_fn: Any = None

        super().__init__(config, device)

    def _initialize_losses(self) -> None:
        """Initialize loss functions using LossBuilder; raise when no reconstruction is declared."""
        from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

        _refuse_undeclared_reconstruction(self.config)
        # Build errors with a real config MUST propagate (pitfall #9): a swallowed
        # builder failure once collapsed the whole stack to a bare L1.
        builder = LossBuilder(self.config, self.device)
        losses = builder.build_reconstruction_losses().build_latent_losses().build()
        self.reconstruction_name, self.reconstruction_loss_fn = _declared_reconstruction(losses)
        self.perceptual_loss_fn = losses.get("perceptual")
        self.adversarial_loss_fn = losses.get("adversarial")

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
        if key in ("kl", "kl_divergence"):
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
        # Weights resolved at the declared name, so stacking never looks up "reconstruction".
        stack_weights: dict[str, float] = {}

        # 1. RECONSTRUCTION LOSS: the declared l1/l2 term, weighted by its own declaration.
        rec_weight = (
            self._get_loss_weight(self.reconstruction_name, epoch, iteration)
            if self.reconstruction_name is not None and self.reconstruction_loss_fn is not None
            else 0.0
        )
        if rec_weight > 0:
            components["reconstruction"] = self.reconstruction_loss_fn(pred, target)
            stack_weights["reconstruction"] = rec_weight

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
                if loss_name in components or loss_name == self.reconstruction_name or loss_name in [
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
        total = self._stack_components(
            components, weights=stack_weights, epoch=epoch, iteration=iteration
        )

        return LossOutput(total=total, components=components, metrics={})


class UnifiedVQVAELossComputer(BaseLossComputer):
    """Unified VQ-VAE loss computer (discrete latent codes).

    Handles vector quantization loss and commitment loss.
    """

    def __init__(self, config: Any, device: torch.device = DEFAULT_LOSS_DEVICE):
        """Initialize VQ-VAE loss computer."""
        self.reconstruction_loss_fn: nn.Module | None = None
        self.reconstruction_name: str | None = None
        self.perceptual_loss_fn: nn.Module | None = None
        self.adversarial_loss_fn: nn.Module | None = None

        super().__init__(config, device)

    def _initialize_losses(self) -> None:
        """Bind the declared reconstruction term (l1/l2) from the weight table.

        The VQ term is weighted by ``lambda_vq`` at compute time; this computer builds no
        modules of its own beyond the reconstruction term.
        """
        _refuse_undeclared_reconstruction(self.config)
        from spectramr.models.losses.registry import create_loss
        from spectramr.models.losses.weights import build_loss_weight_table

        table = build_loss_weight_table(getattr(self.config, "losses", None))
        name = next(
            (n for n in ("l1", "l2") if n in table and table[n].enabled and table[n].weight > 0),
            None,
        )
        if name is not None:
            self.reconstruction_name = name
            self.reconstruction_loss_fn = create_loss(name)

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
            **kwargs: vq_loss (the quantiser term, codebook plus commitment), etc.

        Returns:
            LossOutput with total loss and components
        """
        components: dict[str, torch.Tensor] = {}
        # Weights resolved at the declared name, so stacking never looks up "reconstruction".
        stack_weights: dict[str, float] = {}

        # 1. RECONSTRUCTION LOSS: the declared l1/l2 term (no undeclared L1 fallback, #2280).
        rec_weight = (
            self._get_loss_weight(self.reconstruction_name, epoch, iteration)
            if self.reconstruction_name is not None and self.reconstruction_loss_fn is not None
            else 0.0
        )
        if rec_weight > 0:
            components["reconstruction"] = self.reconstruction_loss_fn(pred, target)
            stack_weights["reconstruction"] = rec_weight

        # 1b. VQ LOSS: the quantiser term the VQ strategy hands in, weighted by lambda_vq (#2280).
        vq_loss = kwargs.get("vq_loss")
        vq_weight = self._get_loss_weight("vq", epoch, iteration)
        if vq_loss is not None and vq_weight > 0:
            components["vq_loss"] = vq_loss
            stack_weights["vq_loss"] = vq_weight

        # 2. PERCEPTUAL LOSS
        lambda_percep = self._get_loss_weight("perceptual", epoch, iteration)
        if lambda_percep > 0 and self.perceptual_loss_fn:
            percep_loss = self.perceptual_loss_fn(pred, target)
            components["perceptual"] = percep_loss

        # 5. DYNAMIC COMPONENT LOSSES (from losses_dict)
        if losses_dict:
            for loss_name, loss_fn in losses_dict.items():
                if loss_name in components or loss_name == self.reconstruction_name or loss_name in [
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
        total = self._stack_components(
            components, weights=stack_weights, epoch=epoch, iteration=iteration
        )

        return LossOutput(total=total, components=components, metrics={})


__all__ = [
    "UnifiedVAELossComputer",
    "UnifiedVQVAELossComputer",
]
