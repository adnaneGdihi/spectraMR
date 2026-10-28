"""LOUPE — Learning Optimal Under-sampling Pattern (PR-11 / M3).

Co-trains a binary k-space mask with the reconstruction backbone.
The mask is parametrised by per-line logits driven through a
Gumbel-sigmoid relaxation (CC-2 primitive) plus a straight-through
estimator. A density penalty pulls the expected sampling rate toward
``acceleration.adaptive.target_rate``.

Reference: Bahadir et al. "Learning-Based Optimization of the Under-sampling
Pattern in MRI" (IPMI 2019, MRM 2020).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE, OwnedParameter
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

from .mixins.utils import pick_present
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class LOUPEStrategy(ReconstructionTrainingStrategy):
    """Joint learnable-mask + reconstruction strategy."""

    # parallel_compat: The mask logits are an owned module, built on every rank's first batch,
    # DDP-wrapped and read via call_owned.
    parallel_compat = frozenset({"replicated"})

    #: Learns the sampling pattern from ``undersampling.adaptive``: the block is
    #: applied in-strategy (witness ``undersampling_block_is_applied``).
    applies_undersampling = True

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        adaptive = getattr(getattr(config, "undersampling", None), "adaptive", None)
        own = (("_mask_logits", OWNED_MODULE),) if getattr(adaptive, "enabled", False) else ()
        return (*super().owned_module_declarations(config), *own)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._adaptive_cfg = self.config.undersampling.adaptive
        if not self._adaptive_cfg.enabled:
            logger.warning(
                "LOUPEStrategy instantiated but acceleration.adaptive.enabled is False — "
                "the mask layer will not be trained. Did you forget to flip the flag?"
            )

        # Per-line logits live alongside the model so they're optimised
        # together. Lazy-initialised on first batch (image size unknown
        # until then). The strategy is NOT an nn.Module, so this is a
        # plain attribute, not a registered buffer/parameter.
        self._mask_logits: nn.Parameter | None = None
        self._steps_seen: int = 0

    def _ensure_mask(self, h: int, w: int, n_coils: int = 1) -> nn.Parameter:
        """Lazy-create mask logits on first call and register them on ``opt_g``.

        Per-coil mask sharing: by default all coils share a single mask
        (LOUPE original); set ``n_coils > 1`` and ``share_across_coils=False``
        in ``model_kwargs`` for per-coil masks (Bahadir 2020 §4.2 extension).

        The mask logits ARE the learnable acquisition parameter, so they must
        be added to the generator optimizer (``self.env.opt_g``) or they never
        train. We dedup against the existing param groups so re-entry is safe.
        """
        if self._mask_logits is None:
            mk = getattr(self.config.model, "model_kwargs", None) or {}
            share_raw = mk.get("share_across_coils", True)
            if not isinstance(share_raw, bool):
                raise TypeError(
                    "model_kwargs.share_across_coils must be a bool, got "
                    f"{type(share_raw).__name__!s}={share_raw!r}"
                )
            share = share_raw
            shape = (w,) if share else (n_coils, w)
            mu = torch.full(
                shape,
                float(torch.logit(torch.tensor(self._adaptive_cfg.target_rate))),
            )
            self._mask_logits = nn.Parameter(mu.to(self.device))
            self.register_owned_module("_mask_logits", OwnedParameter(self._mask_logits))
            self._register_on_opt_g(self._mask_logits)
        return self._mask_logits

    def _materialize_owned_state(self, state: dict[str, Any]) -> None:
        """Rebuild saved mask logits from their shape and register them on ``opt_g``."""
        super()._materialize_owned_state(state)
        saved = state.get("params", {}).get("_mask_logits")
        if saved is not None and self._mask_logits is None:
            self._mask_logits = nn.Parameter(torch.zeros_like(saved, device=self.device))
            self.register_owned_module("_mask_logits", OwnedParameter(self._mask_logits))
            self._register_on_opt_g(self._mask_logits)

    def _register_on_opt_g(self, module: nn.Parameter | nn.Module) -> None:
        """Add any not-yet-registered params of ``module`` to ``self.env.opt_g``."""
        opt = getattr(self.env, "opt_g", None)
        if opt is None:
            return
        params = [module] if isinstance(module, nn.Parameter) else list(module.parameters())
        existing = {p for g in opt.param_groups for p in g["params"]}
        new = [p for p in params if p not in existing]
        if new:
            opt.add_param_group({"params": new, "lr": self._adaptive_cfg.mask_learning_rate})

    def _current_tau(self) -> float:
        """Anneal Gumbel temperature from tau_init → tau_final linearly."""
        ann = self._adaptive_cfg.annealing
        frac = min(1.0, self._steps_seen / max(1, ann.anneal_steps))
        return ann.tau_init * (1.0 - frac) + ann.tau_final * frac

    def _gumbel_sigmoid_mask(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        """Relaxed-Bernoulli sample with straight-through estimator."""
        u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = torch.log(u) - torch.log(1.0 - u)
        soft = torch.sigmoid((logits + gumbel) / tau)
        # Straight-through: hard forward, soft backward.
        hard = (soft > 0.5).float()
        return hard + (soft - soft.detach())

    def _eval_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Deterministic validation mask: the same budget the training mask targets.

        ``soft`` keeps ``sigmoid(logits) > 0.5``; ``hard_topk`` keeps the
        ``round(target_rate * W)`` highest logits, so the eval mask is not empty at
        a low target rate (the logits start at ``logit(target_rate) < 0``).
        """
        budget = self._adaptive_cfg.budget
        if budget == "soft":
            return (torch.sigmoid(logits) > 0.5).to(logits.dtype)
        if budget == "hard_topk":
            w = logits.shape[-1]
            k = max(1, round(self._adaptive_cfg.target_rate * w))
            flat = logits.reshape(-1, w)
            keep = torch.zeros_like(flat).scatter_(-1, flat.topk(k, dim=-1).indices, 1.0)
            return keep.reshape(logits.shape)
        raise ValueError(f"undersampling.adaptive.budget must be soft|hard_topk, got {budget!r}")

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Grade the deterministic learned mask, not the fully sampled input (cohort5 §2.0)."""
        if not self._adaptive_cfg.enabled:
            return super()._validation_forward(input_batch, batch_context, **kwargs)
        n_coils = input_batch.shape[1] if input_batch.ndim == 4 else 1
        logits = self._ensure_mask(input_batch.shape[-2], input_batch.shape[-1], n_coils=n_coils)
        masked = self._apply_learned_mask(input_batch, self._eval_mask(logits.detach()))
        return super()._validation_forward(masked, batch_context, **kwargs)

    def _apply_learned_mask(self, image: torch.Tensor, mask_line: torch.Tensor) -> torch.Tensor:
        """Undersample ``image`` with the learned per-line mask in k-space.

        ``image`` is the dataloader's input in ``[B, C, H, W]`` layout (C may be
        interleaved real/imag). We round-trip through the centered orthonormal
        FFT (never raw ``torch.fft`` on complex k-space) and keep the input
        layout/channel-count so the result drops straight into the recon
        backbone. The per-line mask (shape ``(W,)`` or ``(n_coils, W)``)
        broadcasts over the phase-encode columns of k-space.
        """
        complex_in = torch.is_complex(image)
        even_channels = image.ndim == 4 and not complex_in and image.shape[1] % 2 == 0

        kspace = fft2c(image)  # complex (..., H, W); shares grad with mask
        # Broadcast the per-line mask over the column (phase-encode) axis.
        mask = mask_line.reshape(*([1] * (kspace.ndim - 1)), -1)
        masked = ifft2c(kspace * mask)  # complex (..., H, W)

        if complex_in:
            return masked.reshape(image.shape).to(image.dtype)
        if even_channels:
            # Re-interleave real/imag to restore the [B, C, H, W] channel count.
            # fft2c collapsed C -> C//2 complex; view_as_real splits each back
            # into (real, imag) at a new trailing axis, then we fold it into C.
            b, c_half, h, w = masked.shape
            ri = torch.view_as_real(masked)  # (B, C//2, H, W, 2)
            recon = ri.permute(0, 1, 4, 2, 3).reshape(b, c_half * 2, h, w)
            return recon.to(image.dtype).reshape(image.shape)
        # Pure-real input (zero-phase): magnitude/real channel round-trips 1:1.
        return masked.real.to(image.dtype).reshape(image.shape)

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Apply learned mask in k-space, then dispatch to recon losses + density penalty.

        Signature matches the canonical keyword-only contract documented at
        ``base.py::_compute_losses_impl``. The previous positional-``batch``
        signature broke any call path that reached the parent (audit F1)
        because the orchestrator forwards keyword args including ``input_batch=``.

        The previous body called ``super()`` on the RAW input and never invoked
        ``_ensure_mask`` / ``_gumbel_sigmoid_mask`` / ``_current_tau`` — so the
        learned sampling mask was never created, sampled, or applied (the
        learnable acquisition was dead). We now wire it: ensure + register the
        logits, sample a soft mask, undersample the input in k-space, and feed
        the masked input to the recon backbone.
        """
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        masked_input = pick_present(batch if isinstance(batch, torch.Tensor) else None, input_batch)
        training = bool(self.generator_model.training)
        # The training step's one call_owned output; the mask and the density penalty share it.
        owned: torch.Tensor | None = None

        if self._adaptive_cfg.enabled and isinstance(masked_input, torch.Tensor):
            # [B, C, H, W] — W is the phase-encode line axis the mask indexes.
            n_coils = masked_input.shape[1] if masked_input.ndim == 4 else 1
            h = masked_input.shape[-2]
            w = masked_input.shape[-1]
            logits = self._ensure_mask(h, w, n_coils=n_coils)
            # The logits may have been created before the env optimizer was
            # attached; re-assert registration cheaply (dedup makes it a no-op).
            self._register_on_opt_g(logits)
            # The Gumbel draw and the schedule belong to training only.
            if training:
                owned = self.call_owned("_mask_logits")
                mask = self._gumbel_sigmoid_mask(owned, self._current_tau())
            else:
                mask = self._eval_mask(logits.detach())
            masked_input = self._apply_learned_mask(masked_input, mask)

        # Run base reconstruction forward + losses on the (now-undersampled) input.
        base_losses = super()._compute_losses_impl(
            input_batch=masked_input,
            target_batch=target_batch,
            epoch=epoch,
            **kwargs,
        )
        if not self._adaptive_cfg.enabled:
            return base_losses

        if not training:
            return base_losses
        self._steps_seen += 1

        # Density penalty: pull sigmoid(logits) (expected sampling rate) toward
        # the target acceleration. This is the only gradient signal that fights
        # the recon loss's incentive to sample everything.
        if self._mask_logits is not None:
            if owned is None:
                owned = self.call_owned("_mask_logits")
            expected_rate = torch.sigmoid(owned).mean()
            target = torch.tensor(self._adaptive_cfg.target_rate, device=self.device)
            density_loss = (expected_rate - target).pow(2)
            base_losses["loss_density_penalty"] = (
                self._adaptive_cfg.density_penalty_lambda * density_loss
            )
            # Add to total loss key in convention used by base strategy.
            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base_losses:
                    base_losses[total_key] = (
                        base_losses[total_key] + base_losses["loss_density_penalty"]
                    )
                    break

        return base_losses
