"""Multi-Contrast Contrastive SSL Strategy (Multi-Contrast Item 1).

Plan: TODO/backlog_multi_contrast_remaining.md §Item 1.

InfoNCE / SimCLR-style contrastive loss where positives are co-registered
contrasts of the same slice (``view_a``/``view_b``, e.g. ``m4raw_triplet`` with
``contrastive_views``). Under ``data.multi_contrast.negative_sampling:
cross_subject`` in-batch pairs from the same subject are masked out of the
negatives (the batch's ``subject_index``); ``cross_volume`` keeps every other
sample. Validation reports the InfoNCE loss and top-1 view retrieval.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field
from spectramr.models.capabilities import StrategyCapabilities

from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class MultiContrastContrastiveStrategy(BaseTrainingStrategy):
    """Contrastive SSL on co-registered multi-contrast cohorts."""

    # parallel_compat: The momentum encoder is a frozen copy of the inner generator, updated on
    # every rank from the synchronised weights, so it needs no reduction.
    parallel_compat = frozenset({"replicated"})
    #: The momentum warm-up schedule reads ``_step``; a resume must continue it.
    checkpointed_counters: ClassVar[tuple[str, ...]] = ("_step",)
    capabilities: ClassVar[StrategyCapabilities] = dataclasses.replace(
        BaseTrainingStrategy.capabilities,
        emitted_metrics=frozenset({"infonce", "view_retrieval_top1"}),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._apply_mc_knobs(self.config)
        self._target_encoder: nn.Module | None = None
        self._step = 0

    @staticmethod
    def _resolve_mc_knobs(config: Any) -> dict[str, Any]:
        """Resolve + validate ``config.training.ssl.multi_contrast_contrastive``; no defaults.

        ``training.ssl`` arrives as a dict at runtime, so the block is coerced through
        ``MultiContrastContrastiveConfigSchema`` for its field validation.

        Raises:
            ValueError: the block is absent, ``temperature`` is not positive, or
                ``symmetrize`` and ``use_momentum`` are both on (the second InfoNCE
                direction through a gradient-frozen momentum target is a no-op).
        """
        from spectramr.config.schemas.base import (
            MultiContrastContrastiveConfigSchema,
        )

        # ``ssl`` exists only on the SSL training schema variant.
        ssl_cfg = getattr(config.training, "ssl", None)
        if isinstance(ssl_cfg, dict):
            raw = ssl_cfg.get("multi_contrast_contrastive")
        else:
            raw = getattr(ssl_cfg, "multi_contrast_contrastive", None) if ssl_cfg else None
        if raw is None:
            raise ValueError(
                "MultiContrastContrastiveStrategy requires the nested block "
                "config.training.ssl.multi_contrast_contrastive "
                "(temperature/symmetrize/use_momentum/momentum/warmup_steps). "
                "It is absent — refusing to fall back to defaults that would "
                "silently ignore the YAML."
            )
        if isinstance(raw, MultiContrastContrastiveConfigSchema):
            mc_cfg = raw
        else:
            mc_cfg = MultiContrastContrastiveConfigSchema.model_validate(raw)
        temperature = float(mc_cfg.temperature)
        if temperature <= 0:
            raise ValueError(
                f"multi_contrast_contrastive.temperature must be > 0, got {temperature}"
            )
        symmetrize = bool(mc_cfg.symmetrize)
        use_momentum = bool(mc_cfg.use_momentum)
        if symmetrize and use_momentum:
            raise ValueError(
                "multi_contrast_contrastive: symmetrize=True is incompatible "
                "with use_momentum=True — the momentum target encoder is "
                "gradient-frozen, so the second InfoNCE direction has no "
                "gradient path and symmetrize would silently no-op. Set "
                "either symmetrize=False or use_momentum=False."
            )
        return {
            "temperature": temperature,
            "symmetrize": symmetrize,
            "use_momentum": use_momentum,
            "momentum": float(mc_cfg.momentum),
            "warmup_steps": int(mc_cfg.warmup_steps),
        }

    def _apply_mc_knobs(self, config: Any) -> None:
        """Resolve the knobs and stamp them onto ``self`` + the run log."""
        knobs = self._resolve_mc_knobs(config)
        self._temperature = knobs["temperature"]
        self._symmetrize = knobs["symmetrize"]
        self._use_momentum = knobs["use_momentum"]
        self._momentum = knobs["momentum"]
        # Warm-up steps before momentum encoder fully tracks.
        self._warmup_steps = knobs["warmup_steps"]
        mc = getattr(getattr(config, "data", None), "multi_contrast", None)
        if mc is None:  # a config without the block carries the schema's default
            from spectramr.config.schemas.data import MultiContrastConfigSchema

            mc = MultiContrastConfigSchema()
        self._negative_sampling = mc.negative_sampling
        # Stamp the resolved effective knobs into the run log so the
        # provenance reflects what actually ran (pitfall #15).
        logger.info(
            "multi_contrast_contrastive knobs: temperature=%s symmetrize=%s "
            "use_momentum=%s momentum=%s warmup_steps=%s negative_sampling=%s",
            self._temperature,
            self._symmetrize,
            self._use_momentum,
            self._momentum,
            self._warmup_steps,
            self._negative_sampling,
        )

    def _ensure_momentum_encoder(self) -> None:
        # Use the DI-injected generator (``env.generator``); the
        # previous ``self.context.generator`` lookup hit a non-existent
        # attribute and silently became None — see audit F3.
        gen = self.env.generator
        if gen is None or self._target_encoder is not None or not self._use_momentum:
            return
        from copy import deepcopy

        # The inner module: a copy of a DDP wrapper carries its process group and prefixes
        # every checkpoint key with ``module.``.
        self._target_encoder = deepcopy(unwrap_model(gen))
        for p in self._target_encoder.parameters():
            p.requires_grad = False

    def _materialize_owned_state(self, state: dict[str, Any]) -> None:
        """Rebuild the momentum encoder so its saved EMA weights load over it."""
        super()._materialize_owned_state(state)
        if "_target_encoder" in state.get("modules", {}):
            self._ensure_momentum_encoder()

    @torch.no_grad()
    def _ema_update(self) -> None:
        gen = self.env.generator
        if gen is None or self._target_encoder is None:
            return
        m = self._momentum * min(1.0, self._step / max(1, self._warmup_steps))
        src_params = unwrap_model(gen).parameters()
        for tgt, src in zip(self._target_encoder.parameters(), src_params, strict=False):
            tgt.mul_(m).add_(src, alpha=1 - m)

    @staticmethod
    def _ddp_gather(z: torch.Tensor) -> torch.Tensor:
        """Cross-rank gather so contrastive negatives include the full
        world-size batch. No-op in single-process runs."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return z
        world_size = torch.distributed.get_world_size()
        gathered = [torch.zeros_like(z) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, z.detach())
        rank = torch.distributed.get_rank()
        gathered[rank] = z
        return torch.cat(gathered, dim=0)

    @staticmethod
    def _info_nce(
        z1: torch.Tensor,
        z2: torch.Tensor,
        tau: float,
        pos_offset: int = 0,
        same_subject: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """InfoNCE: cross-entropy with positives at ``pos_offset + i``.

        ``pos_offset`` places the local block inside a DDP-gathered ``z2`` (columns
        ``rank*B``). ``same_subject`` ``[B, N]`` marks non-positive columns from the
        same subject; they are removed from the negatives.
        """
        logits = F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).t() / tau
        labels = torch.arange(z1.shape[0], device=z1.device) + pos_offset
        if same_subject is not None:
            logits = logits.masked_fill(same_subject, float("-inf"))
        return F.cross_entropy(logits, labels)

    def _views(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        view_a, view_b = (read_batch_field(batch, k) for k in ("view_a", "view_b"))
        missing = [k for k, v in (("view_a", view_a), ("view_b", view_b)) if v is None]
        if missing:
            raise RuntimeError(
                f"multi_contrast_contrastive needs co-registered contrast views; the batch has no "
                f"{missing}. Serve them with data.m4raw_triplet.contrastive_views."
            )
        subject = read_batch_field(batch, "subject_index")
        if self._negative_sampling == "cross_subject" and subject is None:
            raise RuntimeError(
                "negative_sampling 'cross_subject' needs the batch's subject_index to mask "
                "same-subject negatives"
            )
        return view_a, view_b, subject if self._negative_sampling == "cross_subject" else None

    @staticmethod
    def _projection(z: Any) -> torch.Tensor:
        return z["projection"] if isinstance(z, dict) else z

    def _same_subject(self, subject: torch.Tensor | None, offset: int) -> torch.Tensor | None:
        """``[B, N]`` mask of same-subject, non-positive pairs against the gathered batch."""
        if subject is None:
            return None
        local = subject.reshape(-1)
        gathered = self._ddp_gather(local)
        rows = torch.arange(local.numel(), device=local.device)
        cols = torch.arange(gathered.numel(), device=local.device)
        positive = cols[None, :] == (rows + offset)[:, None]  # elementwise: no index_put sync
        return (local[:, None] == gathered[None, :]) & ~positive

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Canonical signature; the views come from the batch (``kwargs``)."""
        self._ensure_momentum_encoder()
        gen = self.env.generator
        if gen is None:
            raise RuntimeError("multi_contrast_contrastive strategy: env.generator is None")
        view_a, view_b, subject = self._views(self._resolve_legacy_batch(input_batch, kwargs))
        z_a = self._projection(gen(view_a))
        if self._target_encoder is not None:
            with torch.no_grad():
                z_b = self._projection(self._target_encoder(view_b))
        else:
            z_b = self._projection(gen(view_b))
        # The local block sits at columns rank*B of the gathered negatives.
        dist = torch.distributed
        offset = (
            dist.get_rank() * z_a.shape[0] if dist.is_available() and dist.is_initialized() else 0
        )
        same = self._same_subject(subject, offset)
        loss = self._info_nce(z_a, self._ddp_gather(z_b), self._temperature, offset, same)
        if self._symmetrize:  # __init__ refused symmetrize with a momentum target
            loss = 0.5 * (
                loss + self._info_nce(z_b, self._ddp_gather(z_a), self._temperature, offset, same)
            )
        self._step += 1
        self._ema_update()
        return {"loss_total": loss, "loss_contrastive": loss.detach()}

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """InfoNCE and top-1 view retrieval of the online encoder on the validation pairs."""
        view_a, view_b, subject = self._views(batch_data)
        gen = self.env.generator
        z_a, z_b = self._projection(gen(view_a)), self._projection(gen(view_b))
        same = self._same_subject(subject, 0)
        logits = F.normalize(z_a, dim=-1) @ F.normalize(z_b, dim=-1).t()
        if same is not None:
            logits = logits.masked_fill(same, float("-inf"))
        hits = logits.argmax(dim=1) == torch.arange(z_a.shape[0], device=z_a.device)
        return {
            "infonce": self._info_nce(z_a, z_b, self._temperature, 0, same),
            "view_retrieval_top1": hits.float().mean(),
        }
