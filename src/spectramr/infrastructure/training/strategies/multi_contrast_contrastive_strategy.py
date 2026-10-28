"""Multi-Contrast Contrastive SSL Strategy (Multi-Contrast Item 1).

Plan: TODO/backlog_multi_contrast_remaining.md §Item 1.

InfoNCE / SimCLR-style contrastive loss where positives are co-registered
contrasts of the same subject and negatives are different subjects.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model

from .base import BaseTrainingStrategy
from .mixins.utils import pick_present

logger = logging.getLogger(__name__)


class MultiContrastContrastiveStrategy(BaseTrainingStrategy):
    """Contrastive SSL on co-registered multi-contrast cohorts."""

    # parallel_compat: The momentum encoder is a frozen copy of the inner generator, updated on
    # every rank from the synchronised weights, so it needs no reduction.
    parallel_compat = frozenset({"replicated"})
    #: The momentum warm-up schedule reads ``_step``; a resume must continue it.
    checkpointed_counters: ClassVar[tuple[str, ...]] = ("_step",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._apply_mc_knobs(self.config)
        self._target_encoder: nn.Module | None = None
        self._step = 0

    @staticmethod
    def _resolve_mc_knobs(config: Any) -> dict[str, Any]:
        """Resolve + validate the multi-contrast contrastive knobs from the SSOT.

        Reads ONLY the declared nested home
        (``config.training.ssl.multi_contrast_contrastive``). There is no flat
        ``config.ssl`` fallback (it never existed on ``TrainingSettings``) and
        no ``getattr`` defaults — an absent block raises so a misconfigured run
        fails at startup instead of silently using stand-in defaults that ignore
        the YAML (CLAUDE.md non-negotiables #8 / pitfalls #9, #15).

        Note:
            ``config.training.ssl`` is declared as ``dict[str, Any] | None`` on
            ``TrainingConfigSchema`` (``schemas/training/core.py``), so at
            runtime it is a plain dict (the nested block is a dict too) — NOT an
            ``SSLConfigSchema`` object. We coerce the block into
            ``MultiContrastContrastiveConfigSchema`` here so the schema's own
            validation (``temperature>0``, ``momentum∈[0,1]``,
            ``extra='forbid'``) actually runs at strategy startup regardless of
            whether ``ssl`` arrives as a dict or an already-typed object.

        Raises:
            ValueError: if the nested block is absent, ``temperature`` is not
                positive, or ``symmetrize`` and ``use_momentum`` are both True
                (a gradient-frozen momentum target makes the symmetric second
                InfoNCE direction a silent no-op — reject rather than degrade).
        """
        from spectramr.config.schemas.base import (
            MultiContrastContrastiveConfigSchema,
        )

        # ``ssl`` lives only on the SSL paradigm variant of the training
        # schema, so keep a single (leaf-level) getattr — direct access would
        # AttributeError for non-SSL configs. ``config.training`` is a
        # guaranteed top-level section.
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
        # Coerce a raw dict (the runtime form) through the schema so its
        # field-level validation runs; pass an already-typed object through.
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
        # ``symmetrize`` is incompatible with a momentum target encoder: the
        # target is gradient-frozen, so the second (symmetric) InfoNCE
        # direction routed through it carries no gradient and the advertised
        # ``symmetrize=True`` would silently no-op (audit: dropped knob).
        # Raise rather than degrade (pitfall #9) — the YAML must turn one off.
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
        # Stamp the resolved effective knobs into the run log so the
        # provenance reflects what actually ran (pitfall #15).
        logger.info(
            "multi_contrast_contrastive knobs: temperature=%s symmetrize=%s "
            "use_momentum=%s momentum=%s warmup_steps=%s",
            self._temperature,
            self._symmetrize,
            self._use_momentum,
            self._momentum,
            self._warmup_steps,
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
        z1: torch.Tensor, z2: torch.Tensor, tau: float, pos_offset: int = 0
    ) -> torch.Tensor:
        """Symmetric InfoNCE: cross-entropy with positives on the diagonal.

        ``pos_offset`` shifts the positive column for the local block when
        ``z2`` is a DDP-gathered tensor (local samples sit at columns
        ``rank*B``); 0 in single-process runs.
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / tau
        labels = torch.arange(z1.shape[0], device=z1.device) + pos_offset
        return F.cross_entropy(logits, labels)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Canonical signature; reads the dict-batch from kwargs (audit F1)."""
        self._ensure_momentum_encoder()
        # ``self.env.generator`` instead of ``self.context.generator``
        # (audit F3 silent-fallback fix).
        gen = self.env.generator
        if gen is None:
            raise RuntimeError("multi_contrast_contrastive strategy: env.generator is None")

        # Resolve the dict batch through the SSOT helper, then enforce the
        # contrast-view contract: a dict batch MUST carry both co-registered
        # views. Previously a missing key passed ``None`` straight into the
        # generator (``gen(None)``) producing an obscure crash with no
        # diagnostic — raise descriptively instead (audit F1; pitfall #9).
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if isinstance(batch, dict):
            missing = [k for k in ("view_a", "view_b") if batch.get(k) is None]
            if missing:
                raise RuntimeError(
                    "multi_contrast_contrastive requires co-registered contrast "
                    f"views but the batch is missing {missing}. Expected keys "
                    "'view_a' and 'view_b' (e.g. two contrasts of the same "
                    "subject). No producer under src/spectramr/data/ currently "
                    "emits these keys — wire a multi-contrast dataset/transform "
                    "before running this strategy end-to-end."
                )
            view_a = batch["view_a"]
            view_b = batch["view_b"]
        else:
            # Non-dict batch: a single tensor view is treated as both branches
            # (degenerate single-view contrastive). pick_present keeps tensor
            # operands intact (never ``a or b`` on tensors).
            view_a = view_b = pick_present(batch, input_batch)
            if view_a is None:
                raise RuntimeError(
                    "multi_contrast_contrastive received no usable batch: both "
                    "the resolved batch and input_batch are None."
                )

        z_a = gen(view_a)
        # Use momentum encoder for the second view if available.
        if self._target_encoder is not None:
            with torch.no_grad():
                z_b = self._target_encoder(view_b)
        else:
            z_b = gen(view_b)

        if isinstance(z_a, dict):
            z_a = z_a.get("projection", next(iter(z_a.values())))
        if isinstance(z_b, dict):
            z_b = z_b.get("projection", next(iter(z_b.values())))

        # DDP gather so negatives span the full world-size batch. The local
        # block sits at columns rank*B in the gathered tensor, so the positive
        # labels must be offset by rank*B (the old arange(B) was correct only
        # on rank 0).
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            pos_offset = torch.distributed.get_rank() * z_a.shape[0]
        else:
            pos_offset = 0
        z_b_neg = self._ddp_gather(z_b)

        loss = self._info_nce(z_a, z_b_neg, self._temperature, pos_offset)
        if self._symmetrize:
            # ``symmetrize`` is honored unconditionally here: __init__ has
            # already rejected the symmetrize + use_momentum combination, so
            # when we reach this branch ``_target_encoder`` is guaranteed None
            # and both views went through the same (online) encoder — the
            # second InfoNCE direction has a real gradient path. (Previously
            # this was gated on ``_target_encoder is None``, which silently
            # dropped the term whenever a momentum encoder was active.)
            loss = 0.5 * (
                loss + self._info_nce(z_b, self._ddp_gather(z_a), self._temperature, pos_offset)
            )

        self._step += 1
        self._ema_update()
        return {"loss_total": loss, "loss_contrastive": loss.detach()}
