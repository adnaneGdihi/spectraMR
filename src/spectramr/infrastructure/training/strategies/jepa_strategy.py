"""I-JEPA — Joint Embedding Predictive Architecture (PR-21 / Part-I C).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-21.

Reference: Assran et al. 2023, *I-JEPA: Self-supervised Learning from
Images with a Joint Embedding Predictive Architecture*.

Three trainable / non-trainable pieces:

- ``context_encoder``  — the main generator (trainable).
- ``target_encoder``   — EMA copy of the context encoder (frozen).
- ``predictor``        — a small MLP/Conv head that maps context
  embeddings → predicted target-patch embeddings (trainable).

The strategy:

1. Picks a multi-block mask (one ``context`` block, several ``target``
   blocks) using :func:`multi_block_mask`.
2. Encodes the context block with the online encoder.
3. Encodes the *full* image with the target encoder (no grad), reads
   off the embedding at the target-block locations.
4. Runs the predictor on the context embedding to predict the target
   embedding; loss = smooth-L1 between predictor output and stop-grad
   target.
5. EMA-updates the target encoder.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE

from .base import BaseTrainingStrategy
from .mixins.utils import pick_present


def multi_block_mask(
    H: int,
    W: int,
    n_targets: int = 4,
    ctx_scale: float = 0.85,
    tgt_scale: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample one context mask + ``n_targets`` target masks (all binary).

    Following the I-JEPA recipe: one large context block (~85% area)
    minus the target blocks (~20% area each), with disjoint targets.
    """
    # Mask GEOMETRY (block sizes / offsets) is scalar bookkeeping. Compute it —
    # and the boolean masks — entirely on CPU, then move to ``device`` ONCE at
    # the end. The old code sampled on ``device`` (CUDA in real training) and
    # ``.item()``-synced every draw (~16 blocking host/device syncs per step),
    # a direct performance.md violation. On CPU, ``.item()`` is a free host read.
    cpu = torch.device("cpu")
    rng = torch.Generator(device=cpu).manual_seed(int(torch.randint(0, 1 << 30, (1,)).item()))
    targets: list[torch.Tensor] = []
    occupied = torch.zeros(H, W, dtype=torch.bool, device=cpu)
    for _ in range(n_targets):
        h = max(
            2, int(round(H * (tgt_scale**0.5) * (0.7 + 0.6 * torch.rand(1, generator=rng).item())))
        )
        w = max(
            2, int(round(W * (tgt_scale**0.5) * (0.7 + 0.6 * torch.rand(1, generator=rng).item())))
        )
        y0 = int(torch.randint(0, max(1, H - h + 1), (1,), generator=rng).item())
        x0 = int(torch.randint(0, max(1, W - w + 1), (1,), generator=rng).item())
        m = torch.zeros(H, W, dtype=torch.bool, device=cpu)
        m[y0 : y0 + h, x0 : x0 + w] = True
        targets.append(m)
        occupied = occupied | m
    ctx = ~occupied
    # If context too small, expand by one dilation pass.
    if ctx.float().mean().item() < 0.4:
        ctx = ctx | (~occupied)
    if device != cpu:
        ctx = ctx.to(device)
        targets = [m.to(device) for m in targets]
    return ctx, targets


class JEPAStrategy(BaseTrainingStrategy):
    """I-JEPA SSL strategy with multi-block masking + predictor head + EMA."""

    # Computes its own objective and reads no `losses:` block, so an empty builder stack
    # is the design (LossBuilder.validate) and any declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    # parallel_compat: The predictor is an owned module, built on every rank's first step (DDP
    # broadcasts rank 0's initialisation) and run via call_owned. The target encoder is a
    # frozen EMA copy of the inner generator, updated from the synchronised weights.
    parallel_compat = frozenset({"replicated"})

    momentum: float = 0.996
    n_targets: int = 4
    ctx_scale: float = 0.85
    tgt_scale: float = 0.2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._target_encoder: nn.Module | None = None
        self._predictor: nn.Module | None = None

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("_predictor", OWNED_MODULE))

    def _ensure_modules(self, embed_dim: int) -> None:
        gen = getattr(self.env, "generator", None)
        if gen is None:
            return
        if self._target_encoder is None:
            self._target_encoder = deepcopy(unwrap_model(gen))
            for p in self._target_encoder.parameters():
                p.requires_grad = False
        if self._predictor is None:
            self._predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            ).to(next(gen.parameters()).device)
            self.register_owned_module("_predictor", self._predictor)
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                opt.add_param_group({"params": list(self._predictor.parameters()), "lr": 1e-4})

    def _materialize_owned_state(self, state: dict[str, Any]) -> None:
        """Rebuild the target encoder and predictor, sizing the predictor from its saved weight."""
        super()._materialize_owned_state(state)
        saved = state.get("modules", {}).get("_predictor")
        if saved is not None and self._predictor is None:
            self._ensure_modules(int(saved["0.weight"].shape[1]))

    @torch.no_grad()
    def _ema_update(self) -> None:
        gen = getattr(self.env, "generator", None)
        if gen is None or self._target_encoder is None:
            return
        src = unwrap_model(gen).parameters()
        for tgt_p, src_p in zip(self._target_encoder.parameters(), src, strict=False):
            tgt_p.mul_(self.momentum).add_(src_p.detach(), alpha=1 - self.momentum)

    @staticmethod
    def _embed_to_vector(out: Any) -> torch.Tensor:
        if isinstance(out, dict):
            out = next(iter(out.values()))
        if out.dim() == 4:
            return out.mean(dim=(-1, -2))  # global-pool to (B, C)
        return out.flatten(1)

    def _compute_losses_impl(
        self, batch: Any, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        # The base orchestrator (``BaseTrainingStrategy._compute_losses``)
        # calls ``_compute_losses_impl(input_batch=..., target_batch=...,
        # epoch=..., **kwargs)`` and then back-props the returned
        # ``loss_total``. If we return ``torch.tensor(0.0)`` here on a
        # configuration failure, that tensor is a *leaf with
        # requires_grad=False* and ``loss.backward()`` raises the cryptic
        # ``RuntimeError: element 0 of tensors does not require grad and
        # does not have a grad_fn`` (audit-2026-05-14 E3 on
        # configurable_unet + JEPAStrategy).
        #
        # CLAUDE.md #9 — fail loud instead.
        #
        # The base orchestrator passes ``input_batch`` as a keyword; this
        # strategy's first param is named ``batch`` so the kwarg binds
        # via ``**kwargs`` instead. Resolve from both sites.
        if not isinstance(batch, dict):
            batch_kw = kwargs.get("batch") or kwargs.get("input_batch")
            if isinstance(batch_kw, dict):
                batch = batch_kw
            elif isinstance(batch_kw, torch.Tensor):
                # The orchestrator passed a tensor (``input_batch``); wrap
                # it so the .get() lookups below find the image.
                batch = {"input": batch_kw}

        # F8b (2026-05-17 round 5): The generic ``StrategyContext``
        # dataclass (src/infrastructure/training/utils/strategy_context.py)
        # carries device/config/utilities — NOT the generator. The
        # canonical location is ``self.env.generator`` (also exposed via
        # the ``BaseTrainingStrategy.generator_model`` property which
        # simply returns ``self.env.generator``). The pre-F8b code read
        # from ``self.context.generator`` which never existed on the
        # dataclass, so ``getattr(..., "generator", None)`` always
        # returned ``None`` and the strategy raised "DI did not produce
        # a generator" even when the YAML declared a valid
        # ``model.model_type`` (7 hits in 2026-05-16 smoke).
        gen = getattr(self.env, "generator", None)
        if gen is None:
            raise RuntimeError(
                "JEPAStrategy._compute_losses_impl: "
                "``self.env.generator`` is None — the upstream "
                "ModelBuilder + TrainingEnvironment did not produce a "
                "generator. Check the experiment YAML's ``model:`` "
                "section (model_type, in_channels, out_channels)."
            )
        if not isinstance(batch, dict):
            raise TypeError(
                f"JEPAStrategy expects a dict batch with an ``input`` / "
                f"``image`` key (got {type(batch).__name__}). The "
                "dataloader / strategy collate function must produce a "
                "TrainingBatch-style dict for JEPA pretraining."
            )

        image = pick_present(batch.get("input"), batch.get("image"))
        if image is None:
            raise KeyError(
                "JEPAStrategy: batch is missing both 'input' and 'image' "
                f"keys (available: {sorted(batch.keys())}). The JEPA "
                "context-target pretext task needs the full source image; "
                "set ``data.return_keys`` to include 'input' or 'image'."
            )

        H, W = image.shape[-2:]
        device = image.device
        ctx_mask, tgt_masks = multi_block_mask(
            H,
            W,
            n_targets=self.n_targets,
            ctx_scale=self.ctx_scale,
            tgt_scale=self.tgt_scale,
            device=device,
        )

        masked_input = image * ctx_mask.float().unsqueeze(0).unsqueeze(0)
        ctx_out = self._embed_to_vector(gen(masked_input))
        embed_dim = ctx_out.shape[-1]
        self._ensure_modules(embed_dim)
        if self._target_encoder is None or self._predictor is None:
            # CLAUDE.md #9 — fail loud rather than returning a
            # detached zero that triggers ``does not require grad``
            # downstream during backward.
            raise RuntimeError(
                "JEPAStrategy: _ensure_modules failed to create the "
                "target_encoder / predictor pair. Check that the DI "
                "container produced a non-None generator with at least "
                "one trainable parameter before the strategy was set up."
            )

        with torch.no_grad():
            full_out = self._embed_to_vector(self._target_encoder(image))

        pred = self.call_owned("_predictor", ctx_out)
        loss = F.smooth_l1_loss(pred, full_out.detach())

        # (Removed a dead per-target ``area = m.float().mean()...item()`` loop
        # here — review 2026-07-01: it GPU-synced every step via ``.item()`` and
        # then did ``pass``, i.e. paid a per-step sync for a value never used.)

        self._ema_update()
        return {"loss_total": loss, "loss_jepa": loss.detach()}
