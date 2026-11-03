"""I-JEPA — Joint Embedding Predictive Architecture (PR-21 / Part-I C).

Reference: Assran et al. 2023, *I-JEPA: Self-supervised Learning from
Images with a Joint Embedding Predictive Architecture*.

- ``context_encoder``: the generator (trainable).
- ``target_encoder``: EMA copy of the context encoder (frozen).
- ``predictor``: an MLP mapping the context embedding plus a target block's
  position to that block's target embedding (trainable).

Each step samples one context block and several target blocks
(:func:`multi_block_mask`), encodes the masked image with the online encoder
and the full image with the target encoder, pools both encoder feature maps
over their blocks (``jepa_targets``), and regresses every target-block
embedding from the context embedding and the block geometry (smooth-L1
against the stop-grad target). The target encoder is then EMA-updated.

Validation reports the same objective on a fixed mask (``val_loss``) and a
linear probe of the frozen target encoder's pooled embedding against the
batch's ``contrast_idx`` (``core/metrics/representation_probe.py``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.core.metrics.representation_probe import (
    PROBE_METRIC_KEYS,
    RepresentationProbeAccumulator,
)
from spectramr.core.module_utils import unwrap_model
from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.builders.leaf.adapter_builders import apply_pre_model
from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.models.capabilities import StrategyCapabilities

from .base import BaseTrainingStrategy
from .disentangled_diagnostics import expand_row_names
from .jepa_targets import (
    GEOMETRY_DIM,
    block_geometry,
    encoder_features,
    flatten_volume_rows,
    pool_over_mask,
)
from .mixins.utils import pick_present

#: Validation draws its blocks from this seed so ``val_loss`` is comparable across passes.
_VALIDATION_MASK_SEED = 0


def multi_block_mask(
    H: int,
    W: int,
    n_targets: int = 4,
    ctx_scale: float = 0.85,
    tgt_scale: float = 0.2,
    device: torch.device = torch.device("cpu"),
    seed: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample one context mask + ``n_targets`` target masks (all binary).

    Following the I-JEPA recipe: one large context block (~85% area)
    minus the target blocks (~20% area each), with disjoint targets.
    """
    # Geometry is scalar bookkeeping: sample it on CPU, where ``.item()`` is a free host
    # read, and move the masks to ``device`` once (sampling on CUDA synced every draw).
    cpu = torch.device("cpu")
    if seed is None:
        seed = int(torch.randint(0, 1 << 30, (1,)).item())
    rng = torch.Generator(device=cpu).manual_seed(seed)
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
    capabilities = StrategyCapabilities(emitted_metrics=PROBE_METRIC_KEYS)

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
        self._probe: RepresentationProbeAccumulator | None = None

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("_predictor", OWNED_MODULE))

    def _ensure_modules(self, embed_dim: int, predictor: bool = True) -> None:
        gen = getattr(self.env, "generator", None)
        if gen is None:
            return
        if self._target_encoder is None:
            self._target_encoder = deepcopy(unwrap_model(gen))
            for p in self._target_encoder.parameters():
                p.requires_grad = False
        if predictor and self._predictor is None:
            self._predictor = nn.Sequential(
                nn.Linear(embed_dim + GEOMETRY_DIM, embed_dim * 2),
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
            self._ensure_modules(int(saved["0.weight"].shape[1]) - GEOMETRY_DIM)

    @torch.no_grad()
    def _ema_update(self) -> None:
        gen = getattr(self.env, "generator", None)
        if gen is None or self._target_encoder is None:
            return
        src = unwrap_model(gen).parameters()
        for tgt_p, src_p in zip(self._target_encoder.parameters(), src, strict=False):
            tgt_p.mul_(self.momentum).add_(src_p.detach(), alpha=1 - self.momentum)

    def _embedding_loss(
        self, gen: nn.Module, image: torch.Tensor, seed: int | None = None
    ) -> torch.Tensor:
        """Smooth-L1 between predicted and target embeddings of every target block."""
        H, W = image.shape[-2:]
        ctx_mask, tgt_masks = multi_block_mask(
            H,
            W,
            n_targets=self.n_targets,
            ctx_scale=self.ctx_scale,
            tgt_scale=self.tgt_scale,
            device=image.device,
            seed=seed,
        )
        masked_input = image * ctx_mask.to(image.dtype)[None, None]
        # Copy the target encoder before the forward: a latent cache with grad is not copyable.
        self._ensure_modules(0, predictor=False)
        ctx_feats = encoder_features(gen, gen(masked_input))
        self._ensure_modules(int(ctx_feats.shape[1]))
        if self._target_encoder is None or self._predictor is None:
            raise RuntimeError(
                "JEPAStrategy: _ensure_modules failed to create the target_encoder / "
                "predictor pair; the generator is missing or has no parameters."
            )
        with torch.no_grad():
            tgt_feats = encoder_features(self._target_encoder, self._target_encoder(image))
        ctx_emb = pool_over_mask(ctx_feats, ctx_mask)
        b = ctx_emb.shape[0]
        queries = torch.stack(
            [
                torch.cat([ctx_emb, block_geometry(m).to(ctx_emb.dtype).expand(b, -1)], dim=1)
                for m in tgt_masks
            ]
        )
        targets = torch.stack([pool_over_mask(tgt_feats, m) for m in tgt_masks])
        pred = self.call_owned("_predictor", queries)
        return F.smooth_l1_loss(pred, targets.detach())

    def _compute_losses_impl(
        self, batch: Any, *args: Any, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        # Resolve ``input_batch=`` too; every failure raises (a detached zero fails in backward).
        if not isinstance(batch, dict):
            batch_kw = kwargs.get("batch") or kwargs.get("input_batch")
            if isinstance(batch_kw, dict):
                batch = batch_kw
            elif isinstance(batch_kw, torch.Tensor):
                batch = {"input": batch_kw}

        gen = getattr(self.env, "generator", None)
        if gen is None:
            raise RuntimeError(
                "JEPAStrategy._compute_losses_impl: ``self.env.generator`` is None — the "
                "ModelBuilder did not produce a generator; check the YAML's ``model:`` block."
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

        # train_step hands the pre_model-adapted tensor; a direct dict call gets the same chain.
        prepared = kwargs.get("input_batch")
        if not isinstance(prepared, torch.Tensor):
            prepared = apply_pre_model(self.adapter_chains, image, None)[0]
        loss = self._embedding_loss(gen, prepared)
        self._ema_update()
        return {"loss_total": loss, "loss_jepa": loss.detach()}

    def reset_validation_metrics(self) -> None:
        """Open a validation pass: also start the linear-probe accumulator."""
        super().reset_validation_metrics()
        self._probe = RepresentationProbeAccumulator()

    def finalize_validation(self) -> dict[str, float]:
        """Close the pass: add the frozen-encoder linear probe of the contrast label."""
        summary = dict(super().finalize_validation())
        probe, self._probe = self._probe, None
        if probe is not None and probe.n_rows:
            summary.update(probe.summary())
        return summary

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Registry metrics, the JEPA objective on a fixed mask, and the probe rows.

        Raises:
            KeyError: when the batch carries no ``contrast_idx`` for the probe.
        """
        encoder = self._target_encoder
        if self._predictor is None or encoder is None:
            raise RuntimeError("JEPAStrategy validation before the first training step.")
        metrics = dict(super().validation_step(input_batch, target_batch, **kwargs))
        gen = self.env.generator
        image, _ = apply_pre_model(self.adapter_chains, input_batch, None)  # as train_step
        image = flatten_volume_rows(image.to(self.device, non_blocking=True))
        modes = [(m, m.training) for m in (gen, encoder)]
        for m, _ in modes:
            m.eval()
        try:
            loss = self._embedding_loss(gen, image, seed=_VALIDATION_MASK_SEED)
            emb = encoder_features(encoder, encoder(image)).mean(dim=(-2, -1))
        finally:
            for m, was_training in modes:
                m.train(was_training)
        metrics["val_loss"] = float(loss)
        labels = read_batch_field(batch_data, "contrast_idx") if batch_data is not None else None
        if labels is None:
            raise KeyError(
                "JEPAStrategy validation: the batch carries no 'contrast_idx', which the "
                "frozen-encoder linear probe classifies; use a dataset that stamps it "
                "(m4raw does)."
            )
        labels = torch.as_tensor(labels).reshape(-1)
        n_rows = emb.shape[0]
        if n_rows % labels.numel():
            raise ValueError(f"{labels.numel()} contrast_idx values for {n_rows} rows")
        labels = labels.repeat_interleave(n_rows // labels.numel())
        groups = expand_row_names(read_batch_field(batch_data, "subject_id"), n_rows)
        if self._probe is None:
            self._probe = RepresentationProbeAccumulator()
        self._probe.add(emb, labels.to(emb.device), groups=groups)
        return metrics
