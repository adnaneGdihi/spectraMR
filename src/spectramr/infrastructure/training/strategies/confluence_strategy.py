"""Cross-source confluence — Fréchet-mean consensus (MICCAI MRIxFields2026, B-1.1).

The travelling-volunteer tuple images the SAME subject at every source field
``{0.1,1.5,3,5}`` T with a single 7T target. A source-conditioned renderer
(:class:`~spectramr.models.generators.cross_field_renderer.AnatomyFieldRenderer`, reused
unchanged from B-3.8/B-1.9) produces a per-source 7T prediction ``x_hat_b = G(x_b)``; the
loss combines per-source FIDELITY to the shared target with a CONSENSUS (Fréchet-mean)
regulariser pulling the per-source predictions together::

    L = (1/N) sum_b ||G(x_b) - x_7||_1  +  lambda * mean_{b<b'} ||G(x_b) - G(x_{b'})||_1

The K=N per-source predictors are correlated (rho<1) variance-sigma^2 estimators; their
consensus has variance ``sigma^2 (rho + (1-rho)/N) < sigma^2`` — a strict reduction
(B-1.1's variance-reduction argument). Inference uses the consensus MEAN over the N source
predictions (the variance-reduced estimate).

ONE-KNOB ABLATION: ``lambda_consensus`` (a config knob, sampler-and-loss). ``lambda=0`` is
NOT degenerate — it trains the SAME renderer on all source->7T pairs (a valid multi-source
fidelity translator), so the delta vs ``lambda>0`` isolates exactly the consensus
regulariser's contribution (#17). The multi-source TUPLE is supplied by the dataset's
``multi_source`` pairing policy (``batch['sources']`` ``[B,N,1,H,W]``).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from .reconstruction import ReconstructionTrainingStrategy


def _render_each_source(
    model: Any,
    sources: torch.Tensor,
    field_target: torch.Tensor,
    contrast_id: torch.Tensor | None,
) -> list[torch.Tensor]:
    """Render every source ``[B,N,1,H,W]`` to the target field -> list of N ``[B,1,H,W]``."""
    n = sources.shape[1]
    preds = []
    for i in range(n):
        q = model.encode(sources[:, i])
        preds.append(model.render(q, field_target, contrast_id))
    return preds


def compute_confluence_losses(
    model: Any,
    batch: dict[str, Any],
    *,
    lambda_consensus: float = 1.0,
    contrast_conditioning: bool = True,
) -> dict[str, torch.Tensor]:
    """Per-source 7T fidelity + Fréchet-mean consensus over the travelling-volunteer tuple."""
    sources = batch["sources"]  # [B, N, 1, H, W]
    target = batch["target"]  # [B, 1, H, W]
    b_t = batch["field_strength_target"].reshape(-1).float()
    cid = batch.get("contrast_id") if contrast_conditioning else None
    preds = _render_each_source(model, sources, b_t, cid)  # N x [B,1,H,W]
    n = len(preds)

    loss_fidelity = sum(F.l1_loss(p, target) for p in preds) / n
    # Consensus: mean pairwise L1 between the per-source predictions (the Frechet-mean pull).
    pair_terms = [F.l1_loss(preds[i], preds[j]) for i in range(n) for j in range(i + 1, n)]
    loss_consensus = (
        torch.stack(pair_terms).mean()
        if pair_terms
        else torch.zeros((), device=target.device, dtype=target.dtype)
    )
    total = loss_fidelity + lambda_consensus * loss_consensus
    # The inference deliverable is the variance-reduced CONSENSUS MEAN over the N source
    # predictions; sharpen THAT (not any single source render) against the shared target.
    consensus = torch.stack(preds, dim=0).mean(dim=0)
    return {
        "loss_total": total,
        "loss_fidelity": loss_fidelity,
        "loss_consensus": loss_consensus,
        # Grad-carrying pair for the loss-SSOT seam (popped in _compute_losses_impl).
        "prediction": consensus,
        "target_image": target,
    }


class ConfluenceStrategy(ReconstructionTrainingStrategy):
    """Multi-source Fréchet-mean consensus cross-field synthesis (B-1.1)."""

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = True

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("confluence", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        cfg = getattr(self.config.training, "confluence", None)
        self._cf_lambda_consensus = (
            float(getattr(cfg, "lambda_consensus", 1.0)) if cfg is not None else 1.0
        )
        self._cf_contrast = bool(getattr(cfg, "contrast_conditioning", True)) if cfg else True

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if batch is None or not hasattr(batch, "get"):  # dict OR TrainingBatch (canonical pipeline)
            raise ValueError(
                "ConfluenceStrategy requires a mapping batch (dict/TrainingBatch) from the "
                f"mrixfields dataset; got {type(batch)!r}."
            )
        if batch.get("sources") is None:
            raise ValueError(
                "ConfluenceStrategy requires the multi-source tuple 'sources' "
                "(set data.mrixfields_pairing_policy='multi_source'). Got None."
            )
        out = compute_confluence_losses(
            self.env.generator,
            batch,
            lambda_consensus=getattr(self, "_cf_lambda_consensus", 1.0),
            contrast_conditioning=getattr(self, "_cf_contrast", True),
        )
        # Loss-SSOT seam: fold declarative image losses (hfen/ms_ssim) onto the inline
        # objective, applied to the consensus deliverable. The inline per-source L1
        # fidelity is skipped by the seam (no double-count).
        pred = out.pop("prediction", None)
        target_image = out.pop("target_image", None)
        if pred is not None and target_image is not None:
            aux = self._apply_builder_image_losses(pred, target_image, out)
            if aux is not None:
                out["loss_total"] = out["loss_total"] + aux
            self._last_prediction = pred
            self._last_target = target_image
        return out

    def validation_step(
        self,
        input_batch: Any,
        target_batch: Any,
        sources: Any = None,
        field_strength_target: Any = None,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Fold the tuple + target field into batch_context (gated seam) for the consensus eval."""
        bc = kwargs.pop("batch_context", None) or {}
        if sources is not None:
            bc["sources"] = sources
        if field_strength_target is not None:
            bc["field_strength_target"] = field_strength_target
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """Validation = the variance-reduced CONSENSUS MEAN over the N source predictions.

        This grades the actual deliverable (the consensus estimate), so the metric reflects
        the variance reduction the method claims (#18). Falls back to the single ``input``
        render when the tuple is absent (e.g. a non-multi_source loader)."""
        gen = self.env.generator
        b_t = batch_context.get("field_strength_target", kwargs.get("field_strength_target"))
        if b_t is None:
            raise ValueError(
                "ConfluenceStrategy validation requires per-sample 'field_strength_target' "
                "(set data.expose_field_strength_target). Got None."
            )
        cid = batch_context.get("contrast_id") if self._cf_contrast else None
        sources = batch_context.get("sources")
        if sources is None:
            x0 = batch_context.get("input", input_batch)
            b_t = torch.as_tensor(b_t, device=x0.device).reshape(-1).float()
            out = gen.render(gen.encode(x0), b_t, cid).clamp(0.0, 1.0)
            self._last_visual_pred = out.detach()
            return out
        sources = torch.as_tensor(sources, device=input_batch.device)
        b_t = torch.as_tensor(b_t, device=sources.device).reshape(-1).float()
        preds = _render_each_source(gen, sources, b_t, cid)
        stacked = torch.stack(preds, dim=0)
        consensus = stacked.mean(dim=0).clamp(0.0, 1.0)  # variance-reduced estimate
        # Inter-source dispersion: a collapse to ~0 (encoder driven source-invariant, rho->1)
        # zeroes the consensus loss WITHOUT genuine variance reduction (#20). Expose it so a
        # collapsed run is distinguishable from a real one (a low loss_consensus alone does
        # NOT certify the sigma^2(rho+(1-rho)/N) reduction — rho is driven, not measured).
        self._last_intersource_std = float(stacked.std(dim=0).mean().detach())
        # Cache the graded consensus so the image-capture path saves the actual deliverable
        # (it bypasses validation_step's batch_context and would otherwise save nothing).
        self._last_visual_pred = consensus.detach()
        return consensus
