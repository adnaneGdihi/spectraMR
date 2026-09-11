"""k-Space Implicit Neural Representation (ULF-PR-4 / II.1).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-4.

This training strategy supervises a k-space-domain network on the sampled
k-space points only (the unsampled locations are masked out of the loss).
The configured model maps the under-sampled k-space to a completed k-space;
the objective is a mask-weighted L1 in k-space and never enters the image
domain (the wired arm sets ``losses.output_domain: kspace`` /
``model.model_domain: kspace``).

NOTE (2026-06 audit): the original docstring advertised a per-subject SIREN
coordinate-MLP ``(k_x, k_y) -> (real, imag)`` with a per-subject fit and an
iFFT-to-image step. That coordinate-grid / SIREN forward and the image-domain
iFFT term are NOT implemented here, and no ``kspace_inr`` coordinate-MLP model
is registered — the wired model is a plain k-space CNN. The docstring is
aligned to the actual k-space-only behaviour to avoid the pitfall #15 /
#9 "advertised-but-unimplemented" divergence. Implementing the genuine
per-subject INR (a registered SIREN model + image-domain ``ifft2c`` term)
remains future work tracked in the ULF-PR-4 plan.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.data.batch_types import read_batch_field

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class KSpaceINRStrategy(ReconstructionTrainingStrategy):
    """k-Space INR — mask-weighted L1 supervision on the sampled k-space points.

    See the module docstring for the scope caveat: this is a k-space-domain
    masked-L1 strategy, not (yet) the per-subject SIREN coordinate-MLP fit.
    """

    #: Loss ownership (issue #1918). l1_loss(pred_k, kspace) is a K-SPACE fidelity term. It is not the canonical
    #: image-space l1, so it must NOT be declared inline: doing so would make the
    #: witness exclude a declared image_losses: [l1] from the very check that
    #: would otherwise report it as unreachable.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Provenance stamp (decision: document the alias): this arm runs as a
        # mask-weighted k-space L1, NOT the advertised per-subject SIREN
        # coordinate-MLP + iFFT (that remains roadmap — see the design-decision
        # memo). Stamp it so a run is never mistaken for the INR paradigm.
        logger.info(
            "KSpaceINRStrategy: running as mask-weighted k-space L1 "
            "(per-subject SIREN coordinate-MLP fit is not implemented)."
        )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Loss is just MSE on the sampled k-space points (the rest is masked out)."""
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        if gen is None:
            # No silent zero-loss fallback (CLAUDE.md pitfall #9/#10): a missing
            # generator must fail loudly, not present a zero-gradient step that
            # looks like training.
            raise RuntimeError(
                "k-space INR requires self.env.generator and batch['kspace']; generator is None"
            )

        # ``read_batch_field``, not ``isinstance(batch, dict)``: the loop delivers a
        # ``TrainingBatch`` dataclass, so the isinstance leg is False and the
        # ``else`` handed the WHOLE BATCH OBJECT on as if it were a tensor.
        kspace = read_batch_field(batch, "kspace")
        mask = read_batch_field(batch, "mask")
        if kspace is None:
            raise RuntimeError(
                "k-space INR requires self.env.generator and batch['kspace']; "
                "batch['kspace'] is missing"
            )

        pred_k = gen(kspace, mask=mask) if mask is not None else gen(kspace)
        if mask is not None:
            # Mask-weighted L1 — sum/mask-sum rather than the SSOT F.l1_loss
            # because the standard reduction would average over masked-out pixels.
            loss = ((pred_k - kspace).abs() * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss = F.l1_loss(pred_k, kspace)
        return {"loss_total": loss, "loss_kspace_mae": loss.detach()}
