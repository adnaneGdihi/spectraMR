"""DTN2S — Dual-Traversal Noise2Self strategy.

Implements ``IMPLEMENTATION_SPEC.md`` §10 (Phase 4): self-supervised
denoising with the J-invariance condition enforced via a dual-traversal
mask. Wraps any sequence-aware denoising backbone (e.g. the existing
``NoiseToNoiseStrategy`` generator) so that it can train on a single
noisy volume — no paired clean reference required.

Pipeline (matches Spec §10.1)::

    noisy volume  →  φ_A.linearize  →  s_A
                  →  build_dtn2s_mask(φ_A, φ_B, recv)
                  →  zero out blocked positions in s_A
                  →  backbone(s_A_masked)  →  pred
                  →  loss = MSE(pred, s_B)          (full sequence: the mask
                                                    already guarantees
                                                    J-invariance everywhere)

The mask construction is delegated to
:func:`spectramr.models.blocks.dtn2s_mask.build_dtn2s_mask`.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model
from spectramr.config.schemas.training.strategy_knobs_2026_08 import (
    DTN2STrainingConfigSchema,
)
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.blocks.dtn2s_mask import build_dtn2s_mask, dual_traversal_pair
from spectramr.models.blocks.hilbert_order import HilbertOrder


class DTN2SStrategy(BaseTrainingStrategy):
    """Dual-Traversal Noise2Self self-supervised denoising (Spec §10).

    The model (``env.generator``) must accept a flat sequence
    ``(B, C, N)`` and return a same-shape prediction, and must declare a
    ``receptive_half_width`` no larger than the window (for example
    ``windowed_sequence_denoiser``): the mask certifies J-invariance only for a
    backbone that reads no further than the window, so any other backbone raises
    instead of being trained on a mask that does not cover it. The window is
    ``config.training.dtn2s.receptive_window`` (default 8 tokens).
    """

    @classmethod
    def standalone_infer_blocker(cls, config: object) -> str | None:
        return (
            "the DTN2S backbone consumes a space-filling-curve sequence, which only this "
            "strategy builds; `infer` would feed it the raw image."
        )

    def __init__(self, env=None, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        self._phi_a: HilbertOrder | None = None
        self._phi_b: HilbertOrder | None = None
        self._mask: torch.Tensor | None = None

    def _setup_strategy_specific_components(self) -> None:
        return

    def _resolve_traversals(self, x: torch.Tensor) -> tuple[HilbertOrder, HilbertOrder]:
        spatial = tuple(x.shape[-2:])
        if self._phi_a is None or tuple(self._phi_a.shape) != spatial:
            self._phi_a, self._phi_b = dual_traversal_pair(
                spatial,
                device=x.device,
            )
            self._mask = None  # re-derive on next call
        return self._phi_a, self._phi_b

    def _resolve_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Blocked positions in the **s_A sequence**, not voxel ids.

        ``build_dtn2s_mask`` returns a VOXEL-indexed mask, but ``s_a`` is
        ordered by ``φ_A``, so it must be re-indexed before use — reading it
        directly zeroed a set of positions unrelated to the blocked voxels
        (#1028). ``permutation[p]`` is the voxel at position ``p``, so
        ``blocked[permutation]`` is the same set expressed positionally.
        """
        phi_a, phi_b = self._resolve_traversals(x)
        recv = self._recv()
        if self._mask is None:
            blocked_voxels = build_dtn2s_mask(phi_a, phi_b, recv)
            self._mask = blocked_voxels[phi_a.permutation]
        return self._mask.to(x.device)

    def _recv(self) -> int:
        # Read the TYPED schema field (`DTN2STrainingConfigSchema`,
        # config/schemas/training/strategy_knobs_2026_08.py:355). This used to
        # be `getattr(_dtn2s_cfg, "receptive_window", 8)` on an untyped object,
        # under a comment claiming `dtn2s` was "not (yet) a schema field". The
        # field had landed and the consumer was never repointed, so every arm
        # silently ran 8 no matter what its YAML said (#376). The schema's own
        # docstring records why the old guard could not work: `hasattr` passed
        # under `extra="allow"` while `getattr` then ran against a plain dict.
        #
        # The block is optional (`dtn2s: ... | None = None`), so an arm that
        # declares nothing gets the SCHEMA's default by constructing it -- not a
        # literal repeated here. One owner for the default; a declared value is
        # never substituted, only an absent block is filled.
        dtn2s_cfg = self.config.training.dtn2s or DTN2STrainingConfigSchema()
        return int(dtn2s_cfg.receptive_window)

    def _require_j_invariant_backbone(self, generator: Any) -> None:
        """Raise unless the backbone declares a receptive field inside the mask's window."""
        declared = getattr(unwrap_model(generator), "receptive_half_width", None)
        recv = self._recv()
        if declared is None:
            raise ValueError(
                "DTN2S needs a backbone that declares `receptive_half_width`; "
                f"{type(generator).__name__} does not, so nothing says it reads no further than "
                f"the mask's window ({recv} tokens). Use model_type: windowed_sequence_denoiser."
            )
        if int(declared) > recv:
            raise ValueError(
                f"The backbone reads {int(declared)} tokens either side but the J-invariance mask "
                f"only covers training.dtn2s.receptive_window={recv}: outputs would see their own "
                "target. Narrow the backbone or widen the window."
            )

    def _validation_forward(
        self, input_batch: torch.Tensor, batch_context: dict[str, Any], **kwargs: Any
    ) -> torch.Tensor:
        """The J-invariant denoiser as an image: mask, read in phi_A order, write in phi_B order.

        Validation scores the same estimator training fits, so the blocked voxels are zeroed here
        too; an unmasked pass would let a voxel inform its own prediction.
        """
        del batch_context, kwargs
        generator = self.env.generator
        self._require_j_invariant_backbone(generator)
        phi_a, phi_b = self._resolve_traversals(input_batch)
        mask = self._resolve_mask(input_batch)
        s_a = phi_a.linearize(input_batch).clone()
        s_a[..., mask] = 0.0
        return phi_b.unflatten(generator(s_a), tuple(input_batch.shape[-2:]))

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute DTN2S J-invariant denoising loss.

        ``input_batch`` is the *only* required tensor (single noisy
        volume). ``target_batch`` is ignored — this is unsupervised.
        """
        del target_batch
        generator = self.env.generator
        self._require_j_invariant_backbone(generator)

        phi_a, phi_b = self._resolve_traversals(input_batch)
        mask = self._resolve_mask(input_batch)  # (N,) bool

        # 1) Linearise both traversals
        s_a = phi_a.linearize(input_batch)  # (B, C, N)
        s_b = phi_b.linearize(input_batch)  # (B, C, N) — target

        # 2) Zero blocked positions in s_a (J-invariance enforcement)
        s_a_masked = s_a.clone()
        s_a_masked[..., mask] = 0.0

        # 3) Backbone prediction on the SFC-ordered masked sequence
        pred = generator(s_a_masked)
        if pred.shape != s_a.shape:
            # If backbone returned spatial output, re-linearise
            pred = phi_a.linearize(pred)

        # 4) Loss over the FULL sequence, because J-invariance now holds at
        # every output position — which is the whole point of the dual
        # traversal. Output position i predicts voxel φ_B[i] while the model
        # reads input positions i ± recv, i.e. voxels φ_A[i ± recv]. Either:
        #   * the target voxel is BLOCKED  → its value is zeroed in s_a_masked;
        #   * or it is UNBLOCKED           → by construction of the mask its s_A
        #     position is further than `recv` from i, so it is outside the
        #     window anyway.
        # Restricting the loss to `mask` would instead throw away ~98% of the
        # signal (measured: 1.93% blocked at 64x64, recv=8) and produce NaN
        # whenever the mask is legitimately empty, since `.mean()` of zero
        # elements is NaN. The old `[..., mask]` restriction was equivalent to
        # this line only because the mask was all-True (#1028).
        loss = ((pred - s_b) ** 2).mean()

        return {"g_total_loss": loss, "g_dtn2s_mse": loss.detach()}


__all__ = ["DTN2SStrategy"]
